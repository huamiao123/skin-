import sys
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

from models.egeunet import (
    Grouped_multi_axis_Hadamard_Product_Attention,
    group_aggregation_bridge,
)
from timm.models.layers import trunc_normal_

workspace_root = str(Path(__file__).resolve().parents[2])
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
from pure_transformer_unet.models.encoder import TransformerEncoder


class FusionInject(nn.Module):
    """Protected residual fusion: F' = F_cnn + gamma * proj(F_T)

    gamma mode:
      - scalar: one learnable scalar per scale (default, EGE-Dual baseline)
      - scale_cond: per-sample gamma conditioned on pooled CNN feature
        (descriptor detached to keep loss decoupling). Gate weights
        zero-init with bias=logit(0.1) so it starts exactly as the
        scalar version, then learns to condition on input.
    """
    def __init__(self, t_dim, c_dim, gamma_init=0.1, scale_conditioned=False):
        super().__init__()
        self.scale_conditioned = scale_conditioned
        self.proj = nn.Conv2d(t_dim, c_dim, kernel_size=1, bias=True)
        init_logit = math.log(gamma_init / (1.0 - gamma_init))
        if scale_conditioned:
            hidden = max(c_dim // 4, 8)
            self.gate_fc = nn.Sequential(
                nn.Linear(c_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.gate_fc[-1].weight)
            nn.init.constant_(self.gate_fc[-1].bias, init_logit)
        else:
            self.gamma_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    def forward(self, f_cnn, tok, *, gamma_override=None, injection_mask=None):
        B, N, C = tok.shape
        H, W = f_cnn.shape[2], f_cnn.shape[3]
        if N != H * W:
            raise ValueError(f'token count {N} does not match feature grid {H}x{W}')
        tok = tok.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        proj = self.proj(tok)
        if self.scale_conditioned:
            desc = F.adaptive_avg_pool2d(f_cnn.detach(), 1).flatten(1)
            gamma_native = torch.sigmoid(self.gate_fc(desc)).view(B, 1, 1, 1)
        else:
            gamma_native = torch.sigmoid(self.gamma_logit).reshape(1, 1, 1, 1).expand(B, 1, 1, 1)
        gamma = gamma_native if gamma_override is None else gamma_override
        mask = torch.ones_like(gamma) if injection_mask is None else injection_mask
        effective = gamma * mask
        delta = effective * proj
        ratio = (
            delta.detach().flatten(1).norm(dim=1)
            / (f_cnn.detach().flatten(1).norm(dim=1) + 1e-6)
        )
        return f_cnn + delta, gamma_native, effective, ratio


class EGEDualUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3,
                 c_list=(8, 16, 24, 32, 48, 64),
                 bridge=True, gt_ds=True,
                 t_embed=48, t_depths=(2, 2, 2, 2),
                 t_head_dim=16, t_sr_ratios=(4, 2, 1, 1),
                 t_mlp_ratio=4.0, t_drop_path_rate=0.1,
                 fusion_type='scalar'):
        super().__init__()
        self.bridge = bridge
        self.gt_ds = gt_ds

        # ---------- CNN branch (identical to EGEUNet encoder) ----------
        self.encoder1 = nn.Sequential(nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1))
        self.encoder2 = nn.Sequential(nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1))
        self.encoder3 = nn.Sequential(nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1))
        self.encoder4 = nn.Sequential(Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[3]))
        self.encoder5 = nn.Sequential(Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[4]))
        self.encoder6 = nn.Sequential(Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[5]))
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        # ---------- Transformer branch (parallel, lightweight PTU encoder) ----------
        t_dims = [t_embed * (2 ** i) for i in range(4)]
        t_heads = [d // t_head_dim for d in t_dims]
        self.t_encoder = TransformerEncoder(
            in_chans=input_channels, patch_size=4,
            embed_dims=t_dims, depths=list(t_depths), num_heads=t_heads,
            sr_ratios=list(t_sr_ratios), mlp_ratio=t_mlp_ratio,
            head_dim=t_head_dim, drop=0.0, attn_drop=0.0,
            drop_path_rate=t_drop_path_rate)

        # ---------- protected-residual fusion at 4 scales ----------
        sc = (fusion_type == 'scale_cond')
        self.fuse2 = FusionInject(t_dims[0], c_list[1], scale_conditioned=sc)
        self.fuse3 = FusionInject(t_dims[1], c_list[2], scale_conditioned=sc)
        self.fuse4 = FusionInject(t_dims[2], c_list[3], scale_conditioned=sc)
        self.fuse5 = FusionInject(t_dims[3], c_list[4], scale_conditioned=sc)
        self._gamma_stats = {}

        # ---------- decoder (identical to EGEUNet) ----------
        if bridge:
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
        self.decoder1 = nn.Sequential(Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[4]))
        self.decoder2 = nn.Sequential(Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[3]))
        self.decoder3 = nn.Sequential(Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[2]))
        self.decoder4 = nn.Sequential(nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1))
        self.decoder5 = nn.Sequential(nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1))
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)
        for m in [self.fuse2, self.fuse3, self.fuse4, self.fuse5]:
            if m.scale_conditioned:
                nn.init.zeros_(m.gate_fc[-1].weight)
                nn.init.constant_(m.gate_fc[-1].bias,
                                  math.log(0.1 / (1.0 - 0.1)))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod
    def _validate_control(value, name, x, *, binary=False):
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=x.device, dtype=x.dtype)
        else:
            value = value.to(device=x.device, dtype=x.dtype)
        if tuple(value.shape) != (x.shape[0], 4):
            raise ValueError(f'{name} must have shape [B,4], got {tuple(value.shape)}')
        if not torch.isfinite(value).all() or torch.any(value < 0) or torch.any(value > 1):
            raise ValueError(f'{name} must contain finite values in [0,1]')
        if binary and not torch.all((value == 0) | (value == 1)):
            raise ValueError('fusion_mask must contain only 0 or 1')
        return value

    def forward(self, x, *, fusion_override=None, fusion_mask=None, return_aux=False):
        if not self.bridge or not self.gt_ds or tuple(x.shape[-2:]) != (256, 256):
            raise ValueError('research fusion controls support bridge=True, gt_ds=True, 256x256 only')
        fusion_override = self._validate_control(fusion_override, 'fusion_override', x)
        fusion_mask = self._validate_control(fusion_mask, 'fusion_mask', x, binary=True)

        # parallel transformer branch
        t_feats = self.t_encoder(x)

        # CNN main branch with protected-residual injection
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        descriptor = torch.cat([
            out.detach().mean(dim=(2, 3)),
            out.detach().std(dim=(2, 3), unbiased=False),
        ], dim=1)
        controls = []
        for index in range(4):
            override = None if fusion_override is None else fusion_override[:, index].view(-1, 1, 1, 1)
            mask = None if fusion_mask is None else fusion_mask[:, index].view(-1, 1, 1, 1)
            controls.append((override, mask))
        t2, g2, ge2, r2 = self.fuse2(
            out, t_feats[0][0], gamma_override=controls[0][0], injection_mask=controls[0][1]
        )

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(t2)), 2, 2))
        t3, g3, ge3, r3 = self.fuse3(
            out, t_feats[1][0], gamma_override=controls[1][0], injection_mask=controls[1][1]
        )

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(t3)), 2, 2))
        t4, g4, ge4, r4 = self.fuse4(
            out, t_feats[2][0], gamma_override=controls[2][0], injection_mask=controls[2][1]
        )

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(t4)), 2, 2))
        t5, g5, ge5, r5 = self.fuse5(
            out, t_feats[3][0], gamma_override=controls[3][0], injection_mask=controls[3][1]
        )
        self._gamma_stats = {'g2': g2, 'g3': g3, 'g4': g4, 'g5': g5}

        out = F.gelu(self.encoder6(t5))
        t6 = out

        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        if self.gt_ds:
            gt_pre5 = self.gt_conv1(out5)
            t5 = self.GAB5(t6, t5, gt_pre5)
            gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode='bilinear', align_corners=True)
        else:
            t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5)

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        if self.gt_ds:
            gt_pre4 = self.gt_conv2(out4)
            t4 = self.GAB4(t5, t4, gt_pre4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)
        else:
            t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4)

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        if self.gt_ds:
            gt_pre3 = self.gt_conv3(out3)
            t3 = self.GAB3(t4, t3, gt_pre3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)
        else:
            t3 = self.GAB3(t4, t3)
        out3 = torch.add(out3, t3)

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        if self.gt_ds:
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.GAB2(t3, t2, gt_pre2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)
        else:
            t2 = self.GAB2(t3, t2)
        out2 = torch.add(out2, t2)

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        if self.gt_ds:
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.GAB1(t2, t1, gt_pre1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1)

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear', align_corners=True)

        if self.gt_ds:
            legacy_output = (
                (
                torch.sigmoid(gt_pre5),
                torch.sigmoid(gt_pre4),
                torch.sigmoid(gt_pre3),
                torch.sigmoid(gt_pre2),
                torch.sigmoid(gt_pre1),
                ),
                torch.sigmoid(out0),
            )
        else:
            legacy_output = torch.sigmoid(out0)
        if not return_aux:
            return legacy_output
        return {
            'legacy_output': legacy_output,
            'research_aux': {
                'gamma_native': torch.cat([g2, g3, g4, g5], dim=1).flatten(1).detach(),
                'gamma_effective': torch.cat([ge2, ge3, ge4, ge5], dim=1).flatten(1).detach(),
                'injection_ratio': torch.stack([r2, r3, r4, r5], dim=1).detach(),
                'descriptor': descriptor.detach(),
            },
        }
