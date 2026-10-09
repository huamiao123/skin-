import torch
import torch.nn as nn
import torch.nn.functional as F

from models.egeunet import EGEWaveUNet
from models.wavelet_adapter import WaveletGlobalAdapter, HaarDWT2D, HaarIDWT2D, ConvPosEncoding, LFTransformerBlock, SpatialWaveGate


class WaveletGlobalAdapterWithLL(WaveletGlobalAdapter):
    def forward(self, x, return_ll=False):
        orig_h, orig_w = x.shape[-2:]

        pad_h = orig_h % 2
        pad_w = orig_w % 2
        if pad_h or pad_w:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_pad = x

        ll, lh, hl, hh = self.dwt(x_pad)
        ll_t = self.pos(ll)

        B, C, H, W = ll_t.shape
        z = ll_t.flatten(2).transpose(1, 2)
        for block in self.blocks:
            z = block(z)
        ll_feat = z.transpose(1, 2).reshape(B, C, H, W)

        x_rec = self.idwt(ll_feat, lh, hl, hh)
        x_rec = x_rec[..., :orig_h, :orig_w]

        if self.fusion_type == 'scalar':
            alpha = torch.sigmoid(self.alpha_logit)
            out = x + alpha * (x_rec - x)
            stats = {'alpha': alpha}
        elif self.fusion_type == 'spatial_gate':
            out, gate = self.spatial_gate(x, x_rec)
            stats = {}

        if return_ll:
            return out, stats, ll_feat
        return out, stats


class EGEWaveWeakUNet(EGEWaveUNet):
    def __init__(self, num_classes=1, input_channels=3, c_list=None,
                 bridge=True, gt_ds=True,
                 adapter_dim=None, adapter_heads=4,
                 adapter_depth=1, adapter_mlp_ratio=2.0):
        if c_list is None:
            c_list = [8, 16, 24, 32, 48, 64]
        nn.Module.__init__(self)
        self.bridge = bridge
        self.gt_ds = gt_ds

        self.encoder1 = nn.Sequential(nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1))
        self.encoder2 = nn.Sequential(nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1))
        self.encoder3 = nn.Sequential(nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1))

        from models.egeunet import Grouped_multi_axis_Hadamard_Product_Attention
        self.encoder4 = nn.Sequential(Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[3]))
        self.encoder5 = nn.Sequential(Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[4]))
        self.encoder6 = nn.Sequential(Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[5]))

        from models.egeunet import group_aggregation_bridge
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

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        dim = adapter_dim if adapter_dim is not None else c_list[2]
        self.wave_adapter = WaveletGlobalAdapterWithLL(
            dim=dim,
            num_heads=adapter_heads,
            mlp_ratio=adapter_mlp_ratio,
            depth=adapter_depth,
            alpha_init=0.1,
            fusion_type='spatial_gate',
        )

        self.ll_aux_head = nn.Conv2d(dim, 1, kernel_size=1)
        self.local_aux_head = nn.Conv2d(c_list[2], 1, kernel_size=1)

        from timm.models.layers import trunc_normal_
        import math
        for m in [self.ll_aux_head, self.local_aux_head]:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, math.sqrt(2.0 / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, return_aux=False):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        t3_skip = t3
        t3_deep, wave_stats, ll_feat = self.wave_adapter(t3, return_ll=True)

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(t3_deep)), 2, 2))
        t4 = out

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out

        out = F.gelu(self.encoder6(out))
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
            t3_skip = self.GAB3(t4, t3_skip, gt_pre3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)
        else:
            t3_skip = self.GAB3(t4, t3_skip)
        out3 = torch.add(out3, t3_skip)

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        if self.gt_ds:
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.GAB2(t3_skip, t2, gt_pre2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)
        else:
            t2 = self.GAB2(t3_skip, t2)
        out2 = torch.add(out2, t2)

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        if self.gt_ds:
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.GAB1(t2, t1, gt_pre1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1)

        final_logits = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear', align_corners=True)

        if not return_aux:
            if self.gt_ds:
                return (
                    torch.sigmoid(gt_pre5),
                    torch.sigmoid(gt_pre4),
                    torch.sigmoid(gt_pre3),
                    torch.sigmoid(gt_pre2),
                    torch.sigmoid(gt_pre1),
                ), torch.sigmoid(final_logits)
            else:
                return torch.sigmoid(final_logits)

        ll_logits = self.ll_aux_head(ll_feat)
        local_logits = self.local_aux_head(t3)

        ll_logits_up = F.interpolate(ll_logits, size=final_logits.shape[-2:], mode='bilinear', align_corners=False)
        local_logits_up = F.interpolate(local_logits, size=final_logits.shape[-2:], mode='bilinear', align_corners=False)

        if self.gt_ds:
            deep_sup = (
                torch.sigmoid(gt_pre5),
                torch.sigmoid(gt_pre4),
                torch.sigmoid(gt_pre3),
                torch.sigmoid(gt_pre2),
                torch.sigmoid(gt_pre1),
            )
            return {
                "final": final_logits,
                "final_prob": torch.sigmoid(final_logits),
                "ll": ll_logits_up,
                "local": local_logits_up,
                "deep_supervision": deep_sup,
            }
        else:
            return {
                "final": final_logits,
                "final_prob": torch.sigmoid(final_logits),
                "ll": ll_logits_up,
                "local": local_logits_up,
            }
