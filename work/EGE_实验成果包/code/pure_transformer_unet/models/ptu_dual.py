import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embed import PatchEmbed
from .patch_merging import PatchMerging
from .encoder import TransformerStage
from .decoder import TransformerDecoder


class FusionInject(nn.Module):
    """Protected residual fusion: F' = F_main + sigmoid(gamma_logit) * proj(F_aux)

    f_main: token tensor (B, N, C)
    f_aux: spatial tensor (B, c, H, W)
    """
    def __init__(self, aux_dim, main_dim, gamma_init=0.1):
        super().__init__()
        self.proj = nn.Conv2d(aux_dim, main_dim, kernel_size=1, bias=True)
        init_logit = math.log(gamma_init / (1.0 - gamma_init))
        self.gamma_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    def forward(self, f_main, f_aux):
        proj = self.proj(f_aux).flatten(2).transpose(1, 2)
        gamma = torch.sigmoid(self.gamma_logit)
        return f_main + gamma * proj, gamma


class ConvStage(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBranch(nn.Module):
    """Pure convolution auxiliary branch, no attention.

    Output scales: 64x64x32 -> 32x32x64 -> 16x16x128 -> 8x8x256
    """
    def __init__(self, in_chans=3, channels=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, channels[0], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, channels[0]),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.stages = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.stages.append(ConvStage(channels[i], channels[i + 1]))

    def forward(self, x):
        feats = []
        x = self.stem(x)            # 64x64
        feats.append(x)
        for s in self.stages:
            x = s(x)
            feats.append(x)         # 32x32, 16x16, 8x8
        return feats


class PTUDualUNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1, patch_size=4,
                 embed_dims=(64, 128, 256, 512),
                 depths=(2, 2, 4, 2),
                 num_heads=(2, 4, 8, 16),
                 sr_ratios=(4, 2, 1, 1),
                 decoder_depths=(2, 2, 2),
                 decoder_heads=(8, 4, 2),
                 decoder_sr_ratios=(1, 2, 4),
                 mlp_ratio=4.0, head_dim=32,
                 drop=0.0, attn_drop=0.0, drop_path_rate=0.1,
                 cnn_channels=(32, 64, 128, 256)):
        super().__init__()
        self.embed_dims = embed_dims

        # ---------- Transformer main path (PTU encoder) ----------
        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      patch_size=patch_size)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.t_stages = nn.ModuleList()
        self.t_merges = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            self.t_stages.append(TransformerStage(
                dim=embed_dims[i], depth=depths[i], num_heads=num_heads[i],
                sr_ratio=sr_ratios[i], mlp_ratio=mlp_ratio, head_dim=head_dim,
                drop=drop, attn_drop=attn_drop,
                drop_path=dpr[cur:cur + depths[i]]))
            cur += depths[i]
            if i < len(depths) - 1:
                self.t_merges.append(PatchMerging(embed_dims[i], embed_dims[i + 1]))

        # ---------- CNN auxiliary branch ----------
        self.cnn_branch = ConvBranch(in_chans=in_chans, channels=cnn_channels)

        # ---------- protected-residual fusion at 4 scales ----------
        self.fuse = nn.ModuleList([
            FusionInject(cnn_channels[i], embed_dims[i]) for i in range(4)
        ])
        self._gamma_stats = {}

        # ---------- decoder (unchanged PTU decoder) ----------
        self.decoder = TransformerDecoder(
            embed_dims=embed_dims, decoder_depths=decoder_depths,
            decoder_heads=decoder_heads, decoder_sr_ratios=decoder_sr_ratios,
            mlp_ratio=mlp_ratio, head_dim=head_dim,
            drop=drop, attn_drop=attn_drop, drop_path_rate=drop_path_rate,
            num_classes=num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # CNN auxiliary branch (parallel, from raw input)
        c_feats = self.cnn_branch(x)

        # Transformer main path with fusion injection
        x, H, W = self.patch_embed(x)
        feats = []
        gamma_stats = {}
        for i, stage in enumerate(self.t_stages):
            x = stage(x, H, W)
            x, g = self.fuse[i](x, c_feats[i])
            gamma_stats[f'g{i+1}'] = g
            feats.append((x, H, W))
            if i < len(self.t_merges):
                x, H, W = self.t_merges[i](x, H, W)
        self._gamma_stats = gamma_stats

        out = self.decoder(feats)
        return out
