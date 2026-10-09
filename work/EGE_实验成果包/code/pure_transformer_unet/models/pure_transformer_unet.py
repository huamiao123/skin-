import math
import torch
import torch.nn as nn

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class PureTransformerUNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1, patch_size=4,
                 embed_dims=(64, 128, 256, 512),
                 depths=(2, 2, 4, 2),
                 num_heads=(2, 4, 8, 16),
                 sr_ratios=(4, 2, 1, 1),
                 decoder_depths=(2, 2, 2),
                 decoder_heads=(8, 4, 2),
                 decoder_sr_ratios=(1, 2, 4),
                 mlp_ratio=4.0, head_dim=32,
                 drop=0.0, attn_drop=0.0, drop_path_rate=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(
            in_chans=in_chans, patch_size=patch_size,
            embed_dims=embed_dims, depths=depths, num_heads=num_heads,
            sr_ratios=sr_ratios, mlp_ratio=mlp_ratio, head_dim=head_dim,
            drop=drop, attn_drop=attn_drop, drop_path_rate=drop_path_rate)
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
        feats = self.encoder(x)
        out = self.decoder(feats)
        return out
