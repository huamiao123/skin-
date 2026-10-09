import torch
import torch.nn as nn

from .patch_expand import PatchExpand, FinalPatchExpand_X4
from .encoder import TransformerStage


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dims=(64, 128, 256, 512),
                 decoder_depths=(2, 2, 2),
                 decoder_heads=(8, 4, 2),
                 decoder_sr_ratios=(1, 2, 4),
                 mlp_ratio=4.0, head_dim=32,
                 drop=0.0, attn_drop=0.0, drop_path_rate=0.1,
                 num_classes=1):
        super().__init__()
        self.embed_dims = embed_dims
        n_up = len(decoder_depths)
        self.expands = nn.ModuleList()
        self.fuses = nn.ModuleList()
        self.stages = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(decoder_depths))]
        cur = 0
        for j in range(n_up):
            up_dim = embed_dims[3 - j]
            out_dim = embed_dims[2 - j]
            self.expands.append(PatchExpand(up_dim, dim_scale=2))
            self.fuses.append(nn.Linear(out_dim * 2, out_dim))
            self.stages.append(TransformerStage(
                dim=out_dim, depth=decoder_depths[j],
                num_heads=decoder_heads[j], sr_ratio=decoder_sr_ratios[j],
                mlp_ratio=mlp_ratio, head_dim=head_dim,
                drop=drop, attn_drop=attn_drop,
                drop_path=dpr[cur:cur + decoder_depths[j]]))
            cur += decoder_depths[j]

        self.norm = nn.LayerNorm(embed_dims[0], eps=1e-6)
        self.final_expand = FinalPatchExpand_X4(embed_dims[0], dim_scale=4)
        self.head = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)

    def forward(self, feats):
        x = feats[3][0]
        H, W = feats[3][1], feats[3][2]
        for j in range(len(self.stages)):
            x, H, W = self.expands[j](x, H, W)
            x = torch.cat([x, feats[2 - j][0]], dim=-1)
            x = self.fuses[j](x)
            x = self.stages[j](x, H, W)

        x = self.norm(x)
        x, H, W = self.final_expand(x, H, W)
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        out = self.head(x)
        return out
