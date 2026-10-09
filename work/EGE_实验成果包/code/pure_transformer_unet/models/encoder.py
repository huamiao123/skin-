import torch
import torch.nn as nn

from .patch_embed import PatchEmbed
from .patch_merging import PatchMerging
from .transformer_block import TransformerBlock


class TransformerStage(nn.Module):
    def __init__(self, dim, depth, num_heads, sr_ratio, mlp_ratio=4.0,
                 head_dim=32, drop=0.0, attn_drop=0.0,
                 drop_path=0.0):
        super().__init__()
        if isinstance(drop_path, (float, int)):
            drop_path = [float(drop_path)] * depth
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim, num_heads=num_heads, sr_ratio=sr_ratio,
                mlp_ratio=mlp_ratio, head_dim=head_dim,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path[i])
            for i in range(depth)
        ])

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, in_chans=3, patch_size=4,
                 embed_dims=(64, 128, 256, 512),
                 depths=(2, 2, 4, 2),
                 num_heads=(2, 4, 8, 16),
                 sr_ratios=(4, 2, 1, 1),
                 mlp_ratio=4.0, head_dim=32,
                 drop=0.0, attn_drop=0.0, drop_path_rate=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      patch_size=patch_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            self.stages.append(TransformerStage(
                dim=embed_dims[i], depth=depths[i], num_heads=num_heads[i],
                sr_ratio=sr_ratios[i], mlp_ratio=mlp_ratio, head_dim=head_dim,
                drop=drop, attn_drop=attn_drop,
                drop_path=dpr[cur:cur + depths[i]]))
            cur += depths[i]
            if i < len(depths) - 1:
                self.merges.append(PatchMerging(embed_dims[i], embed_dims[i + 1]))

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x, H, W)
            feats.append((x, H, W))
            if i < len(self.merges):
                x, H, W = self.merges[i](x, H, W)
        return feats
