import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W
