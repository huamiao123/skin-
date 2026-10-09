import torch
import torch.nn as nn

from einops import rearrange


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // dim_scale, eps=eps)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, 2 * C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C // 2)
        x = x.view(B, -1, C // 2)
        x = self.norm(x)
        return x, H * 2, W * 2


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, (dim_scale ** 2) * dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, 16 * C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=4, p2=4, c=C)
        x = x.view(B, -1, C)
        x = self.norm(x)
        return x, H * 4, W * 4
