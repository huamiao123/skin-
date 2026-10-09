import torch
import torch.nn as nn


class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim, eps=1e-6):
        super().__init__()
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim, eps=eps)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2
