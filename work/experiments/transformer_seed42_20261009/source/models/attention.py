import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, head_dim=32,
                 qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, self.inner_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_sp = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            x_sr = self.sr(x_sp).reshape(B, C, -1).permute(0, 2, 1)
            x_sr = self.norm(x_sr)
            kv = self.kv(x_sr)
        else:
            kv = self.kv(x)

        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.inner_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def flops(self, H, W, q_len, kv_len):
        flops = 0
        flops += q_len * self.inner_dim * self.dim
        flops += kv_len * self.inner_dim * self.dim * 2
        flops += q_len * kv_len * self.inner_dim * 2
        flops += q_len * self.inner_dim * self.dim
        return flops
