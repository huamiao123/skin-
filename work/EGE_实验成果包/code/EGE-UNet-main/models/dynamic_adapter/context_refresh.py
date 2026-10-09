import torch
import torch.nn as nn


class ContextRefresh2D(nn.Module):
    def __init__(self, dim=48, num_heads=4, use_deformable=False):
        super().__init__()
        self.use_deformable = use_deformable

        self.norm_query = nn.LayerNorm(dim, eps=1e-6)
        self.norm_memory = nn.LayerNorm(dim, eps=1e-6)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm_out = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, query, memory, padding_mask=None):
        residual = query

        q = self.norm_query(query)
        kv = self.norm_memory(memory)

        attn_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
        )

        query = residual + attn_out
        query = query + self.ffn(self.norm_out(query))
        return query
