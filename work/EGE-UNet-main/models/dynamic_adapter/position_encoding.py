import torch
import torch.nn as nn


def build_2d_sincos_position_embedding(h, w, embed_dim, num_tokens=0, temperature=10000.):
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
    grid_w, grid_h = grid_w.flatten(), grid_h.flatten()

    assert embed_dim % 4 == 0
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)

    out_w = torch.einsum('m,d->md', [grid_w, omega])
    out_h = torch.einsum('m,d->md', [grid_h, omega])

    pos_emb = torch.cat([
        torch.sin(out_w), torch.cos(out_w),
        torch.sin(out_h), torch.cos(out_h)
    ], dim=1)[None, :, :]

    pe_token = torch.zeros([1, num_tokens, embed_dim], dtype=torch.float32)
    pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    pos_embed.requires_grad = False
    return pos_embed
