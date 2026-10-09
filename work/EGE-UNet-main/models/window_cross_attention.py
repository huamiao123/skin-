import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """
    Args:
        x: [B, C, H, W]
        window_size: int

    Returns:
        [B * num_windows, window_size * window_size, C]
    """
    B, C, H, W = x.shape

    assert H % window_size == 0
    assert W % window_size == 0

    x = x.permute(0, 2, 3, 1).contiguous()

    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C
    )

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    windows = x.view(
        -1,
        window_size * window_size,
        C
    )

    return windows


def window_reverse(
    windows,
    window_size,
    B,
    H,
    W,
    C
):
    """
    Args:
        windows:
            [B * num_windows, window_size * window_size, C]

    Returns:
        x:
            [B, C, H, W]
    """
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        C
    )

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, C)

    return x.permute(0, 3, 1, 2).contiguous()


class WindowCrossAttention(nn.Module):
    def __init__(
        self,
        dim=24,
        num_heads=4,
        window_size=8,
        mlp_ratio=2.0,
        dropout=0.0
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.window_size = window_size

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        hidden_dim = int(dim * mlp_ratio)

        self.norm_ffn = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.q_pos = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim
        )

        self.kv_pos = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim
        )

        self.out_pos = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim
        )

    def forward(
        self,
        q_map,
        kv_map
    ):
        B, C, H, W = q_map.shape

        assert kv_map.shape == q_map.shape

        q_map = q_map + self.q_pos(q_map)
        kv_map = kv_map + self.kv_pos(kv_map)

        q_tokens = window_partition(
            q_map,
            self.window_size
        )

        kv_tokens = window_partition(
            kv_map,
            self.window_size
        )

        q_norm = self.norm_q(q_tokens)
        kv_norm = self.norm_kv(kv_tokens)

        attn_out, _ = self.attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            need_weights=False
        )

        x = q_tokens + attn_out

        x = x + self.ffn(
            self.norm_ffn(x)
        )

        x = window_reverse(
            x,
            self.window_size,
            B,
            H,
            W,
            C
        )

        x = x + self.out_pos(x)

        return x
