import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_adapter.position_encoding import build_2d_sincos_position_embedding
from .dynamic_adapter.window_transformer import WindowTransformerBlock
from .dynamic_adapter.transformer_block import TransformerBlock
from .dynamic_adapter.context_refresh import ContextRefresh2D


class ProtectedResidualHaltingAdapter(nn.Module):
    def __init__(
        self,
        in_channels=24,
        dim=48,
        num_heads=4,
        window_size=8,
        total_depth=12,
        shallow_blocks=3,
        final_keep_ratio=0.5,
        mlp_ratio=2.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.shallow_blocks = shallow_blocks
        self.total_depth = total_depth
        self.final_keep_ratio = final_keep_ratio
        deep_depth = total_depth - shallow_blocks

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

        self.shallow = nn.ModuleList([
            WindowTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path,
            )
            for _ in range(shallow_blocks)
        ])

        self.router = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

        self.deep_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            )
            for _ in range(deep_depth)
        ])

        refresh_positions = [5, 8]
        self.context1 = None
        self.context2 = None
        if refresh_positions[0] < total_depth:
            self.context1 = ContextRefresh2D(dim=dim, num_heads=num_heads)
        if refresh_positions[1] < total_depth:
            self.context2 = ContextRefresh2D(dim=dim, num_heads=num_heads)

        self.proj_out = nn.Sequential(
            nn.Conv2d(dim, in_channels, kernel_size=1),
            nn.GroupNorm(min(4, in_channels), in_channels),
            nn.GELU(),
        )

        self.gamma_logit = nn.Parameter(torch.tensor(-2.1972246))

        self._pos_embed = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_pos_embed(self, H, W, device):
        if self._pos_embed is None or self._pos_embed.shape[1] != H * W or self._pos_embed.device != device:
            self._pos_embed = build_2d_sincos_position_embedding(
                H, W, self.dim, num_tokens=0
            ).to(device)
        return self._pos_embed

    def get_retention_ratio(self, epoch):
        if not self.training:
            return self.final_keep_ratio
        if epoch is None:
            raise ValueError("epoch must be provided during PRH training")
        if epoch < 20:
            return 1.0
        elif epoch < 40:
            return 1.0 - 0.5 * ((epoch - 20) / 20.0)
        else:
            return self.final_keep_ratio

    def forward(self, t3, epoch=None):
        B, C_in, H, W = t3.shape
        N = H * W
        D = self.dim

        x = self.proj_in(t3)
        x = x.flatten(2).transpose(1, 2)

        pos_embed = self._get_pos_embed(H, W, x.device)
        x = x + pos_embed

        for blk in self.shallow:
            x = blk(x, H, W)

        base = x
        assert base.shape == (B, N, D)

        router_input = base.detach()
        edge_logits = self.router(router_input).squeeze(-1)
        edge_prob = torch.sigmoid(edge_logits)

        rho = self.get_retention_ratio(epoch)
        K = max(1, min(N, int(round(N * rho))))

        active_idx = torch.topk(
            edge_prob,
            k=K,
            dim=1,
            largest=True,
            sorted=False,
        ).indices

        idx_expand = active_idx.unsqueeze(-1).expand(-1, -1, D)

        active_base = torch.gather(base, dim=1, index=idx_expand)
        active_prob = torch.gather(
            edge_prob, dim=1, index=active_idx
        ).unsqueeze(-1)

        active = active_base
        for i, blk in enumerate(self.deep_blocks):
            active = blk(active)

            if self.context1 is not None and i == 2:
                active = self.context1(
                    query=active, memory=base
                )
            if self.context2 is not None and i == 5:
                active = self.context2(
                    query=active, memory=base
                )

        deep = active

        delta_active = deep - active_base
        correction_active = active_prob * delta_active

        correction = torch.zeros_like(base)
        correction = correction.scatter(
            dim=1,
            index=idx_expand,
            src=correction_active,
        )

        gamma = torch.sigmoid(self.gamma_logit)
        out = base + gamma * correction

        out = out.transpose(1, 2).reshape(B, D, H, W)
        out = self.proj_out(out)
        assert out.shape == (B, C_in, H, W)

        return {
            "feature": out,
            "edge_logits": edge_logits,
            "edge_prob": edge_prob,
            "active_idx": active_idx,
            "retention_ratio": rho,
            "gamma": gamma,
        }
