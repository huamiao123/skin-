import torch
import torch.nn as nn
import torch.nn.functional as F

from .position_encoding import build_2d_sincos_position_embedding
from .window_transformer import WindowTransformerBlock
from .transformer_block import TransformerBlock
from .edge_router import EdgeRouter
from .token_ops import (
    build_keep_mask, spatial_expand,
    gather_active_tokens, reconstruct_tokens
)
from .context_refresh import ContextRefresh2D


class EGEHRViTAdapter(nn.Module):
    def __init__(
        self,
        in_channels=24,
        dim=48,
        num_heads=4,
        window_size=8,
        total_depth=12,
        halt_after=3,
        mlp_ratio=2.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        no_halting=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.halt_after = halt_after
        self.total_depth = total_depth
        self.no_halting = no_halting
        deep_depth = total_depth - halt_after

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

        self.shallow_blocks = nn.ModuleList([
            WindowTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path,
            )
            for _ in range(halt_after)
        ])

        self.router = EdgeRouter(dim=dim)

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
        if 5 < deep_depth:
            self.context1 = ContextRefresh2D(dim=dim, num_heads=num_heads)
        if 8 < deep_depth:
            self.context2 = ContextRefresh2D(dim=dim, num_heads=num_heads)

        self.proj_out = nn.Sequential(
            nn.Conv2d(dim, in_channels, kernel_size=1),
            nn.GroupNorm(min(4, in_channels), in_channels),
            nn.GELU(),
        )

        self.gamma = nn.Parameter(torch.tensor(0.1))

        self.aux_head = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1),
        )

        self._pos_embed = None
        self._init_weights()

        self.router_call_count = 0
        self.reconstruct_call_count = 0
        self._token_log_done = False

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

    def forward(self, t3, threshold=0.5, min_keep_ratio=0.15, keep_all=False):
        B, C_in, H, W = t3.shape

        if self.no_halting:
            keep_all = True

        x = self.proj_in(t3)
        x = x.flatten(2).transpose(1, 2)

        expected_tokens = H * W

        pos_embed = self._get_pos_embed(H, W, x.device)
        x = x + pos_embed

        for blk in self.shallow_blocks:
            x = blk(x, H, W)

        assert x.shape[1] == expected_tokens, (
            f"Token count changed after shallow blocks: "
            f"{x.shape[1]} != {expected_tokens}"
        )

        memory = x.clone()

        aux_logits_flat = None
        aux_feat = x.transpose(1, 2).reshape(B, self.dim, H, W)
        aux_logits = self.aux_head(aux_feat)

        if keep_all:
            for i, blk in enumerate(self.deep_blocks):
                x = blk(x)
                assert x.shape[1] == expected_tokens, (
                    f"Token count changed after deep block {i + 1}: "
                    f"{x.shape[1]} != {expected_tokens}"
                )
            x = x + memory
            assert x.shape[1] == expected_tokens
        else:
            self.router_call_count += 1

            router_logits, router_prob = self.router(x)

            keep_mask = build_keep_mask(
                router_prob,
                threshold=threshold,
                min_keep_ratio=min_keep_ratio,
            )
            keep_mask = spatial_expand(keep_mask, H, W)

            x_active, active_idx, padding_mask = gather_active_tokens(x, keep_mask)

            for i, blk in enumerate(self.deep_blocks):
                x_active = blk(x_active, padding_mask)

                if self.context1 is not None and i == 2:
                    x_active = self.context1(
                        query=x_active, memory=memory, padding_mask=padding_mask
                    )
                if self.context2 is not None and i == 5:
                    x_active = self.context2(
                        query=x_active, memory=memory, padding_mask=padding_mask
                    )

            self.reconstruct_call_count += 1
            x = reconstruct_tokens(memory, x_active, active_idx, padding_mask)

        if self.no_halting and not self._token_log_done:
            print(
                f"[EGEHRViTAdapter NoHalting] Stage3: [{B}, {C_in}, {H}, {W}], "
                f"tokens: {expected_tokens}, "
                f"after shallow({self.halt_after})={expected_tokens}, "
                f"after deep({len(self.deep_blocks)})={expected_tokens}, "
                f"router_called={self.router_call_count}, "
                f"reconstruct_called={self.reconstruct_call_count}"
            )
            self._token_log_done = True

        x = x.transpose(1, 2).reshape(B, self.dim, H, W)
        transformer_feature = self.proj_out(x)

        output = t3 + self.gamma * transformer_feature

        stats = {}
        if not keep_all:
            stats["router_logits"] = router_logits
            stats["router_prob"] = router_prob
            stats["keep_mask"] = keep_mask
            stats["retention_ratio"] = keep_mask.float().mean()
        stats["aux_logits"] = aux_logits

        return output, stats
