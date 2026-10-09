import torch
import torch.nn as nn
import torch.nn.functional as F

from models.window_cross_attention import WindowCrossAttention
from models.egeunet import group_aggregation_bridge


class BoundaryGuidedCrossTransformerBridge(nn.Module):
    def __init__(
        self,
        dim_high=32,
        dim_low=24,
        window_size=8,
        num_heads=4,
    ):
        super().__init__()

        self.local_gab = group_aggregation_bridge(
            dim_high,
            dim_low
        )

        self.high_project = nn.Conv2d(
            dim_high,
            dim_low,
            kernel_size=1
        )

        self.transformer = WindowCrossAttention(
            dim=dim_low,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=2.0,
            dropout=0.0
        )

        self.gamma = nn.Parameter(
            torch.tensor(0.1)
        )

        self.mask_scale = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(
        self,
        x_high,
        x_low,
        mask_logits,
        edge_logits
    ):
        local_feature = self.local_gab(
            x_high,
            x_low,
            mask_logits
        )

        high_feature = self.high_project(
            x_high
        )

        high_feature = F.interpolate(
            high_feature,
            size=x_low.shape[-2:],
            mode="bilinear",
            align_corners=True
        )

        mask_prob = torch.sigmoid(
            mask_logits
        )

        beta = torch.tanh(
            self.mask_scale
        )

        high_feature = high_feature * (
            1.0 + beta * mask_prob
        )

        transformer_feature = self.transformer(
            q_map=x_low,
            kv_map=high_feature
        )

        edge_prob = torch.sigmoid(
            edge_logits
        )

        edge_gate = 0.5 + edge_prob

        output = (
            local_feature
            + self.gamma
            * edge_gate
            * transformer_feature
        )

        return output
