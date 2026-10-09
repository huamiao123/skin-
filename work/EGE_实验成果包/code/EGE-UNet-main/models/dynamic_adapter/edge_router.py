import torch
import torch.nn as nn


class EdgeRouter(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):
        x = self.norm(x)
        logits = self.mlp(x).squeeze(-1)
        prob = torch.sigmoid(logits)
        return logits, prob
