import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj1 = nn.Conv2d(8, 8, 1)
        self.proj2 = nn.Conv2d(16, 8, 1)
        self.proj3 = nn.Conv2d(24, 8, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(
                24,
                24,
                kernel_size=3,
                padding=1,
                groups=24
            ),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv2d(
                24,
                1,
                kernel_size=1
            )
        )

    def forward(
        self,
        t1,
        t2,
        t3
    ):
        target_size = t3.shape[-2:]

        f1 = self.proj1(t1)
        f1 = F.interpolate(
            f1,
            size=target_size,
            mode="bilinear",
            align_corners=True
        )

        f2 = self.proj2(t2)
        f2 = F.interpolate(
            f2,
            size=target_size,
            mode="bilinear",
            align_corners=True
        )

        f3 = self.proj3(t3)

        x = torch.cat(
            [f1, f2, f3],
            dim=1
        )

        edge_logits = self.fuse(x)

        return edge_logits
