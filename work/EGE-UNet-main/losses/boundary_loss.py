import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_to_boundary(
    mask,
    kernel_size=5
):
    padding = kernel_size // 2

    dilated = F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    eroded = -F.max_pool2d(
        -mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    boundary = (
        dilated - eroded
    ).clamp(0, 1)

    return boundary


def dice_loss(
    probability,
    target,
    eps=1e-6
):
    dims = (1, 2, 3)

    intersection = torch.sum(
        probability * target,
        dim=dims
    )

    denominator = (
        torch.sum(
            probability,
            dim=dims
        )
        +
        torch.sum(
            target,
            dim=dims
        )
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return 1.0 - dice.mean()
