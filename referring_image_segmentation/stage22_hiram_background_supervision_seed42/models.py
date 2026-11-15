"""Only auxiliary supervision varies; architecture is exactly stage17 HiRAM."""
from common import *
from torch import nn
from torch.nn import functional as F
from stage17_mambavision_decoder_hiram_seed42.models import (
    DecoderProbe, FrozenEncoder, region_targets as historical_targets,
    region_loss_numerators, segmentation_loss, tensor_hash,
)

NAMES = ('original', 'scaled_original', 'background_supervised')


@torch.no_grad()
def region_targets(mask):
    boundary, interior = historical_targets(mask)
    residual = 1. - boundary - interior
    assert residual.min() >= -1e-6 and residual.max() <= 1. + 1e-6
    background = residual.clamp(0., 1.)
    return boundary, interior, background


def auxiliary_numerators(regions, targets):
    boundary, interior, background = targets
    old = region_loss_numerators(regions, (boundary, interior))
    bg = regions[0].new_zeros((), dtype=torch.float32)
    for pred in regions:
        bg = bg + F.kl_div(pred[:, 0:1].float(), background.float(), reduction='sum')
    return old[0], old[1], bg


def auxiliary_loss(mode, nums, denoms):
    boundary, interior, bg = [n/d for n,d in zip(nums,denoms)]
    old = boundary + interior
    if mode == 'original': return old
    if mode == 'scaled_original': return old * 1.5
    if mode == 'background_supervised': return old + bg
    raise ValueError(mode)
