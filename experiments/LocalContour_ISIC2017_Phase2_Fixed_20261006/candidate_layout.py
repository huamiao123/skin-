"""Single source of truth for the dense, sorted normal-offset layout."""
from __future__ import annotations

RADIUS = 32
CANDIDATE_COUNT = 2 * RADIUS + 1
ZERO_INDEX = RADIUS


def assert_layout(valid, has_contour):
    if valid.shape[-1] != CANDIDATE_COUNT:
        raise ValueError(f'Expected {CANDIDATE_COUNT} candidates, got {valid.shape[-1]}')
    if has_contour.any() and not valid[has_contour, :, ZERO_INDEX].all():
        raise ValueError('A contour node has an invalid zero-offset candidate')
