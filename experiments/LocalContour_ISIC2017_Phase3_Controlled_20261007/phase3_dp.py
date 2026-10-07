"""Exact 65-state cyclic DP with a zero-first tie order and original-index output."""
from __future__ import annotations

import numpy as np
import torch

ZERO_FIRST = np.array([32] + [j for radius in range(1, 33) for j in (32-radius, 32+radius)], dtype=np.int64)
OFFSETS = np.arange(-32, 33, dtype=np.float64)


def choose_argmax(scores, valid, alpha=0.0):
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    adjusted = np.where(valid, scores-alpha*np.abs(OFFSETS), -np.inf)
    return ZERO_FIRST[np.argmax(adjusted[:, ZERO_FIRST], axis=1)]


def choose_closed_dp(scores, valid, alpha, lam, device=None):
    """Minimize -score+alpha|offset|+lambda cyclic squared step cost.

    The first state in ZERO_FIRST wins exact ties. Infeasible states retain inf.
    """
    if lam == 0: return choose_argmax(scores, valid, alpha)
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    score = np.asarray(scores, dtype=np.float64)[:, ZERO_FIRST]
    validity = np.asarray(valid, dtype=bool)[:, ZERO_FIRST]
    if score.ndim != 2 or score.shape[1] != 65 or validity.shape != score.shape:
        raise ValueError("Expected N x 65 scores and valid")
    if not validity.any(axis=1).all(): raise ValueError("Every node needs a valid candidate")
    offsets = OFFSETS[ZERO_FIRST]
    adjusted = np.where(validity, -score+alpha*np.abs(offsets), np.inf)
    row_shift = np.min(adjusted, axis=1, keepdims=True)
    cost = torch.as_tensor(adjusted-row_shift, device=device, dtype=torch.float64)
    shift = torch.as_tensor(offsets, device=device, dtype=torch.float64)
    transition = lam*(shift[:, None]-shift[None, :]).square()
    n, k = cost.shape
    value = torch.full((k, k), float("inf"), device=device, dtype=torch.float64)
    index = torch.arange(k, device=device)
    value[index, index] = cost[0]
    parents = torch.empty((n, k, k), device=device, dtype=torch.int16)
    for row in range(1, n):
        choices = value[:, :, None] + transition[None, :, :]
        minimum, parent = torch.min(choices, dim=1)
        value = minimum + cost[row][None, :]
        parents[row] = parent.to(torch.int16)
    closing = value + transition.transpose(0, 1)
    flat = int(torch.argmin(closing).item())
    start, last = divmod(flat, k)
    parents_cpu = parents[:, start, :].cpu().numpy()
    path = np.empty(n, dtype=np.int64)
    path[-1] = last
    for row in range(n-1, 0, -1):
        path[row-1] = int(parents_cpu[row, path[row]])
    assert path[0] == start
    return ZERO_FIRST[path]
