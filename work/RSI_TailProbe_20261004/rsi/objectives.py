"""The four P2-IMA-M-v2 objectives, with image-then-reference reduction.

All loss arithmetic is FP32, including under autocast. Missing references and
invalid pixels are removed before computation, rather than merely multiplied
by zero after a potentially non-finite operation.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


DICE_SMOOTH = 1e-6


def _presence(present: Tensor, shape: tuple[int, int], device: torch.device) -> Tensor:
    present = torch.as_tensor(present, device=device)
    if tuple(present.shape) != shape:
        raise ValueError(f"present must have shape {shape}, got {tuple(present.shape)}")
    if not torch.all((present == 0) | (present == 1)):
        raise ValueError("present must contain only zero or one")
    present = present.bool()
    if not present.any(dim=1).all():
        raise ValueError("every image must contain at least one present reference")
    return present


def per_image_reference_mean(values: Tensor, present: Tensor) -> Tensor:
    """Each image receives equal weight, independent of its reference count."""
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("values must have nonempty shape [B,R]")
    p = _presence(present, tuple(values.shape), values.device)
    safe = torch.where(p, values, torch.zeros_like(values))
    if not torch.isfinite(safe).all():
        raise ValueError("present reference values must be finite")
    return safe.sum(dim=1) / p.sum(dim=1)


def image_reference_mean(values: Tensor, present: Tensor) -> Tensor:
    return per_image_reference_mean(values, present).mean()


def per_reference_loss(
    logits: Tensor,
    masks: Tensor,
    present: Tensor,
    valid: Tensor | None = None,
    *,
    return_components: bool = False,
) -> Tensor | dict[str, Tensor]:
    """Return ell[B,R], or its BCE/SoftDice components.

    logits: [B,1,H,W], masks: [B,R,1,H,W], present: [B,R].
    valid accepts shared [B,1,H,W] or per-reference [B,R,1,H,W].
    Absent entries are returned as zero and must still be reduced using present.
    """
    if logits.ndim != 4 or logits.shape[1] != 1 or masks.ndim != 5:
        raise ValueError("expected logits[B,1,H,W] and masks[B,R,1,H,W]")
    b, r, c, h, w = masks.shape
    if tuple(logits.shape) != (b, 1, h, w) or c != 1 or min(b, r, h, w) <= 0:
        raise ValueError("logit/mask shape mismatch or empty dimensions")
    p = _presence(present, (b, r), logits.device)
    masks = masks.to(device=logits.device)
    if valid is None:
        v = torch.ones((b, r, 1, h, w), dtype=torch.bool, device=logits.device)
    else:
        v = torch.as_tensor(valid, device=logits.device)
        if tuple(v.shape) == (b, 1, h, w):
            v = v[:, None].expand(-1, r, -1, -1, -1)
        elif tuple(v.shape) != (b, r, 1, h, w):
            raise ValueError("valid must have shape [B,1,H,W] or [B,R,1,H,W]")
        if not torch.all((v == 0) | (v == 1)):
            raise ValueError("valid must contain only zero or one")
        v = v.bool()
    effective = v & p[:, :, None, None, None]
    count = effective.sum(dim=(2, 3, 4))
    if not (count[p] > 0).all():
        raise ValueError("each present reference must have at least one valid pixel")
    expanded_logits = logits[:, None].expand(-1, r, -1, -1, -1)
    if not torch.isfinite(expanded_logits[effective]).all():
        raise ValueError("valid logits must be finite")
    if not torch.all((masks[effective] == 0) | (masks[effective] == 1)):
        raise ValueError("valid mask pixels must be binary")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        z = torch.where(effective, expanded_logits.float(), 0.0)
        y = torch.where(effective, masks.float(), 0.0)
        v_float = effective.float()
        bce = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * v_float).sum((2, 3, 4))
        bce = bce / count.clamp_min(1)
        probability = torch.sigmoid(z) * v_float
        intersection = (probability * y).sum((2, 3, 4))
        denominator = probability.sum((2, 3, 4)) + y.sum((2, 3, 4))
        soft_dice = (2 * intersection + DICE_SMOOTH) / (denominator + DICE_SMOOTH)
        loss = 0.5 * bce + 0.5 * (1 - soft_dice)
        loss = torch.where(p, loss, 0.0)
        bce = torch.where(p, bce, 0.0)
        soft_dice = torch.where(p, soft_dice, 0.0)
    if return_components:
        return {"loss": loss, "bce": bce, "soft_dice": soft_dice}
    return loss


def objective_from_losses(
    current: Tensor,
    anchor: Tensor,
    present: Tensor,
    *,
    method: str = "d0",
    weight: float = 0.0,
    q: Tensor | float | None = None,
    epsilon_l: float = 0.0,
) -> dict[str, Any]:
    """Compute a differentiable loss/risk and detached canonical diagnostics.

    Anchor and q are always detached. Risk remains differentiable so callers
    can measure its gradient contribution without an optimizer step.
    """
    if current.ndim != 2 or tuple(anchor.shape) != tuple(current.shape):
        raise ValueError("current and anchor losses must have matching [B,R] shapes")
    if not math.isfinite(float(weight)) or weight < 0 or not math.isfinite(float(epsilon_l)):
        raise ValueError("weight must be finite/nonnegative and epsilon_l finite")
    method = method.lower().replace("-", "_")
    method = {"meanhinge": "mean_hinge", "abshard": "abs_hard"}.get(method, method)
    if method not in {"d0", "rsi", "mean_hinge", "abs_hard"}:
        raise ValueError(f"unknown objective: {method}")
    p = _presence(present, tuple(current.shape), current.device)
    with torch.autocast(device_type=current.device.type, enabled=False):
        ell = torch.where(p, current.float(), 0.0)
        ell0 = torch.where(p, anchor.detach().to(device=current.device, dtype=torch.float32), 0.0)
        if not torch.isfinite(ell).all() or not torch.isfinite(ell0).all():
            raise ValueError("present reference losses must be finite")
        d = torch.where(p, ell - ell0 - epsilon_l, 0.0)
        seg = image_reference_mean(ell, p)
        mean_d = per_image_reference_mean(d, p)
        relative_activation = d > 0
        absolute_activation = None
        detached_q = None
        if q is not None:
            detached_q = torch.as_tensor(q, device=ell.device, dtype=torch.float32).detach()
            if detached_q.numel() != 1 or not torch.isfinite(detached_q).all():
                raise ValueError("q must be one finite, fixed scalar")
            absolute_activation = ell > detached_q
        if method == "rsi":
            risk = image_reference_mean(torch.relu(d), p)
            activation = relative_activation
        elif method == "mean_hinge":
            risk = torch.relu(mean_d).mean()
            activation = (mean_d > 0)[:, None].expand_as(p)
        elif method == "abs_hard":
            if detached_q is None:
                raise ValueError("abs_hard requires the fixed training-side quantile q")
            risk = image_reference_mean(torch.relu(ell - detached_q), p)
            activation = absolute_activation
        else:
            risk = seg * 0
            activation = torch.zeros_like(p)
        weighted_risk = float(weight) * risk
        loss = seg + weighted_risk
        with torch.no_grad():
            j = per_image_reference_mean(torch.relu(d), p) - torch.relu(mean_d)
            # Small negative roundoff is retained, making identity checks possible.
            j_identity = (per_image_reference_mean(torch.abs(d), p) - torch.abs(mean_d)) / 2
            d_min = torch.where(p, d, torch.inf).min(dim=1).values
            d_max = torch.where(p, d, -torch.inf).max(dim=1).values
            activation_rate = image_reference_mean(activation.float(), p)
            diagnostics = {
                "rho": image_reference_mean(relative_activation.float(), p),
                "J": j.mean(),
                "J_per_image": j,
                "J_identity_per_image": j_identity,
                "kappa_pm": ((d_min < -1e-6) & (d_max > 1e-6)).float().mean(),
                "activation_rate": activation_rate,
                "mean_weight": 1 + float(weight) * activation_rate,
                "absolute_activation_rate": (
                    image_reference_mean(absolute_activation.float(), p)
                    if absolute_activation is not None else ell.new_tensor(float("nan"))
                ),
            }
        return {"loss": loss, "seg": seg, "risk": risk, "weighted_risk": weighted_risk,
                "per_reference_loss": ell, "anchor_loss": ell0, "d": d.detach(),
                "q": detached_q, **diagnostics}


def objective(
    logits: Tensor,
    anchor_logits: Tensor,
    masks: Tensor,
    present: Tensor,
    valid: Tensor | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    current = per_reference_loss(logits, masks, present, valid)
    anchor = per_reference_loss(anchor_logits.detach(), masks, present, valid)
    return objective_from_losses(current, anchor, present, **kwargs)


def weighted_reference_quantile(
    losses: Tensor,
    present: Tensor,
    quantile: float = 0.75,
) -> Tensor:
    """Smallest sorted loss whose image-then-reference mass reaches quantile.

    Call once on the entire canonical B training split, not separately on each
    minibatch. The returned FP32 scalar is detached and fixed throughout D.
    """
    if losses.ndim != 2 or not 0 < quantile <= 1:
        raise ValueError("expected losses[B,R] and quantile in (0,1]")
    p = _presence(present, tuple(losses.shape), losses.device)
    with torch.no_grad():
        values = losses.detach().float()[p]
        if not torch.isfinite(values).all():
            raise ValueError("present losses must be finite")
        weights = (1.0 / (len(losses) * p.sum(dim=1).double()))[:, None].expand_as(losses)[p]
        order = torch.argsort(values, stable=True)
        cumulative = weights[order].cumsum(0)
        # Normalization prevents a representational error at quantile=1.
        cumulative = cumulative / cumulative[-1]
        index = torch.searchsorted(cumulative, cumulative.new_tensor(quantile), right=False)
        return values[order[index.clamp_max(len(order) - 1)]].clone().detach()
