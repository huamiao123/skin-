"""Original-coordinate mask metrics and paired RSI diagnostics.

Call evaluate_reference only after cropping letterbox padding and interpolating
logits to the original mask dimensions. Thresholding precedes no resize here.
HD95 uses 4-neighbour pixel boundaries, Euclidean nearest distances in both
directions, and the 95th percentile of their concatenation.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.special import expit


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image(value: Any, name: str) -> np.ndarray:
    a = _array(value)
    if a.ndim < 2 or (a.ndim > 2 and np.prod(a.shape[:-2]) != 1):
        raise ValueError(f"{name} must be one two-dimensional image")
    a = a.reshape(a.shape[-2:])
    if min(a.shape) == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return a


def _mask(value: Any, name: str) -> np.ndarray:
    a = _image(value, name)
    if not np.all((a == 0) | (a == 1)):
        raise ValueError(f"{name} must contain binary zero/one values")
    return a.astype(bool)


def pixel_boundary(mask: np.ndarray) -> np.ndarray:
    """Inner four-neighbour pixel boundary, including the image edge."""
    connectivity = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    return mask & ~binary_erosion(mask, structure=connectivity, border_value=0)


def binary_mask_metrics(prediction: Any, target: Any) -> dict[str, float | int]:
    pred, gt = _mask(prediction, "prediction"), _mask(target, "target")
    if pred.shape != gt.shape:
        raise ValueError("prediction/target original-coordinate shape mismatch")
    diagonal = math.hypot(*gt.shape)
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    pred_empty, gt_empty = not pred.any(), not gt.any()
    both_empty = pred_empty and gt_empty
    one_empty = pred_empty != gt_empty
    dice = 1.0 if both_empty else (2 * tp / (2 * tp + fp + fn))
    iou = 1.0 if both_empty else (tp / (tp + fp + fn))
    out: dict[str, float | int] = {
        "dice": dice, "iou": iou,
        "thresholded_jaccard": iou if iou >= 0.65 else 0.0,
        "pred_empty": int(pred_empty), "gt_empty": int(gt_empty),
        "both_empty": int(both_empty), "one_empty": int(one_empty),
        "tp": tp, "fp": fp, "fn": fn,
        "diagonal_px": diagonal,
    }
    if both_empty or one_empty:
        bf1 = 1.0 if both_empty else 0.0
        hd95 = 0.0 if both_empty else diagonal
        out.update(bf1=bf1, bf1_025=bf1, bf1_1pct=bf1,
                   hd95=hd95, hd95_normalized=hd95 / diagonal)
        return out
    pb, gb = pixel_boundary(pred), pixel_boundary(gt)
    p_to_g = distance_transform_edt(~gb)[pb]
    g_to_p = distance_transform_edt(~pb)[gb]
    for fraction, key in [(0.005, "bf1"), (0.0025, "bf1_025"), (0.01, "bf1_1pct")]:
        tolerance = max(1.0, fraction * diagonal)
        precision = float(np.mean(p_to_g <= tolerance))
        recall = float(np.mean(g_to_p <= tolerance))
        out[key] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hd95 = float(np.percentile(np.concatenate([p_to_g, g_to_p]), 95))
    out.update(hd95=hd95, hd95_normalized=hd95 / diagonal)
    return out


def evaluate_reference(
    logits_orig: Any,
    mask_orig: Any,
    anchor_logits_orig: Any | None = None,
    d0_logits_orig: Any | None = None,
) -> dict[str, float | int]:
    """Metrics for one complete original-size reference; hard threshold=0.5.

    Zero logits are foreground because sigmoid(0)==0.5. The reported composite
    loss is on original pixels; epoch rho/J must instead use the canonical
    letterbox-view training objective, with its valid-pixel mask.
    """
    z = _image(logits_orig, "logits").astype(np.float64)
    gt = _mask(mask_orig, "reference")
    if z.shape != gt.shape:
        raise ValueError("logits must already be restored to original mask size")
    pred = z >= 0
    out = binary_mask_metrics(pred, gt)
    probability = expit(z)
    bce = float(np.mean(np.logaddexp(0, z) - z * gt))
    soft_dice = float((2 * np.sum(probability * gt) + 1e-6) /
                      (np.sum(probability) + np.sum(gt) + 1e-6))
    out.update(bce=bce, soft_dice=soft_dice, loss=0.5 * bce + 0.5 * (1 - soft_dice))
    if anchor_logits_orig is not None:
        anchor = evaluate_reference(anchor_logits_orig, gt)
        ap = _image(anchor_logits_orig, "anchor_logits") >= 0
        out.update(dice_anchor=anchor["dice"], iou_anchor=anchor["iou"], loss_anchor=anchor["loss"],
                   gain_dice=out["dice"] - anchor["dice"], gain_loss=anchor["loss"] - out["loss"],
                   fp_added=int(np.count_nonzero(pred & ~ap & ~gt)),
                   fn_added=int(np.count_nonzero(~pred & ap & gt)),
                   fp_removed=int(np.count_nonzero(~pred & ap & ~gt)),
                   fn_removed=int(np.count_nonzero(pred & ~ap & gt)),
                   changed_pixel_fraction=float(np.mean(pred != ap)))
    if d0_logits_orig is not None:
        d0 = evaluate_reference(d0_logits_orig, gt)
        out.update(dice_D0=d0["dice"], loss_D0=d0["loss"], gain_vs_D0=out["dice"] - d0["dice"])
    return out


def update_diagnostics(
    message: Any | None,
    logits: Any,
    anchor_logits: Any,
    valid: Any | None = None,
) -> dict[str, float]:
    """RMS of actual message and logit update, and changed hard-mask fraction.

    The message RMS includes its whole feature tensor. Logit statistics exclude
    padding when valid is supplied. Use original logits for original-pixel
    update statistics and record the view explicitly in the export metadata.
    """
    z, z0 = _array(logits).astype(float), _array(anchor_logits).astype(float)
    if z.shape != z0.shape:
        raise ValueError("current/anchor logit shape mismatch")
    v = np.ones_like(z, dtype=bool) if valid is None else _array(valid)
    if v.shape != z.shape or not np.all((v == 0) | (v == 1)):
        raise ValueError("valid must be binary and match logits")
    v = v.astype(bool)
    if not v.any() or not np.isfinite(z[v]).all() or not np.isfinite(z0[v]).all():
        raise ValueError("update statistics require finite valid pixels")
    out = {"logit_delta_rms": float(np.sqrt(np.mean((z[v] - z0[v]) ** 2))),
           "changed_pixel_fraction": float(np.mean((z[v] >= 0) != (z0[v] >= 0)))}
    if message is not None:
        m = _array(message).astype(float)
        if not m.size or not np.isfinite(m).all():
            raise ValueError("message must be nonempty and finite")
        out["message_rms"] = float(np.sqrt(np.mean(m ** 2)))
    return out


def gain_statistics(
    per_image_gains: Sequence[Sequence[float]],
    *,
    epsilon_d: float = 0.005,
    changed: Sequence[bool] | None = None,
) -> dict[str, float | int]:
    """Reference-wise gains then image-wise harm indicators and worst tails."""
    if len(per_image_gains) == 0 or not math.isfinite(epsilon_d) or epsilon_d < 0:
        raise ValueError("need nonempty image gains and nonnegative epsilon_d")
    gains = [np.asarray(g, dtype=float).reshape(-1) for g in per_image_gains]
    if any(len(g) == 0 or not np.isfinite(g).all() for g in gains):
        raise ValueError("each image must contain finite, present-reference gains")
    worst = np.array([g.min() for g in gains])
    best = np.array([g.max() for g in gains])
    sign_flip = (worst < -epsilon_d) & (best > epsilon_d)
    multi = np.array([len(g) >= 2 for g in gains])
    sign_flip = sign_flip & multi
    exceed = (worst < -epsilon_d) | (best > epsilon_d)
    tail_count = max(1, math.ceil(0.1 * len(gains)))
    out: dict[str, float | int] = {
        "n_images": len(gains), "n_references": sum(map(len, gains)),
        "mean_gain": float(np.mean([g.mean() for g in gains])),
        "G_plus": float(np.mean([np.maximum(g, 0).mean() for g in gains])),
        "H_minus": float(np.mean([np.maximum(-g, 0).mean() for g in gains])),
        "H_epsilon": float(np.mean([np.maximum(-g - epsilon_d, 0).mean() for g in gains])),
        "any_harm": float(np.mean(worst < -epsilon_d)),
        "worst_reference_mean": float(worst.mean()),
        "worst_tail_10pct": float(np.sort(worst)[:tail_count].mean()),
        "worst_tail_n": tail_count,
        "sign_flip": float(sign_flip[multi].mean()) if multi.any() else float("nan"),
        "sign_flip_n": int(multi.sum()),
        "sign_flip_exceed_subset": float(sign_flip[multi & exceed].mean()) if (multi & exceed).any() else float("nan"),
        "sign_flip_exceed_n": int((multi & exceed).sum()),
    }
    if changed is not None:
        changed_array = np.asarray(changed, dtype=bool)
        if changed_array.shape != multi.shape:
            raise ValueError("changed must have one indicator per image")
        selected = multi & changed_array
        out["sign_flip_changed_subset"] = float(sign_flip[selected].mean()) if selected.any() else float("nan")
        out["sign_flip_changed_n"] = int(selected.sum())
    for eps in (0.0, 0.002, 0.005, 0.01):
        suffix = str(eps).replace(".", "p")
        out[f"H_epsilon_{suffix}"] = float(np.mean([np.maximum(-g - eps, 0).mean() for g in gains]))
        out[f"any_harm_{suffix}"] = float(np.mean(worst < -eps))
    return out


def gain_loss_dice_cross_table(
    per_image_loss_gains: Sequence[Sequence[float]],
    per_image_dice_gains: Sequence[Sequence[float]],
    *,
    epsilon_d: float = 0.005,
    epsilon_loss: float = 0.0,
) -> dict[str, float]:
    """Image/reference-weighted 3x3 tolerance table and zero-tolerance signs."""
    if len(per_image_loss_gains) != len(per_image_dice_gains) or len(per_image_loss_gains) == 0:
        raise ValueError("paired nonempty loss and Dice gains are required")
    results: Counter[str] = Counter()
    n = len(per_image_loss_gains)
    classify = lambda x, eps: "positive" if x > eps else "negative" if x < -eps else "neutral"
    for loss, dice in zip(per_image_loss_gains, per_image_dice_gains):
        if len(loss) != len(dice) or not len(loss):
            raise ValueError("every image requires matching loss/Dice reference counts")
        for gl, gd in zip(loss, dice):
            if not math.isfinite(gl) or not math.isfinite(gd):
                raise ValueError("gains must be finite")
            mass = 1 / (n * len(loss))
            results[f"loss_{classify(gl, epsilon_loss)}_dice_{classify(gd, epsilon_d)}"] += mass
            results[f"zero_loss_{classify(gl, 0)}_dice_{classify(gd, 0)}"] += mass
    for prefix in ("", "zero_"):
        for loss_sign in ("positive", "neutral", "negative"):
            for dice_sign in ("positive", "neutral", "negative"):
                results.setdefault(f"{prefix}loss_{loss_sign}_dice_{dice_sign}", 0.0)
    return dict(results)


def paired_bootstrap(
    current: Sequence[float],
    comparator: Sequence[float],
    *,
    group_ids: Sequence[str] | None = None,
    repeats: int = 2000,
    seed: int = 17,
) -> dict[str, float | int | str]:
    """Resample images or entire known-case groups, preserving method pairing.

    Inputs must already be reference-averaged per image. Group resampling
    retains every image in each sampled group; the estimator is image-weighted.
    Missing group IDs remain distinct image units, never a shared missing group.
    """
    a, b = np.asarray(current, float), np.asarray(comparator, float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired finite per-image vectors of equal length are required")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if group_ids is None:
        units = [np.array([i]) for i in range(len(a))]
        unit_name = "image"
    else:
        if len(group_ids) != len(a):
            raise ValueError("group_ids length mismatch")
        by_group: dict[str, list[int]] = {}
        for i, group in enumerate(group_ids):
            missing = (group is None or not str(group).strip() or
                       (isinstance(group, (float, np.floating)) and math.isnan(group)))
            key = f"missing:{i}" if missing else f"known:{group}"
            by_group.setdefault(key, []).append(i)
        units = [np.asarray(indices) for indices in by_group.values()]
        unit_name = "known_case_group"
    difference = a - b
    group_sums = np.array([difference[u].sum() for u in units])
    group_counts = np.array([len(u) for u in units])
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats)
    for j in range(repeats):
        indices = rng.integers(len(units), size=len(units))
        draws[j] = group_sums[indices].sum() / group_counts[indices].sum()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"difference": float(difference.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "bootstrap_unit": unit_name, "bootstrap_units": len(units), "bootstrap_repeats": repeats,
            "bootstrap_seed": seed}
