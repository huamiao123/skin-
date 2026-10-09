from __future__ import annotations

import numpy as np


def overlap_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, np.ndarray | float]:
    """Compute per-image and pooled binary overlap metrics.

    Inputs are [N,H,W] or [N,1,H,W]. Both-empty Dice/IoU is defined as 1.
    """
    def as_batch(array: np.ndarray) -> np.ndarray:
        result = np.asarray(array)
        if result.ndim == 4 and result.shape[1] == 1:
            result = result[:, 0]
        if result.ndim != 3 or min(result.shape) <= 0:
            raise ValueError("input must be non-empty [N,H,W] or [N,1,H,W]")
        return result

    p, y = as_batch(probabilities), as_batch(targets)
    if p.shape != y.shape:
        raise ValueError("prediction and target shapes differ")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0,1]")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("predictions must be finite probabilities in [0,1]")
    if not np.isfinite(y).all() or not ((y == 0) | (y == 1)).all():
        raise ValueError("targets must be strictly binary")

    pred, gt = p >= threshold, y.astype(bool)
    axis = (1, 2)
    tp = np.sum(pred & gt, axis=axis, dtype=np.int64)
    fp = np.sum(pred & ~gt, axis=axis, dtype=np.int64)
    fn = np.sum(~pred & gt, axis=axis, dtype=np.int64)
    tn = np.sum(~pred & ~gt, axis=axis, dtype=np.int64)
    dice_den, iou_den = 2 * tp + fp + fn, tp + fp + fn
    dice = np.divide(2 * tp, dice_den, out=np.ones(tp.shape, dtype=float), where=dice_den > 0)
    iou = np.divide(tp, iou_den, out=np.ones(tp.shape, dtype=float), where=iou_den > 0)
    pooled_dice_den = int(2 * tp.sum() + fp.sum() + fn.sum())
    pooled_iou_den = int(tp.sum() + fp.sum() + fn.sum())
    return {
        "dice": dice,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_empty": np.sum(gt, axis=axis) == 0,
        "pred_empty": np.sum(pred, axis=axis) == 0,
        "macro_dice": float(dice.mean()),
        "macro_iou": float(iou.mean()),
        "pooled_foreground_dice": float(2 * tp.sum() / pooled_dice_den) if pooled_dice_den else 1.0,
        "pooled_foreground_iou": float(tp.sum() / pooled_iou_den) if pooled_iou_den else 1.0,
        "pooled_sensitivity": float(tp.sum() / (tp.sum() + fn.sum())) if tp.sum() + fn.sum() else 1.0,
        "pooled_specificity": float(tn.sum() / (tn.sum() + fp.sum())) if tn.sum() + fp.sum() else 1.0,
        "thresholded_jaccard": float(np.where(iou < 0.65, 0.0, iou).mean()),
    }

