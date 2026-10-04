"""Exact original-pixel mask metrics with boundary-only nearest queries.

This independent implementation keeps rsi.metrics.binary_mask_metrics verbatim
except for its two EDT distance arrays. cKDTree queries the full four-neighbour
pixel boundaries with eps=0 and workers=1, without subsampling or resizing.
The frozen rsi/config sources are not changed.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from rsi.metrics import _mask, pixel_boundary


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
    pred_coords, gt_coords = np.argwhere(pb), np.argwhere(gb)
    p_to_g = cKDTree(gt_coords).query(pred_coords, k=1, eps=0, workers=1)[0]
    g_to_p = cKDTree(pred_coords).query(gt_coords, k=1, eps=0, workers=1)[0]
    for fraction, key in [(0.005, "bf1"), (0.0025, "bf1_025"), (0.01, "bf1_1pct")]:
        tolerance = max(1.0, fraction * diagonal)
        precision = float(np.mean(p_to_g <= tolerance))
        recall = float(np.mean(g_to_p <= tolerance))
        out[key] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hd95 = float(np.percentile(np.concatenate([p_to_g, g_to_p]), 95))
    out.update(hd95=hd95, hd95_normalized=hd95 / diagonal)
    return out
