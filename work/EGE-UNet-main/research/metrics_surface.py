from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from surface_distance import metrics as surface_metrics


SURFACE_IMPLEMENTATION = "surface-distance==0.1"


@dataclass(frozen=True)
class SurfaceResult:
    bf1_2px: float
    hd95_px: float
    assd_dirmean_px: float
    surface_status: str
    gt_empty: bool
    pred_empty: bool

    def as_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


def pixel_boundary_4n(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be 2-D")
    eroded = binary_erosion(binary, structure=generate_binary_structure(2, 1), border_value=0)
    return binary & ~eroded


def boundary_f1_tolerance(pred: np.ndarray, gt: np.ndarray, tolerance_px: float = 2.0) -> float:
    if tolerance_px < 0 or not np.isfinite(tolerance_px):
        raise ValueError("tolerance_px must be finite and non-negative")
    pb, gb = pixel_boundary_4n(pred), pixel_boundary_4n(gt)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0
    distance_to_gt = distance_transform_edt(~gb)
    distance_to_pred = distance_transform_edt(~pb)
    precision = float((distance_to_gt[pb] <= tolerance_px).mean())
    recall = float((distance_to_pred[gb] <= tolerance_px).mean())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_surface(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tolerance_px: float = 2.0,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> SurfaceResult:
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError("pred and gt must be same-shaped 2-D masks")
    pred_empty, gt_empty = not pred.any(), not gt.any()
    if pred_empty and gt_empty:
        return SurfaceResult(1.0, float("nan"), float("nan"), "both_empty", True, True)
    if pred_empty or gt_empty:
        return SurfaceResult(0.0, float("inf"), float("inf"), "one_empty", gt_empty, pred_empty)

    distances = surface_metrics.compute_surface_distances(gt, pred, spacing_mm=spacing)
    hd95 = float(surface_metrics.compute_robust_hausdorff(distances, 95.0))
    directed = surface_metrics.compute_average_surface_distance(distances)
    assd = float((directed[0] + directed[1]) / 2.0)
    bf1 = boundary_f1_tolerance(pred, gt, tolerance_px)
    return SurfaceResult(bf1, hd95, assd, "finite", False, False)

