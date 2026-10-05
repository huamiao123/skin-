"""CPU geometry for the preregistered local-candidate contour diagnostic.

Coordinates are ``(x, y)`` in model-input pixels; integer coordinates are pixel
centres and the visible image rectangle is [-.5, W-.5] x [-.5, H-.5]. GT is
accepted only by diagnostic/metric functions, never by candidate generation.
The representation is the largest predicted exterior closed contour. Other
components and holes are reported rather than removed from evaluation cases.
All reconstructed selectors share straight connections and pixel-centre fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import contourpy
import numpy as np
from scipy import ndimage, signal

CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class ContourGeometry:
    points: np.ndarray
    normals: np.ndarray
    shape: tuple[int, int]
    search_radius: float
    source_polygon: np.ndarray
    source_components: int
    source_holes: int
    source_perimeter: float
    edge_nodes: np.ndarray
    contour_self_intersections: int
    strip_self_intersections: int
    degenerate_strip_count: int
    source_contour_count: int


@dataclass(frozen=True)
class CandidateField:
    points: np.ndarray
    offsets: np.ndarray
    scores: np.ndarray
    valid: np.ndarray
    sample_offsets: np.ndarray
    sample_scores: np.ndarray
    sample_valid: np.ndarray

    @property
    def local_costs(self) -> np.ndarray:
        """Probability negative log cost; invalid candidates are +infinity."""
        costs = -np.log(np.clip(self.scores, 1e-6, 1.0))
        return np.where(self.valid, costs, np.inf)


def _mask(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError("A mask must be a two-dimensional input-coordinate array")
    return value.astype(bool, copy=False)


def _component_count(mask: np.ndarray) -> int:
    return int(ndimage.label(mask, structure=CONNECTIVITY_8)[1])


def polygon_area(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    q = np.roll(p, -1, axis=0)
    return float(0.5 * np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]))


def mask_contours(mask: np.ndarray) -> list[np.ndarray]:
    """All subpixel exterior/hole contours, including image-border closures.

    These are contourpy binary 0.5 isocontours. A saddle between diagonally
    touching pixels may have multiple geometric loops despite a single
    8-connected pixel component; both counts are reported, without GT removal.
    """
    value = _mask(mask)
    if not value.any():
        return []
    h, w = value.shape
    generator = contourpy.contour_generator(
        x=np.arange(-1, w + 1, dtype=np.float64),
        y=np.arange(-1, h + 1, dtype=np.float64),
        z=np.pad(value.astype(np.float64), 1),
        line_type="Separate",
        corner_mask=False,
    )
    contours = []
    for line in generator.lines(0.5):
        line = np.asarray(line, dtype=np.float64)
        if len(line) > 1 and np.allclose(line[0], line[-1], atol=1e-12, rtol=0):
            line = line[:-1]
        if len(line) >= 3:
            contours.append(line)
    return contours


def polygon_perimeter(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())


def resample_closed(points: np.ndarray, nodes: int) -> np.ndarray:
    """Uniform arc-length samples; the closing segment is part of the length."""
    points = np.asarray(points, dtype=np.float64)
    if nodes < 3 or points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("A closed polygon and at least three nodes are required")
    if np.allclose(points[0], points[-1], atol=1e-12, rtol=0):
        points = points[:-1]
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    good = lengths > 1e-12
    points, lengths = points[good], lengths[good]
    total = lengths.sum()
    if len(points) < 3 or total <= 0:
        raise ValueError("Degenerate closed contour")
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    positions = np.arange(nodes, dtype=np.float64) * (total / nodes)
    segment = np.searchsorted(cumulative, positions, side="right") - 1
    fraction = (positions - cumulative[segment]) / lengths[segment]
    return points[segment] + fraction[:, None] * (
        points[(segment + 1) % len(points)] - points[segment]
    )


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def self_intersection_count(points: np.ndarray) -> int:
    """Proper crossings of nonadjacent edges; touching/collinear are separate."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 4:
        return 0
    a, b = points, np.roll(points, -1, axis=0)
    i, j = np.triu_indices(n, 1)
    eligible = (j != i + 1) & ~((i == 0) & (j == n - 1))
    i, j = i[eligible], j[eligible]
    ab, cd = b[i] - a[i], b[j] - a[j]
    c1 = _cross(ab, a[j] - a[i])
    c2 = _cross(ab, b[j] - a[i])
    c3 = _cross(cd, a[i] - a[j])
    c4 = _cross(cd, b[i] - a[j])
    return int(np.count_nonzero((c1 * c2 < -1e-12) & (c3 * c4 < -1e-12)))


def _spoke_crossing_count(starts: np.ndarray, ends: np.ndarray) -> int:
    """Proper crossings of finite normal search segments, including neighbours."""
    i, j = np.triu_indices(len(starts), 1)
    ab, cd = ends[i] - starts[i], ends[j] - starts[j]
    c1 = _cross(ab, starts[j] - starts[i])
    c2 = _cross(ab, ends[j] - starts[i])
    c3 = _cross(cd, starts[i] - starts[j])
    c4 = _cross(cd, ends[i] - starts[j])
    return int(np.count_nonzero((c1 * c2 < -1e-12) & (c3 * c4 < -1e-12)))


def strip_quadrilaterals(geometry: ContourGeometry) -> np.ndarray:
    lo = geometry.points - geometry.search_radius * geometry.normals
    hi = geometry.points + geometry.search_radius * geometry.normals
    return np.stack((lo, np.roll(lo, -1, axis=0), np.roll(hi, -1, axis=0), hi), axis=1)


def extract_geometry(
    prediction_mask: np.ndarray, nodes: int = 128, search_radius: float = 16.0
) -> ContourGeometry | None:
    """No GT input; return None for an empty prediction rather than inventing it."""
    value = _mask(prediction_mask)
    if nodes not in (128, 256):
        raise ValueError("The planned node settings are 128 or one 256-node sensitivity")
    if search_radius not in (16.0, 32.0):
        raise ValueError("The planned radii are 16 or one 32-pixel sensitivity")
    if nodes == 256 and search_radius == 32.0:
        raise ValueError("The plan allows one sensitivity, not combined node/range changes")
    contours = mask_contours(value)
    if not contours:
        return None
    source = max(contours, key=lambda p: abs(polygon_area(p)))
    # Positive signed area yields the right-hand normal pointing outwards.
    if polygon_area(source) < 0:
        source = source[::-1]
    points = resample_closed(source, nodes)
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    length = np.linalg.norm(tangent, axis=1)
    if np.any(length <= 1e-12):
        raise ValueError("A contour normal is degenerate; record this case as failure")
    normals = np.stack((tangent[:, 1], -tangent[:, 0]), axis=1) / length[:, None]
    h, w = value.shape
    endpoints = points[:, None, :] + np.array([-search_radius, search_radius])[None, :, None] * normals[:, None, :]
    edge_nodes = np.any(
        (endpoints[..., 0] < -0.5) | (endpoints[..., 0] > w - 0.5)
        | (endpoints[..., 1] < -0.5) | (endpoints[..., 1] > h - 0.5), axis=1
    )
    components = _component_count(value)
    holes = _component_count(ndimage.binary_fill_holes(value, structure=CONNECTIVITY_8) & ~value)
    lo, hi = points - search_radius * normals, points + search_radius * normals
    quads = np.stack((lo, np.roll(lo, -1, axis=0), np.roll(hi, -1, axis=0), hi), axis=1)
    degenerate = sum(abs(polygon_area(q)) <= 1e-8 or self_intersection_count(q) > 0 for q in quads)
    strip_crossings = self_intersection_count(lo) + self_intersection_count(hi) + _spoke_crossing_count(lo, hi)
    return ContourGeometry(
        points, normals, value.shape, float(search_radius), source, components,
        holes, polygon_perimeter(source), edge_nodes,
        self_intersection_count(points), strip_crossings, int(degenerate), len(contours),
    )


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd fill including points on an edge, without integer rounding."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    polygon = np.asarray(polygon, dtype=np.float64)
    inside = np.zeros(len(points), dtype=bool)
    on_edge = np.zeros(len(points), dtype=bool)
    x, y = points[:, 0], points[:, 1]
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        delta = b - a
        rel = points - a
        length2 = float(delta @ delta)
        if length2 > 1e-24:
            projection = rel @ delta / length2
            on_edge |= (np.abs(_cross(rel, delta)) <= 1e-8 * max(1.0, np.sqrt(length2))) & (projection >= -1e-10) & (projection <= 1 + 1e-10)
        crossing = (a[1] > y) != (b[1] > y)
        if b[1] != a[1]:
            intercept = a[0] + (y - a[1]) * delta[0] / delta[1]
            inside ^= crossing & (x < intercept)
    return inside | on_edge


def rasterize(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """All G1--G4 paths use this same input-coordinate centre fill."""
    h, w = shape
    result = np.zeros((h, w), dtype=bool)
    polygon = np.asarray(points, dtype=np.float64)
    if len(polygon) < 3:
        return result
    xmin = max(0, int(np.ceil(polygon[:, 0].min())))
    xmax = min(w - 1, int(np.floor(polygon[:, 0].max())))
    ymin = max(0, int(np.ceil(polygon[:, 1].min())))
    ymax = min(h - 1, int(np.floor(polygon[:, 1].max())))
    if xmax < xmin or ymax < ymin:
        return result
    yy, xx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
    result[ymin:ymax + 1, xmin:xmax + 1] = points_in_polygon(
        np.stack((xx.ravel(), yy.ravel()), axis=1), polygon
    ).reshape(yy.shape)
    return result


def generate_candidates(score_map: np.ndarray, geometry: ContourGeometry) -> CandidateField:
    """At most three >=2px-separated score peaks plus the mandatory zero offset.

    score_map must be a finite [0,1] boundary probability map. Scores outside
    the visible image are unavailable. A clipped border sample uses the nearest
    border pixel only within the visible half-pixel extent.
    """
    scores = np.asarray(score_map, dtype=np.float64)
    if scores.shape != geometry.shape or not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise ValueError("Boundary score map must be finite probabilities at the mask input size")
    radius = int(geometry.search_radius)
    sample_offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    samples = geometry.points[:, None, :] + sample_offsets[None, :, None] * geometry.normals[:, None, :]
    h, w = geometry.shape
    available = (samples[..., 0] >= -0.5) & (samples[..., 0] <= w - 0.5) & (samples[..., 1] >= -0.5) & (samples[..., 1] <= h - 0.5)
    sample_scores = ndimage.map_coordinates(scores, [samples[..., 1], samples[..., 0]], order=1, mode="nearest", prefilter=False)
    sample_scores = np.where(available, sample_scores, -np.inf)
    n = len(geometry.points)
    offsets = np.zeros((n, 4), dtype=np.float64)
    candidate_scores = np.full((n, 4), -np.inf, dtype=np.float64)
    valid = np.zeros((n, 4), dtype=bool)
    for row in range(n):
        profile = sample_scores[row]
        # Index 0 is always the original point; it is independent of any GT.
        candidate_scores[row, 0] = profile[radius]
        valid[row, 0] = True
        peaks = list(signal.find_peaks(profile, distance=2)[0])
        finite = np.flatnonzero(available[row])
        if len(finite):
            first, last = int(finite[0]), int(finite[-1])
            if first < last and profile[first] > profile[first + 1]:
                peaks.append(first)
            if last > first and profile[last] > profile[last - 1]:
                peaks.append(last)
        ordered = sorted(set(peaks), key=lambda index: (-profile[index], abs(sample_offsets[index]), sample_offsets[index]))
        chosen: list[int] = []
        for index in ordered:
            if not np.isfinite(profile[index]) or any(abs(index - old) < 2 for old in chosen):
                continue
            chosen.append(index)
            if len(chosen) == 3:
                break
        cursor = 1
        for index in chosen:
            if sample_offsets[index] == 0:
                continue
            offsets[row, cursor] = sample_offsets[index]
            candidate_scores[row, cursor] = profile[index]
            valid[row, cursor] = True
            cursor += 1
    if not np.isfinite(candidate_scores[:, 0]).all():
        raise ValueError("The original prediction contour lies outside the visible image")
    candidate_points = geometry.points[:, None, :] + offsets[..., None] * geometry.normals[:, None, :]
    return CandidateField(candidate_points, offsets, candidate_scores, valid, sample_offsets, sample_scores, available)


def path_objective(indices: np.ndarray, local_costs: np.ndarray, offsets: np.ndarray, smooth_lambda: float) -> float:
    indices = np.asarray(indices, dtype=np.int64)
    rows = np.arange(len(indices))
    selected = offsets[rows, indices]
    return float(local_costs[rows, indices].sum() + smooth_lambda * np.square(selected - np.roll(selected, -1)).sum())


def closed_dp(local_costs: np.ndarray, offsets: np.ndarray, smooth_lambda: float) -> tuple[np.ndarray, float]:
    """Exact cyclic DP for sum(local) + lambda sum((s_i-s_{i+1})**2).

    Enumerating the initial candidate ensures that the final-first transition
    is included. Invalid candidates have +infinity cost. Equal paths are chosen
    deterministically by the first initial state and numpy's first argmin.
    """
    costs = np.asarray(local_costs, dtype=np.float64)
    shifts = np.asarray(offsets, dtype=np.float64)
    if costs.ndim != 2 or shifts.shape != costs.shape or len(costs) == 0 or not np.isfinite(shifts).all() or np.isnan(costs).any() or np.isneginf(costs).any():
        raise ValueError("DP costs/offsets must be nonempty matching NxK arrays")
    if not np.isfinite(smooth_lambda) or smooth_lambda < 0 or np.any(~np.isfinite(costs).any(axis=1)):
        raise ValueError("Nonnegative finite lambda and a valid candidate at every node are required")
    n, k = costs.shape
    best_value, best_path = np.inf, None
    for start in range(k):
        if not np.isfinite(costs[0, start]):
            continue
        value = np.full(k, np.inf)
        value[start] = costs[0, start]
        parents = np.full((n, k), -1, dtype=np.int64)
        for row in range(1, n):
            transition = smooth_lambda * np.square(shifts[row - 1, :, None] - shifts[row, None, :])
            choices = value[:, None] + transition
            parents[row] = np.argmin(choices, axis=0)
            value = choices[parents[row], np.arange(k)] + costs[row]
        closing = value + smooth_lambda * np.square(shifts[-1] - shifts[0, start])
        last = int(np.argmin(closing))
        objective = float(closing[last])
        if objective < best_value:
            path = np.empty(n, dtype=np.int64)
            path[-1] = last
            for row in range(n - 1, 0, -1):
                path[row - 1] = parents[row, path[row]]
            best_value, best_path = objective, path
    if best_path is None:
        raise ValueError("No feasible closed candidate path")
    return best_path, best_value


def dense_boundary_samples(mask: np.ndarray, max_step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("Positive finite boundary sampling step required")
    all_points = []
    all_weights = []
    for contour in mask_contours(mask):
        a, b = contour, np.roll(contour, -1, axis=0)
        for start, end in zip(a, b):
            length = float(np.linalg.norm(end - start))
            count = max(1, int(np.ceil(length / max_step)))
            fraction = np.arange(count, dtype=np.float64) / count
            all_points.append(start + fraction[:, None] * (end - start))
            all_weights.append(np.full(count, length / count))
    if not all_points:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.concatenate(all_points), np.concatenate(all_weights)


def dense_boundary_points(mask: np.ndarray, max_step: float = 0.5) -> np.ndarray:
    return dense_boundary_samples(mask, max_step)[0]


def distance_to_boundary(points: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Exact Euclidean point-to-subpixel-segment distances for offline G4 cost."""
    points = np.asarray(points, dtype=np.float64)
    shape = points.shape[:-1]
    flat = points.reshape(-1, 2)
    contours = mask_contours(gt_mask)
    if not contours:
        return np.full(shape, np.inf)
    a = np.concatenate(contours)
    b = np.concatenate([np.roll(contour, -1, axis=0) for contour in contours])
    delta = b - a
    length2 = np.square(delta).sum(axis=1)
    result = np.empty(len(flat), dtype=np.float64)
    for begin in range(0, len(flat), 64):
        rel = flat[begin:begin + 64, None, :] - a[None, :, :]
        fraction = np.clip(np.sum(rel * delta[None, :, :], axis=-1) / length2[None, :], 0, 1)
        residual = rel - fraction[..., None] * delta[None, :, :]
        result[begin:begin + 64] = np.sqrt(np.square(residual).sum(axis=-1).min(axis=1))
    return result.reshape(shape)


def normal_gt_intersections(geometry: ContourGeometry, gt_mask: np.ndarray) -> dict[str, Any]:
    contours = mask_contours(gt_mask)
    counts = np.zeros(len(geometry.points), dtype=np.int64)
    single_offsets = np.full(len(geometry.points), np.nan)
    collinear = np.zeros(len(geometry.points), dtype=bool)
    intersections: list[np.ndarray] = []
    if contours:
        a = np.concatenate(contours)
        b = np.concatenate([np.roll(contour, -1, axis=0) for contour in contours])
        edges = b - a
        for row, (point, normal) in enumerate(zip(geometry.points, geometry.normals)):
            relative = a - point
            denominator = _cross(normal, edges)
            parallel = np.abs(denominator) <= 1e-10
            projected_a = relative @ normal
            projected_b = (b - point) @ normal
            overlaps_range = (np.minimum(projected_a, projected_b) <= geometry.search_radius) & (np.maximum(projected_a, projected_b) >= -geometry.search_radius)
            collinear[row] = bool(np.any(parallel & (np.abs(_cross(relative, normal)) <= 1e-8) & overlaps_range))
            proper = ~parallel
            t = _cross(relative[proper], edges[proper]) / denominator[proper]
            u = _cross(relative[proper], normal) / denominator[proper]
            t = np.sort(t[(u >= -1e-8) & (u <= 1 + 1e-8) & (np.abs(t) <= geometry.search_radius + 1e-8)])
            if len(t):
                t = t[np.r_[True, np.diff(t) > 1e-7]]
            intersections.append(t)
            counts[row] = len(t)
            if len(t) == 1 and not collinear[row]:
                single_offsets[row] = t[0]
    else:
        intersections = [np.empty(0) for _ in geometry.points]
    valid = (counts == 1) & ~collinear
    return {"counts": counts, "single_offsets": single_offsets, "valid": valid, "collinear": collinear, "intersections": intersections}


def coverage_diagnostics(geometry: ContourGeometry, candidates: CandidateField, gt_mask: np.ndarray, tolerance: float = 2.0) -> dict[str, Any]:
    gt = _mask(gt_mask)
    if gt.shape != geometry.shape or tolerance != 2.0:
        raise ValueError("Use matching input coordinates and the fixed 2-input-pixel tolerance")
    dense_gt, dense_weights = dense_boundary_samples(gt, max_step=0.5)
    in_band = np.zeros(len(dense_gt), dtype=bool)
    for quad in strip_quadrilaterals(geometry):
        remaining = np.flatnonzero(~in_band)
        if not len(remaining):
            break
        relevant = dense_gt[remaining]
        within_box = (relevant[:, 0] >= quad[:, 0].min()) & (relevant[:, 0] <= quad[:, 0].max()) & (relevant[:, 1] >= quad[:, 1].min()) & (relevant[:, 1] <= quad[:, 1].max())
        selected = remaining[within_box]
        in_band[selected] |= points_in_polygon(dense_gt[selected], quad)
    intersections = normal_gt_intersections(geometry, gt)
    valid = intersections["valid"]
    target_offsets = intersections["single_offsets"]
    distance = np.abs(candidates.offsets - target_offsets[:, None])
    distance = np.where(candidates.valid, distance, np.inf)
    covered = valid & np.any(distance <= tolerance, axis=1)
    local = np.argmax(np.where(candidates.valid, candidates.scores, -np.inf), axis=1)
    selected_distance = distance[np.arange(len(local)), local]
    wrong = covered & (selected_distance > tolerance)
    def fraction(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None
    result = {
        "search_band_coverage": float(dense_weights[in_band].sum() / dense_weights.sum()) if len(dense_weights) else None,
        "search_band_covered_length_input_px": float(dense_weights[in_band].sum()),
        "search_band_total_length_input_px": float(dense_weights.sum()),
        "search_band_coverage_weighting": "subpixel_gt_segment_arc_length",
        "search_band_covered_dense_points": int(in_band.sum()),
        "search_band_total_dense_points": len(in_band),
        "dense_boundary_max_step_input_px": 0.5,
        "total_nodes": len(valid),
        "effective_normal_nodes": int(valid.sum()),
        "effective_normal_fraction": float(valid.mean()),
        "no_intersection_nodes": int((intersections["counts"] == 0).sum()),
        "multiple_intersection_nodes": int((intersections["counts"] > 1).sum()),
        "collinear_intersection_nodes": int(intersections["collinear"].sum()),
        "edge_clipped_nodes": int(geometry.edge_nodes.sum()),
        "candidate_covered_effective_nodes": int(covered.sum()),
        "candidate_coverage_effective": fraction(int(covered.sum()), int(valid.sum())),
        "candidate_coverage_all_nodes": float(covered.mean()),
        "local_wrong_among_covered_nodes": int(wrong.sum()),
        "local_wrong_rate_given_candidate_coverage": fraction(int(wrong.sum()), int(covered.sum())),
        "prediction_components": geometry.source_components,
        "prediction_holes": geometry.source_holes,
        "prediction_contour_count": geometry.source_contour_count,
        "gt_components": _component_count(gt),
        "gt_holes": _component_count(ndimage.binary_fill_holes(gt, structure=CONNECTIVITY_8) & ~gt),
        "gt_contour_count": len(mask_contours(gt)),
        "component_hole_connectivity": 8,
        "contour_extraction": "contourpy binary 0.5 isocontours; saddle loops may differ from 8-connected pixel components",
        "contour_self_intersections": geometry.contour_self_intersections,
        "strip_self_intersections": geometry.strip_self_intersections,
        "degenerate_strip_count": geometry.degenerate_strip_count,
        # Full per-node records allow auditing validity without GT case removal.
        "node_valid": valid,
        "node_gt_offset": target_offsets,
        "node_gt_intersection_counts": intersections["counts"],
        "node_gt_intersections": intersections["intersections"],
        "node_candidate_covered": covered,
        "node_local_wrong": wrong,
    }
    return result


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    value = _mask(mask)
    return value & ~ndimage.binary_erosion(value, structure=np.ones((3, 3), dtype=bool), border_value=0)


def segmentation_metrics(prediction: np.ndarray, gt_mask: np.ndarray, boundary_tolerance: float = 2.0) -> dict[str, float]:
    prediction, gt = _mask(prediction), _mask(gt_mask)
    if prediction.shape != gt.shape or boundary_tolerance != 2.0:
        raise ValueError("Metrics require common input coordinates and fixed BF1 tolerance 2")
    intersection = int(np.count_nonzero(prediction & gt))
    pred_count, gt_count = int(prediction.sum()), int(gt.sum())
    union = pred_count + gt_count - intersection
    dice = 2 * intersection / (pred_count + gt_count) if pred_count + gt_count else 1.0
    iou = intersection / union if union else 1.0
    pb, gb = boundary_pixels(prediction), boundary_pixels(gt)
    if not pb.any() and not gb.any():
        precision, recall, bf1 = 1.0, 1.0, 1.0
    elif not pb.any() or not gb.any():
        precision, recall, bf1 = 0.0, 0.0, 0.0
    else:
        precision = float((ndimage.distance_transform_edt(~gb)[pb] <= boundary_tolerance).mean())
        recall = float((ndimage.distance_transform_edt(~pb)[gb] <= boundary_tolerance).mean())
        bf1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"dice": float(dice), "iou": float(iou), "bf1": float(bf1), "boundary_precision": precision, "boundary_recall": recall}


def evaluate_selectors(prediction_mask: np.ndarray, score_map: np.ndarray, gt_mask: np.ndarray, smooth_lambda: float, *, nodes: int = 128, search_radius: float = 16.0) -> dict[str, Any]:
    """G4 uses GT solely after the fixed candidates exist, for offline diagnosis.

    No per-node scores are called Dice upper bounds. Every chosen complete path
    is rasterized and rescored. Empty predictions stay in the experiment with
    empty outputs and explicitly unavailable candidate diagnostics.
    """
    prediction, gt = _mask(prediction_mask), _mask(gt_mask)
    if prediction.shape != gt.shape:
        raise ValueError("Prediction and GT must share model-input coordinates")
    geometry = extract_geometry(prediction, nodes, search_radius)
    if geometry is None:
        masks = {f"G{index}": prediction.copy() for index in range(5)}
        coverage = {
            "search_band_coverage": 0.0 if gt.any() else None,
            "search_band_covered_dense_points": 0,
            "search_band_total_dense_points": len(dense_boundary_points(gt)),
            "total_nodes": 0, "effective_normal_nodes": 0,
            "effective_normal_fraction": 0.0, "no_intersection_nodes": 0,
            "multiple_intersection_nodes": 0, "collinear_intersection_nodes": 0,
            "edge_clipped_nodes": 0, "candidate_covered_effective_nodes": 0,
            "candidate_coverage_effective": None, "candidate_coverage_all_nodes": None,
            "local_wrong_among_covered_nodes": 0,
            "local_wrong_rate_given_candidate_coverage": None,
            "prediction_components": 0, "prediction_holes": 0,
            "prediction_contour_count": 0,
            "gt_components": _component_count(gt),
            "gt_holes": _component_count(ndimage.binary_fill_holes(gt, structure=CONNECTIVITY_8) & ~gt),
            "gt_contour_count": len(mask_contours(gt)),
            "component_hole_connectivity": 8,
            "contour_self_intersections": 0, "strip_self_intersections": 0,
            "degenerate_strip_count": 0,
        }
        return {"status": "empty_prediction_no_contour", "geometry": None, "candidates": None, "masks": masks, "metrics": {name: segmentation_metrics(mask, gt) for name, mask in masks.items()}, "coverage": coverage, "g4_deployable": False, "g4_is_dice_upper_bound": False}
    candidates = generate_candidates(score_map, geometry)
    rows = np.arange(nodes)
    zero = np.zeros(nodes, dtype=np.int64)
    local = np.argmax(np.where(candidates.valid, candidates.scores, -np.inf), axis=1)
    dp, dp_value = closed_dp(candidates.local_costs, candidates.offsets, smooth_lambda)
    gt_costs = np.where(candidates.valid, distance_to_boundary(candidates.points, gt), np.inf)
    if gt.any():
        diagnostic, diagnostic_value = closed_dp(gt_costs, candidates.offsets, smooth_lambda)
        status = "ok"
    else:
        diagnostic, diagnostic_value, status = zero, None, "empty_gt_no_g4_boundary"
    indices = {"G1": zero, "G2": local, "G3": dp, "G4": diagnostic}
    masks = {"G0": prediction.copy()}
    masks.update({name: rasterize(candidates.points[rows, selected], prediction.shape) for name, selected in indices.items()})
    paths = {name: candidates.points[rows, selected] for name, selected in indices.items()}
    return {"status": status, "geometry": geometry, "candidates": candidates, "indices": indices, "paths": paths, "path_self_intersections": {name: self_intersection_count(path) for name, path in paths.items()}, "masks": masks, "metrics": {name: segmentation_metrics(mask, gt) for name, mask in masks.items()}, "coverage": coverage_diagnostics(geometry, candidates, gt), "g3_objective": dp_value, "g4_objective": diagnostic_value, "g4_deployable": False, "g4_is_dice_upper_bound": False}
