"""Meaningful CPU engineering checks, not evidence of method performance."""

from itertools import product

import numpy as np
import pytest
from scipy.spatial import cKDTree

from contour.geometry import (
    closed_dp, coverage_diagnostics, dense_boundary_points,
    dense_boundary_samples, distance_to_boundary, evaluate_selectors,
    extract_geometry, generate_candidates, mask_contours,
    normal_gt_intersections, path_objective, rasterize,
    segmentation_metrics, self_intersection_count,
)


def disk(shape=(128, 128), radius=30, centre=(64, 64)):
    yy, xx = np.indices(shape)
    return (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= radius ** 2


@pytest.mark.parametrize("seed", [1, 7, 17])
def test_closed_dp_matches_complete_path_enumeration(seed):
    rng = np.random.default_rng(seed)
    costs = rng.uniform(0, 2, (5, 3))
    offsets = rng.normal(size=(5, 3))
    costs[2, 1] = np.inf
    strength = 0.73
    path, value = closed_dp(costs, offsets, strength)
    exhaustive = min(
        path_objective(np.array(p), costs, offsets, strength)
        for p in product(range(3), repeat=5)
    )
    assert value == pytest.approx(exhaustive, abs=1e-12)
    assert path_objective(path, costs, offsets, strength) == pytest.approx(value, abs=1e-12)


def test_dp_includes_last_to_first_connection():
    costs = np.array([[0, 5], [0, 5], [9, 0]], dtype=float)
    offsets = np.tile([0.0, 3.0], (3, 1))
    path, value = closed_dp(costs, offsets, 1.0)
    # An open chain chooses (0,0,1), while the omitted closing edge changes it.
    assert tuple(path) == (0, 0, 0)
    assert value == 9


def test_dp_rejects_node_without_candidate_and_handles_one_node():
    with pytest.raises(ValueError):
        closed_dp(np.array([[np.inf, np.inf]]), np.zeros((1, 2)), 0.1)
    path, value = closed_dp(np.array([[3.0, 2.0]]), np.array([[0.0, 9.0]]), 100)
    assert tuple(path) == (1,)
    assert value == 2.0


@pytest.mark.parametrize("kind", ["square", "disk", "border", "single_pixel"])
def test_subpixel_contour_rasterization_roundtrip(kind):
    mask = np.zeros((32, 40), dtype=bool)
    if kind == "square":
        mask[6:27, 9:30] = True
    elif kind == "disk":
        mask = disk((32, 40), 11, (20, 16))
    elif kind == "border":
        mask[:10, :19] = True
    else:
        mask[17, 21] = True
    contours = mask_contours(mask)
    assert len(contours) == 1
    assert np.array_equal(rasterize(contours[0], mask.shape), mask)


def test_candidates_have_zero_backup_and_separated_gt_free_peaks():
    mask = disk()
    geometry = extract_geometry(mask)
    yy, xx = np.indices(mask.shape)
    # Concentric clear scoring ridges create local alternatives independently of GT.
    radial = np.sqrt((xx - 64) ** 2 + (yy - 64) ** 2)
    score = np.exp(-((radial - 26) / 0.6) ** 2) * 0.8 + np.exp(-((radial - 34) / 0.6) ** 2) * 0.9
    candidates = generate_candidates(score, geometry)
    assert candidates.valid[:, 0].all()
    assert np.array_equal(candidates.offsets[:, 0], np.zeros(128))
    assert np.array_equal(candidates.points[:, 0], geometry.points)
    for row in range(128):
        peaks = candidates.offsets[row, 1:][candidates.valid[row, 1:]]
        assert len(peaks) <= 3
        assert all(abs(a - b) >= 2 for a, b in product(peaks, repeat=2) if a != b)
    with pytest.raises(TypeError):
        generate_candidates(score, geometry, gt_mask=mask)


def test_dense_band_coverage_avoids_128_node_density_bias():
    mask = np.zeros((512, 512), dtype=bool)
    mask[40:472, 40:472] = True
    geometry = extract_geometry(mask)
    candidates = generate_candidates(np.zeros(mask.shape), geometry)
    result = coverage_diagnostics(geometry, candidates, mask)
    # Most GT boundary points are farther than 2px from the sparse node centres.
    dense = dense_boundary_points(mask)
    sparse_distance = cKDTree(geometry.points).query(dense)[0]
    assert (sparse_distance <= 2).mean() < 0.5
    assert result["search_band_coverage"] > 0.999
    assert result["search_band_total_dense_points"] > 128


def test_dense_sampling_weights_sum_to_true_perimeter():
    mask = disk((64, 64), 15, (32, 32))
    p1, w1 = dense_boundary_samples(mask, max_step=0.5)
    p2, w2 = dense_boundary_samples(mask, max_step=0.1)
    assert len(p2) > len(p1)
    assert w1.sum() == pytest.approx(w2.sum(), abs=1e-10)


def test_normal_intersections_separate_missing_and_multiple_cases():
    prediction = disk(radius=30)
    geometry = extract_geometry(prediction)
    single = normal_gt_intersections(geometry, disk(radius=34))
    assert single["valid"].mean() > 0.95
    absent = normal_gt_intersections(geometry, disk(radius=8))
    assert not absent["valid"].any()
    assert (absent["counts"] == 0).all()
    annulus = disk(radius=34) & ~disk(radius=26)
    multiple = normal_gt_intersections(geometry, annulus)
    assert (multiple["counts"] >= 2).mean() > 0.95
    assert not multiple["valid"].any()


def test_exact_point_to_boundary_distance_not_sparse_node_distance():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True
    distance = distance_to_boundary(np.array([[20, 9.5], [20, 7.5], [20, 10.0]]), mask)
    assert np.allclose(distance, [0, 2, 0.5], atol=1e-12)


def test_bf1_tolerance_has_input_pixel_units_and_empty_convention():
    gt = np.zeros((30, 30), dtype=bool)
    gt[8:22, 8:22] = True
    shifted2 = np.zeros_like(gt)
    shifted2[8:22, 10:24] = True
    shifted3 = np.zeros_like(gt)
    shifted3[8:22, 11:25] = True
    assert segmentation_metrics(shifted2, gt)["bf1"] == 1.0
    assert segmentation_metrics(shifted3, gt)["bf1"] < 1.0
    assert segmentation_metrics(np.zeros_like(gt), np.zeros_like(gt))["bf1"] == 1.0
    assert segmentation_metrics(np.zeros_like(gt), gt)["bf1"] == 0.0
    with pytest.raises(ValueError):
        segmentation_metrics(gt, gt, boundary_tolerance=3)


def test_g4_cannot_change_candidates_and_is_not_dice_bound():
    prediction = disk(radius=30)
    score = np.full(prediction.shape, 0.25)
    first = evaluate_selectors(prediction, score, disk(radius=34), 0.001)
    second = evaluate_selectors(prediction, score, disk(radius=26), 0.001)
    assert np.array_equal(first["candidates"].points, second["candidates"].points)
    assert np.array_equal(first["candidates"].scores, second["candidates"].scores)
    assert np.array_equal(first["masks"]["G3"], second["masks"]["G3"])
    assert not first["g4_deployable"]
    assert not first["g4_is_dice_upper_bound"]


def test_multicomponent_hole_and_edge_cases_are_recorded_not_dropped():
    mask = np.zeros((100, 100), dtype=bool)
    mask[:40, :40] = True
    mask[15:20, 15:20] = False
    mask[70:80, 70:80] = True
    result = evaluate_selectors(mask, np.full(mask.shape, 0.2), mask, 0.001)
    assert result["status"] == "ok"
    assert result["coverage"]["prediction_components"] == 2
    assert result["coverage"]["prediction_holes"] == 1
    assert result["coverage"]["edge_clipped_nodes"] > 0
    assert np.array_equal(result["masks"]["G0"], mask)
    assert result["metrics"]["G1"]["dice"] < result["metrics"]["G0"]["dice"]


def test_empty_prediction_is_kept_without_an_invented_candidate():
    mask = np.zeros((128, 128), dtype=bool)
    result = evaluate_selectors(mask, np.full(mask.shape, 0.2), disk(), 0.001)
    assert result["status"] == "empty_prediction_no_contour"
    assert result["candidates"] is None
    assert all(not output.any() for output in result["masks"].values())
    assert result["coverage"]["search_band_coverage"] == 0


def test_self_crossing_report_and_one_sensitivity_guard():
    bowtie = np.array([[0, 0], [3, 3], [0, 3], [3, 0]], dtype=float)
    assert self_intersection_count(bowtie) == 1
    with pytest.raises(ValueError, match="one sensitivity"):
        extract_geometry(disk(), nodes=256, search_radius=32)


def test_crossing_normal_strips_are_reported_without_gt_filter():
    # Radius 16 searches pass through this small object's opposite side.
    geometry = extract_geometry(disk(radius=9))
    assert geometry.strip_self_intersections > 0
    assert geometry.degenerate_strip_count > 0
    assert len(geometry.points) == 128


def test_diagonal_pixel_components_use_registered_eight_connectivity():
    mask = np.zeros((128, 128), dtype=bool)
    mask[55:60, 55:60] = True
    mask[60:65, 60:65] = True
    result = evaluate_selectors(mask, np.full(mask.shape, 0.2), mask, 0.001)
    assert result["coverage"]["prediction_components"] == 1
    assert result["coverage"]["gt_components"] == 1
    assert result["coverage"]["component_hole_connectivity"] == 8
    assert result["coverage"]["prediction_contour_count"] == len(mask_contours(mask))
    assert result["coverage"]["gt_contour_count"] == len(mask_contours(mask))
    assert np.array_equal(result["masks"]["G0"], mask)
