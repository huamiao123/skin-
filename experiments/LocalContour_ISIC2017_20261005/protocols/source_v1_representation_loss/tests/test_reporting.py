"""Synthetic inference checks; these do not measure the proposed method."""

import json

import numpy as np
import pytest

from contour.reporting import default_reporting_rules, paired_bootstrap, summarize


def comparison(result, method, baseline, metric):
    return next(row for row in result["comparisons"] if (row["method"], row["baseline"], row["metric"]) == (method, baseline, metric))


def test_paired_bootstrap_cancels_large_common_image_difficulty():
    rng = np.random.default_rng(21)
    baseline = rng.uniform(0.2, 0.8, (37, 3))
    values = np.stack([baseline + method * 0.005 for method in range(5)], axis=1)
    result = paired_bootstrap(values, [f"image_{i:03d}" for i in range(37)])
    row = comparison(result, "G4", "G0", "dice")
    assert row["difference"] == pytest.approx(0.020, abs=1e-12)
    assert row["ci_low"] == pytest.approx(0.020, abs=1e-12)
    assert row["ci_high"] == pytest.approx(0.020, abs=1e-12)
    assert row["harmed_images"] == 0


def test_shared_draws_preserve_linear_relation_between_comparison_intervals():
    difficulty = np.linspace(-0.02, 0.03, 31)
    values = np.full((31, 5, 3), 0.6)
    values[:, 3] += difficulty[:, None]
    values[:, 4] += 2 * difficulty[:, None]
    result = paired_bootstrap(values, [str(i) for i in range(31)])
    g3 = comparison(result, "G3", "G0", "dice")
    g4 = comparison(result, "G4", "G0", "dice")
    assert g4["ci_low"] == pytest.approx(2 * g3["ci_low"], abs=1e-12)
    assert g4["ci_high"] == pytest.approx(2 * g3["ci_high"], abs=1e-12)
    assert result["draw_indices_sha256"] == paired_bootstrap(values, [str(i) for i in range(31)])["draw_indices_sha256"]


def fixture_rows():
    rows = []
    for index in range(150):
        diagnostics = {
            "search_band_coverage": 0.95, "effective_normal_nodes": 120,
            "total_nodes": 128, "candidate_covered_effective_nodes": 108,
            "local_wrong_among_covered_nodes": 30, "no_intersection_nodes": 4,
            "multiple_intersection_nodes": 4, "collinear_intersection_nodes": 0,
            "edge_clipped_nodes": 8, "prediction_components": 2,
            "prediction_holes": 1, "gt_components": 2, "gt_holes": 0,
            "contour_self_intersections": 0, "strip_self_intersections": 1,
            "degenerate_strip_count": 1, "status": "difficult_case_retained",
        }
        for method, gain in zip(("G0", "G1", "G2", "G3", "G4"), (0, -0.0002, -0.001, 0.002, 0.007)):
            rows.append({"image_id": f"ISIC_{index:07d}", "role": "cal50" if index < 50 else "locked100", "method": method, "dice": 0.8 + gain, "iou": 0.7 + gain, "bf1": 0.7 + gain, **diagnostics})
    return rows


def test_complete_report_retains_all_difficult_images_and_requires_complete_pairs(tmp_path):
    rows = fixture_rows()
    prereg = {"reporting_rules": default_reporting_rules(), "recorded_before_results": True}
    result = summarize(rows, prereg, {"nodes": 128}, tmp_path / "complete")
    assert result["n_rows"] == 750
    assert result["decision"]["status"] == "CANDIDATE_NEXT_SMALL_COMPARISON"
    assert result["scope_results"]["locked100"]["coverage"]["multiple_prediction_component_images"] == 100
    assert result["scope_results"]["locked100"]["coverage"]["strip_self_intersections_images"] == 100
    assert result["all_five_selectors_all_images_retained"]
    saved = json.loads((tmp_path / "complete" / "decision.json").read_text())
    assert not saved["independent_test_claim"]
    reconstructed_rows = [dict(row) for row in rows]
    for row in reconstructed_rows:
        if row["method"] == "G1":
            row["dice"] = 0.806
    reconstructed = summarize(reconstructed_rows, prereg, {"nodes": 128}, tmp_path / "reconstruction_explains")
    assert reconstructed["decision"]["status"] == "KEEP_RECONSTRUCTION_SIMPLE"
    assert reconstructed["decision"]["reconstruction_explained_fraction"] == pytest.approx(6 / 7)
    with pytest.raises(ValueError, match="retain all five"):
        summarize(rows[:-1], prereg, {"nodes": 128}, tmp_path / "incomplete")
    with pytest.raises(ValueError, match="Record the complete"):
        summarize(rows, {}, {"nodes": 128}, tmp_path / "unregistered")
