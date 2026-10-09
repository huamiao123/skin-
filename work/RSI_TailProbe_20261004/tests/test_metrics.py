import csv
import math

import numpy as np
import pytest

from rsi.metrics import (
    binary_mask_metrics,
    evaluate_reference,
    gain_loss_dice_cross_table,
    gain_statistics,
    paired_bootstrap,
    update_diagnostics,
)
from rsi.summarize import summarize_pilot, validate_rows


def test_empty_conventions_and_original_image_diagonal_penalty():
    empty = np.zeros((3, 4), dtype=bool)
    full = np.ones((3, 4), dtype=bool)
    both = binary_mask_metrics(empty, empty)
    assert both["dice"] == both["iou"] == both["bf1"] == 1
    assert both["hd95"] == both["hd95_normalized"] == 0
    one = binary_mask_metrics(empty, full)
    assert one["dice"] == one["iou"] == one["bf1"] == 0
    assert one["hd95"] == 5
    assert one["hd95_normalized"] == 1
    assert one["pred_empty"] == one["one_empty"] == 1


def test_hd95_is_percentile_of_merged_bidirectional_nearest_distances():
    pred, gt = np.zeros((6, 3), bool), np.zeros((6, 3), bool)
    pred[1, 1] = gt[1, 1] = gt[4, 1] = True
    result = binary_mask_metrics(pred, gt)
    # Directed distances [0] and [0,3], merged [0,0,3].
    assert result["hd95"] == pytest.approx(2.7)
    assert result["hd95"] != pytest.approx(2.85)  # maximum directed 95th
    assert result["bf1"] == pytest.approx(2 / 3)
    assert result["dice"] == pytest.approx(2 / 3)
    assert result["iou"] == 0.5
    assert result["thresholded_jaccard"] == 0


def test_boundary_tolerance_is_original_diagonal_based_with_sensitivity():
    p, g = np.zeros((400, 400), bool), np.zeros((400, 400), bool)
    p[200, 200] = g[200, 202] = True
    result = binary_mask_metrics(p, g)
    assert result["bf1"] == 1  # 0.5% diagonal = 2.828 px
    assert result["bf1_025"] == 0  # 0.25% diagonal = 1.414 px
    assert result["bf1_1pct"] == 1
    assert result["hd95"] == 2
    assert result["hd95_normalized"] == pytest.approx(2 / math.hypot(400, 400))


def test_logits_threshold_one_pixel_images_and_paired_error_changes():
    current = np.array([[0.0, -1.0, 1.0, -1.0]])
    anchor = np.array([[-1.0, 1.0, -1.0, 1.0]])
    reference = np.array([[1, 0, 0, 1]])
    result = evaluate_reference(current, reference, anchor, current)
    assert result["dice"] == 0.5
    assert result["gain_dice"] == 0
    assert result["gain_vs_D0"] == 0
    assert all(result[k] == 1 for k in ("fp_added", "fn_added", "fp_removed", "fn_removed"))
    assert result["changed_pixel_fraction"] == 1
    tiny = evaluate_reference(np.array([[0.0]]), np.array([[1]]))
    assert tiny["dice"] == 1
    assert tiny["bce"] == pytest.approx(math.log(2))
    assert tiny["soft_dice"] == pytest.approx((1 + 1e-6) / (1.5 + 1e-6))
    with pytest.raises(ValueError, match="restored"):
        evaluate_reference(np.zeros((3, 3)), np.ones((2, 2)))


def test_update_rms_and_mask_change_exclude_padding():
    result = update_diagnostics(np.array([3.0, 4.0]), np.array([[[[1., -1., np.nan]]]]),
                                np.array([[[[-1., -1., np.nan]]]]), np.array([[[[1, 1, 0]]]]))
    assert result["message_rms"] == pytest.approx(math.sqrt(12.5))
    assert result["logit_delta_rms"] == pytest.approx(math.sqrt(2))
    assert result["changed_pixel_fraction"] == 0.5


def test_gain_harm_image_then_reference_reduction_and_sign_before_mean():
    result = gain_statistics([[0.1, -0.1], [0.2]], changed=[True, True])
    assert result["mean_gain"] == pytest.approx(0.1)
    assert result["G_plus"] == pytest.approx(0.125)
    assert result["H_minus"] == pytest.approx(0.025)
    assert result["H_epsilon"] == pytest.approx(0.02375)
    assert result["any_harm"] == 0.5  # image-average gain is zero for image 1
    assert result["sign_flip"] == 1  # denominator: all eligible multi-reference images
    assert result["sign_flip_n"] == 1
    assert result["sign_flip_changed_subset"] == 1
    assert result["mean_gain"] == pytest.approx(result["G_plus"] - result["H_minus"])
    assert result["H_epsilon_0p0"] == result["H_minus"]
    assert result["H_epsilon_0p01"] < result["H_epsilon_0p005"]


def test_worst_tail_uses_ceil_and_reference_order_and_copy_invariance():
    result = gain_statistics([[float(x), float(x) + 1] for x in range(-10, 1)])
    assert result["worst_tail_n"] == 2
    assert result["worst_tail_10pct"] == -9.5
    base = gain_statistics([[0.1, -0.1], [0.2, -0.05]])
    repeated = gain_statistics([[0.1, -0.1, 0.1, -0.1], [0.2, -0.05, 0.2, -0.05]])
    reversed_refs = gain_statistics([[-0.1, 0.1], [-0.05, 0.2]])
    for key in ("mean_gain", "G_plus", "H_epsilon", "any_harm", "sign_flip", "worst_tail_10pct"):
        assert base[key] == pytest.approx(repeated[key])
        assert base[key] == pytest.approx(reversed_refs[key])


def test_loss_hard_dice_tolerance_cross_table_uses_image_equal_weights():
    table = gain_loss_dice_cross_table([[0.1, 0.1], [-0.1]], [[0.02, -0.001], [-0.02]])
    assert table["loss_positive_dice_positive"] == 0.25
    assert table["loss_positive_dice_neutral"] == 0.25
    assert table["loss_negative_dice_negative"] == 0.5
    assert table["zero_loss_positive_dice_negative"] == 0.25
    assert sum(v for k, v in table.items() if not k.startswith("zero_")) == 1


def test_paired_bootstrap_preserves_groups_and_missing_groups_are_distinct():
    fixed = paired_bootstrap([0.5, 0.6], [0.3, 0.4], repeats=100, seed=5)
    assert fixed["difference"] == pytest.approx(0.2)
    assert fixed["ci_low"] == pytest.approx(0.2)
    assert fixed["ci_high"] == pytest.approx(0.2)
    grouped = paired_bootstrap([0.4, 0.6, 0.5], [0.3, 0.4, 0.2], group_ids=["case1", "case1", ""], repeats=100, seed=5)
    assert grouped["bootstrap_units"] == 2
    assert grouped["difference"] == pytest.approx(0.2)
    missing = paired_bootstrap([0.4, 0.6], [0.3, 0.4], group_ids=["", ""], repeats=100)
    assert missing["bootstrap_units"] == 2
    nan_missing = paired_bootstrap([0.4, 0.6], [0.3, 0.4], group_ids=[np.nan, np.nan], repeats=100)
    assert nan_missing["bootstrap_units"] == 2


def rows_for_summary():
    rows = []
    for method, dice_by_image in [("d0", [[0.8, 0.7], [0.6, 0.7, 0.8]]),
                                  ("rsi", [[0.8, 0.72], [0.61, 0.69, 0.8]])]:
        for i, dices in enumerate(dice_by_image):
            for r, dice in enumerate(dices):
                rows.append(dict(protocol_id="P2-IMA-M-v2", seed=17, run_id=method, method=method,
                                 checkpoint_hash=f"hash-{method}", image_id=f"img{i}", group_id=f"case{i}",
                                 subset="M", reference_id=f"ref{i}-{r}", dice=dice, iou=dice / (2 - dice),
                                 dice_anchor=0.7, loss=0.2, loss_anchor=0.3, tool="T1" if r == 0 else "T3",
                                 message_rms=0.1, logit_delta_rms=0.2, changed_pixel_fraction=0.1,
                                 **{"lambda": 1 if method == "rsi" else 0}))
        rows.append(dict(rows[-len(dice_by_image[-1])], subset="H", reference_id="H-distinct", dice=0.75, iou=0.6))
    return rows


def test_summary_exports_original_metrics_reference_sources_and_comparators(tmp_path):
    table = summarize_pilot(rows_for_summary(), tmp_path, bootstrap_repeats=20)
    m = [r for r in table if r["subset"] == "M" and r["method"] == "d0"][0]
    assert m["dice"] == pytest.approx((0.75 + 0.7) / 2)
    assert m["workpoint_selected"]
    assert "dice_vs_D0_ci_low" in m
    assert "H_epsilon_vs_D0_ci_low" in m
    expected = ["pilot_main_table.csv", "per_reference_gains.csv", "source_sensitivity.csv",
                "update_diagnostics.csv", "per_reference_gains_vs_anchor.csv", "per_reference_gains_vs_D0.csv", "gain_harm_tradeoff.png"]
    assert all((tmp_path / name).is_file() for name in expected)
    with (tmp_path / "source_sensitivity.csv").open() as f:
        sources = list(csv.DictReader(f))
    paired = [r for r in sources if r["report"] == "paired_M_references_on_source_images" and r["run_id"] == "d0"][0]
    assert float(paired["dice"]) == pytest.approx(0.7)  # exactly image 1, not all M images
    assert float(paired["source_minus_M_difference"]) == pytest.approx(0.05)
    assert any(r["report"] == "reference_count" and r["stratum"] == "m>=3" for r in sources)
    assert any(r["report"] == "fixed_two_references" for r in sources)


def test_summary_rejects_duplicate_keys_test_rows_and_inconsistent_update(tmp_path):
    rows = rows_for_summary()
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows(rows + [rows[0]])
    with pytest.raises(ValueError, match="test scoring"):
        summarize_pilot(rows, tmp_path, split="test")
    with pytest.raises(ValueError, match="requested"):
        summarize_pilot([dict(r, split="test") for r in rows], tmp_path)
    rows[0]["message_rms"] = 2
    with pytest.raises(ValueError, match="must agree"):
        summarize_pilot(rows, tmp_path, bootstrap_repeats=2)


def test_workpoint_selection_dice_tolerance_then_harm_and_ties():
    from rsi.summarize import _select_workpoints
    common = dict(protocol_id="P2", seed=17, subset="M")
    main = [dict(common, method="d0", dice=0.8, H_epsilon=0.03),
            dict(common, method="rsi", dice=0.797, H_epsilon=0.0, **{"lambda": 3}),
            dict(common, method="rsi", dice=0.799, H_epsilon=0.01, **{"lambda": 1}),
            dict(common, method="rsi", dice=0.799, H_epsilon=0.01, **{"lambda": 0.3}),
            dict(common, method="shrink", dice=0.8, H_epsilon=0.02, alpha=1/3),
            dict(common, method="shrink", dice=0.8, H_epsilon=0.02, alpha=2/3)]
    _select_workpoints(main)
    assert main[0]["workpoint_selected"]
    assert not main[1]["workpoint_selected"]  # fails absolute Dice tolerance
    assert not main[2]["workpoint_selected"]
    assert main[3]["workpoint_selected"]  # same Dice/harm, smaller lambda
    assert not main[4]["workpoint_selected"]
    assert main[5]["workpoint_selected"]  # same Dice/harm, larger alpha
