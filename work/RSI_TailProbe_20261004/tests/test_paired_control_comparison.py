"""Hand-derived CPU tests for direct paired controls and reranked worst tails."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/paired_control_comparison.py"
SPEC = importlib.util.spec_from_file_location("paired_controls", TOOL_PATH)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def reference_rows(gains, groups):
    return [{"image_id": f"I{index:02d}", "reference_id": reference,
             "group_id": groups[index], "dice": .5 + gain, "dice_anchor": .5}
            for index, gain in enumerate(gains) for reference in ("r1", "r2")]


def hand_points():
    groups = ["case-A"] * 10 + ["case-B"]
    current_rows = reference_rows([-.3] + [.2] * 9 + [-.5], groups)
    control_rows = reference_rows([-.4] + [.1] * 9 + [-.1], groups)
    return tool.summarize_references(current_rows), tool.summarize_references(control_rows)


def test_hand_derived_means_and_group_resampled_tail_are_reranked():
    current, control = hand_points()
    # The two case units produce AA(20 images), AB/BA(11), BB(2).
    # Tail differences are +.1, -.15, -.15, -.4; each method ranks independently.
    draws = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    result = tool.compare_points(current, control, repeats=4, draws=draws)
    assert result["dice_difference"] == pytest.approx(.6 / 11)
    assert result["G_plus_difference"] == pytest.approx(.9 / 11)
    assert result["H_epsilon_difference"] == pytest.approx(.3 / 11)
    assert result["worst_tail_10pct_rsi"] == pytest.approx(-.4)
    assert result["worst_tail_10pct_control"] == pytest.approx(-.25)
    assert result["worst_tail_10pct_difference"] == pytest.approx(-.15)
    assert result["worst_tail_10pct_ci_low"] == pytest.approx(-.38125)
    assert result["worst_tail_10pct_ci_high"] == pytest.approx(.08125)
    assert result["worst_tail_n_observed"] == 2  # ceil(0.1*11), not floor.
    assert result["bootstrap_units"] == 2
    assert (result["resampled_image_count_min"], result["resampled_image_count_max"]) == (2, 20)
    assert result["worst_tail_reranked_each_draw"]


def test_tail_of_each_method_is_not_tail_of_differences_or_mean_worst_reference():
    groups = ["one-case"] * 10
    current = tool.summarize_references(reference_rows([-.3, -.2] + [.1] * 8, groups))
    control = tool.summarize_references(reference_rows([-.2, -.4] + [.1] * 8, groups))
    result = tool.compare_points(current, control, repeats=3)
    assert result["worst_tail_10pct_difference"] == pytest.approx(.1)
    assert result["worst_tail_10pct_ci_low"] == pytest.approx(.1)
    assert result["worst_tail_10pct_ci_high"] == pytest.approx(.1)
    assert tool.tail_mean(current["worst"] - control["worst"]) == pytest.approx(-.1)
    assert (current["worst"] - control["worst"]).mean() == pytest.approx(.01)


def test_fixed_rng_and_reference_permutation_are_reproducible():
    current, control = hand_points()
    first = tool.compare_points(current, control, repeats=16, seed=17)
    second = tool.compare_points(current, control, repeats=16, seed=17)
    assert first == second
    groups = ["case-A"] * 10 + ["case-B"]
    rows = reference_rows([-.3] + [.2] * 9 + [-.5], groups)
    shuffled = np.random.default_rng(99).permutation(len(rows))
    reordered = tool.summarize_references([rows[int(index)] for index in shuffled])
    assert tool.compare_points(reordered, control, repeats=16, seed=17) == first
    assert first["worst_tail_10pct_ci_low"] == pytest.approx(-.4)
    assert first["worst_tail_10pct_ci_high"] == pytest.approx(.1)


def test_reference_mean_precedes_image_mean_and_missing_groups_are_independent():
    rows = [dict(image_id="i1", reference_id="r1", group_id="", dice=.8, dice_anchor=.5),
            dict(image_id="i1", reference_id="r2", group_id="", dice=.2, dice_anchor=.5),
            dict(image_id="i2", reference_id="r1", group_id="", dice=.8, dice_anchor=.5),
            dict(image_id="i2", reference_id="r2", group_id="", dice=.8, dice_anchor=.5),
            dict(image_id="i2", reference_id="r3", group_id="", dice=.8, dice_anchor=.5)]
    point = tool.summarize_references(rows)
    assert point["values"]["dice"] == pytest.approx(.65)
    assert point["values"]["G_plus"] == pytest.approx(.225)
    units, kind = tool.make_units(point["image_ids"], point["group_ids"])
    assert kind == "image" and len(units) == 2


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def make_inputs(tmp_path):
    groups = ["case-A"] * 10 + ["case-B"]
    specifications = [("RSI-1", "rsi", 1., [-.3] + [.2] * 9 + [-.5]),
                      ("D0", "d0", 0., [-.4] + [.1] * 9 + [-.1])]
    rows, main = [], []
    for run_id, method, weight, gains in specifications:
        references = reference_rows(gains, groups)
        point = tool.summarize_references(references)
        meta = {"protocol_id": "toy", "seed": 17, "run_id": run_id, "method": method,
                "checkpoint_hash": run_id + "-selected", "action": 1., "lambda": weight,
                "alpha": 1., "subset": "M", "split": "val"}
        rows.extend([{**meta, **row} for row in references])
        main.append({**meta, **point["values"], "n_images": 11, "n_references": 22,
                     "workpoint_selected": run_id == "D0", "workpoint_status": "toy-existing-flag"})
    # Numerically invalid H fields are deliberately never parsed or selected.
    rows.append({**rows[0], "subset": "H", "dice": "not-read"})
    main.append({**main[0], "subset": "H", "dice": "not-read", "workpoint_selected": True})
    input_path, main_path = tmp_path / "per_reference_gains.csv", tmp_path / "pilot_main_table.csv"
    write_csv(input_path, rows); write_csv(main_path, main)
    return input_path, main_path


def test_complete_CPU_pipeline_uses_existing_M_flags_and_keeps_original_bytes(tmp_path):
    input_path, main_path = make_inputs(tmp_path)
    originals = (input_path.read_bytes(), main_path.read_bytes())
    output = tmp_path / "paired"
    audit = tool.finalize(input_path, main_path, output, repeats=16, seed=17)
    assert audit["status"] == "complete" and audit["comparison_count"] == 1
    assert audit["selected_rsi_from_existing_main_table"] == []  # H=True must not change M=False.
    assert audit["test_scoring_locked"] and not audit["workpoints_reselected"]
    assert not audit["new_lambda_or_alpha"] and not audit["GPU_compute"]
    with (output / "paired_control_comparison.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["control_run_id"] == "D0"
    assert row["rsi_workpoint_selected"] == "False"
    assert float(row["worst_tail_10pct_difference"]) == pytest.approx(-.15)
    previous = (output / "paired_control_comparison.csv").read_bytes()
    rerun = tool.finalize(input_path, main_path, output, repeats=16, seed=17)
    assert (output / "paired_control_comparison.csv").read_bytes() == previous
    assert len(rerun["prior_output_backups"]) == 2
    assert (input_path.read_bytes(), main_path.read_bytes()) == originals
    recorded = json.loads((output / "paired_control_comparison.json").read_text())
    assert recorded["output_csv_sha256"] == tool.file_hash(output / "paired_control_comparison.csv")


def test_group_and_reference_pairing_mismatches_are_rejected():
    current, control = hand_points()
    control["group_ids"] = ["different"] * len(control["group_ids"])
    with pytest.raises(ValueError, match="case groups"):
        tool.compare_points(current, control, repeats=3)
    current, control = hand_points()
    control["reference_keys"] = set()
    with pytest.raises(ValueError, match="reference identities"):
        tool.compare_points(current, control, repeats=3)


def test_test_scores_and_original_mislabeled_B_are_rejected(tmp_path):
    input_path, main_path = make_inputs(tmp_path)
    with input_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["split"] = "test"
    write_csv(input_path, rows)
    with pytest.raises(PermissionError, match="validation"):
        tool.read_existing_points(input_path, main_path)
    rows[0]["split"] = "val"
    rows[0]["run_id"] = "B"; rows[0]["method"] = "d0"
    write_csv(input_path, rows)
    with pytest.raises(ValueError, match="corrected B"):
        tool.read_existing_points(input_path, main_path)


def test_interrupted_publication_restores_previous_CSV_and_audit(tmp_path, monkeypatch):
    input_path, main_path = make_inputs(tmp_path)
    output = tmp_path / "paired"
    tool.finalize(input_path, main_path, output, repeats=16, seed=17)
    originals = {name: (output / name).read_bytes() for name in
                 ("paired_control_comparison.csv", "paired_control_comparison.json")}
    real_replace = tool.os.replace
    def fail_second_output(source, target):
        if Path(target).name == "paired_control_comparison.json":
            raise OSError("simulated publication failure")
        return real_replace(source, target)
    monkeypatch.setattr(tool.os, "replace", fail_second_output)
    with pytest.raises(OSError, match="publication"):
        tool.finalize(input_path, main_path, output, repeats=16, seed=17)
    assert {name: (output / name).read_bytes() for name in originals} == originals
