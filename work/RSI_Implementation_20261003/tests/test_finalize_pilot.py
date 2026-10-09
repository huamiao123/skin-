"""CPU regressions for metadata repair outside the frozen training bundle."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/finalize_pilot.py"
SPEC = importlib.util.spec_from_file_location("pilot_finalizer", TOOL_PATH)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def make_export_fixture(tmp_path, *, complete_t2):
    exports = tmp_path / "original_exports"
    runs = tmp_path / "runs"
    output = tmp_path / "summary"
    exports.mkdir()
    cfg = {"protocol_id": "toy-protocol", "test_scoring_locked": True,
           "epochs": {"A_CNN": 100, "A_T": 100, "B": 40, "D": 40},
           "audited_counts": {subset: {"references": {"val": 2}, "images": {"val": 1}}
                              for subset in ("M", "H", "T1")}}
    specifications = [("B", "d0", 0., 1., .81 if complete_t2 else .79)]
    if complete_t2:
        specifications += [("D0", "d0", 0., 1., .82), ("RSI-1", "rsi", 1., 1., .821),
                           ("MeanHinge-1", "mean_hinge", 1., 1., .82),
                           ("AbsHard-1", "abs_hard", 1., 1., .82)]
        specifications += [(f"Shrink-D0-{alpha:.6f}", "fixed_shrink", 0., alpha, .8 + .02 * alpha)
                           for alpha in (0., 1/3, 2/3, 1.)]
    stages = ["A_CNN", "A_T", "B"] + (["D0", "RSI-1", "MeanHinge-1", "AbsHard-1"] if complete_t2 else [])
    hashes = {stage: str(index + 1) * 64 for index, stage in enumerate(stages)}
    methods = {rid: (method, weight) for rid, method, weight, _, _ in specifications}
    for stage in stages:
        directory = runs / stage
        directory.mkdir(parents=True)
        actual_stage = stage if stage in ("A_CNN", "A_T", "B") else "D"
        budget = cfg["epochs"][actual_stage]
        parent = {"A_T": "A_CNN", "B": "A_T", "D": "B"}.get(actual_stage)
        done = {"stage": actual_stage, "seed": 17, "epochs": budget, "best_epoch": 1,
                "best_sha256": hashes[stage], "test_scoring_locked": True,
                "signature": {"config": cfg, "seed": 17, "manifest_hash": "fixed-manifest",
                              "source_bundle_hash": tool.frozen_sources()["source_bundle_hash"],
                              "init_checkpoint_hash": hashes[parent] if parent else None,
                              "method": methods.get(stage, ("d0", 0.))[0],
                              "weight": methods.get(stage, ("d0", 0.))[1],
                              "q": .45 if actual_stage == "D" else None}}
        (directory / "DONE.json").write_text(json.dumps(done))
        tool.write_csv(directory / "epoch_diagnostics.csv",
                       [{"epoch": epoch, "run_id": stage} for epoch in range(1, budget + 1)])
    for run_id, method, weight, alpha, dice in specifications:
        rows = []
        checkpoint = hashes["D0"] if run_id.startswith("Shrink-D0-") else hashes[run_id]
        for subset in ("M", "H", "T1"):
            for reference in ("ref-1", "ref-2"):
                row = {"protocol_id": "toy-protocol", "seed": 17, "run_id": run_id,
                       "checkpoint_hash": checkpoint, "manifest_hash": "fixed-manifest",
                       "image_id": "image-1", "group_id": "case-1", "subset": subset,
                       "split": "val", "reference_id": reference, "action": alpha,
                       "alpha": alpha, "lambda": weight, "method": method,
                       "dice": dice, "iou": dice / (2 - dice), "dice_anchor": .8,
                       "loss": .45, "loss_anchor": .5,
                       "message_rms": .01, "logit_delta_rms": .02,
                       "changed_pixel_fraction": .03}
                rows.append(row)
                if run_id == "B":
                    rows.append(dict(row, run_id="Anchor", method="anchor", action=0., alpha=0.,
                                     dice=.8, iou=.8 / 1.2, loss=.5, message_rms=0.,
                                     logit_delta_rms=0., changed_pixel_fraction=0.))
        tool.write_csv(exports / f"{run_id}.csv", rows)
    return exports, output, runs


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_full_t2_repairs_only_B_and_keeps_inputs_and_prior_outputs(tmp_path):
    exports, output, runs = make_export_fixture(tmp_path, complete_t2=True)
    originals = {path.name: path.read_bytes() for path in exports.glob("*.csv")}
    output.mkdir()
    (output / "pilot_main_table.csv").write_text("previous invalid summary\n")
    before_sources = tool.frozen_sources()
    audit = tool.finalize(exports, output, runs, bootstrap_repeats=5)
    assert {path.name: path.read_bytes() for path in exports.glob("*.csv")} == originals
    assert tool.frozen_sources() == before_sources
    assert audit["scope"] == "T2"
    assert sum(item["correction"]["rows_corrected"] for item in audit["original_csvs"]) == 6
    corrected = read_csv(output / "corrected_inputs/B.csv")
    assert all(row["method"] == "B" for row in corrected if row["run_id"] == "B")
    assert all(row["method"] == "anchor" for row in corrected if row["run_id"] == "Anchor")
    main = read_csv(output / "pilot_main_table.csv")
    m = [row for row in main if row["subset"] == "M"]
    assert len(m) == 10
    assert {row["run_id"] for row in m if row["method"] == "d0"} == {"D0"}
    assert next(row for row in m if row["run_id"] == "B")["method"] == "b"
    assert next(row for row in m if row["run_id"] == "B")["workpoint_selected"] == "False"
    backup = Path(audit["prior_output_backup"])
    assert (backup / "pilot_main_table.csv").read_text() == "previous invalid summary\n"
    for name in ("pilot_main_table.csv", "per_reference_gains.csv", "epoch_diagnostics.csv",
                 "update_diagnostics.csv", "source_sensitivity.csv", "gain_harm_tradeoff.png", "decision_log.md"):
        assert (output / name).is_file()
    decision = json.loads((output / "decision_inputs.json").read_text())
    assert decision["test_scoring_locked"] and decision["final_continue_or_stop_decision_pending"]
    assert decision["workpoint_comparison_available"]


def test_failed_T1_has_no_D0_and_no_selected_workpoint(tmp_path):
    exports, output, runs = make_export_fixture(tmp_path, complete_t2=False)
    audit = tool.finalize(exports, output, runs, bootstrap_repeats=5)
    assert audit["scope"] == "T1"
    main = read_csv(output / "pilot_main_table.csv")
    assert not any(row["method"] == "d0" for row in main)
    assert not any(row["workpoint_selected"] == "True" for row in main)
    decision = json.loads((output / "decision_inputs.json").read_text())
    assert not decision["T1_average_gain_gate"]
    assert not decision["workpoint_comparison_available"]
    assert "当前没有D0" in (output / "decision_log.md").read_text()
    assert read_csv(output / "per_reference_gains_vs_D0.csv") == []


def test_rejects_test_rows_before_any_output_or_input_mutation(tmp_path):
    exports, output, runs = make_export_fixture(tmp_path, complete_t2=False)
    path = exports / "B.csv"
    rows = read_csv(path)
    rows[0]["split"] = "test"
    tool.write_csv(path, rows)
    original = path.read_bytes()
    with pytest.raises(PermissionError, match="val"):
        tool.finalize(exports, output, runs, bootstrap_repeats=5)
    assert path.read_bytes() == original and not output.exists()


def test_rejects_changed_training_bundle_before_output(tmp_path):
    exports, output, runs = make_export_fixture(tmp_path, complete_t2=False)
    path = runs / "B/DONE.json"
    done = json.loads(path.read_text())
    done["signature"]["source_bundle_hash"] = "changed"
    path.write_text(json.dumps(done))
    with pytest.raises(ValueError, match="source bundle"):
        tool.finalize(exports, output, runs, bootstrap_repeats=5)
    assert not output.exists()


@pytest.mark.parametrize("target", ["training_child", "export_child", "frozen_source"])
def test_output_replacement_cannot_touch_training_inputs_or_frozen_sources(tmp_path, target):
    exports, output, runs = make_export_fixture(tmp_path, complete_t2=False)
    dangerous = {"training_child": runs / "B", "export_child": exports / "nested",
                 "frozen_source": tool.PROJECT / "rsi"}[target]
    source_before = tool.frozen_sources()
    input_before = {p.name: p.read_bytes() for p in exports.glob("*.csv")}
    with pytest.raises(ValueError, match="protected"):
        tool.finalize(exports, dangerous, runs, bootstrap_repeats=5)
    assert tool.frozen_sources() == source_before
    assert {p.name: p.read_bytes() for p in exports.glob("*.csv")} == input_before
    assert (runs / "B/DONE.json").is_file()


def test_rejects_mislabeled_alpha_against_completed_training(tmp_path):
    exports, output, runs = make_export_fixture(tmp_path, complete_t2=True)
    path = exports / "RSI-1.csv"
    rows = read_csv(path)
    rows[0]["alpha"] = rows[0]["action"] = .5
    tool.write_csv(path, rows)
    with pytest.raises(ValueError, match="training objective"):
        tool.finalize(exports, output, runs, bootstrap_repeats=5)
    assert not output.exists()


def test_rejects_D_stage_q_mismatch(tmp_path):
    exports, output, runs = make_export_fixture(tmp_path, complete_t2=True)
    path = runs / "RSI-1/DONE.json"
    done = json.loads(path.read_text())
    done["signature"]["q"] = .46
    path.write_text(json.dumps(done))
    with pytest.raises(ValueError, match="same fixed q"):
        tool.finalize(exports, output, runs, bootstrap_repeats=5)
    assert not output.exists()
