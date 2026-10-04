#!/usr/bin/env python3
"""Plot the complete seed17 D40 grid from immutable CPU-only evidence.

Two figures show augmented segmentation/total objectives and original-image
validation hard Dice. Every panel includes the same actual D0 trajectory.
Incomplete training or paired provenance produces no figures or metadata.
--validate-only never writes outputs. This tool does not train or select points.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools.finalize_pilot import frozen_sources, review_completed_training, review_matrix
from tools.paired_control_comparison import read_existing_points

SOURCE_BUNDLE = "36825cbb7a90064f3db3cfd97eb16ec62b29aa1947e17e88c7ad4759024268fb"
FROZEN_TOOLS = {
    "tools/finalize_pilot.py": "95de6989fc16e1de9ae477ab49c5252137907856a8f1f894c3c09d032bf766d8",
    "tools/paired_control_comparison.py": "738f854195ee119fee387dc655c91a2a509387349dc0210d9169abaf02914ddb",
}
FAMILIES = (("RSI", "rsi", "RSI", "#2166ac"),
            ("MeanHinge", "mean_hinge", "Mean-Hinge", "#b35806"),
            ("AbsHard", "abs_hard", "Abs-Hard", "#1b7837"))
WEIGHTS = (.3, 1., 3.)
D_RUNS = {"D0", *(f"{name}-{weight:g}" for name, *_ in FAMILIES for weight in WEIGHTS)}
FULL_RUNS = {"Anchor", "B", *D_RUNS,
             *(f"Shrink-D0-{alpha:.6f}" for alpha in (0., 1/3, 2/3, 1.))}
PAIRED_FILES = ("finalization_audit.json", "pilot_main_table.csv", "per_reference_gains.csv",
                "paired_controls/paired_control_comparison.csv",
                "paired_controls/paired_control_comparison.json")
RECOVERY_LEDGER = PROJECT / "outputs/recovery_20261004/timing_scope_review.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def snapshot(path, inputs):
    data = Path(path).read_bytes()
    inputs.append({"path": str(Path(path).resolve()), "bytes": len(data),
                   "sha256": hashlib.sha256(data).hexdigest()})
    return data


def finite(row, key):
    value = float(row[key])
    require(math.isfinite(value), f"Nonfinite {key}")
    return value


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_d_stage(run_dir, name, method, weight, cfg, inputs):
    directory = run_dir / name
    done = json.loads(snapshot(directory / "DONE.json", inputs))
    data = snapshot(directory / "epoch_diagnostics.csv", inputs)
    require(data.endswith(b"\n"), "A complete D stage cannot have an incomplete CSV record")
    raw = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    require([int(row["epoch"]) for row in raw] == list(range(1, 41)), "Require every D40 epoch exactly once")
    signature = done["signature"]
    require(done["run"] == name and done["seed"] == 17 and done["stage"] == "D"
            and done["epochs"] == 40 and done["test_scoring_locked"] is True
            and signature["config"] == cfg and signature["seed"] == 17
            and signature["stage"] == "D" and signature["method"] == method
            and signature["weight"] == weight and signature["source_bundle_hash"] == SOURCE_BUNDLE,
            "D-stage signature differs from the registered grid")
    rows = []
    for source in raw:
        require(source["run_id"] == name and source["seed"] == "17" and source["stage"] == "D"
                and source["method"] == method and finite(source, "weight") == weight
                and finite(source, "q") == signature["q"], "Epoch identity/objective differs from DONE")
        seg, total, risk = (finite(source, key) for key in
                            ("augmented_train_seg", "augmented_train_loss", "augmented_train_weighted_risk"))
        require(seg >= 0 and risk >= 0 and math.isclose(total, seg + risk, rel_tol=1e-7, abs_tol=1e-9),
                "Augmented segmentation plus weighted risk differs from the total objective")
        if method == "d0":
            require(risk == 0., "D0 weighted risk must be zero")
        dice = finite(source, "val_dice")
        require(0 <= dice <= 1, "Original-image validation Dice is invalid")
        rows.append(dict(epoch=int(source["epoch"]), augmented_seg=seg, augmented_total=total,
                         augmented_weighted_risk=risk, val_dice=dice,
                         epoch_body_seconds=finite(source, "epoch_seconds")))
    best = max(row["val_dice"] for row in rows)
    earliest = next(row["epoch"] for row in rows if row["val_dice"] == best)
    require(done["best_epoch"] == earliest and done["best_val_dice"] == best,
            "Selected checkpoint differs from original hard-Dice/earliest-tie rule")
    wall = finite(done, "wall_seconds")
    require(wall > 0, "DONE must retain a measured cumulative loop duration")
    return dict(run_id=name, method=method, weight=weight, committed_epochs=40,
                best_epoch=earliest, best_val_dice=best, selected_checkpoint_sha256=done["best_sha256"],
                done_cumulative_loop_accounting_seconds=wall,
                sum_committed_epoch_body_seconds=math.fsum(row["epoch_body_seconds"] for row in rows),
                rows=rows)


def readiness(config_path, input_dir):
    cfg = yaml.safe_load(config_path.read_text())
    require(cfg["test_scoring_locked"] is True and cfg["strict_test_scoring_lock"] is True
            and cfg["epochs"] == dict(A_CNN=100, A_T=100, B=40, D=40)
            and cfg["conditional_weight_grid"] == list(WEIGHTS), "Require the registered locked configuration")
    run_dir = Path(cfg["run_root"]).resolve() / "seed17"
    missing = [str(run_dir / name / "DONE.json") for name in sorted(D_RUNS)
               if not (run_dir / name / "DONE.json").is_file()]
    missing += [str(input_dir / name) for name in PAIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        return dict(status="not_ready", missing_complete_inputs=missing, outputs_written=False), None
    refs = read_csv(input_dir / "per_reference_gains.csv")
    actual_runs = {row["run_id"] for row in refs}
    if actual_runs != FULL_RUNS:
        return dict(status="not_ready", missing_full_matrix_runs=sorted(FULL_RUNS - actual_runs),
                    unexpected_full_matrix_runs=sorted(actual_runs - FULL_RUNS), outputs_written=False), None
    require(all(row["seed"] == "17" and row["split"] == "val" for row in refs), "Only seed17 development val is allowed")
    sources = frozen_sources()
    require(sources["source_bundle_hash"] == SOURCE_BUNDLE, "Frozen training source/config bundle changed")
    require(record(config_path)["sha256"] == sources["source_sha256"]["configs/p2_ima_m_v2.yaml"], "Config path differs from frozen config")
    inputs = [record(config_path), record(__file__), record(cfg["manifest"]), record(cfg["audit"])]
    for name, expected in FROZEN_TOOLS.items():
        item = record(PROJECT / name)
        require(item["sha256"] == expected, "Frozen CPU validator changed")
        inputs.append(item)
    inputs.extend(record(input_dir / name) for name in PAIRED_FILES)
    audit = json.loads((input_dir / "finalization_audit.json").read_text())
    require(audit["scope"] == "T2" and audit["seed"] == 17 and audit["test_scoring_locked"] is True
            and audit["original_csv_bytes_unchanged"] is True and audit["original_training_source"] == sources
            and audit["tool_sha256"] == FROZEN_TOOLS["tools/finalize_pilot.py"], "Finalization provenance differs")
    for original in audit["original_csvs"]:
        require(record(original["path"])["sha256"] == original["sha256"], "An original export changed after finalization")
        inputs.append(record(original["path"]))
    scope, stages, weights = review_matrix(refs, "T2")
    require(scope == "T2" and len(stages) == 13 and weights == list(WEIGHTS), "Require all thirteen completed training stages")
    completions, epoch_rows, proof = review_completed_training(run_dir, stages, refs, sources)
    require(proof == audit["training_completion_evidence"] and len(epoch_rows) == 640,
            "Full training proof differs from finalization")
    for item in proof:
        inputs.extend((record(item["done_path"]), record(item["epoch_csv"])))
        best = record(run_dir / item["run_id"] / "best.pth")
        require(best["sha256"] == item["best_checkpoint_sha256"], "Actual selected checkpoint differs from DONE")
        inputs.append(best)
    manifest_hash = record(cfg["manifest"])["sha256"]
    data_audit = json.loads(Path(cfg["audit"]).read_text())
    require(data_audit["file_audit_complete"] is True and data_audit["scope"] == "all_selected_M_H_T1"
            and data_audit["final_references_sha256"] == manifest_hash, "Data audit does not bind the frozen manifest")
    q_path = run_dir / "B/abs_hard_q.json"
    q = json.loads(q_path.read_text())
    require(q["checkpoint_hash"] == completions["B"]["best_sha256"] and q["manifest_hash"] == manifest_hash
            and q["source"] == "B_canonical_M_train" and q["quantile"] == .75
            and q["fixed"] is True and q["gradient"] is False
            and all(completions[name]["signature"]["q"] == q["q"] for name in D_RUNS), "Actual D grid does not share its fixed B train q")
    inputs.append(record(q_path))
    points = read_existing_points(input_dir / "per_reference_gains.csv", input_dir / "pilot_main_table.csv")
    require(len(points) == 16 and all(p["n_images"] == 223 and p["n_references"] == 471 for p in points),
            "Full M matrix must contain sixteen actual 223-image/471-reference points")
    paired = json.loads((input_dir / "paired_controls/paired_control_comparison.json").read_text())
    require(paired["status"] == "complete" and paired["subset"] == "M" and paired["split"] == "val"
            and paired["existing_point_count"] == 16 and paired["comparison_count"] == 39
            and paired["bootstrap_seed"] == 17 and paired["bootstrap_repeats"] == 2000
            and paired["test_scoring_locked"] is True and paired["workpoints_reselected"] is False
            and paired["tool_sha256"] == FROZEN_TOOLS["tools/paired_control_comparison.py"], "Paired matrix evidence is incomplete")
    expected_inputs = {str((input_dir / name).resolve()): record(input_dir / name)["sha256"]
                       for name in ("per_reference_gains.csv", "pilot_main_table.csv")}
    require({str(Path(item["path"]).resolve()): item["sha256"] for item in paired["input_files"]} == expected_inputs,
            "Paired sidecar is bound to different main/reference files")
    require(paired["output_csv_sha256"] == record(input_dir / "paired_controls/paired_control_comparison.csv")["sha256"],
            "Paired CSV changed")
    d_stages = {"D0": read_d_stage(run_dir, "D0", "d0", 0., cfg, inputs)}
    for name, method, _, _ in FAMILIES:
        for weight in WEIGHTS:
            run = f"{name}-{weight:g}"
            d_stages[run] = read_d_stage(run_dir, run, method, weight, cfg, inputs)
    require(RECOVERY_LEDGER.is_file(), "Preserve and include the known MH-0.3 recovery cost ledger")
    ledger = json.loads(RECOVERY_LEDGER.read_text())
    require(ledger["status"] == "timing_scope_verified" and ledger["run"] == "MeanHinge-0.3"
            and ledger["source_bundle_hash"] == SOURCE_BUNDLE
            and ledger["measured"]["DONE_cumulative_loop_wall_seconds"] == d_stages["MeanHinge-0.3"]["done_cumulative_loop_accounting_seconds"],
            "Recovery ledger does not match the actual completed MH-0.3 duration")
    ledger_done = next(item for item in ledger["inputs"] if item["path"] == str((run_dir / "MeanHinge-0.3/DONE.json").resolve()))
    require(ledger_done["sha256"] == record(ledger_done["path"])["sha256"], "Recovery ledger refers to a different completed MH-0.3 run")
    inputs.append(record(RECOVERY_LEDGER))
    # Record hashes once, rejecting inconsistent duplicate snapshots.
    unique = {}
    for item in inputs:
        require(item["path"] not in unique or item == unique[item["path"]], "An input changed during readiness")
        unique[item["path"]] = item
    return dict(status="ready", actual_D40_runs=10, full_training_stages=13, full_M_points=16,
                committed_D_epochs=400, outputs_written=False), dict(cfg=cfg, sources=sources, inputs=list(unique.values()),
                stages=d_stages, recovery_ledger=ledger)


def draw(evidence, view):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 12.3))
    fig.subplots_adjust(left=.065, right=.975, bottom=.17, top=.865, hspace=.42, wspace=.30)
    d0 = evidence["stages"]["D0"]
    x = np.arange(1, 41)
    for row, (name, _, title, color) in enumerate(FAMILIES):
        for column, weight in enumerate(WEIGHTS):
            ax = axes[row, column]
            stage = evidence["stages"][f"{name}-{weight:g}"]
            ax.set_title(f"{title}  $\\lambda={weight:g}$  |  40/40 committed", fontsize=11, pad=9)
            if view == "training":
                ax.plot(x, [r["augmented_seg"] for r in d0["rows"]], color="#707070", lw=1.15, ls=":")
                ax.plot(x, [r["augmented_seg"] for r in stage["rows"]], color=color, lw=1.55)
                ax.plot(x, [r["augmented_total"] for r in stage["rows"]], color=color, lw=1.25, ls="--")
                ax.set_ylabel("Augmented objective / segmentation loss")
            else:
                ax.plot(x, [r["val_dice"] for r in d0["rows"]], color="#707070", lw=1.1, ls=":")
                ax.plot(x, [r["val_dice"] for r in stage["rows"]], color=color, lw=1.55)
                ax.scatter([d0["best_epoch"]], [d0["best_val_dice"]], marker="s", s=32,
                           color="#707070", edgecolor="white", linewidth=.5, zorder=4)
                ax.scatter([stage["best_epoch"]], [stage["best_val_dice"]], marker="*", s=125,
                           color=color, edgecolor="white", linewidth=.5, zorder=5)
                ax.axvline(stage["best_epoch"], color=color, lw=.8, ls="--", alpha=.25)
                ax.set_ylabel("Validation hard Dice (original image)")
                ax.text(.985, .035, f"Best ep {stage['best_epoch']}: {stage['best_val_dice']:.6f}",
                        ha="right", va="bottom", transform=ax.transAxes, fontsize=9,
                        bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=2))
            ax.set_xlim(.5, 40.5)
            ax.set_xlabel("Epoch within D stage")
            ax.grid(alpha=.20, lw=.6)
            ax.margins(y=.16)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    subject = "Augmented training objectives" if view == "training" else "Original-image validation hard Dice"
    fig.suptitle(f"Seed17 registered weight grid | {subject}", fontsize=17, y=.969)
    handles = [Line2D([0], [0], color="#707070", lw=1.4, ls=":", label="Actual D0 trajectory (same reference in every panel)")]
    if view == "training":
        handles += [Line2D([0], [0], color="#2166ac", lw=1.6, label="Augmented segmentation composite"),
                    Line2D([0], [0], color="#2166ac", lw=1.4, ls="--", label="Augmented total objective: segmentation + weighted risk")]
    else:
        handles += [Line2D([0], [0], color="#2166ac", lw=1.6, label="Actual method trajectory"),
                    Line2D([0], [0], color="#2166ac", marker="*", ms=10, ls="None", label="Method selected best epoch"),
                    Line2D([0], [0], color="#707070", marker="s", ms=6, ls="None", label="D0 selected best epoch")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .935), ncol=2,
               frameon=False, fontsize=10, columnspacing=2.0)
    notes = ["Actual unsmoothed 40-epoch logs only. Each panel has an independent y-axis range; compare values and labels, not visual slopes.",
             "Segmentation composite: 0.5 BCE + 0.5 (1 - SoftDice), valid letterbox region. Training uses augmented views.",
             "Validation: original-coordinate hard Dice, mean over references then images; already used for development checkpoint selection.",
             "40 committed epochs per D task. MH-0.3 resumed after interruption; discarded physical work is recorded separately. Test remains locked."]
    for index, note in enumerate(notes):
        fig.text(.065, .077 - index * .018, note, fontsize=9.2, color="#444444", ha="left", va="top")
    return fig


def publish(evidence, gate, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    records, figures = [], []
    with tempfile.TemporaryDirectory(prefix=".weight-grid-curves-", dir=output_dir) as temporary:
        staged = Path(temporary)
        try:
            for view in ("training", "validation"):
                figure = draw(evidence, view)
                figures.append(figure)
                for extension in ("png", "pdf"):
                    name = f"weight_grid_{view}_curves_seed17.{extension}"
                    figure.savefig(staged / name, dpi=300, facecolor="white", format=extension,
                                   metadata={"Creator": "plot_weight_grid_learning_curves.py"})
                    item = record(staged / name); item["path"] = str(output_dir / name)
                    records.append(item)
            for item in evidence["inputs"]:
                require(record(item["path"])["sha256"] == item["sha256"], "An input changed before publication")
            require(frozen_sources() == evidence["sources"], "Frozen source/config changed before publication")
            metadata = dict(schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(),
                status="complete", seed=17, protocol_id=evidence["cfg"]["protocol_id"], readiness=gate,
                tool=record(__file__), inputs=evidence["inputs"], frozen_sources=evidence["sources"], outputs=records,
                training_fields=["augmented_train_seg", "augmented_train_loss", "augmented_train_weighted_risk"],
                validation_field="val_dice", validation_coordinate_system="original image",
                validation_reduction="mean over references within image, then mean over images",
                selection="existing DONE selected epoch by M validation hard Dice; earlier tie",
                common_reference="same actual D0 epoch trajectory repeated in each panel",
                smoothing="none", independent_y_axis_ranges=True, workpoints_reselected=False,
                inference_run=False, GPU_used=False, test_inputs_read=False, test_scoring_locked=True,
                timing_field="DONE.wall_seconds", timing_scope="cumulative measured active loop accounting across committed calls",
                timing_exclusions="per-call setup, inactive gaps, exports, some final writes, and discarded interrupted work; not full process elapsed",
                uniform_GPU_hours_claimed=False, physical_compute_cost_complete=False,
                recovery_ledger_path=str(RECOVERY_LEDGER), recovery_cost_scope=evidence["recovery_ledger"],
                stages=[{key: value for key, value in stage.items() if key != "rows"}
                        for stage in evidence["stages"].values()])
            name = "weight_grid_learning_curves_seed17.json"
            (staged / name).write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            for item in [Path(row["path"]).name for row in records] + [name]:
                target = output_dir / item
                if target.exists():
                    target.rename(target.with_name(item + ".before_" + suffix))
                os.replace(staged / item, target)
        finally:
            for figure in figures:
                plt.close(figure)
    return dict(status="complete", outputs_written=True, actual_D40_runs=10, output_dir=str(output_dir), test_scoring_locked=True)


def run(config_path, input_dir, output_dir, *, validate_only=False):
    config_path, input_dir, output_dir = map(lambda path: Path(path).resolve(), (config_path, input_dir, output_dir))
    require(output_dir.parent == PROJECT / "outputs" and output_dir.name.startswith("weight_grid_learning_curves"),
            "Figures must use their own outputs/weight_grid_learning_curves* namespace")
    gate, evidence = readiness(config_path, input_dir)
    if evidence is None:
        return {**gate, "test_scoring_locked": True}
    if validate_only:
        return {**gate, "status": "validated_only", "outputs_written": False, "test_scoring_locked": True}
    return publish(evidence, gate, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/p2_ima_m_v2.yaml")
    parser.add_argument("--input-dir", type=Path, default=PROJECT / "outputs/seed17")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/weight_grid_learning_curves_seed17")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.input_dir, args.output_dir, validate_only=args.validate_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "not_ready" and not args.validate_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
