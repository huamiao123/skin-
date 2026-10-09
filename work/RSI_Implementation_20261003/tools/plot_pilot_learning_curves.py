#!/usr/bin/env python3
"""Export pilot learning curves from immutable local epoch logs; no model inference.

Usage:
    python tools/plot_pilot_learning_curves.py --seed 17

The segmentation composite is plotted from ``augmented_train_seg``.  Risk-stage
total objectives are separate dashed curves, and validation Dice is measured in
the original image coordinates.  Stage wall times come only from DONE.json.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
STAGES = (
    ("A_CNN", "A: CNN", "A_CNN"),
    ("A_T", "A: Transformer", "A_T"),
    ("B", "B: interaction", "B"),
    ("D0", "D0", "D"),
    ("RSI-1", "RSI", "D"),
    ("MeanHinge-1", "Mean-Hinge", "D"),
    ("AbsHard-1", "Abs-Hard", "D"),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot(path: Path, inputs: list[dict]) -> bytes:
    data = path.read_bytes()
    inputs.append({"path": str(path.resolve()), "bytes": len(data), "sha256": sha256(data)})
    return data


def finite_number(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} at epoch {row.get('epoch')}")
    return value


def read_stage(run_root: Path, seed: int, name: str, title: str, budget: int,
               inputs: list[dict]) -> dict:
    directory = run_root / f"seed{seed}" / name
    csv_path = directory / "epoch_diagnostics.csv"
    done_path = directory / "DONE.json"
    done = json.loads(snapshot(done_path, inputs)) if done_path.exists() else None
    discarded_trailing_fragment = False
    rows = []
    if csv_path.exists():
        data = snapshot(csv_path, inputs)
        # An active writer can leave a partial final row. Only newline-terminated
        # records are eligible; a completed stage must have a complete file.
        if data and not data.endswith(b"\n"):
            if done is not None:
                raise ValueError(f"Completed stage has an unterminated CSV: {csv_path}")
            data = data[:data.rfind(b"\n") + 1]
            discarded_trailing_fragment = True
        for row in csv.DictReader(io.StringIO(data.decode("utf-8"))):
            epoch_value = finite_number(row, "epoch")
            epoch = int(epoch_value)
            if epoch != epoch_value or epoch != len(rows) + 1:
                raise ValueError(f"Epoch sequence is not contiguous in {csv_path}")
            if int(row["seed"]) != seed or row["run_id"] != name:
                raise ValueError(f"Seed/run identity mismatch in {csv_path}")
            seg = finite_number(row, "augmented_train_seg")
            total = finite_number(row, "augmented_train_loss")
            weighted_risk = finite_number(row, "augmented_train_weighted_risk")
            if not math.isclose(total, seg + weighted_risk, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"Composite/objective loss mismatch in {csv_path}:{epoch}")
            dice = finite_number(row, "val_dice")
            if not 0 <= dice <= 1:
                raise ValueError(f"Invalid validation Dice in {csv_path}:{epoch}")
            rows.append({"epoch": epoch, "train_seg": seg, "train_total": total,
                         "train_weighted_risk": weighted_risk, "val_dice": dice,
                         "logged_best_epoch": int(finite_number(row, "best_epoch"))})
    if len(rows) > budget:
        raise ValueError(f"Stage exceeds configured budget: {name}")
    status = "DONE" if done is not None else ("ONGOING" if rows else "NOT_STARTED")
    selected = None
    wall_seconds = None
    if done is not None:
        if int(done["seed"]) != seed or done["run"] != name:
            raise ValueError(f"Seed/run identity mismatch in {done_path}")
        if int(done["epochs"]) != budget or len(rows) != budget:
            raise ValueError(f"DONE/CSV/configured budget disagree for {name}")
        if done.get("test_scoring_locked") is not True:
            raise ValueError(f"Test lock is not explicit in {done_path}")
        selected = int(done["best_epoch"])
        wall_seconds = float(done["wall_seconds"])
        if not math.isfinite(wall_seconds) or wall_seconds < 0:
            raise ValueError(f"Invalid measured wall time in {done_path}")
    elif rows:
        selected = rows[-1]["logged_best_epoch"]
    selected_dice = None
    if rows:
        best_from_csv = max(rows, key=lambda row: row["val_dice"])["epoch"]
        if selected != best_from_csv:
            raise ValueError(f"Best-epoch metadata disagrees with Dice/earlier tie rule: {name}")
        selected_dice = rows[selected - 1]["val_dice"]
        if done is not None and not math.isclose(selected_dice, float(done["best_val_dice"]),
                                               rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Best Dice disagrees with DONE: {name}")
    return {"name": name, "title": title, "budget_epochs": budget,
            "status": status, "logged_epochs": len(rows), "best_epoch": selected,
            "best_val_dice": selected_dice, "measured_done_wall_seconds": wall_seconds,
            "wall_source": str(done_path) if done is not None else None,
            "discarded_incomplete_trailing_csv_fragment": discarded_trailing_fragment,
            "csv_path": str(csv_path), "done_path": str(done_path), "rows": rows}


def draw(stages: list[dict], seed: int, weight: float, loss_config: dict) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.titleweight": "semibold"})
    fig, axes = plt.subplots(2, 4, figsize=(19.8, 10.2))
    fig.subplots_adjust(left=.055, right=.955, bottom=.135, top=.805,
                        wspace=.53, hspace=.43)
    train_color, val_color, total_color = "#2166ac", "#d95f02", "#67a9cf"
    for ax, stage in zip(axes.flat, stages):
        rows = stage["rows"]
        title = stage["title"]
        if stage["name"] in {"RSI-1", "MeanHinge-1", "AbsHard-1"}:
            title += f" ($\\lambda={weight:g}$)"
        state = f"{stage['status']} {stage['logged_epochs']}/{stage['budget_epochs']}"
        ax.set_title(f"{title}\n{state}", fontsize=12, pad=9)
        ax.set_xlabel("Epoch within stage")
        ax.set_ylabel("Train composite loss (aug.)", color=train_color)
        ax.tick_params(axis="y", colors=train_color)
        ax.grid(alpha=.2, linewidth=.6)
        ax.set_xlim(.5, stage["budget_epochs"] + .5)
        if not rows:
            ax.text(.5, .5, "No complete epoch records", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        x = [row["epoch"] for row in rows]
        ax.plot(x, [row["train_seg"] for row in rows], color=train_color, lw=1.6)
        if any(abs(row["train_weighted_risk"]) > 1e-12 for row in rows):
            ax.plot(x, [row["train_total"] for row in rows], color=total_color,
                    lw=1.3, ls="--")
        ax.margins(y=.1)
        val_ax = ax.twinx()
        val_ax.spines["top"].set_visible(False)
        val_ax.set_ylabel("Val Dice (original image)", color=val_color)
        val_ax.tick_params(axis="y", colors=val_color)
        val_ax.plot(x, [row["val_dice"] for row in rows], color=val_color, lw=1.4)
        val_ax.ticklabel_format(style="plain", useOffset=False, axis="y")
        val_ax.margins(y=.16)
        selected = stage["best_epoch"]
        selected_dice = stage["best_val_dice"]
        val_ax.axvline(selected, color="#777777", lw=.9, ls=":", zorder=0)
        val_ax.scatter([selected], [selected_dice], marker="*", s=110,
                       color=val_color, edgecolor="white", linewidth=.6, zorder=4)
        prefix = "Best" if stage["status"] == "DONE" else "Best so far"
        ax.text(.98, .04, f"{prefix}: ep {selected}, Dice {selected_dice:.5f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 2})
    wall_ax = axes.flat[7]
    y = np.arange(len(stages))
    wall_ax.set_title("Measured stage wall time\nDONE.json only", fontsize=12, pad=9)
    minutes = [s["measured_done_wall_seconds"] / 60
               if s["measured_done_wall_seconds"] is not None else 0 for s in stages]
    wall_ax.barh(y, minutes, color="#8da0cb", height=.66)
    wall_ax.set_yticks(y, ["A-CNN", "A-T", "B", "D0", "RSI", "Mean-H.", "Abs-H."])
    wall_ax.invert_yaxis()
    wall_ax.set_xlabel("Elapsed wall time (minutes)")
    wall_ax.tick_params(axis="y", labelsize=9, pad=2)
    wall_ax.grid(axis="x", alpha=.2, linewidth=.6)
    maximum = max(minutes) if any(minutes) else 1
    wall_ax.set_xlim(0, maximum * 1.28)
    for i, (stage, value) in enumerate(zip(stages, minutes)):
        label = f"{value:.1f} min" if stage["status"] == "DONE" else "Not DONE; no total"
        wall_ax.text(value + maximum * .025, i, label, va="center", fontsize=9)
    fig.suptitle(f"RSI pilot learning curves | development seed {seed}", fontsize=18, y=.963)
    handles = [Line2D([0], [0], color=train_color, lw=1.8,
                      label="Train segmentation composite (augmented view)"),
               Line2D([0], [0], color=total_color, lw=1.5, ls="--",
                      label="Train total objective (segmentation + weighted risk)"),
               Line2D([0], [0], color=val_color, lw=1.5,
                      label="Validation Dice (original image coordinates)"),
               Line2D([0], [0], color=val_color, marker="*", ms=11, ls="None",
                      label="Best epoch selected by validation Dice")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .925),
               ncol=2, frameon=False, columnspacing=2.8, fontsize=11)
    composite = (f"Composite = {float(loss_config['bce']):g} BCE + "
                 f"{float(loss_config['soft_dice']):g} (1 - SoftDice), letterbox valid region.")
    notes = [composite + " Training curves use augmented views; no smoothing.",
             "Validation: original-image hard Dice, reference mean within image, then image mean; "
             "used for development checkpoint selection.",
             "Panels use independent y-axis ranges. Stage wall times are recorded elapsed times, "
             "not GPU utilization or a uniform GPU-hour budget. Test scoring remains locked."]
    for offset, note in enumerate(notes):
        fig.text(.055, .09 - offset * .024, note, ha="left", va="top", fontsize=10,
                 color="#444444")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/p2_ima_m_v2.yaml")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs")
    args = parser.parse_args()
    inputs = []
    config_path = args.config.resolve()
    cfg = yaml.safe_load(snapshot(config_path, inputs))
    if args.seed not in cfg["development_seeds"]:
        raise ValueError("This tool only exports predeclared development seeds")
    if cfg.get("test_scoring_locked") is not True or cfg.get("strict_test_scoring_lock") is not True:
        raise ValueError("Both configured test locks must be explicit")
    run_root = (args.run_root or Path(cfg["run_root"])).resolve()
    weight = float(cfg["pilot_weight"])
    if weight != 1.0:
        raise ValueError("The seven pilot directories are the predeclared lambda=1 runs")
    stages = [read_stage(run_root, args.seed, name, title, int(cfg["epochs"][budget_key]), inputs)
              for name, title, budget_key in STAGES]
    script_path = Path(__file__).resolve()
    snapshot(script_path, inputs)
    generated = datetime.now(timezone.utc).isoformat()
    figure = draw(stages, args.seed, weight, cfg["loss"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pilot_learning_curves_seed{args.seed}"
    output_records = []
    temporaries = []
    try:
        for ext in ("png", "pdf"):
            destination = args.output_dir / f"{stem}.{ext}"
            temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.{ext}")
            temporaries.append(temporary)
            figure.savefig(temporary, dpi=300, facecolor="white", format=ext,
                           metadata={"Creator": "plot_pilot_learning_curves.py"})
            output_records.append({"path": str(destination.resolve()),
                                   "bytes": temporary.stat().st_size,
                                   "sha256": sha256(temporary.read_bytes())})
        # A simultaneous append must never produce a falsely stable snapshot.
        for record in inputs:
            if sha256(Path(record["path"]).read_bytes()) != record["sha256"]:
                raise RuntimeError(f"Input changed during figure generation; rerun: {record['path']}")
        metadata = {
            "schema_version": 1, "generated_at_utc": generated, "seed": args.seed,
            "protocol_id": cfg["protocol_id"], "run_root": str(run_root),
            "scope": "development train/validation learning curves and DONE elapsed wall times only",
            "checkpoint_selection_scope": "M validation original-image macro mean-rater hard Dice; earlier epoch on tie",
            "test_scoring_locked": True, "test_inputs_read": False, "model_inference_run": False,
            "gpu_used": False, "smoothing": "none", "independent_y_axis_ranges": True,
            "training_view": "augmented training images; letterbox valid region for segmentation loss",
            "training_composite_field": "augmented_train_seg",
            "training_total_objective_field": "augmented_train_loss",
            "training_loss_definition": {"bce_weight": float(cfg["loss"]["bce"]),
                                         "one_minus_soft_dice_weight": float(cfg["loss"]["soft_dice"]),
                                         "total": "segmentation composite + weighted risk"},
            "validation_field": "val_dice", "validation_coordinate_system": "original image",
            "validation_reduction": "mean over references within each image, then mean over images",
            "probability_threshold": float(cfg["threshold"]),
            "wall_time_definition": "individual stage elapsed wall_seconds from DONE.json only; no inferred ongoing total",
            "uniform_gpu_hours_claimed": False, "stage_epoch_budgets_aggregated": False,
            "all_seven_stages_done": all(s["status"] == "DONE" for s in stages),
            "inputs": inputs, "stages": [{k: v for k, v in s.items() if k != "rows"} for s in stages],
            "outputs": output_records,
        }
        json_destination = args.output_dir / f"{stem}.json"
        json_temporary = json_destination.with_name(f".{json_destination.stem}.{os.getpid()}.tmp.json")
        temporaries.append(json_temporary)
        json_temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        for ext, temporary in zip(("png", "pdf"), temporaries[:2]):
            temporary.replace(args.output_dir / f"{stem}.{ext}")
        json_temporary.replace(json_destination)
    finally:
        plt.close(figure)
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"outputs": [str(args.output_dir / f"{stem}.{ext}") for ext in ("png", "pdf", "json")],
                      "all_seven_stages_done": metadata["all_seven_stages_done"],
                      "logged_epochs": {s["name"]: s["logged_epochs"] for s in stages}}, indent=2))


if __name__ == "__main__":
    main()
