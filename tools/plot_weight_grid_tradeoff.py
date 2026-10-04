#!/usr/bin/env python3
"""Publish a readable tradeoff figure from the actual complete seed17 matrix.

Reuse the full scientific readout's evidence gate without running its statistics.
Never select a lambda/alpha, alter scores, or overwrite the original autofigure.
--validate-only creates no output. Coordinates are existing M validation values.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Bbox
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools.full_scientific_readout import file_record, readiness, read_csv, require, verify_record
from tools.finalize_pilot import frozen_sources
from tools.paired_control_comparison import method_name, selected

GUARD_PATH = PROJECT / "tools/full_scientific_readout.py"
GUARD_SHA256 = "9c196bc8666a31db89d31485d00b67e8cd9f85cc8f7cdf12a36dc9f46e826dc8"
STYLE = {
    "rsi": ("#2166ac", "o", "RSI"),
    "mean_hinge": ("#b35806", "^", "Mean-Hinge (MH)"),
    "abs_hard": ("#1b7837", "s", "Abs-Hard (AH)"),
    "shrink": ("#762a83", "D", "Fixed-Shrink (S)"),
    "b": ("#666666", "X", "B context"),
    "anchor": ("#222222", "o", "Anchor / S0"),
    "d0": ("#222222", "P", "D0 / S1"),
}
STEM = "gain_harm_tradeoff_readable"


def short_id(row):
    method = method_name(row["method"])
    if method in {"anchor", "d0", "b"}:
        return {"anchor": "Anchor / S0", "d0": "D0 / S1", "b": "B"}[method]
    if method == "shrink":
        alpha = float(row["alpha"])
        return {0.: "Anchor / S0", 1/3: "S1/3", 2/3: "S2/3", 1.: "D0 / S1"}[alpha]
    weight = float(row["lambda"])
    prefix = {"rsi": "RSI", "mean_hinge": "MH", "abs_hard": "AH"}[method]
    return prefix + (".3" if weight == .3 else f"{weight:g}")


def actual_m_points(input_dir, evidence):
    main = read_csv(input_dir / "pilot_main_table.csv")
    points = [dict(row) for row in main if row["subset"] == "M"]
    require(len(points) == 16 and len({row["run_id"] for row in points}) == 16,
            "Require all sixteen existing M points")
    require(all(row["seed"] == "17" and row["split"] == "val" for row in points),
            "Only development seed17 M validation coordinates may be plotted")
    for row in points:
        for key in ("dice", "G_plus", "H_epsilon", "lambda", "alpha"):
            value = float(row[key])
            require(math.isfinite(value), "Nonfinite original tradeoff coordinate")
        require(float(row["H_epsilon"]) >= 0 and float(row["G_plus"]) >= 0,
                "Hard gain/harm coordinates must be nonnegative")
        row["display_label"] = short_id(row)
        row["display_workpoint_selected"] = selected(row["workpoint_selected"])
    by_id = {row["run_id"]: row for row in points}
    for shrink, endpoint in (("Shrink-D0-0.000000", "Anchor"), ("Shrink-D0-1.000000", "D0")):
        require(all(float(by_id[shrink][key]) == float(by_id[endpoint][key])
                    for key in ("dice", "G_plus", "H_epsilon")),
                "Shrink endpoint coordinates must coincide exactly with Anchor/D0")
    require({point["metadata"]["run_id"] for point in evidence["points"]} == set(by_id),
            "Scientific guard and existing main table identify different points")
    displayed = [row for row in points if row["run_id"] not in
                 {"Shrink-D0-0.000000", "Shrink-D0-1.000000"}]
    return points, displayed, by_id


def overlap(a, b):
    width = max(0., min(a.x1, b.x1) - max(a.x0, b.x0))
    height = max(0., min(a.y1, b.y1) - max(a.y0, b.y0))
    return width * height


def place_labels(fig, ax, displayed, y_key):
    """Deterministic, nonoverlapping text boxes inside axes; no coordinate edits."""
    renderer = fig.canvas.get_renderer()
    font = FontProperties(family="DejaVu Sans", size=9.5)
    area = ax.get_window_extent(renderer).shrunk(.98, .98)
    positions = {row["run_id"]: ax.transData.transform((float(row["H_epsilon"]), float(row[y_key])))
                 for row in displayed}
    markers = [Bbox.from_bounds(x - 10., y - 10., 20., 20.) for x, y in positions.values()]
    def density(row):
        point = positions[row["run_id"]]
        distances = [float(np.linalg.norm(point - other)) for key, other in positions.items()
                     if key != row["run_id"]]
        return min(distances), row["display_label"]
    placed, annotations, provenance = [], [], []
    pixels_per_point = fig.dpi / 72.
    for row in sorted(displayed, key=density):
        x, y = positions[row["run_id"]]
        text = row["display_label"]
        width, height, _ = renderer.get_text_width_height_descent(text, font, ismath=False)
        width += 9.; height += 10.
        candidates = []
        # Prefer nearby labels, then use an interior grid if a dense cluster
        # requires longer leaders. Candidate order is stable, without RNG.
        for radius in (20., 32., 48., 68., 92., 125., 170., 220., 285.):
            for angle in (45., -45., 135., -135., 0., 90., 180., -90.):
                radians = math.radians(angle)
                candidates.append((x + radius * math.cos(radians), y + radius * math.sin(radians)))
        candidates += [(cx, cy) for cy in np.linspace(area.y0 + height / 2, area.y1 - height / 2, 13)
                       for cx in np.linspace(area.x0 + width / 2, area.x1 - width / 2, 17)]
        valid = []
        for index, (cx, cy) in enumerate(candidates):
            box = Bbox.from_bounds(cx - width / 2, cy - height / 2, width, height)
            if box.x0 < area.x0 or box.x1 > area.x1 or box.y0 < area.y0 or box.y1 > area.y1:
                continue
            if any(overlap(box, other) > 0 for other in placed + markers):
                continue
            valid.append((math.hypot(cx - x, cy - y), index, cx, cy, box))
        require(bool(valid), "Cannot place all tradeoff labels without overlapping text or markers")
        _, _, cx, cy, box = min(valid, key=lambda item: item[:2])
        color = STYLE[method_name(row["method"])][0]
        annotation = ax.annotate(text, xy=(float(row["H_epsilon"]), float(row[y_key])),
            xytext=((cx - x) / pixels_per_point, (cy - y) / pixels_per_point), textcoords="offset points",
            ha="center", va="center", fontsize=9.5, color=color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=.92),
            arrowprops=dict(arrowstyle="-", color=color, lw=.7, shrinkA=3., shrinkB=5.),
            annotation_clip=True, zorder=6)
        placed.append(box); annotations.append(annotation)
        provenance.append(dict(run_id=row["run_id"], label=text, axes_panel=y_key,
            label_center_display_pixels=[float(cx), float(cy)],
            label_bbox_display_pixels=[float(v) for v in box.extents], data_coordinates_unchanged=True))
    return annotations, provenance


def draw(points, displayed, by_id):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 9.0))
    fig.subplots_adjust(left=.075, right=.975, bottom=.285, top=.875, wspace=.27)
    harm = [float(row["H_epsilon"]) for row in points]
    shrink = sorted((row for row in points if method_name(row["method"]) == "shrink"),
                    key=lambda row: float(row["alpha"]))
    threshold = float(by_id["D0"]["dice"]) - .002
    for ax, y_key, title, label in zip(axes, ("G_plus", "dice"),
            ("Gain retained versus reference harm", "Average Dice versus reference harm"),
            ("G+ (positive reference Dice gain)", "Macro mean-rater hard Dice")):
        values = [float(row[y_key]) for row in points]
        if y_key == "dice":
            values.append(threshold)
        span_x = max(harm) - min(harm) or .005
        span_y = max(values) - min(values) or .005
        ax.set_xlim(min(harm) - .10 * span_x, max(harm) + .15 * span_x)
        ax.set_ylim(min(values) - .18 * span_y, max(values) + .28 * span_y)
        ax.plot([float(row["H_epsilon"]) for row in shrink], [float(row[y_key]) for row in shrink],
                color=STYLE["shrink"][0], lw=1.15, ls="--", alpha=.7, zorder=1)
        if y_key == "dice":
            ax.axhline(threshold, color="#888888", lw=1.05, ls=":", zorder=0)
        for row in displayed:
            color, marker, _ = STYLE[method_name(row["method"])]
            x, y = float(row["H_epsilon"]), float(row[y_key])
            ax.scatter([x], [y], color=color, marker=marker, s=66, edgecolor="white", linewidth=.7, zorder=3)
            if row["display_workpoint_selected"]:
                ax.scatter([x], [y], s=133, facecolor="none", edgecolor="#111111", linewidth=1.05, zorder=4)
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel("Hε (εD = 0.005); smaller values are favorable")
        ax.set_ylabel(label)
        ax.grid(alpha=.18, linewidth=.6)
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    handles = [Line2D([0], [0], color=color, marker=marker, lw=0, markersize=8, label=label)
               for method, (color, marker, label) in STYLE.items() if method not in {"anchor", "d0"}]
    handles += [Line2D([0], [0], color="#222222", marker="o", lw=0, markersize=7, label="Anchor/S0 and D0/S1 endpoints"),
                Line2D([0], [0], color="#111111", marker="o", markerfacecolor="none", lw=0,
                       markersize=10, label="Existing M working-point flag"),
                Line2D([0], [0], color=STYLE["shrink"][0], ls="--", lw=1.1, label="Observed Shrink points connected"),
                Line2D([0], [0], color="#888888", ls=":", lw=1.1, label="D0 Dice − 0.002 eligibility line")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .235), ncol=3,
               frameon=False, fontsize=9.8, columnspacing=2.0)
    fig.suptitle("Seed17 registered development tradeoff | 16 actual M points", fontsize=18, y=.965)
    notes = ["All coordinates are unchanged original-image validation metrics, averaged over references within image, then over 223 images.",
             "Labels: RSI/MH/AH .3, 1, 3 are the registered weights; S1/3 and S2/3 are registered Shrink scales. B is context.",
             "Anchor/S0 and D0/S1 coincide exactly; each label retains two points in provenance. Endpoint rings follow the Anchor/D0 flag.",
             "Shrink segments connect observed points for display; they do not represent inferred or evaluated intermediate outputs.",
             "Working-point flags and D0 − 0.002 denote existing eligibility/selection, not proof of efficacy. Development selection reused this validation set; test remains locked."]
    for index, note in enumerate(notes):
        fig.text(.075, .12 - index * .021, note, fontsize=9.2, color="#444444", ha="left", va="top")
    fig.canvas.draw()
    placements = []
    for ax, y_key in zip(axes, ("G_plus", "dice")):
        _, panel = place_labels(fig, ax, displayed, y_key)
        placements.extend(panel)
    return fig, placements


def publish(input_dir, output_dir, evidence, gate):
    points, displayed, by_id = actual_m_points(input_dir, evidence)
    inputs = list(evidence["inputs"]) + [file_record(__file__), file_record(GUARD_PATH)]
    original = input_dir / "gain_harm_tradeoff.png"
    if original.is_file():
        inputs.append(file_record(original))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, placements = draw(points, displayed, by_id)
    try:
        with tempfile.TemporaryDirectory(prefix=".readable-tradeoff-", dir=output_dir) as temporary:
            staged, outputs = Path(temporary), []
            for extension in ("png", "pdf"):
                name = f"{STEM}.{extension}"
                fig.savefig(staged / name, dpi=300, facecolor="white", format=extension,
                            metadata={"Creator": "plot_weight_grid_tradeoff.py"})
                item = file_record(staged / name); item["path"] = str(output_dir / name)
                outputs.append(item)
            name = f"{STEM}_points.csv"
            with (staged / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in points for key in row)))
                writer.writeheader(); writer.writerows(points)
            item = file_record(staged / name); item["path"] = str(output_dir / name); outputs.append(item)
            for item in inputs:
                verify_record(item)
            require(file_record(GUARD_PATH)["sha256"] == GUARD_SHA256, "Frozen scientific evidence guard changed")
            require(frozen_sources() == evidence["sources"], "Frozen scientific source/config changed during drawing")
            review = dict(schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(), status="complete",
                seed=17, subset="M", split="val", original_M_points=16, combined_display_glyphs=14,
                inputs=inputs, outputs=outputs, readiness=gate, selected_flags_source="existing pilot_main_table.csv",
                coordinates_modified=False, workpoints_reselected=False, new_lambda_or_alpha=False,
                original_autofigure_preserved=True, original_score_CSVs_modified=False, models_constructed=False,
                GPU_used=False, test_inputs_read=False, test_scoring_locked=True,
                eligibility_line=dict(definition="selected D0 Dice minus 0.002", value=float(by_id["D0"]["dice"]) - .002,
                                      scientific_success_implied=False),
                shrink_line="observed registered points connected for display; no inferred intermediate evaluation",
                exact_combined_endpoints={"Anchor / S0": ["Anchor", "Shrink-D0-0.000000"],
                                          "D0 / S1": ["D0", "Shrink-D0-1.000000"]},
                combined_endpoint_ring_selection_source="Anchor/D0 existing flag; separate Shrink endpoint flags retained in points table",
                label_layout=dict(method="deterministic inside-axes pixel-space search with leader lines",
                                  text_boxes_nonoverlapping=True, labels_do_not_cover_point_markers=True,
                                  plot_data_positions_unchanged=True, placements=placements),
                points=points, limits="Checkpoint/workpoint-selected development validation; eligibility is not proof of efficacy")
            name = f"{STEM}.json"
            (staged / name).write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            for name in [Path(item["path"]).name for item in outputs] + [name]:
                target = output_dir / name
                if target.exists():
                    target.rename(target.with_name(name + ".before_" + suffix))
                os.replace(staged / name, target)
    finally:
        plt.close(fig)
    return dict(status="complete", output_dir=str(output_dir), actual_M_points=16, original_scores_unchanged=True,
                test_scoring_locked=True)


def run(input_dir, run_dir, output_dir=None, *, validate_only=False):
    input_dir, run_dir = Path(input_dir).resolve(), Path(run_dir).resolve()
    output_dir = Path(output_dir or input_dir).resolve()
    require(output_dir == PROJECT / "outputs/seed17" or
            (output_dir.parent == PROJECT / "outputs" and output_dir.name.startswith("tradeoff_readable")),
            "Readable figures must use their own output filenames in seed17 or an own tradeoff_readable directory")
    require(file_record(GUARD_PATH)["sha256"] == GUARD_SHA256, "Scientific evidence guard changed before validation")
    gate, evidence = readiness(input_dir, run_dir)
    if evidence is None:
        return {**gate, "outputs_written": False, "test_scoring_locked": True}
    actual_m_points(input_dir, evidence)
    if validate_only:
        return {**gate, "status": "validated_only", "outputs_written": False}
    return publish(input_dir, output_dir, evidence, gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT / "outputs/seed17")
    parser.add_argument("--run-dir", type=Path, default=Path("/home/featurize/rsi_runs/P2-IMA-M-v2/seed17"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run(args.input_dir, args.run_dir, args.output_dir, validate_only=args.validate_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "not_ready" and not args.validate_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
