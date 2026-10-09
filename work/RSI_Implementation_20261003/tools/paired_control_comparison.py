"""Direct paired RSI/control development intervals from existing M val rows.

The worst-tail interval reranks each method's worst-reference gain on every
resampled image cohort. It is neither a mean-worst-reference interval nor the
tail of per-image differences. No model, torch, training source or test score
is needed, and selected workpoints are read from the existing main table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
EPSILON = 0.005
METRICS = ("dice", "G_plus", "H_epsilon", "worst_tail_10pct")
TAIL_DEFINITION = ("Each resampled cohort: per-image minimum reference Dice gain versus the shared CNN anchor; "
                   "independently sort each method and mean the first max(1, ceil(0.1 * resampled_image_count)) images")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def method_name(value) -> str:
    name = str(value).lower().replace("-", "_")
    return {"meanhinge": "mean_hinge", "abshard": "abs_hard", "fixed_shrink": "shrink",
            "shrink_d0": "shrink"}.get(name, name)


def finite(value, label) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {label}")
    return result


def point_key(row) -> tuple:
    return (str(row["protocol_id"]), str(row["seed"]), str(row["run_id"]), str(row["checkpoint_hash"]),
            *(finite(row[key], key) for key in ("action", "lambda", "alpha")))


def selected(value) -> bool:
    if value is True or value in ("True", "true", "1"):
        return True
    if value is False or value in ("False", "false", "0", "", None):
        return False
    raise ValueError(f"Invalid workpoint selection flag: {value}")


def tail_mean(worst) -> float:
    values = np.asarray(worst, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Tail requires finite per-image worst-reference gains")
    count = max(1, math.ceil(0.1 * len(values)))
    return float(np.sort(values)[:count].mean())


def summarize_references(rows) -> dict:
    """Canonical image/reference identities; hard metrics remain image equal."""
    by_image = defaultdict(list)
    seen = set()
    for row in rows:
        key = (str(row["image_id"]), str(row["reference_id"]))
        if key in seen:
            raise ValueError(f"Repeated image/reference row: {key}")
        seen.add(key)
        by_image[key[0]].append(row)
    image_ids = sorted(by_image)
    if not image_ids:
        raise ValueError("A point has no M validation references")
    anchor_by_reference, group_ids, vectors, worst = {}, [], [], []
    for image_id in image_ids:
        refs = sorted(by_image[image_id], key=lambda row: str(row["reference_id"]))
        groups = {str(row.get("group_id") or "") for row in refs}
        if len(groups) != 1:
            raise ValueError(f"Group identity varies within image: {image_id}")
        group_ids.append(groups.pop())
        dice, gains = [], []
        for row in refs:
            value = finite(row["dice"], "Dice")
            anchor = finite(row["dice_anchor"], "anchor Dice")
            if not 0 <= value <= 1 or not 0 <= anchor <= 1:
                raise ValueError("Hard Dice must be in[0,1]")
            dice.append(value)
            gains.append(value - anchor)
            anchor_by_reference[(image_id, str(row["reference_id"]))] = anchor
        gains = np.asarray(gains)
        vectors.append((np.mean(dice), np.maximum(gains, 0).mean(), np.maximum(-gains - EPSILON, 0).mean()))
        worst.append(gains.min())
    vectors = np.asarray(vectors, dtype=float)
    worst = np.asarray(worst, dtype=float)
    return {"image_ids": image_ids, "group_ids": group_ids, "reference_keys": seen,
            "anchor_by_reference": anchor_by_reference, "vectors": vectors, "worst": worst,
            "values": dict(zip(METRICS, [*vectors.mean(axis=0), tail_mean(worst)])),
            "n_images": len(image_ids), "n_references": len(rows)}


def make_units(image_ids, group_ids) -> tuple[list[np.ndarray], str]:
    if len(image_ids) != len(group_ids):
        raise ValueError("Image/group vectors differ")
    grouped = defaultdict(list)
    any_known = False
    for index, (image_id, group) in enumerate(zip(image_ids, group_ids)):
        known = bool(str(group).strip())
        any_known |= known
        key = ("known", str(group)) if known else ("image", image_id)
        grouped[key].append(index)
    return [np.asarray(indices, dtype=int) for indices in grouped.values()], "known_case_group" if any_known else "image"


def compare_points(current, control, *, repeats=2000, seed=17, draws=None) -> dict:
    if current["image_ids"] != control["image_ids"] or current["group_ids"] != control["group_ids"]:
        raise ValueError("Methods do not share the same images and case groups")
    if current["reference_keys"] != control["reference_keys"]:
        raise ValueError("Methods do not share all reference identities")
    if current["anchor_by_reference"] != control["anchor_by_reference"]:
        raise ValueError("Methods do not share the same per-reference CNN anchor")
    units, unit_name = make_units(current["image_ids"], current["group_ids"])
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    if draws is None:
        draws = np.random.default_rng(seed).integers(len(units), size=(repeats, len(units)))
    else:
        draws = np.asarray(draws)
        if draws.shape != (repeats, len(units)) or not np.issubdtype(draws.dtype, np.integer) or (draws < 0).any() or (draws >= len(units)).any():
            raise ValueError("Bootstrap draws must sample every original case unit with replacement")
    observed = np.array([current["values"][metric] - control["values"][metric] for metric in METRICS])
    bootstrap = np.empty((repeats, len(METRICS)))
    counts = np.array([len(unit) for unit in units])
    sum_differences = np.array([(current["vectors"][unit] - control["vectors"][unit]).sum(axis=0) for unit in units])
    bootstrap[:, :3] = sum_differences[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    resampled_sizes = counts[draws].sum(axis=1)
    for index, draw in enumerate(draws):
        image_indices = np.concatenate([units[int(unit)] for unit in draw])
        bootstrap[index, 3] = tail_mean(current["worst"][image_indices]) - tail_mean(control["worst"][image_indices])
    low, high = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    result = {"difference_orientation": "RSI minus control", "epsilon_d": EPSILON,
              "n_images": current["n_images"], "n_references": current["n_references"],
              "bootstrap_unit": unit_name, "bootstrap_units": len(units), "bootstrap_repeats": repeats,
              "bootstrap_seed": seed, "resampled_image_count_min": int(resampled_sizes.min()),
              "resampled_image_count_max": int(resampled_sizes.max()),
              "worst_tail_n_observed": max(1, math.ceil(.1 * current["n_images"])),
              "worst_tail_definition": TAIL_DEFINITION, "worst_tail_reranked_each_draw": True,
              "interval_scope": "exploratory selected-development validation; not independent confirmation"}
    for index, metric in enumerate(METRICS):
        result.update({metric + "_rsi": current["values"][metric], metric + "_control": control["values"][metric],
                       metric + "_difference": float(observed[index]), metric + "_ci_low": float(low[index]),
                       metric + "_ci_high": float(high[index])})
    return result


def read_existing_points(input_path: Path, main_path: Path) -> list[dict]:
    with input_path.open(newline="", encoding="utf-8") as stream:
        refs = list(csv.DictReader(stream))
    with main_path.open(newline="", encoding="utf-8") as stream:
        main = list(csv.DictReader(stream))
    if any(row.get("split") != "val" for row in [*refs, *main]):
        raise PermissionError("Only explicitly exported validation rows are allowed")
    refs = [row for row in refs if row["subset"] == "M"]
    main = [row for row in main if row["subset"] == "M"]
    if not refs or not main:
        raise ValueError("M validation inputs are absent")
    groups = defaultdict(list)
    for row in refs:
        if row["run_id"] == "B" and method_name(row["method"]) == "d0":
            raise ValueError("Use finalizer-corrected B rows, not the original mislabeled export")
        groups[point_key(row)].append(row)
    main_by_key = {}
    for row in main:
        key = point_key(row)
        if key in main_by_key:
            raise ValueError("Repeated main-table point")
        main_by_key[key] = row
    if set(main_by_key) != set(groups):
        raise ValueError("Main table and reference exports do not identify the same existing M points")
    if len({key[:2] for key in groups}) != 1:
        raise ValueError("One protocol and seed are required per paired comparison")
    points = []
    for key in sorted(groups):
        row = main_by_key[key]
        if method_name(row["method"]) not in {"rsi", "d0", "mean_hinge", "abs_hard", "shrink", "b", "anchor"}:
            raise ValueError("Unexpected method in the existing pilot main table")
        if {method_name(ref["method"]) for ref in groups[key]} != {method_name(row["method"])}:
            raise ValueError("Reference method differs from main table")
        point = summarize_references(groups[key])
        for metric, value in point["values"].items():
            if not math.isclose(value, finite(row[metric], metric), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Main-table {metric} does not match these reference inputs")
        if point["n_images"] != int(row["n_images"]) or point["n_references"] != int(row["n_references"]):
            raise ValueError("Main-table reference counts differ")
        point["metadata"] = {column: row[column] for column in ("protocol_id", "seed", "run_id", "checkpoint_hash", "action", "lambda", "alpha")}
        point["method"] = method_name(row["method"])
        point["workpoint_selected"] = selected(row["workpoint_selected"])
        point["workpoint_status"] = row.get("workpoint_status", "")
        points.append(point)
    return points


def comparison_table(points, *, repeats=2000, seed=17) -> list[dict]:
    rsi = [point for point in points if point["method"] == "rsi"]
    controls = [point for point in points if point["method"] != "rsi"]
    simple = {"d0", "mean_hinge", "abs_hard", "shrink"}
    if not rsi or not any(point["method"] in simple for point in controls):
        raise ValueError("At least one existing RSI and simple-control candidate are required")
    rows = []
    for current in rsi:
        for control in controls:
            result = compare_points(current, control, repeats=repeats, seed=seed)
            label = "simple_control" if control["method"] in simple else "baseline_context"
            metadata = {"protocol_id": current["metadata"]["protocol_id"], "seed": current["metadata"]["seed"]}
            for prefix, point in (("rsi", current), ("control", control)):
                metadata.update({prefix + "_" + key: value for key, value in point["metadata"].items() if key not in ("protocol_id", "seed")})
                metadata.update({prefix + "_method": point["method"], prefix + "_workpoint_selected": point["workpoint_selected"],
                                 prefix + "_workpoint_status": point["workpoint_status"]})
            coordinate_dominance = (control["values"]["dice"] >= current["values"]["dice"] and
                                    control["values"]["H_epsilon"] <= current["values"]["H_epsilon"] and
                                    (control["values"]["dice"] > current["values"]["dice"] or
                                     control["values"]["H_epsilon"] < current["values"]["H_epsilon"]))
            rows.append({**metadata, "comparison_class": label,
                         "both_existing_workpoints_selected": current["workpoint_selected"] and control["workpoint_selected"],
                         "control_coordinate_dominates_rsi": coordinate_dominance,
                         "dominance_scope": "coordinate screen only; no prespecified substantive margin, assess intervals, G+ and tail",
                         **result})
    return rows


def finalize(input_path: Path, main_path: Path, output_dir: Path, *, repeats=2000, seed=17) -> dict:
    input_path, main_path, output_dir = map(lambda path: Path(path).resolve(), (input_path, main_path, output_dir))
    protected = [PROJECT / name for name in ("rsi", "configs", "tools", "tests", ".git")]
    protected.append(Path("/home/featurize/rsi_runs"))
    if any(output_dir.is_relative_to(path) or path.is_relative_to(output_dir) for path in protected):
        raise ValueError("Output overlaps a protected training or source directory")
    targets = [output_dir / "paired_control_comparison.csv", output_dir / "paired_control_comparison.json"]
    if any(path == target for path in (input_path, main_path) for target in targets):
        raise ValueError("Output may not overwrite a comparison input")
    inputs = [{"path": str(path), "sha256": file_hash(path)} for path in (input_path, main_path)]
    points = read_existing_points(input_path, main_path)
    rows = comparison_table(points, repeats=repeats, seed=seed)
    selected_rsi = [point["metadata"] for point in points if point["method"] == "rsi" and point["workpoint_selected"]]
    selected_controls = [point["metadata"] for point in points if point["method"] in {"d0", "mean_hinge", "abs_hard", "shrink"} and point["workpoint_selected"]]
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "complete", "subset": "M", "split": "val",
             "test_scoring_locked": True, "input_files": inputs, "tool_path": str(Path(__file__).resolve()),
             "tool_sha256": file_hash(Path(__file__)), "existing_point_count": len(points), "comparison_count": len(rows),
             "selected_rsi_from_existing_main_table": selected_rsi, "selected_simple_controls_from_existing_main_table": selected_controls,
             "workpoints_reselected": False, "new_lambda_or_alpha": False, "epsilon_d": EPSILON,
             "bootstrap_seed": seed, "bootstrap_repeats": repeats, "worst_tail_definition": TAIL_DEFINITION,
             "difference_orientation": "RSI minus control; positive Dice/G+/tail and negative H_epsilon are favorable",
             "interval_scope": "exploratory M validation used for checkpoint and workpoint selection",
             "final_continue_or_stop_decision_pending": True, "GPU_compute": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    backups = []
    staging = Path(tempfile.mkdtemp(prefix=".paired-comparison-", dir=output_dir))
    try:
        csv_path = staging / targets[0].name
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        for item in inputs:
            if file_hash(Path(item["path"])) != item["sha256"]:
                raise RuntimeError("Comparison inputs changed while computing intervals")
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        for target in targets:
            if target.exists():
                backup = target.with_name(target.name + ".before_" + suffix)
                backups.append({"path": str(backup), "original_sha256": file_hash(target)})
        audit.update(prior_output_backups=backups, output_csv_sha256=file_hash(csv_path))
        (staging / targets[1].name).write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        published, moved = [], []
        try:
            for target, staged in zip(targets, (csv_path, staging / targets[1].name)):
                if target.exists():
                    backup = target.with_name(target.name + ".before_" + suffix)
                    target.rename(backup)
                    moved.append((target, backup))
                os.replace(staged, target)
                published.append(target)
        except Exception:
            for target in published:
                target.unlink()
            for target, backup in reversed(moved):
                backup.rename(target)
            raise
        return audit
    finally:
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "outputs/seed17/per_reference_gains.csv")
    parser.add_argument("--main-table", type=Path, default=PROJECT / "outputs/seed17/pilot_main_table.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/seed17/paired_controls")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    args = parser.parse_args()
    audit = finalize(args.input, args.main_table, args.output_dir,
                     repeats=args.bootstrap_repeats, seed=args.bootstrap_seed)
    print(json.dumps({"status": audit["status"], "comparisons": audit["comparison_count"],
                      "output_dir": str(args.output_dir), "test_scoring_locked": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
