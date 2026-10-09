"""Export exploratory pilot tables from complete per-reference evaluation rows.

Input schema: run_id, checkpoint_hash, image_id, subset, reference_id, dice,
iou, dice_anchor are required. action defaults to 1. Optional metadata and
metric columns are preserved. Each unique run/checkpoint/image/subset/reference/
action key must occur exactly once. This module never loads model predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import gain_loss_dice_cross_table, gain_statistics, paired_bootstrap


REQUIRED = ("run_id", "checkpoint_hash", "image_id", "subset", "reference_id",
            "dice", "iou", "dice_anchor")
METRIC_COLUMNS = ("dice", "iou", "thresholded_jaccard", "bf1", "bf1_025", "bf1_1pct",
                  "hd95", "hd95_normalized", "loss", "bce", "soft_dice", "loss_orig",
                  "pred_empty", "gt_empty", "both_empty", "one_empty", "dice_anchor", "dice_D0",
                  "gain_dice", "gain_loss", "gain_vs_D0", "fp_added", "fn_added", "fp_removed", "fn_removed",
                  "message_rms", "logit_delta_rms", "changed_pixel_fraction")


def _number(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {key} for image {row.get('image_id')}")
    return result


def validate_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    seen = set()
    for raw in rows:
        row = dict(raw)
        for key in REQUIRED:
            if key not in row or row[key] is None or row[key] == "":
                raise ValueError(f"missing required export column: {key}")
        row.setdefault("action", 1)
        if row["action"] in (None, ""):
            row["action"] = 1
        for key in ("run_id", "checkpoint_hash", "image_id", "subset", "reference_id"):
            row[key] = str(row[key])
        unique = tuple(str(row[k]) for k in ("run_id", "checkpoint_hash", "image_id", "subset", "reference_id", "action"))
        if unique in seen:
            raise ValueError(f"duplicate per-reference export key: {unique}")
        seen.add(unique)
        for key in METRIC_COLUMNS + ("loss_anchor", "loss_D0", "lambda", "alpha"):
            if row.get(key) not in (None, ""):
                row[key] = _number(row, key)
        for key in ("dice", "iou", "dice_anchor", "dice_D0"):
            if row.get(key) is not None and not 0 <= row[key] <= 1:
                raise ValueError(f"{key} must be in [0,1]")
        expected = row["dice"] - row["dice_anchor"]
        if row.get("gain_dice") is not None and not math.isclose(row["gain_dice"], expected, abs_tol=1e-7):
            raise ValueError("gain_dice disagrees with dice - dice_anchor")
        row["gain_dice"] = expected
        if row.get("loss") is not None and row.get("loss_anchor") is not None:
            expected_loss = row["loss_anchor"] - row["loss"]
            if row.get("gain_loss") is not None and not math.isclose(row["gain_loss"], expected_loss, abs_tol=1e-7):
                raise ValueError("gain_loss disagrees with loss_anchor - loss")
            row["gain_loss"] = expected_loss
        if row.get("dice_D0") is not None:
            row["gain_vs_D0"] = row["dice"] - row["dice_D0"]
        normalized.append(row)
    if not normalized:
        raise ValueError("no per-reference rows to summarize")
    return normalized


def _method(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("method", "")).lower().replace("-", "_")
    if explicit:
        return {"meanhinge": "mean_hinge", "abshard": "abs_hard",
                "fixed_shrink": "shrink", "shrink_d0": "shrink"}.get(explicit, explicit)
    name = str(row["run_id"]).lower().replace("-", "_")
    for match, method in [("shrink", "shrink"), ("abs", "abs_hard"), ("mean", "mean_hinge"),
                          ("rsi", "rsi"), ("d0", "d0"), ("anchor", "anchor")]:
        if match in name:
            return method
    return "B" if name in {"b", "run_b"} else name


def _images(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[row["image_id"]].append(row)
    return dict(sorted(result.items()))


def _aggregate(rows: list[dict[str, Any]], *, epsilon_d: float = 0.005) -> dict[str, Any]:
    images = _images(rows)
    per_image = list(images.values())
    gains = [[r["gain_dice"] for r in refs] for refs in per_image]
    has_changes = all(_number(r, "changed_pixel_fraction") is not None for r in rows)
    changed = [any(_number(r, "changed_pixel_fraction") > 0 for r in refs) for refs in per_image] if has_changes else None
    result = gain_statistics(gains, epsilon_d=epsilon_d, changed=changed)
    for column in METRIC_COLUMNS:
        values = [[_number(r, column) for r in refs] for refs in per_image]
        present = [v is not None for refs in values for v in refs]
        if any(present) and not all(present):
            raise ValueError(f"partial metric column {column}; export all references or none")
        if all(present):
            result[column] = float(np.mean([np.mean(refs) for refs in values]))
    if all(_number(r, "pred_empty") is not None for r in rows):
        result["empty_prediction_count"] = sum(bool(rs[0]["pred_empty"]) for rs in per_image)
        result["empty_prediction_fraction"] = result["empty_prediction_count"] / len(per_image)
    if all("gain_loss" in r for r in rows):
        result.update(gain_loss_dice_cross_table(
            [[r["gain_loss"] for r in refs] for refs in per_image], gains, epsilon_d=epsilon_d))
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        fields = ["status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_key(row: Mapping[str, Any], *, subset: bool = True) -> tuple[str, ...]:
    keys = ["protocol_id", "seed", "run_id", "checkpoint_hash", "action", "lambda", "alpha"]
    if subset:
        keys.append("subset")
    return tuple(str(row.get(key, "")) for key in keys)


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("protocol_id", "seed", "run_id", "checkpoint_hash", "subset", "action", "lambda", "alpha")
    return {**{key: row.get(key, "") for key in keys}, "method": _method(row)}


def _attach_d0(rows: list[dict[str, Any]]) -> None:
    """Join a selected D0 checkpoint when dice_D0 was not already exported."""
    d0: dict[tuple[str, ...], float] = {}
    key = lambda r: tuple(str(r.get(k, "")) for k in ("protocol_id", "seed", "image_id", "subset", "reference_id"))
    for row in rows:
        if _method(row) == "d0":
            k = key(row)
            if k in d0 and not math.isclose(d0[k], row["dice"], abs_tol=1e-12):
                raise ValueError("multiple distinct D0 outputs: supply selected-checkpoint dice_D0 explicitly")
            d0[k] = row["dice"]
    for row in rows:
        if row.get("dice_D0") is None and key(row) in d0:
            row["dice_D0"] = d0[key(row)]
        if row.get("dice_D0") is not None:
            row["gain_vs_D0"] = row["dice"] - row["dice_D0"]


def _fixed_two(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected = []
    for image_id, refs in _images(rows).items():
        if len(refs) < 2:
            continue
        refs = sorted(refs, key=lambda r: r["reference_id"])
        digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        indices = rng.choice(len(refs), size=2, replace=False)
        selected.extend(refs[int(i)] for i in indices)
    return selected


def _select_workpoints(main: list[dict[str, Any]]) -> None:
    families: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    d0 = {}
    for row in main:
        row["workpoint_selected"] = False
        if row["subset"] != "M":
            continue
        seed_key = (str(row["protocol_id"]), str(row["seed"]))
        if row["method"] == "d0":
            if seed_key in d0:
                raise ValueError("pilot selection requires one selected D0 checkpoint per seed")
            d0[seed_key] = row["dice"]
            row["workpoint_selected"] = True
        if row["method"] in {"rsi", "mean_hinge", "abs_hard", "shrink"}:
            families[(*seed_key, row["method"])].append(row)
    for family_key, candidates in families.items():
        baseline = d0.get(family_key[:2])
        if baseline is None:
            for row in candidates:
                row["workpoint_status"] = "D0 comparator missing"
            continue
        eligible = [row for row in candidates if row["dice"] >= baseline - 0.002]
        if family_key[-1] == "rsi":
            eligible = [row for row in eligible if float(row.get("lambda") or 0) > 0]
        if eligible:
            def ranking(row: dict[str, Any]) -> tuple[float, float, float]:
                final = -(float(row.get("alpha") or 0)) if family_key[-1] == "shrink" else float(row.get("lambda") or 0)
                return row["H_epsilon"], -row["dice"], final
            best = min(eligible, key=ranking)
            best["workpoint_selected"] = True
        for row in candidates:
            row["workpoint_status"] = "eligible" if row in eligible else "outside Dice tolerance; retain D0 if no eligible candidate"


def summarize_pilot(
    rows: Iterable[Mapping[str, Any]] | str | Path,
    output_dir: str | Path,
    *,
    split: str = "val",
    bootstrap_repeats: int = 2000,
    bootstrap_seed: int = 17,
    fixed_two_seed: int = 17,
) -> list[dict[str, Any]]:
    """Write the five pilot CSVs and gain/harm figure from exported val rows.

    Intervals are explicitly exploratory after checkpoint/workpoint selection.
    Canonical epoch and gradient diagnostics are exported by the training loop.
    """
    if split not in {"val", "train"}:
        raise ValueError("pilot summarizer refuses test scoring; only train/val are permitted")
    if isinstance(rows, (str, Path)):
        with Path(rows).open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    rows = validate_rows(rows)
    for row in rows:
        if row.get("split") not in (None, "", split):
            raise ValueError("input rows do not match the requested train/val split")
        row["split"] = split
    _attach_d0(rows)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_run_key(row)].append(row)
    main, sources, updates = [], [], []
    for refs in groups.values():
        meta = _metadata(refs[0])
        aggregate = {**meta, **_aggregate(refs), "split": split,
                     "interval_scope": "exploratory selected-development outputs"}
        images = _images(refs)
        per_image = list(images.values())
        current_dice = [np.mean([r["dice"] for r in rs]) for rs in per_image]
        anchor_dice = [np.mean([r["dice_anchor"] for r in rs]) for rs in per_image]
        group_ids = [rs[0].get("group_id", "") for rs in per_image]
        for rs in per_image:
            if len({str(r.get("group_id", "")) for r in rs}) != 1:
                raise ValueError("group_id differs across references of one image")
        interval = paired_bootstrap(current_dice, anchor_dice, group_ids=group_ids if any(group_ids) else None,
                                    repeats=bootstrap_repeats, seed=bootstrap_seed)
        aggregate.update({f"dice_vs_anchor_{k}": v for k, v in interval.items()})
        if all("dice_D0" in r for r in refs):
            d0_dice = [np.mean([r["dice_D0"] for r in rs]) for rs in per_image]
            interval_d0 = paired_bootstrap(current_dice, d0_dice, group_ids=group_ids if any(group_ids) else None,
                                          repeats=bootstrap_repeats, seed=bootstrap_seed)
            aggregate.update({f"dice_vs_D0_{k}": v for k, v in interval_d0.items()})
            # Reduce each image's complete reference set before paired sampling.
            # Harm is relative to the same anchor for both current and D0.
            current_gains = [np.array([r["dice"] - r["dice_anchor"] for r in rs]) for rs in per_image]
            d0_gains = [np.array([r["dice_D0"] - r["dice_anchor"] for r in rs]) for rs in per_image]
            statistics = {
                "G_plus": lambda g: np.maximum(g, 0).mean(),
                "H_epsilon": lambda g: np.maximum(-g - 0.005, 0).mean(),
                "any_harm": lambda g: float(g.min() < -0.005),
                "worst_reference": lambda g: g.min(),
            }
            for name, statistic in statistics.items():
                interval_statistic = paired_bootstrap(
                    [statistic(g) for g in current_gains], [statistic(g) for g in d0_gains],
                    group_ids=group_ids if any(group_ids) else None,
                    repeats=bootstrap_repeats, seed=bootstrap_seed)
                aggregate.update({f"{name}_vs_D0_{k}": v for k, v in interval_statistic.items()})
        main.append(aggregate)
        sources.append({**meta, "report": "native_source_subset", "image_subset": meta["subset"],
                        "reference_subset": meta["subset"], **_aggregate(refs)})
        for label, predicate in [("m=2", lambda n: n == 2), ("m>=3", lambda n: n >= 3)]:
            selected = [r for rs in per_image if predicate(len(rs)) for r in rs]
            if selected:
                sources.append({**meta, "report": "reference_count", "stratum": label, **_aggregate(selected)})
        fixed = _fixed_two(refs, fixed_two_seed)
        if fixed:
            sources.append({**meta, "report": "fixed_two_references", "fixed_two_seed": fixed_two_seed, **_aggregate(fixed)})
        for tool in sorted({str(r.get("tool", "")) for r in refs} - {""}):
            selected = [r for r in refs if str(r.get("tool")) == tool]
            sources.append({**meta, "report": "tool_sensitivity", "source_tool": tool, **_aggregate(selected)})
        for image_id, rs in images.items():
            update = {**meta, "image_id": image_id, "group_id": rs[0].get("group_id", ""), "reference_count": len(rs)}
            for key in ("message_rms", "logit_delta_rms", "changed_pixel_fraction"):
                values = [_number(r, key) for r in rs]
                if any(v is not None for v in values):
                    if not all(v is not None for v in values):
                        raise ValueError(f"partial per-image update statistic {key}")
                    if not np.allclose(values, values[0], atol=1e-8, rtol=1e-6):
                        raise ValueError(f"{key} must agree across references of the same output")
                    update[key] = values[0]
            updates.append(update)
    # Same model outputs, same H/T1 images: compare independently selected source
    # references with M references on that exact image intersection.
    by_run: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for refs in groups.values():
        by_run[_run_key(refs[0], subset=False)][str(refs[0]["subset"])] = refs
    for subsets in by_run.values():
        if "M" not in subsets:
            continue
        m_images = _images(subsets["M"])
        for subset in ("H", "T1"):
            if subset not in subsets:
                continue
            source_images = _images(subsets[subset])
            if not set(source_images).issubset(m_images):
                raise ValueError(f"{subset} image set must be a subset of the same-run M images")
            matched_m = [r for image_id in source_images for r in m_images[image_id]]
            meta = _metadata(subsets[subset][0])
            sensitivity = {**meta, "report": "paired_M_references_on_source_images",
                           "image_subset": subset, "reference_subset": "M", **_aggregate(matched_m)}
            ids = list(source_images)
            native = [np.mean([r["dice"] for r in source_images[i]]) for i in ids]
            comparator = [np.mean([r["dice"] for r in m_images[i]]) for i in ids]
            group_ids = [source_images[i][0].get("group_id", "") for i in ids]
            sensitivity.update({f"source_minus_M_{k}": v for k, v in paired_bootstrap(
                native, comparator, group_ids=group_ids if any(group_ids) else None,
                repeats=bootstrap_repeats, seed=bootstrap_seed).items()})
            sources.append(sensitivity)
    _select_workpoints(main)
    out = Path(output_dir)
    _write_csv(out / "pilot_main_table.csv", main)
    _write_csv(out / "per_reference_gains.csv", rows)
    _write_csv(out / "source_sensitivity.csv", sources)
    _write_csv(out / "update_diagnostics.csv", updates)
    # Keep explicit comparator tables in addition to the complete joint export.
    _write_csv(out / "per_reference_gains_vs_anchor.csv", rows)
    _write_csv(out / "per_reference_gains_vs_D0.csv", [r for r in rows if "dice_D0" in r])
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plotted = [row for row in main if row["subset"] == "M"] or main
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter([r["G_plus"] for r in plotted], [r["H_epsilon"] for r in plotted],
                         c=[r["dice"] for r in plotted], cmap="viridis", edgecolors="black")
    for row in plotted:
        label = f"{row['run_id']} (seed {row['seed']})"
        if row.get("lambda") not in (None, ""):
            label += f" λ={row['lambda']:g}"
        if row.get("alpha") not in (None, ""):
            label += f" α={row['alpha']:g}"
        ax.annotate(label, (row["G_plus"], row["H_epsilon"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Mean positive Dice gain G+", ylabel="Mean harm Hε (ε=0.005)",
           title=f"{split}: exploratory selected-development tradeoff")
    fig.colorbar(scatter, ax=ax, label="Macro mean-reference Dice")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gain_harm_tradeoff.png", dpi=180)
    plt.close(fig)
    return main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    args = parser.parse_args()
    summarize_pilot(args.input, args.output_dir, split=args.split,
                    bootstrap_repeats=args.bootstrap_repeats, bootstrap_seed=args.bootstrap_seed)


if __name__ == "__main__":
    main()
