"""CPU readout of the complete registered seed17 matrix; never select experiments.

Run only after finalizer, sensitivity and paired-control commands finish.
--validate-only writes nothing, including when the matrix is not yet ready.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools.finalize_pilot import frozen_sources, review_matrix, review_completed_training
from tools.paired_control_comparison import (
    METRICS, TAIL_DEFINITION, compare_points, finite, make_units, method_name, read_existing_points,
    selected, summarize_references,
)

SOURCE_BUNDLE = "36825cbb7a90064f3db3cfd97eb16ec62b29aa1947e17e88c7ad4759024268fb"
REPEATS, SEED, EPSILON = 2000, 17, .005
BOUNDARY = ("bf1", "bf1_025", "bf1_1pct", "hd95", "hd95_normalized")
SIDECAR = (*BOUNDARY, "any_harm", "sign_flip")
RUNS = {"Anchor", "B", "D0", *(f"{family}-{weight}" for family in
        ("RSI", "MeanHinge", "AbsHard") for weight in ("0.3", "1", "3")),
        *(f"Shrink-D0-{alpha}" for alpha in ("0.000000", "0.333333", "0.666667", "1.000000"))}
SHRINK_ALPHAS = dict(zip(("Shrink-D0-0.000000", "Shrink-D0-0.333333",
                         "Shrink-D0-0.666667", "Shrink-D0-1.000000"), (0., 1/3, 2/3, 1.)))
COUNTS = {"M": (223, 471), "H": (47, 104), "T1": (20, 49)}
FILES = ("per_reference_gains.csv", "pilot_main_table.csv", "update_diagnostics.csv",
         "epoch_diagnostics.csv", "source_sensitivity.csv", "finalization_audit.json",
         "decision_inputs.json", "paired_controls/paired_control_comparison.csv",
         "paired_controls/paired_control_comparison.json",
         "sensitivity/source_threshold_sensitivity.csv",
         "sensitivity/source_threshold_sensitivity.json")


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def csv_rows_for_json(rows):
    """CSV missing conditional rates/correlations remain JSON null, never zero."""
    metadata = {"protocol_id", "run_id", "checkpoint_hash", "subset", "method", "report",
                "image_subset", "reference_subset", "stratum", "source_tool", "reference_selection",
                "selection_use", "split", "interval_scope", "image_id", "group_id", "reference_id", "workpoint_status"}
    return [{key: (None if key not in metadata and
                  (value is None or str(value).strip().lower() in {"", "nan", "none", "null"}) else value)
             for key, value in row.items()} for row in rows]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_record(record):
    require(file_record(record["path"])["sha256"] == record["sha256"],
            "Input digest differs: " + record["path"])


def cohort_vectors(rows):
    """Image-equal boundary means and all-multi-reference incidence indicators."""
    images = defaultdict(list)
    for row in rows:
        images[row["image_id"]].append(row)
    ids, groups, values = [], [], []
    seen = set()
    for image in sorted(images):
        refs = images[image]
        require(len(refs) >= 2, "Incidence cohort must retain multiple references")
        for row in refs:
            identity = (image, row["reference_id"])
            require(identity not in seen, "Repeated reference in incidence/boundary cohort")
            seen.add(identity)
        case = {str(row.get("group_id") or "") for row in refs}
        require(len(case) == 1, "Group varies within image")
        gains = [finite(row["dice"], "Dice") - finite(row["dice_anchor"], "anchor") for row in refs]
        harm = min(gains) < -EPSILON
        ids.append(image); groups.append(case.pop())
        values.append([*[np.mean([finite(row[k], k) for row in refs]) for k in BOUNDARY],
                       float(harm), float(harm and max(gains) > EPSILON)])
    require(bool(ids), "Empty scientific cohort")
    return ids, groups, np.asarray(values, dtype=float)


def sidecar_comparison(current, control, *, repeats=REPEATS, seed=SEED, draws=None):
    ids, groups, a = cohort_vectors(current)
    other_ids, other_groups, b = cohort_vectors(control)
    require(ids == other_ids and groups == other_groups, "Paired sidecar cohorts differ")
    require({(r["image_id"], r["reference_id"]) for r in current} ==
            {(r["image_id"], r["reference_id"]) for r in control}, "Paired reference identities differ")
    require({(r["image_id"], r["reference_id"]): finite(r["dice_anchor"], "anchor") for r in current} ==
            {(r["image_id"], r["reference_id"]): finite(r["dice_anchor"], "anchor") for r in control},
            "Paired per-reference anchors differ")
    units, unit_name = make_units(ids, groups)
    if draws is None:
        draws = np.random.default_rng(seed).integers(len(units), size=(repeats, len(units)))
    draws = np.asarray(draws)
    require(draws.shape == (repeats, len(units)) and np.issubdtype(draws.dtype, np.integer)
            and not (draws < 0).any() and not (draws >= len(units)).any(), "Invalid paired bootstrap draws")
    counts = np.asarray([len(unit) for unit in units])
    sums = np.asarray([(a[unit] - b[unit]).sum(axis=0) for unit in units])
    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    result = {"n_images": len(ids), "n_references": len(current), "bootstrap_unit": unit_name,
              "bootstrap_units": len(units), "bootstrap_repeats": repeats, "bootstrap_seed": seed,
              "all_images_multi_reference": True, "epsilon_D": EPSILON,
              "incidence_denominator": "all images; every included image retains >=2 references",
              "any_harm_definition": "min reference Dice-current minus Dice-anchor < -0.005",
              "sign_flip_definition": "min gain < -0.005 and max gain > 0.005",
              "difference_orientation": "RSI minus control",
              "interval_scope": "exploratory selected-development validation, unadjusted"}
    for index, metric in enumerate(SIDECAR):
        result.update({metric + "_rsi": float(a[:, index].mean()),
                       metric + "_control": float(b[:, index].mean()),
                       metric + "_difference": float((a - b)[:, index].mean()),
                       metric + "_ci_low": float(low[index]), metric + "_ci_high": float(high[index])})
        if metric in ("any_harm", "sign_flip"):
            result.update({metric + "_rsi_count": int(a[:, index].sum()),
                           metric + "_control_count": int(b[:, index].sum())})
    return result


def applied_message_rows(updates):
    output = []
    seen = set()
    for original in updates:
        row = dict(original)
        identity = (row["run_id"], row["subset"], row["image_id"])
        require(identity not in seen, "Duplicate image update row")
        seen.add(identity)
        raw, alpha, action = (finite(row[k], k) for k in ("message_rms", "alpha", "action"))
        require(raw >= 0 and alpha == action, "Invalid raw message/action")
        if row["run_id"] == "Anchor":
            require(alpha == 0 and raw == 0, "Anchor message must be zero")
        elif not row["run_id"].startswith("Shrink-D0-"):
            require(alpha == 1, "Ordinary message scale must equal one")
        row.update(raw_message_rms=raw, applied_message_rms_scale=abs(alpha),
                   applied_message_rms=abs(alpha) * raw)
        output.append(row)
    return output


def readiness(input_dir, run_dir):
    missing = [name for name in FILES if not (input_dir / name).is_file()]
    if missing:
        return {"status": "not_ready", "missing_completed_inputs": missing}, None
    refs = read_csv(input_dir / "per_reference_gains.csv")
    actual_runs = {r["run_id"] for r in refs}
    if actual_runs != RUNS:
        return {"status": "not_ready", "expected_M_points": 16,
                "actual_runs": sorted(actual_runs), "missing_runs": sorted(RUNS - actual_runs),
                "unexpected_runs": sorted(actual_runs - RUNS)}, None
    require(all(r.get("split") == "val" and r.get("seed") == "17" for r in refs), "Only seed17 val rows are allowed")
    inputs = [file_record(input_dir / name) for name in FILES]
    inputs.extend(file_record(PROJECT / name) for name in
                  ("data_manifests/final_references.csv", "data_manifests/file_audit.json",
                   "tools/finalize_pilot.py", "tools/paired_control_comparison.py", Path(__file__).relative_to(PROJECT)))
    audit = json.loads((input_dir / "finalization_audit.json").read_text())
    require(audit["scope"] == "T2" and audit["seed"] == 17 and audit["test_scoring_locked"]
            and audit["original_csv_bytes_unchanged"] and not audit["models_constructed"] and not audit["GPU_compute"],
            "Finalizer provenance is incomplete or unlocked")
    sources = frozen_sources()
    require(sources["source_bundle_hash"] == SOURCE_BUNDLE and audit["original_training_source"] == sources,
            "Frozen source/config bundle differs")
    require(file_record(audit["tool_path"])["sha256"] == audit["tool_sha256"], "Finalizer tool changed")
    for record in audit["original_csvs"]:
        verify_record(record)
        inputs.append(file_record(record["path"]))
    manifest_path = PROJECT / "data_manifests/final_references.csv"
    manifest_hash = file_record(manifest_path)["sha256"]
    data_audit = json.loads((PROJECT / "data_manifests/file_audit.json").read_text())
    require(data_audit["file_audit_complete"] and data_audit["scope"] == "all_selected_M_H_T1"
            and data_audit["final_references_sha256"] == manifest_hash and not data_audit["models_evaluated_on_test"],
            "Dataset audit does not bind the full frozen manifest")
    scope, stages, weights = review_matrix(refs, "T2")
    require(scope == "T2" and len(stages) == 13 and weights == [.3, 1., 3.], "Require all13 full training stages and symmetric weights")
    completions, epoch_rows, training = review_completed_training(run_dir, stages, refs, sources)
    config = yaml.safe_load((PROJECT / "configs/p2_ima_m_v2.yaml").read_text())
    require(config == completions["A_CNN"]["signature"]["config"]
            and config["test_scoring_locked"] and config["conditional_weight_grid"] == [.3, 1., 3.]
            and config["epochs"] == {"A_CNN": 100, "A_T": 100, "B": 40, "D": 40},
            "Complete training signatures do not match the frozen registered YAML configuration")
    require(training == audit["training_completion_evidence"], "Finalizer completion evidence differs")
    for item in training:
        inputs.extend([file_record(item["done_path"]), file_record(item["epoch_csv"])])
        checkpoint = file_record(run_dir / item["run_id"] / "best.pth")
        require(checkpoint["sha256"] == item["best_checkpoint_sha256"], "Selected checkpoint file differs")
        inputs.append(checkpoint)
    require(len(epoch_rows) == 640, "Expected 100+100+40+10*40 committed epochs")
    groups = defaultdict(list)
    for row in refs:
        require(row["manifest_hash"] == manifest_hash, "Export manifest mismatch")
        if row["run_id"] in SHRINK_ALPHAS:
            require(float(row["alpha"]) == SHRINK_ALPHAS[row["run_id"]], "Shrink run label and actual alpha differ")
        groups[(row["run_id"], row["subset"])].append(row)
    manifest = read_csv(manifest_path)
    expected = {subset: {(r["image_id"], r["reference_id"]) for r in manifest
                         if r["subset"] == subset and r["split"] == "val"} for subset in COUNTS}
    manifest_groups = {(r["subset"], r["image_id"], r["reference_id"]): r["group_id"] for r in manifest if r["split"] == "val"}
    require(len(refs) == 16 * 624 and set(groups) == {(run, subset) for run in RUNS for subset in COUNTS},
            "Require exactly48 full source cohorts")
    for (_, subset), rows in groups.items():
        identities = {(r["image_id"], r["reference_id"]) for r in rows}
        require(identities == expected[subset] and len(identities) == len(rows)
                and (len({r["image_id"] for r in rows}), len(rows)) == COUNTS[subset], "Manifest reference cohort differs")
        require(all(r["group_id"] == manifest_groups[(subset, r["image_id"], r["reference_id"])] for r in rows),
                "Export grouping differs from the known-case manifest")
    for subset in COUNTS:
        anchor = {(r["image_id"], r["reference_id"]): r for r in groups[("Anchor", subset)]}
        d0 = {(r["image_id"], r["reference_id"]): r for r in groups[("D0", subset)]}
        for run in RUNS:
            for row in groups[(run, subset)]:
                key = (row["image_id"], row["reference_id"])
                require(float(row["dice_anchor"]) == float(anchor[key]["dice"])
                        and float(row["loss_anchor"]) == float(anchor[key]["loss"])
                        and float(row["dice_D0"]) == float(d0[key]["dice"]), "Shared anchor/D0 metrics differ")
        for run, baseline in (("Shrink-D0-0.000000", anchor), ("Shrink-D0-1.000000", d0)):
            for row in groups[(run, subset)]:
                other = baseline[(row["image_id"], row["reference_id"])]
                require(all(float(row[k]) == float(other[k]) for k in
                            ("dice", "iou", *BOUNDARY, "loss", "loss_orig", "changed_pixel_fraction", "logit_delta_rms")),
                        "Actual Shrink endpoint differs from anchor or D0")
    points = read_existing_points(input_dir / "per_reference_gains.csv", input_dir / "pilot_main_table.csv")
    require(len(points) == 16, "Main table lacks16 existing M points")
    pair_audit = json.loads((input_dir / "paired_controls/paired_control_comparison.json").read_text())
    require(pair_audit["status"] == "complete" and pair_audit["existing_point_count"] == 16
            and pair_audit["comparison_count"] == 39 and pair_audit["bootstrap_repeats"] == REPEATS
            and pair_audit["bootstrap_seed"] == SEED and not pair_audit["workpoints_reselected"]
            and pair_audit["subset"] == "M" and pair_audit["split"] == "val"
            and pair_audit["test_scoring_locked"], "Full existing paired comparison sidecar is not complete")
    for record in pair_audit["input_files"]:
        verify_record(record)
    require({(str(Path(r["path"]).resolve()), r["sha256"]) for r in pair_audit["input_files"]} ==
            {(str((input_dir / name).resolve()), file_record(input_dir / name)["sha256"])
             for name in ("per_reference_gains.csv", "pilot_main_table.csv")},
            "Paired intervals were computed from different final inputs")
    require(Path(pair_audit["tool_path"]).resolve() == PROJECT / "tools/paired_control_comparison.py"
            and pair_audit["tool_sha256"] == file_record(pair_audit["tool_path"])["sha256"],
            "Paired helper provenance changed")
    require(pair_audit["output_csv_sha256"] == file_record(input_dir / "paired_controls/paired_control_comparison.csv")["sha256"],
            "Paired CSV digest differs")
    pairs = read_csv(input_dir / "paired_controls/paired_control_comparison.csv")
    by_run = {p["metadata"]["run_id"]: p for p in points}
    require(len(pairs) == 39 and {(r["rsi_run_id"], r["control_run_id"]) for r in pairs} ==
            {(a["metadata"]["run_id"], b["metadata"]["run_id"]) for a in points for b in points
             if a["method"] == "rsi" and b["method"] != "rsi"}, "M paired comparison matrix differs")
    for pair in pairs:
        for prefix in ("rsi", "control"):
            point = by_run[pair[prefix + "_run_id"]]
            require(pair[prefix + "_checkpoint_hash"] == point["metadata"]["checkpoint_hash"]
                    and selected(pair[prefix + "_workpoint_selected"]) == point["workpoint_selected"]
                    and all(float(pair[prefix + "_" + k]) == float(point["metadata"][k]) for k in ("action", "lambda", "alpha")),
                    "Existing paired comparison metadata/selection differs")
            require(all(math.isclose(finite(pair[k + "_" + prefix], k), point["values"][k], abs_tol=1e-12)
                        for k in METRICS), "Existing paired point values differ")
        require(int(pair["bootstrap_repeats"]) == REPEATS and int(pair["bootstrap_seed"]) == SEED
                and float(pair["epsilon_d"]) == EPSILON, "Existing paired sampling protocol differs")
        current = by_run[pair["rsi_run_id"]]
        units, unit_name = make_units(current["image_ids"], current["group_ids"])
        require(int(pair["n_images"]) == 223 and int(pair["n_references"]) == 471
                and len(units) == int(pair["bootstrap_units"]) == 222 and pair["bootstrap_unit"] == unit_name
                and int(pair["worst_tail_n_observed"]) == 23 and selected(pair["worst_tail_reranked_each_draw"])
                and pair["worst_tail_definition"] == TAIL_DEFINITION, "Existing paired grouping or reranked-tail definition differs")
        for metric in METRICS:
            require(finite(pair[metric + "_ci_low"], metric) <= finite(pair[metric + "_ci_high"], metric)
                    and math.isclose(finite(pair[metric + "_difference"], metric),
                                     float(pair[metric + "_rsi"]) - float(pair[metric + "_control"]), abs_tol=1e-12),
                    "Existing paired interval or difference is invalid")
    sensitivity_audit = json.loads((input_dir / "sensitivity/source_threshold_sensitivity.json").read_text())
    require(sensitivity_audit["status"] == "complete" and sensitivity_audit["input_reference_rows"] == 9984
            and sensitivity_audit["test_scoring_locked"]
            and sensitivity_audit["configuration"]["manifest_sha256"] == manifest_hash
            and sensitivity_audit["configuration"]["fixed_two_seed"] == 17
            and sensitivity_audit["configuration"]["threshold"] == .5
            and sensitivity_audit["configuration"]["epsilon_D"] == [0., .002, .005, .01], "Full fixed threshold sidecar differs")
    for record in sensitivity_audit["inputs"]:
        verify_record(record)
        inputs.append(file_record(record["path"]))
    require({str(Path(r["path"]).resolve()) for r in sensitivity_audit["inputs"]} ==
            {str(p.resolve()) for p in (input_dir / "corrected_inputs").glob("*.csv")},
            "Sensitivity sidecar does not use this finalizer's corrected input directory")
    require(sensitivity_audit["tool_sha256"] == file_record(PROJECT / "tools/source_threshold_sensitivity.py")["sha256"]
            and Path(sensitivity_audit["configuration"]["path"]).resolve() == PROJECT / "configs/p2_ima_m_v2.yaml"
            and sensitivity_audit["configuration"]["sha256"] == file_record(PROJECT / "configs/p2_ima_m_v2.yaml")["sha256"],
            "Sensitivity tool or configuration provenance changed")
    require(sensitivity_audit["output_sha256"] == file_record(input_dir / "sensitivity/source_threshold_sensitivity.csv")["sha256"],
            "Sensitivity CSV digest differs")
    sensitivity_rows = read_csv(input_dir / "sensitivity/source_threshold_sensitivity.csv")
    require(len(sensitivity_audit["inputs"]) == 15 and sum(r["reference_rows"] for r in sensitivity_audit["inputs"]) == 9984
            and sensitivity_audit["summary_rows"] == len(sensitivity_rows) == 768
            and {r["run_id"] for r in sensitivity_rows} == RUNS,
            "Sensitivity matrix lacks the full native/tool/fixed-two registered points")
    updates = read_csv(input_dir / "update_diagnostics.csv")
    require(len(updates) == 16 * 290, "Full image update table is incomplete")
    require({(r["run_id"], r["subset"], r["image_id"]) for r in updates} ==
            {(r["run_id"], r["subset"], r["image_id"]) for r in refs}, "Update images differ from reference exports")
    by_image = defaultdict(list)
    for row in refs:
        by_image[(row["run_id"], row["subset"], row["image_id"])].append(row)
    for update in updates:
        current = by_image[(update["run_id"], update["subset"], update["image_id"])]
        require(int(update["reference_count"]) == len(current), "Update reference count differs")
        for ref in current:
            require(all(update[k] == ref[k] for k in ("protocol_id", "seed", "checkpoint_hash", "group_id")),
                    "Update checkpoint/group metadata differs from reference inputs")
            require(method_name(update["method"]) == method_name(ref["method"]), "Update method differs")
            require(all(float(update[k]) == float(ref[k]) for k in
                        ("action", "lambda", "alpha", "message_rms", "logit_delta_rms", "changed_pixel_fraction")),
                    "Update action or raw diagnostics differ from reference inputs")
    derived_epochs = read_csv(input_dir / "epoch_diagnostics.csv")
    fields = set().union(*(r.keys() for r in [*derived_epochs, *epoch_rows]))
    canonical_epoch = lambda row: tuple((field, row.get(field, "")) for field in sorted(fields))
    require([canonical_epoch(r) for r in derived_epochs] == [canonical_epoch(r) for r in epoch_rows],
            "Aggregate epoch table does not match all13 original committed logs")
    for record in inputs:
        verify_record(record)
    return {"status": "ready", "M_points": 16, "full_training_stages": 13,
            "source_cohorts": 48, "reference_rows": 9984, "epochs": 640,
            "positive_weights": weights, "test_scoring_locked": True}, dict(
                inputs=inputs, sources=sources, refs=refs, groups=groups, points=points,
                updates=updates, completions=completions)


def comparisons_from_existing(input_dir, evidence):
    points = evidence["points"]
    selected_rsi = [p for p in points if p["method"] == "rsi" and p["workpoint_selected"]]
    require(len(selected_rsi) <= 1, "Existing main table has multiple selected RSI points")
    currents = selected_rsi or [p for p in points if p["method"] == "rsi"]
    controls = [p for p in points if p["method"] in {"d0", "mean_hinge", "abs_hard", "shrink"}]
    paired = read_csv(input_dir / "paired_controls/paired_control_comparison.csv")
    pair_map = {(r["rsi_run_id"], r["control_run_id"]): r for r in paired}
    require(len(paired) == len(pair_map) == 39, "Repeated/missing paired comparisons")
    comparisons = []
    for current in currents:
        rid = current["metadata"]["run_id"]
        for cohort in ("M", "H", "T1", "M_references_on_H_images", "M_references_on_T1_images"):
            reference_subset = "M" if "_on_" in cohort else cohort
            image_subset = cohort.split("_on_")[1].split("_")[0] if "_on_" in cohort else cohort
            ids = {r["image_id"] for r in evidence["groups"][(rid, image_subset)]}
            a_rows = [r for r in evidence["groups"][(rid, reference_subset)] if r["image_id"] in ids]
            a = summarize_references(a_rows)
            units, _ = make_units(a["image_ids"], a["group_ids"])
            draws = np.random.default_rng(SEED).integers(len(units), size=(REPEATS, len(units)))
            for control in controls:
                cid = control["metadata"]["run_id"]
                b_rows = [r for r in evidence["groups"][(cid, reference_subset)] if r["image_id"] in ids]
                b = summarize_references(b_rows)
                if cohort == "M":
                    existing = pair_map[(rid, cid)]
                    for metric in METRICS:
                        require(math.isclose(float(existing[metric + "_difference"]),
                                             a["values"][metric] - b["values"][metric], abs_tol=1e-12),
                                "Existing M direct difference differs")
                    core = {k: existing[k] for k in existing if k.startswith(tuple(m + "_" for m in METRICS))
                            or k in ("worst_tail_definition", "worst_tail_reranked_each_draw", "worst_tail_n_observed")}
                    for k in list(core):
                        if k.endswith(("_rsi", "_control", "_difference", "_ci_low", "_ci_high")):
                            core[k] = finite(core[k], k)
                else:
                    core = compare_points(a, b, repeats=REPEATS, seed=SEED, draws=draws)
                comparisons.append({"cohort": cohort, "image_subset": image_subset,
                                    "reference_subset": reference_subset, "rsi_run_id": rid,
                                    "control_run_id": cid, "rsi_workpoint_selected": current["workpoint_selected"],
                                    "control_workpoint_selected": control["workpoint_selected"],
                                    "core_M_intervals_reused": cohort == "M",
                                    "selection_use": "existing M flags only; all source comparisons descriptive",
                                    **core, **sidecar_comparison(a_rows, b_rows, draws=draws)})
    return selected_rsi, controls, comparisons


def write_outputs(input_dir, output_dir, evidence, gate):
    selected_rsi, controls, comparisons = comparisons_from_existing(input_dir, evidence)
    main = read_csv(input_dir / "pilot_main_table.csv")
    sensitivity = read_csv(input_dir / "sensitivity/source_threshold_sensitivity.csv")
    sources = read_csv(input_dir / "source_sensitivity.csv")
    applied = applied_message_rows(evidence["updates"])
    message_summary = []
    grouped = defaultdict(list)
    for row in applied:
        grouped[(row["run_id"], row["subset"])].append(row)
    for (run, subset), rows in sorted(grouped.items()):
        message_summary.append({"run_id": run, "subset": subset, "n_images": len(rows),
                                "alpha": float(rows[0]["alpha"]),
                                "raw_message_rms": float(np.mean([r["raw_message_rms"] for r in rows])),
                                "applied_message_rms": float(np.mean([r["applied_message_rms"] for r in rows]))})
    review = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
              "scope": "complete registered seed17 development matrix; checkpoint/workpoint-selected validation",
              "readiness": gate, "inputs": evidence["inputs"], "frozen_sources": evidence["sources"],
              "workpoints_reselected": False, "H_T1_used_for_selection": False,
              "test_scoring_locked": True, "test_scores_read": False, "models_constructed": False,
              "GPU_compute": False, "raw_csvs_modified": False, "main_table_modified": False,
              "new_lambda_or_alpha": False, "seed29_started": False, "scientific_pass_declared": False,
              "root_continue_or_stop_decision_pending": True,
              "selected_RSI": [p["metadata"] for p in selected_rsi],
              "no_eligible_RSI_workpoint": not bool(selected_rsi),
              "no_selected_RSI_fallback": "Report all existing RSI candidates descriptively; no substitute selection" if not selected_rsi else None,
              "selected_simple_controls": [p["metadata"] for p in controls if p["workpoint_selected"]],
              "existing_main_table": csv_rows_for_json(main), "direct_control_comparisons": comparisons,
              "threshold_tool_fixed_two": csv_rows_for_json(sensitivity),
              "rater_count_strata": csv_rows_for_json([r for r in sources if r["report"] == "reference_count"]),
              "source_sensitivity_existing": csv_rows_for_json(sources),
              "applied_message_summary": message_summary,
              "limits": ["All intervals exploratory and unadjusted, using the validation set already used for checkpoint/workpoint selection.",
                         "Pairing retains reference sets and methods together within known case groups; missing identities use image units.",
                         "Worst10% reference-gain tails are reranked within each method after every bootstrap draw, with ceil(0.1N).",
                         "H/T1 and exactly matched M-reference image cohorts describe source plus case-cohort sensitivity, never tuning.",
                         "A crossing-zero interval does not establish equivalence; coordinate dominance without a substantive advantage does not prove superiority.",
                         "Eligibility under Dice tolerance is an engineering selection flag, not scientific success.",
                         "Soft-risk improvement does not ensure every hard-Dice reference is safe.",
                         "Known group separation and duplicate audit do not prove independence of unknown patient/lesion identities."]}
    md = "# seed17 完整对称网格科学复核\n\n完整16个M工作点、13项固定预算训练及48个M/H/T1来源队列已验收。下表沿用原主表工作点标记，没有重选λ/α；所有区间来自已参与checkpoint和工作点选择的开发验证集，属未作多重比较校正的探索性证据。最终继续/停止、seed29与正式验证仍由根代理判断，test保持锁定。\n\n"
    md += "| M方法 | 选中 | Dice | G+ | Hε | 最差10%参考收益 |\n|---|---|---:|---:|---:|---:|\n"
    for row in main:
        if row["subset"] == "M":
            md += "| " + row["run_id"] + " | " + ("是" if selected(row["workpoint_selected"]) else "否") + " | "
            md += " | ".join(f"{float(row[k]):.6f}" for k in ("dice", "G_plus", "H_epsilon", "worst_tail_10pct")) + " |\n"
    if not selected_rsi:
        md += "\nRSI没有被主表标记为合格选中工作点；下面保留全部RSI候选的描述性比较，不替代主表选择。\n"
    md += "\n对现有选中RSI（若无则保留全部候选），以下发生率和边界补充使用与核心对照相同的病例/图像配对单位、2000次bootstrap、seed17。M核心Dice/G+/Hε/逐图最小参考收益尾部区间复用原paired_controls结果，没有重复计算或更换尾部定义。\n\n"
    md += "| RSI/简单对照 | Dice差 [95%CI] | Hε差 [95%CI] | 最差尾部差 [95%CI] | any-harm差 pp [95%CI] | sign-flip差 pp [95%CI] |\n|---|---|---|---|---|---|\n"
    for row in comparisons:
        if row["cohort"] != "M":
            continue
        md += "| " + row["rsi_run_id"] + " / " + row["control_run_id"] + " | "
        cells = []
        for metric in ("dice", "H_epsilon", "worst_tail_10pct", "any_harm", "sign_flip"):
            scale = 100 if metric in ("any_harm", "sign_flip") else 1
            cells.append(f"{scale * float(row[metric + '_difference']):+.6f} [{scale * float(row[metric + '_ci_low']):+.6f}, {scale * float(row[metric + '_ci_high']):+.6f}]")
        md += " | ".join(cells) + " |\n"
    md += "\n差值均为RSI减对照：Dice/G+/尾部较大有利，Hε/伤害/符号冲突较小有利。末位小数的坐标优劣和区间跨0都不能证明实质胜出或严格等价；必须共同核对G+保留、负尾部、实际简单控制和来源队列。\n\n"
    md += "| 固定来源队列/对照 | 图/参考数 | Dice差 [95%CI] | Hε差 | BF1差 | HD95像素差 |\n|---|---|---|---:|---:|---:|\n"
    for row in comparisons:
        if row["cohort"] != "M":
            md += f"| {row['cohort']} / {row['control_run_id']} ({row['rsi_run_id']}) | {row['n_images']}/{row['n_references']} | {float(row['dice_difference']):+.6f} [{float(row['dice_ci_low']):+.6f}, {float(row['dice_ci_high']):+.6f}] | {float(row['H_epsilon_difference']):+.6f} | {row['bf1_difference']:+.6f} | {row['hd95_difference']:+.6f} |\n"
    md += "\nH为47图/104参考、严格T1为20图/49参考；同一批图像的M参考分别为109/55。来源原生参考与相同图像M参考的对照差可区分参考来源和病例队列敏感性；这些描述性风险不参与调参。阈值ε=0/.002/.005/.01、m=2/m>=3、固定seed17两参考、soft/hard分歧及风险ρ/J/κ详见JSON，不选择额外阈值或样本。\n\n"
    md += "原message_rms是alpha之前的模块消息；新增full_update_diagnostics_applied_message.csv按abs(alpha)*raw计算实际施加消息，Anchor为0，普通工作点action=1，Fixed-Shrink按注册alpha缩放。所有原始导出、原主表和原更新表保持不变，真实tail重算的logit/掩码变化也不替换。\n"
    provenance = {"definition": "applied_message_rms=abs(alpha)*raw_message_rms",
                  "model_contract": "forward_pair returns message before alpha; tail receives f8 + alpha*message",
                  "raw_update_input": file_record(input_dir / "update_diagnostics.csv"),
                  "contract_sources": [file_record(PROJECT / f) for f in ("rsi/models.py", "rsi/export.py", "rsi/metrics.py")],
                  "original_rows_modified": False, "metric_scores_modified": False,
                  "GPU_compute": False, "test_scoring_locked": True,
                  "summary": message_summary, "rows": len(applied)}
    names = ("full_scientific_review.json", "full_scientific_review.md", "full_scientific_comparisons.csv",
             "full_update_diagnostics_applied_message.csv", "full_update_diagnostics_applied_message_provenance.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scientific-readout-", dir=output_dir) as temporary:
        staged = Path(temporary)
        for name, rows in ((names[2], comparisons), (names[3], applied)):
            with (staged / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                writer.writeheader(); writer.writerows(rows)
        provenance["output_sha256"] = file_record(staged / names[3])["sha256"]
        provenance["output_path"] = str(output_dir / names[3])
        review["applied_message_provenance"] = provenance
        review["comparison_csv_sha256"] = file_record(staged / names[2])["sha256"]
        for name, data in ((names[0], review), (names[4], provenance)):
            (staged / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        (staged / names[1]).write_text(md, encoding="utf-8")
        for record in evidence["inputs"]:
            verify_record(record)
        require(frozen_sources() == evidence["sources"], "Frozen science changed during readout")
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        published, backups = [], []
        try:
            for name in names:
                target = output_dir / name
                if target.exists():
                    backup = target.with_name(name + ".before_" + suffix)
                    target.rename(backup)
                    backups.append((target, backup))
                os.replace(staged / name, target)
                published.append(target)
        except Exception:
            for target in published:
                target.unlink()
            for target, backup in reversed(backups):
                backup.rename(target)
            raise
    return {"status": "complete", "comparison_rows": len(comparisons),
            "selected_RSI": [p["metadata"]["run_id"] for p in selected_rsi],
            "output_dir": str(output_dir), "test_scoring_locked": True,
            "root_continue_or_stop_decision_pending": True}


def run(input_dir, run_dir, output_dir=None, *, validate_only=False):
    input_dir, run_dir = Path(input_dir).resolve(), Path(run_dir).resolve()
    output_dir = Path(output_dir or input_dir).resolve()
    require(output_dir.is_relative_to(PROJECT / "outputs"), "Readout outputs must stay inside project outputs")
    require(not output_dir.is_relative_to(PROJECT / "outputs/per_reference"), "Cannot place derived CSVs beside raw exports")
    require(output_dir == input_dir or
            (not output_dir.is_relative_to(input_dir) and not input_dir.is_relative_to(output_dir)),
            "Custom output must be separate from finalizer inputs and sidecars")
    gate, evidence = readiness(input_dir, run_dir)
    if evidence is None:
        return {**gate, "outputs_written": False, "test_scoring_locked": True}
    if validate_only:
        return {**gate, "status": "validated_only", "outputs_written": False}
    return write_outputs(input_dir, output_dir, evidence, gate)


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
