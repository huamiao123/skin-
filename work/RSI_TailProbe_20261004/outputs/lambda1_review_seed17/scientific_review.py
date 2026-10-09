"""Independent CPU analysis of completed lambda1 validation exports only."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from tools.paired_control_comparison import summarize_references, compare_points, make_units

OUT = Path(__file__).resolve().parent
REPEATS = 2000
SEED = 17
CONTROLS = ("Shrink-D0-0.333333", "Shrink-D0-0.666667")
BOUNDARY = ("bf1", "bf1_025", "bf1_1pct", "hd95", "hd95_normalized")
CORE = ("dice", "G_plus", "H_epsilon", "worst_tail_10pct")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def file_record(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["run_id"], row["subset"])].append(row)
    return groups


def by_reference(rows):
    indexed = {(row["image_id"], row["reference_id"]): row for row in rows}
    assert len(indexed) == len(rows), "Duplicate reference identity"
    return indexed


def image_means(rows, fields):
    images = defaultdict(list)
    for row in rows:
        images[row["image_id"]].append(row)
    ids = sorted(images)
    groups = []
    matrix = []
    for image_id in ids:
        refs = images[image_id]
        current = {row["group_id"] for row in refs}
        assert len(current) == 1
        groups.append(current.pop())
        matrix.append([np.mean([float(row[field]) for row in refs]) for field in fields])
    return ids, groups, np.asarray(matrix)


def boundary_comparison(current, control):
    ids, group_ids, a = image_means(current, BOUNDARY)
    ids_b, groups_b, b = image_means(control, BOUNDARY)
    assert ids == ids_b and group_ids == groups_b
    units, unit_name = make_units(ids, group_ids)
    draws = np.random.default_rng(SEED).integers(len(units), size=(REPEATS, len(units)))
    counts = np.asarray([len(unit) for unit in units])
    delta_sums = np.asarray([(a[unit] - b[unit]).sum(axis=0) for unit in units])
    samples = delta_sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    result = {"bootstrap_unit": unit_name, "bootstrap_units": len(units)}
    for index, field in enumerate(BOUNDARY):
        result.update({field + "_rsi": float(a[:, index].mean()),
                       field + "_control": float(b[:, index].mean()),
                       field + "_difference": float((a - b)[:, index].mean()),
                       field + "_ci_low": float(low[index]), field + "_ci_high": float(high[index])})
    return result


def incidence_comparison(current, control):
    def vectors(refs):
        images = defaultdict(list)
        for row in refs:
            images[row["image_id"]].append(row)
        ids, case_groups, values = [], [], []
        for image_id in sorted(images):
            rs = images[image_id]
            assert len(rs) >= 2
            ids.append(image_id)
            case_groups.append(rs[0]["group_id"])
            gains = [float(r["dice"]) - float(r["dice_anchor"]) for r in rs]
            harm = min(gains) < -.005
            values.append((harm, harm and max(gains) > .005))
        return ids, case_groups, np.asarray(values, dtype=float)
    ids, groups_a, a = vectors(current)
    ids_b, groups_b, b = vectors(control)
    assert ids == ids_b and groups_a == groups_b
    units, unit_name = make_units(ids, groups_a)
    draws = np.random.default_rng(SEED).integers(len(units), size=(REPEATS, len(units)))
    counts = np.asarray([len(unit) for unit in units])
    delta_sums = np.asarray([(a[unit] - b[unit]).sum(axis=0) for unit in units])
    samples = delta_sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    result = {"rsi_run_id": "RSI-1", "control_run_id": "Shrink-D0-0.666667",
              "subset": "M", "epsilon_D": .005, "n_images": len(ids),
              "n_references": len(current), "bootstrap_unit": unit_name,
              "bootstrap_units": len(units), "bootstrap_repeats": REPEATS,
              "bootstrap_seed": SEED, "all_images_multi_reference": True,
              "difference_orientation": "RSI minus alpha2/3; negative is lower incidence",
              "any_harm_definition": "per image min reference Dice gain < -0.005",
              "sign_flip_definition": "per image min reference Dice gain < -0.005 and max reference Dice gain > 0.005",
              "interval_scope": "multiple exploratory selected-validation metrics; not a new selection criterion"}
    for index, metric in enumerate(("any_harm", "sign_flip")):
        result.update({metric + "_rsi_count": int(a[:, index].sum()),
                       metric + "_control_count": int(b[:, index].sum()),
                       metric + "_rsi": float(a[:, index].mean()),
                       metric + "_control": float(b[:, index].mean()),
                       metric + "_difference": float((a - b)[:, index].mean()),
                       metric + "_ci_low": float(low[index]), metric + "_ci_high": float(high[index])})
    return result


def applied_update_table():
    original = OUT / "update_diagnostics.csv"
    updates = read_csv(original)
    assert len(updates) == 2900
    for row in updates:
        raw, alpha, action = (float(row[field]) for field in ("message_rms", "alpha", "action"))
        assert math.isfinite(raw) and raw >= 0 and math.isfinite(alpha) and action == alpha
        if row["run_id"] == "Anchor":
            assert alpha == 0 and raw == 0
        elif not row["run_id"].startswith("Shrink-D0-"):
            assert alpha == 1
        row["raw_message_rms"] = raw
        row["applied_message_rms_scale"] = abs(alpha)
        row["applied_message_rms"] = abs(alpha) * raw
    output = OUT / "update_diagnostics_applied_message.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(updates[0]))
        writer.writeheader(); writer.writerows(updates)
    summary = []
    grouped_updates = defaultdict(list)
    for row in updates:
        grouped_updates[(row["run_id"], row["subset"])].append(row)
    for (run, subset), current in sorted(grouped_updates.items()):
        summary.append({"run_id": run, "subset": subset, "n_images": len(current),
                        "alpha": float(current[0]["alpha"]),
                        "raw_message_rms_mean": float(np.mean([r["raw_message_rms"] for r in current])),
                        "applied_message_rms_mean": float(np.mean([r["applied_message_rms"] for r in current]))})
    source_lines = {"rsi/models.py": [376, 383], "rsi/export.py": [37, 63], "rsi/metrics.py": [129, 155]}
    provenance = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(updates),
                  "input": file_record(original), "output": file_record(output),
                  "definition": "applied_message_rms = abs(alpha) * raw_message_rms; whole feature tensor",
                  "model_contract": "message is returned before scaling; tail receives f8 + alpha * message",
                  "ordinary_action1": "applied_message_rms equals raw_message_rms",
                  "Anchor_action0": "applied_message_rms is zero",
                  "FixedShrink_alpha0": "raw module message may be nonzero; applied message is exactly zero",
                  "original_update_and_per_reference_csvs_modified": False,
                  "metric_score_columns_modified": False, "models_constructed": False,
                  "GPU_compute": False, "test_scoring_locked": True,
                  "contract_source_files": [{**file_record(PROJECT / name), "lines": lines}
                                            for name, lines in source_lines.items()],
                  "derivation_script": file_record(Path(__file__)), "per_run_source_summary": summary}
    (OUT / "update_diagnostics_applied_message_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return provenance


def main():
    paths = [OUT / "per_reference_gains.csv", OUT / "pilot_main_table.csv",
             OUT / "source_sensitivity.csv", OUT / "finalization_audit.json",
             OUT / "export_completion_gate.json",
             OUT / "sensitivity/source_threshold_sensitivity.csv",
             OUT / "sensitivity/source_threshold_sensitivity.json",
             OUT / "paired_controls/paired_control_comparison.csv",
             OUT / "paired_controls/paired_control_comparison.json",
             PROJECT / "data_manifests/final_references.csv",
             PROJECT / "outputs/partial_RSI_D0_seed17/partial_scientific_review.json",
             PROJECT / "tools/paired_control_comparison.py",
             PROJECT / "tools/source_threshold_sensitivity.py", OUT / "update_diagnostics.csv",
             PROJECT / "rsi/models.py", PROJECT / "rsi/export.py", PROJECT / "rsi/metrics.py", Path(__file__)]
    inputs = [file_record(path) for path in paths]
    rows = read_csv(paths[0])
    assert len(rows) == 6240 and all(row["split"] == "val" and row["seed"] == "17" for row in rows)
    groups = grouped(rows)
    runs = {key[0] for key in groups}
    assert runs == {"Anchor", "B", "D0", "RSI-1", "MeanHinge-1", "AbsHard-1",
                    "Shrink-D0-0.000000", "Shrink-D0-0.333333", "Shrink-D0-0.666667", "Shrink-D0-1.000000"}
    manifest = read_csv(PROJECT / "data_manifests/final_references.csv")
    expected = {subset: {(r["image_id"], r["reference_id"]) for r in manifest
                         if r["split"] == "val" and r["subset"] == subset}
                for subset in ("M", "H", "T1")}
    assert all(set(by_reference(refs)) == expected[subset] for (_, subset), refs in groups.items())
    primitive_fields = ("dice", "iou", "thresholded_jaccard", *BOUNDARY, "loss", "loss_orig",
                        "bce", "soft_dice", "tp", "fp", "fn", "changed_pixel_fraction", "logit_delta_rms")
    endpoint_checks = []
    for subset in ("M", "H", "T1"):
        anchor = by_reference(groups[("Anchor", subset)])
        d0 = by_reference(groups[("D0", subset)])
        for run in runs:
            for identity, row in by_reference(groups[(run, subset)]).items():
                assert float(row["dice_anchor"]) == float(anchor[identity]["dice"])
                assert float(row["loss_anchor"]) == float(anchor[identity]["loss"])
                assert float(row["dice_D0"]) == float(d0[identity]["dice"])
        for run, control in (("Shrink-D0-0.000000", anchor), ("Shrink-D0-1.000000", d0)):
            diffs = [abs(float(row[field]) - float(control[identity][field]))
                     for identity, row in by_reference(groups[(run, subset)]).items()
                     for field in primitive_fields]
            endpoint_checks.append({"run_id": run, "subset": subset,
                                    "maximum_absolute_metric_difference": max(diffs),
                                    "fields": list(primitive_fields)})
            assert max(diffs) < 1e-10
    paired_main = read_csv(OUT / "paired_controls/paired_control_comparison.csv")
    comparisons = []
    for cohort in ("M", "H", "T1", "M_references_on_H_images", "M_references_on_T1_images"):
        subset = cohort if cohort in ("M", "H", "T1") else "M"
        image_subset = cohort.split("_on_")[1].split("_")[0] if "_on_" in cohort else subset
        ids = {r["image_id"] for r in groups[("RSI-1", image_subset)]}
        current = [r for r in groups[("RSI-1", subset)] if r["image_id"] in ids]
        a = summarize_references(current)
        for run in CONTROLS:
            control = [r for r in groups[(run, subset)] if r["image_id"] in ids]
            b = summarize_references(control)
            if cohort == "M":
                original = next(r for r in paired_main if r["control_run_id"] == run)
                core = {key: original[key] for key in ("bootstrap_unit", "worst_tail_definition")}
                core.update({key: int(original[key]) for key in ("bootstrap_units", "bootstrap_repeats",
                                                                "bootstrap_seed", "worst_tail_n_observed")})
                core.update({f"{metric}_{suffix}": float(original[f"{metric}_{suffix}"])
                             for metric in CORE for suffix in ("rsi", "control", "difference", "ci_low", "ci_high")})
                core["core_intervals_reused_from_existing_M_comparison"] = True
            else:
                core = compare_points(a, b, repeats=REPEATS, seed=SEED)
                core["core_intervals_reused_from_existing_M_comparison"] = False
            descriptive_fields = ("message_rms", "logit_delta_rms", "changed_pixel_fraction")
            _, _, current_means = image_means(current, descriptive_fields)
            _, _, control_means = image_means(control, descriptive_fields)
            descriptive = {field + "_rsi": float(current_means[:, index].mean())
                           for index, field in enumerate(descriptive_fields)}
            descriptive.update({field + "_control": float(control_means[:, index].mean())
                                for index, field in enumerate(descriptive_fields)})
            descriptive.update(raw_message_rms_rsi=descriptive["message_rms_rsi"],
                               raw_message_rms_control=descriptive["message_rms_control"],
                               applied_message_rms_rsi=descriptive["message_rms_rsi"],
                               applied_message_rms_control=abs(float(control[0]["alpha"])) * descriptive["message_rms_control"])
            comparisons.append({"cohort": cohort, "rsi_run_id": "RSI-1", "control_run_id": run,
                                "image_subset": image_subset, "reference_subset": subset,
                                "n_images": len(ids), "n_references": len(current),
                                "difference_orientation": "RSI minus Fixed-Shrink",
                                "bootstrap_repeats": REPEATS, "bootstrap_seed": SEED,
                                "selection_scope": "M only; H/T1 and matched M references are descriptive fixed cohorts",
                                **core, **boundary_comparison(current, control), **descriptive})
    sensitivity = read_csv(OUT / "sensitivity/source_threshold_sensitivity.csv")
    source_rows = [row for row in sensitivity if row["reference_selection"] == "all_references"
                   and row["stratum"] == "native_subset"]
    selected_main_fields = ("run_id", "subset", "n_images", "n_references", "dice", "G_plus", "H_epsilon",
                            "any_harm", "sign_flip", "worst_tail_10pct", *BOUNDARY,
                            "workpoint_selected", "workpoint_status")
    main_table = [{key: row[key] for key in selected_main_fields}
                  for row in read_csv(OUT / "pilot_main_table.csv")]
    grid_path = PROJECT / "outputs/weight_grid_training_seed17.json"
    grid_bytes = grid_path.read_bytes()
    grid = json.loads(grid_bytes)
    paired_m = [row for row in comparisons if row["cohort"] == "M"]
    incidence = incidence_comparison(groups[("RSI-1", "M")], groups[("Shrink-D0-0.666667", "M")])
    applied_provenance = applied_update_table()
    result = {
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "scope": "INTERMEDIATE seed17 lambda1 plus all four actual Fixed-Shrink validation exports; registered symmetric grid pending",
        "test_scoring_locked": True, "test_inputs_read": False, "GPU_compute": False,
        "models_constructed": False, "final_decision_made": False,
        "H_T1_used_for_selection": False, "new_weights_or_alphas_added": False,
        "validation_selection_bias": "All results and 95% bootstrap intervals reuse checkpoint/workpoint-selection validation; exploratory and unadjusted, not independent confirmation",
        "bootstrap_unit": "known case group when available, all references and methods retained together; unknown identities use image units",
        "missing_identity_limitation": "Known groups do not prove complete patient/lesion identity metadata",
        "cohort_counts": {"M": {"images": 223, "references": 471}, "H": {"images": 47, "references": 104},
                          "T1": {"images": 20, "references": 49}},
        "exact_manifest_reference_identity_match_all_30_run_subsets": True,
        "same_anchor_and_selected_D0_metrics_all_references": True,
        "endpoint_checks": endpoint_checks, "inputs": inputs, "main_table": main_table,
        "registered_grid_snapshot": {"path": str(grid_path), "sha256": hashlib.sha256(grid_bytes).hexdigest(),
                                     "status": grid["status"], "active": grid.get("active"),
                                     "completed": grid.get("completed"), "exports_performed": grid.get("exports_performed")},
        "finalizer_flag_interpretation": "decision_inputs.extra_grid_has_not_run describes this export matrix only; actual registered grid training is already running and is not included here",
        "main_contrast": paired_m, "fixed_source_cohort_comparisons": comparisons,
        "incidence_RSI_vs_alpha2thirds": incidence,
        "applied_message_rms_provenance": applied_provenance,
        "native_source_threshold_rows": source_rows,
        "tool_stratification_epsilon_005": [row for row in sensitivity if row["stratum"] == "tool"
                                            and row["reference_selection"] == "all_references"
                                            and float(row["epsilon_D"]) == .005],
        "fixed_two_epsilon_005": [row for row in sensitivity if row["reference_selection"] == "fixed_two_seed17"
                                  and float(row["epsilon_D"]) == .005],
        "message_rms_definition": "Stored message_rms is raw module message before alpha scaling; equal across Fixed-Shrink alphas. Logit delta and hard-mask changes reflect applied alpha and recomputed tail",
        "interpretation": [
            "RSI lambda1 improves H_epsilon and worst tail against D0/Mean-Hinge/Abs-Hard lambda1 on M, but also retains less G_plus; this alone does not establish a distinct mechanism.",
            "Actual alpha2/3 closely matches RSI lambda1 on M Dice, G_plus, H_epsilon and worst tail; all four direct paired intervals cross zero. A coordinate-only dominance flag from last-digit differences is not substantive dominance.",
            "The M-selected alpha1/3 Fixed-Shrink retains lower Dice/G_plus but lower H_epsilon and a better worst tail; both it and RSI are eligible by the registered engineering Dice tolerance. Eligibility is not scientific superiority.",
            "H/T1 native references and M references on the same source image cohorts describe source and case-mix sensitivity; none selects a checkpoint, lambda or alpha.",
            "Soft/hard harm disagreement and persistent sign flips limit claims about per-reference hard Dice safety; no sample deletion or test release follows from these validation scores.",
            "Full registered symmetric weight grid and any subsequent predeclared seed29 comparison remain outside this interim result; no final continue/stop decision is made."
        ],
    }
    for record in inputs:
        assert file_record(Path(record["path"]))["sha256"] == record["sha256"], "Review input changed"
    csv_path = OUT / "source_fixed_cohort_comparisons.csv"
    fields = list(dict.fromkeys(key for row in comparisons for key in row))
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(comparisons)
    result["source_comparison_csv"] = file_record(csv_path)
    (OUT / "lambda1_scientific_review.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "scope": result["scope"],
                      "comparison_count": len(comparisons), "endpoint_max_error": max(r["maximum_absolute_metric_difference"] for r in endpoint_checks),
                      "source_comparison_csv": str(csv_path), "main_contrast": paired_m}, indent=2))


if __name__ == "__main__":
    main()
