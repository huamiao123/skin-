"""Build reviewer-facing Phase-3 tables and conservative protocol decision."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from train_local import ROOT

TARGET = ROOT / "results_phase3/controlled"
SEEDS = (17, 23, 42)
METHODS = ("CNN", "G1", "S64_A", "S64_D", "S96_A", "S96_D", "N0_A", "N0_D",
           "T8_A", "T8_D", "T32_A", "T32_D", "TG_A", "TG_D")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader(); writer.writerows(rows)


def main():
    summary = list(csv.DictReader((TARGET / "dev100_summary.csv").open()))
    per_image = list(csv.DictReader((TARGET / "dev100_per_image.csv").open()))
    comparison = json.loads((TARGET / "comparison.json").read_text())
    ambiguity = json.loads((TARGET / "ambiguity_diagnostic.json").read_text())
    oracle = json.loads((TARGET / "oracle_GT_distance_summary.json").read_text())
    resource = json.loads((TARGET / "resource_profile.json").read_text())
    runtime = json.loads((TARGET / "runtime_profile.json").read_text())
    audit = json.loads((TARGET / "audit_report.json").read_text())
    assert audit["status"] == "PASS" and resource["models"] == 18
    by = {(row["method"], int(row["seed"])): row for row in summary}
    table = []
    for method in METHODS:
        values = []
        for seed in SEEDS:
            row = by[(method, seed)]
            base = by[("CNN", seed)]
            g1 = by[("G1", seed)]
            item = dict(method=method, seed=seed, images=int(row["images"]),
                        dice=float(row["dice"]), iou=float(row["iou"]),
                        bf1=float(row["bf1"]), hd95=float(row["hd95"]), assd=float(row["assd"]),
                        delta_dice_vs_CNN=float(row["dice"])-float(base["dice"]),
                        delta_dice_vs_G1=float(row["dice"])-float(g1["dice"]))
            values.append(item); table.append(item)
        for aggregation in ("mean", "sample_std"):
            table.append(dict(method=method, seed=aggregation, images=100,
                              **{key: float(np.mean([row[key] for row in values])) if aggregation == "mean"
                                 else float(np.std([row[key] for row in values], ddof=1))
                                 for key in ("dice", "iou", "bf1", "hd95", "assd",
                                             "delta_dice_vs_CNN", "delta_dice_vs_G1")}))
    write_csv(TARGET / "main_table.csv", table)
    lookup = {(row["method"], int(row["seed"]), row["image_id"]): row for row in per_image}
    ids = (ROOT / "splits/development_readout.txt").read_text().splitlines()
    comparisons = []
    for label, record in comparison["comparisons"].items():
        for metric, stats in record["metrics"].items():
            comparisons.append(dict(label=label, model=record["model"], baseline=record["baseline"],
                                    metric=metric, mean_difference=stats["mean"],
                                    seed17_difference=stats["seed_means"][0],
                                    seed23_difference=stats["seed_means"][1],
                                    seed42_difference=stats["seed_means"][2],
                                    images=stats["images"], unit="paired_image_id",
                                    ci95_low=stats["ci95"][0], ci95_high=stats["ci95"][1],
                                    ci975_bonferroni_low=stats["ci975_bonferroni"][0],
                                    ci975_bonferroni_high=stats["ci975_bonferroni"][1],
                                    interval_type="preregistered_primary_10000_image_bootstrap"))
    auxiliary = (("A_TG_vs_S64", "TG_A", "S64_A"), ("A_TG_vs_S96", "TG_A", "S96_A"),
                 ("A_TG_vs_T8", "TG_A", "T8_A"), ("A_TG_vs_T32", "TG_A", "T32_A"),
                 ("D_TG_vs_N0", "TG_D", "N0_D"), ("D_TG_vs_T32", "TG_D", "T32_D"),
                 ("A_N0_vs_S64", "N0_A", "S64_A"))
    rng = np.random.default_rng(20261007)
    for label, method, baseline in auxiliary:
        paired = np.array([[float(lookup[(method, seed, image_id)]["dice"])-
                            float(lookup[(baseline, seed, image_id)]["dice"])
                            for seed in SEEDS] for image_id in ids])
        values = paired.mean(axis=1)
        sample = rng.integers(0, 100, size=(10000, 100))
        ci = np.quantile(values[sample].mean(axis=1), (.025, .975))
        comparisons.append(dict(label=label, model=method, baseline=baseline, metric="dice",
                                mean_difference=float(values.mean()),
                                seed17_difference=float(paired[:, 0].mean()),
                                seed23_difference=float(paired[:, 1].mean()),
                                seed42_difference=float(paired[:, 2].mean()),
                                images=100, unit="paired_image_id", ci95_low=float(ci[0]),
                                ci95_high=float(ci[1]), ci975_bonferroni_low=None,
                                ci975_bonferroni_high=None,
                                interval_type="exploratory_10000_image_bootstrap"))
    write_csv(TARGET / "paired_comparisons.csv", comparisons)
    ambiguity_rows = []
    for row in ambiguity["dev100"]:
        ambiguity_rows.append(dict(seed=row["seed"], group="separated_peak_low_margin_fixable_S64_failed",
                                   threshold=ambiguity["thresholds"][str(row["seed"])]["low_margin_p25"],
                                   threshold_source="fit1800_S64_score_p25", images=row["images"],
                                   nodes=row["low_margin_fixable_s64_failed"],
                                   TG_corrected_nodes=row["low_margin_fixable_s64_failed_tg_corrected"],
                                   TG_correction_rate=row["focal_correction_rate"],
                                   total_contour_nodes=row["nodes"],
                                   candidate_reachable_nodes=row["candidate_reachable"]))
    write_csv(TARGET / "ambiguity_groups.csv", ambiguity_rows)
    c1 = comparison["comparisons"]["C1"]
    c2 = comparison["comparisons"]["C2"]
    boundary = sum(json.loads(path.read_text())["chosen"]["D"]["smooth_lambda"] == 1.0
                   for path in TARGET.glob("*_chosen_config.json"))
    decision = dict(status="HOLD", go=False, independent_test_opened=False,
                    c1_delta_dice=c1["metrics"]["dice"], c2_delta_dice=c2["metrics"]["dice"],
                    c1_correct_to_incorrect_increase=c1["metrics"]["correct_to_incorrect_rate"]["mean"],
                    best_simple_family=comparison["best_simple_family"],
                    exact_full65_dp=True, models=18, calibration_configs=18,
                    dp_lambda_upper_boundary_configs=boundary,
                    triggers=["C1 mean +0.000396 below preregistered +0.001 Dice gate",
                              "C2 mean -0.001020 versus cal50-selected S96_D",
                              "C1 correct-to-incorrect rate +0.0351 exceeds +0.005 guardrail",
                              "primary 95% and Bonferroni intervals include zero and useful positive gains",
                              f"{boundary}/18 DP lambdas selected at upper grid boundary"],
                    interpretation="No GO for independent test; uncertainty is too wide for a precise REDIRECT/equivalence claim.",
                    next_step="Audit conservative selection and harmful moves; if continuing, preregister at most one targeted confirmation before test.",
                    evidence=["comparison.json", "main_table.csv", "paired_comparisons.csv", "movement_budget.csv",
                              "ambiguity_groups.csv", "oracle_GT_distance_summary.json", "audit_report.json"])
    (TARGET / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    def mean(method, metric="dice"):
        return float(next(row[metric] for row in table if row["method"] == method and row["seed"] == "mean"))
    report = f"""# LocalContour Phase 3 controlled development report

**Decision: HOLD.** The preregistered cross-node value claim did not pass the development gate. This is a controlled development result on 100 ISIC2017 validation images, not an independent test finding. Official test600 remains unopened.

The new train2000 was split by SHA256 image ID into fit1800 and stop200; public metadata has no patient/lesion group ID, so patient independence is unknown. Frozen CNN and boundary head were inherited from earlier work with possible historical validation exposure. New S64/S96/N0/T8/T32/TG gradients used fit only, checkpoints used stop only, and all postprocessing parameters used cal50 only. The archived Fixed checkpoints were separately re-evaluated and are not mixed into this table.

| dev100 method | Dice, 3-seed mean | BF1, 3-seed mean |
|---|---:|---:|
| CNN | {mean('CNN'):.6f} | {mean('CNN','bf1'):.6f} |
| G1 zero reconstruction | {mean('G1'):.6f} | {mean('G1','bf1'):.6f} |
| S64_A | {mean('S64_A'):.6f} | {mean('S64_A','bf1'):.6f} |
| S64_D | {mean('S64_D'):.6f} | {mean('S64_D','bf1'):.6f} |
| S96_A | {mean('S96_A'):.6f} | {mean('S96_A','bf1'):.6f} |
| S96_D | {mean('S96_D'):.6f} | {mean('S96_D','bf1'):.6f} |
| N0_A | {mean('N0_A'):.6f} | {mean('N0_A','bf1'):.6f} |
| N0_D | {mean('N0_D'):.6f} | {mean('N0_D','bf1'):.6f} |
| T32_A | {mean('T32_A'):.6f} | {mean('T32_A','bf1'):.6f} |
| TG_A | {mean('TG_A'):.6f} | {mean('TG_A','bf1'):.6f} |
| TG_D | {mean('TG_D'):.6f} | {mean('TG_D','bf1'):.6f} |

**C1:** TG_A−N0_A = {c1['metrics']['dice']['mean']:+.6f} Dice, paired image-bootstrap 95% CI {c1['metrics']['dice']['ci95']}; seed differences {c1['metrics']['dice']['seed_means']}. The +0.001 engineering gate was missed. TG_A increased correct-to-incorrect rate by {c1['metrics']['correct_to_incorrect_rate']['mean']:+.4f} relative to N0_A on 92 images with a defined denominator, beyond the +0.005 guardrail, even though mean BF1 improved.

**C2:** TG_D−S96_D = {c2['metrics']['dice']['mean']:+.6f} Dice, 95% CI {c2['metrics']['dice']['ci95']}; seed differences {c2['metrics']['dice']['seed_means']}. S96_D was selected as BestSimple only from the three family means on cal50. All `_D` methods used exact full-65 cyclic DP over the same 8 alpha × 7 lambda grid. {boundary}/18 selected lambda=1.0, the grid's upper boundary.

At cal50-matched 10% movement, dev100 mean Dice was N0 {np.mean([float(r['dice']) for r in csv.DictReader((TARGET/'movement_budget.csv').open()) if r['family']=='N0' and float(r['target_movement'])==.1]):.6f} and TG {np.mean([float(r['dice']) for r in csv.DictReader((TARGET/'movement_budget.csv').open()) if r['family']=='TG' and float(r['target_movement'])==.1]):.6f}. At the 25% target, actual dev movement differs materially between methods; the comparison is near-budget only. The 50% target was unreachable for many model/seed combinations and was not forced with negative alpha. Full exploratory alpha curves are in `alpha_curve.csv`; alpha=0 causes substantial harmful overmovement in several methods.

The fit-defined separated-peak low-margin, fixable, S64-failed group contained {', '.join(str(r['low_margin_fixable_s64_failed']) for r in ambiguity['dev100'])} nodes across the three seed-specific analyses. TG corrected only {', '.join(str(r['low_margin_fixable_s64_failed_tg_corrected']) for r in ambiguity['dev100'])} of them (about 2.5–3.5%). The same-input GT-distance Oracle reached Dice {oracle['macro_dice']:.6f}; {oracle['candidate_node_reach']:.1%} of contour nodes had a <=2px candidate and {oracle['search_band_boundary_pixel_coverage']:.1%} of GT boundary pixels were in the search-band quadrilateral union. This Oracle uses GT and is neither deployable nor a strict Dice upper bound.

On ten fixed cal50 images, seed17 full-system pure-compute mean latency was {runtime['measurements']['N0_A_total']['mean_ms']:.1f} ms for N0_A, {runtime['measurements']['TG_A_total']['mean_ms']:.1f} ms for TG_A, {runtime['measurements']['TG_D_total']['mean_ms']:.1f} ms for TG_D, and {runtime['measurements']['S96_D_total']['mean_ms']:.1f} ms for S96_D; startup and disk I/O are excluded. Most time was spent constructing candidates and preserving the original prediction context, while the N0/TG refiner itself was below 1.1 ms on average. This ten-image seed17 profile is not a population latency guarantee; see `runtime_profile.json` for p50/p95 and per-image values.

The independent audit passed: 18 checkpoints and score caches had matching hashes, all 18 cal50 configurations were locked, 4,200 dev rows covered the same 100 IDs, and 120 fixed image/method/seed recomputations matched every stored Dice/IoU/BF1/HD95/ASSD exactly. Operational logs have no test access. New-model training did not hit the 40-epoch budget edge. These checks do not remove historical upstream validation exposure or image-level-only split limitations.

The main intervals remain wide enough to include modest positive effects, so REDIRECT/equivalence is not established. The evidence supports pausing the global-contour necessity claim and, if pursued, pre-registering one focused correction/harm experiment before any independent test. No P5 final lock was issued.
"""
    (TARGET / "development_report.md").write_text(report)
    (TARGET / "audit_report.md").write_text(
        "# Phase 3 audit\n\nPASS: 18 models, 18 hash-locked cal50 configs, 4,200 dev100 rows, "
        "120 exact metric recomputations, zero logged test accesses. See `audit_report.json` for IDs and hashes.\n\n"
        "Limit: the frozen upstream CNN/boundary head has possible historical validation exposure; "
        "public metadata does not establish patient-level independence.\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__": main()
