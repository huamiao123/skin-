"""Prespecified image-paired Phase-3 C1/C2 comparisons after locked dev readout."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from train_local import ROOT

RNG_SEED = 20261007


def main():
    target = ROOT / "results_phase3/controlled"
    rows = list(csv.DictReader((target / "dev100_per_image.csv").open()))
    ids = (ROOT / "splits/development_readout.txt").read_text().splitlines()
    assert len(ids) == len(set(ids)) == 100
    keyed = {(row["method"], int(row["seed"]), row["image_id"]): row for row in rows}
    scores = {}
    for family in ("S64", "S96", "N0"):
        scores[family] = float(np.mean([
            json.loads((target / f"{family}_seed{seed}_chosen_config.json").read_text())["chosen"]["D"]["cal_macro_dice"]
            for seed in (17, 23, 42)]))
    best_simple = max(("S64", "S96", "N0"), key=lambda family: (scores[family], -("S64", "S96", "N0").index(family)))
    comparisons = {"C1": ("TG_A", "N0_A"), "C2": ("TG_D", f"{best_simple}_D")}
    rng = np.random.default_rng(RNG_SEED)
    outcome = dict(best_simple_family=best_simple, best_simple_cal50_family_scores=scores,
                   bootstrap_unit="image_id; seed outcomes paired within image",
                   bootstrap_replicates=10000, bootstrap_seed=RNG_SEED, comparisons={})
    for label, (new, baseline) in comparisons.items():
        metrics = {}
        for metric in ("dice", "iou", "bf1", "hd95", "assd", "correct_to_incorrect_rate"):
            def number(method, seed, image_id):
                value = keyed[(method, seed, image_id)].get(metric)
                return float(value) if value not in (None, "") else float("nan")
            paired = np.array([[number(new, seed, image_id)-number(baseline, seed, image_id)
                                for seed in (17, 23, 42)] for image_id in ids], dtype=np.float64)
            if not np.isfinite(paired).all():
                if metric != "correct_to_incorrect_rate": raise RuntimeError(f"Non-finite paired {metric}")
                paired = paired[np.isfinite(paired).all(axis=1)]
                if not len(paired): continue
            image_means = paired.mean(axis=1)
            sample = rng.integers(0, len(paired), size=(10000, len(paired)))
            replicates = image_means[sample].mean(axis=1)
            metrics[metric] = dict(images=len(paired), mean=float(image_means.mean()),
                                   seed_means=[float(value) for value in paired.mean(axis=0)],
                                   ci95=[float(value) for value in np.quantile(replicates, [.025, .975])],
                                   ci975_bonferroni=[float(value) for value in np.quantile(replicates, [.0125, .9875])],
                                   positive_seed_count=int((paired.mean(axis=0) > 0).sum()))
        dice_by_image = np.array([[float(keyed[(new, seed, image_id)]["dice"])-
                                   float(keyed[(baseline, seed, image_id)]["dice"])
                                   for seed in (17, 23, 42)] for image_id in ids]).mean(axis=1)
        order = np.argsort(dice_by_image)
        risk = dict(median_delta_dice=float(np.median(dice_by_image)),
                    improved_images=int((dice_by_image > 1e-12).sum()),
                    declined_images=int((dice_by_image < -1e-12).sum()),
                    unchanged_images=int((np.abs(dice_by_image) <= 1e-12).sum()),
                    decline_over_1pp=int((dice_by_image < -.01).sum()),
                    decline_over_3pp=int((dice_by_image < -.03).sum()),
                    worst_five=[dict(image_id=ids[i], delta_dice=float(dice_by_image[i])) for i in order[:5]],
                    best_five=[dict(image_id=ids[i], delta_dice=float(dice_by_image[i])) for i in order[-5:][::-1]],
                    mean_without_best_five=float(np.delete(dice_by_image, order[-5:]).mean()))
        guardrail = bool(metrics["bf1"]["mean"] >= -.002 and metrics["hd95"]["mean"] <= .25 and
                         ("correct_to_incorrect_rate" not in metrics or metrics["correct_to_incorrect_rate"]["mean"] <= .005))
        outcome["comparisons"][label] = dict(model=new, baseline=baseline, metrics=metrics,
                                               risk=risk, boundary_risk_guardrail=guardrail)
    c1 = outcome["comparisons"]["C1"]["metrics"]["dice"]
    c2 = outcome["comparisons"]["C2"]["metrics"]["dice"]
    # Mechanism and risk analyses are still required for GO. This is only the metric gate.
    outcome["dice_gate"] = bool(all(value["mean"] >= .001 and value["positive_seed_count"] >= 2
                                    for value in (c1, c2)))
    outcome["decision"] = "HOLD_PENDING_MECHANISM_AND_RISK_AUDIT"
    (target / "comparison.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__": main()
