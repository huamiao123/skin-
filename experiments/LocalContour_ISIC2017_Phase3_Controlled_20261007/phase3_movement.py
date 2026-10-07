"""Match _A movement budgets using cal50 predictions, then evaluate locked alphas."""
from __future__ import annotations

import csv
import json

import numpy as np

from evaluation import (full_metrics, repair_metrics, mask_contours, polygon_area,
                        build_prediction_context, compose_prediction_context)
from phase3_access import log_access
from phase3_calibrate import ROOT, CACHE, FAMILIES, source_scores, sha
from phase3_dp import choose_argmax
from phase3_fast_raster import rasterize_exact

TARGETS = (.10, .25, .50)


def node_move_threshold(score, validity):
    offset = np.abs(np.arange(65)-32).astype(np.float64)
    offset[32] = 1.0
    difference = (np.asarray(score, np.float64)-np.asarray(score, np.float64)[:, 32:33])/offset[None, :]
    difference = np.where(validity, difference, -np.inf)
    difference[:, 32] = -np.inf
    return difference.max(axis=1)


def records(role):
    allowed = ("calibration", "development_readout")
    if role not in allowed: raise PermissionError(role)
    ids = set((ROOT / "splits" / f"{role}.txt").read_text().splitlines())
    chosen = [row for row in json.loads((CACHE / "COMPLETE.json").read_text())["records"]
              if row["image_id"] in ids]
    if len(chosen) != len(ids): raise RuntimeError("Role/cache mismatch")
    return chosen


def main():
    target = ROOT / "results_phase3/controlled"
    cal = records("calibration")
    valid = np.load(CACHE / "candidates_R32/valid.npy", mmap_mode="r")
    has = np.load(CACHE / "candidates_R32/has_contour.npy", mmap_mode="r")
    configurations = []
    for family in FAMILIES:
        for seed in (17, 23, 42):
            scores, source_path = source_scores("controlled", family, seed, cal)
            cases = [(score, np.asarray(valid[row["index"]], bool))
                     for row, score in zip(cal, scores) if has[row["index"]]]
            thresholds = [node_move_threshold(score, validity) for score, validity in cases]
            def movement(alpha):
                return float(np.mean([np.mean(value > alpha) for value in thresholds])) if thresholds else 0.0
            maximum = movement(0.0)
            for budget in TARGETS:
                if maximum < budget:
                    alpha = 0.0; attainable = False
                else:
                    attainable = True; lo = 0.0; hi = .02
                    while movement(hi) > budget and hi < 1e6: hi *= 2
                    for _ in range(32):
                        middle = (lo+hi)/2
                        if movement(middle) > budget: lo = middle
                        else: hi = middle
                    alpha = hi
                configurations.append(dict(family=family, seed=seed, target_movement=budget,
                                           alpha=float(alpha), cal_movement=movement(alpha),
                                           cal_max_movement=maximum, attainable=attainable,
                                           scores_sha256=sha(source_path)))
    locked = target / "movement_config.json"
    locked.write_text(json.dumps(configurations, indent=2) + "\n")
    (target / "movement_config.sha256").write_text(sha(locked)+"\n")
    print("LOCKED movement budgets before dev GT access", flush=True)
    if sha(locked) != (target / "movement_config.sha256").read_text().strip():
        raise RuntimeError("Movement config changed")
    dev = records("development_readout")
    log_access("movement_budget_dev", "development_readout", len(dev), True)
    prediction = np.load(CACHE / "prediction.npy", mmap_mode="r")
    truth = np.load(CACHE / "gt.npy", mmap_mode="r")
    points = np.load(CACHE / "candidates_R32/points.npy", mmap_mode="r")
    distance = np.load(CACHE / "candidates_R32/gt_distance.npy", mmap_mode="r")
    contexts = {}
    for record in dev:
        i = record["index"]
        if has[i]:
            pred = np.asarray(prediction[i], bool)
            source = max(mask_contours(pred), key=lambda p: abs(polygon_area(p)))
            contexts[i] = build_prediction_context(pred, source)
    rows = []
    for config in configurations:
        family, seed = config["family"], config["seed"]
        scores, path = source_scores("controlled", family, seed, dev)
        if sha(path) != config["scores_sha256"]: raise RuntimeError("Score hash changed")
        for record, score in zip(dev, scores):
            i = record["index"]
            pred = np.asarray(prediction[i], bool)
            gt = np.asarray(truth[i], bool)
            if has[i]:
                validity = np.asarray(valid[i], bool)
                indices = choose_argmax(score, validity, config["alpha"])
                polygon = np.asarray(points[i], np.float32)[np.arange(256), indices]
                mask = compose_prediction_context(rasterize_exact(polygon, pred.shape), contexts[i])
                repair = repair_metrics(indices, validity, np.asarray(distance[i], np.float32))
            else: mask = pred; repair = {}
            rows.append(dict(family=family, seed=seed, image_id=record["image_id"],
                             target_movement=config["target_movement"], alpha=config["alpha"],
                             attainable=config["attainable"], **full_metrics(mask, gt), **repair))
        print(f"movement dev {family} seed{seed}", flush=True)
    output = target / "movement_budget.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
