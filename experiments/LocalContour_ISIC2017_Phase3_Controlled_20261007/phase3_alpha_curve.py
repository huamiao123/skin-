"""Exploratory dev100 alpha sensitivity after all main configurations are locked."""
from __future__ import annotations

import csv
import json

import numpy as np

from evaluation import (full_metrics, repair_metrics, mask_contours, polygon_area,
                        build_prediction_context, compose_prediction_context)
from phase3_access import log_access
from phase3_calibrate import ROOT, CACHE, FAMILIES, ALPHAS, source_scores, sha
from phase3_dp import choose_argmax
from phase3_fast_raster import rasterize_exact


def main():
    target = ROOT / "results_phase3/controlled"
    # No alpha curve is allowed to create or change main chosen_config files.
    for family in FAMILIES:
        for seed in (17, 23, 42):
            stem = f"{family}_seed{seed}_chosen_config"
            config = target / f"{stem}.json"
            lock = target / f"{stem}.sha256"
            if not lock.is_file() or sha(config) != lock.read_text().strip():
                raise RuntimeError("Main configurations must be hash-locked before sensitivity readout")
    ids = set((ROOT / "splits/development_readout.txt").read_text().splitlines())
    records = [row for row in json.loads((CACHE / "COMPLETE.json").read_text())["records"]
               if row["image_id"] in ids]
    if len(records) != 100: raise RuntimeError("dev100 role mismatch")
    log_access("alpha_sensitivity", "development_readout", len(records), True)
    pred = np.load(CACHE / "prediction.npy", mmap_mode="r")
    gt = np.load(CACHE / "gt.npy", mmap_mode="r")
    root = CACHE / "candidates_R32"
    points = np.load(root / "points.npy", mmap_mode="r")
    valid = np.load(root / "valid.npy", mmap_mode="r")
    distance = np.load(root / "gt_distance.npy", mmap_mode="r")
    has = np.load(root / "has_contour.npy", mmap_mode="r")
    contexts = {}
    for row in records:
        i = row["index"]
        if has[i]:
            value = np.asarray(pred[i], bool)
            source = max(mask_contours(value), key=lambda p: abs(polygon_area(p)))
            contexts[i] = build_prediction_context(value, source)
    output = []
    for family in FAMILIES:
        for seed in (17, 23, 42):
            scores, _ = source_scores("controlled", family, seed, records)
            for alpha in ALPHAS:
                rows = []
                for record, score in zip(records, scores):
                    i = record["index"]
                    value = np.asarray(pred[i], bool)
                    if has[i]:
                        validity = np.asarray(valid[i], bool)
                        indices = choose_argmax(score, validity, alpha)
                        polygon = np.asarray(points[i], np.float32)[np.arange(256), indices]
                        mask = compose_prediction_context(rasterize_exact(polygon, value.shape), contexts[i])
                        repair = repair_metrics(indices, validity, np.asarray(distance[i], np.float32))
                    else: mask = value; repair = {}
                    rows.append(dict(**full_metrics(mask, np.asarray(gt[i], bool)), **repair))
                result = dict(family=family, seed=seed, alpha=alpha, images=len(rows),
                              scope="exploratory_dev100_sensitivity_not_selection")
                for metric in ("dice", "bf1", "moved_fraction", "mean_abs_offset", "correct_to_incorrect_rate"):
                    values = [row[metric] for row in rows if row.get(metric) is not None]
                    result[metric] = float(np.mean(values)) if values else None
                    result[metric+"_images"] = len(values)
                output.append(result)
            print(f"alpha curve {family} seed{seed}", flush=True)
    path = target / "alpha_curve.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys()); writer.writeheader(); writer.writerows(output)


if __name__ == "__main__": main()
