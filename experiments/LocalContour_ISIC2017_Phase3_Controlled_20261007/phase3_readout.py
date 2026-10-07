"""One-shot dev100 readout from immutable cal50-chosen configurations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation import (full_metrics, repair_metrics, mask_contours, polygon_area,
                        build_prediction_context, compose_prediction_context)
from phase3_calibrate import FAMILIES, ROOT, CACHE, sha, source_scores
from phase3_access import log_access
from phase3_dp import choose_argmax, choose_closed_dp
from phase3_fast_raster import rasterize_exact


def main(layer):
    target = ROOT / "results_phase3" / layer
    families = tuple(f for f in FAMILIES if layer != "archive" or f != "N0")
    records = json.loads((CACHE / "COMPLETE.json").read_text())["records"]
    ids = set((ROOT / "splits/development_readout.txt").read_text().splitlines())
    selected = [record for record in records if record["image_id"] in ids]
    if len(selected) != 100: raise RuntimeError("dev100 role mismatch")
    # All configurations must exist and be hash-locked before a single dev GT is opened.
    configurations = {}
    for family in families:
        for seed in (17, 23, 42):
            stem = f"{family}_seed{seed}_chosen_config"
            path = target / (stem + ".json")
            expected = (target / (stem + ".sha256")).read_text().strip()
            if sha(path) != expected: raise RuntimeError(f"Unlocked configuration: {path}")
            config = json.loads(path.read_text())
            if config["calibration_role_sha256"] != sha(ROOT / "splits/calibration.txt"):
                raise RuntimeError("Calibration role changed")
            configurations[(family, seed)] = config
    log_access(f"locked_dev_readout_{layer}", "development_readout", len(selected), True)
    prediction = np.load(CACHE / "prediction.npy", mmap_mode="r")
    gt = np.load(CACHE / "gt.npy", mmap_mode="r")
    candidate = CACHE / "candidates_R32"
    points = np.load(candidate / "points.npy", mmap_mode="r")
    valid = np.load(candidate / "valid.npy", mmap_mode="r")
    distance = np.load(candidate / "gt_distance.npy", mmap_mode="r")
    has = np.load(candidate / "has_contour.npy", mmap_mode="r")
    contexts = {}
    for record in selected:
        index = record["index"]
        if has[index]:
            pred = np.asarray(prediction[index], bool)
            source = max(mask_contours(pred), key=lambda p: abs(polygon_area(p)))
            contexts[index] = build_prediction_context(pred, source)

    def rebuild(pred, point, indices, index):
        polygon = point[np.arange(len(indices)), indices]
        return compose_prediction_context(rasterize_exact(polygon, pred.shape), contexts[index])
    output = []
    for family in families:
        for seed in (17, 23, 42):
            config = configurations[(family, seed)]
            scores, path = source_scores(layer, family, seed, selected)
            if sha(path) != config["scores_sha256"]: raise RuntimeError("Score cache changed")
            for record, score in zip(selected, scores):
                index = record["index"]
                pred = np.asarray(prediction[index], bool)
                truth = np.asarray(gt[index], bool)
                point = np.asarray(points[index], np.float32)
                validity = np.asarray(valid[index], bool)
                dist = np.asarray(distance[index], np.float32)
                methods = {}
                if family == "S64":
                    methods["CNN"] = (pred, None)
                    if has[index]:
                        zero = np.full(256, 32, dtype=np.int64)
                        methods["G1"] = (rebuild(pred, point, zero, index), zero)
                    else: methods["G1"] = (pred, None)
                for mode in ("A", "D"):
                    choice = config["chosen"][mode]
                    if has[index]:
                        indices = (choose_argmax(score, validity, choice["alpha"]) if mode == "A"
                                   else choose_closed_dp(score, validity, choice["alpha"], choice["smooth_lambda"]))
                        mask = rebuild(pred, point, indices, index)
                    else: indices = None; mask = pred
                    methods[f"{family}_{mode}"] = (mask, indices)
                for name, (mask, indices) in methods.items():
                    row = dict(layer=layer, family=family, seed=seed, image_id=record["image_id"],
                               method=name, **full_metrics(mask, truth))
                    if indices is not None: row.update(repair_metrics(indices, validity, dist))
                    output.append(row)
            print(f"read out {layer} {family} seed{seed}", flush=True)
    path = target / "dev100_per_image.csv"
    fields = list(dict.fromkeys(key for row in output for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(output)
    summary = []
    for family in families:
        for seed in (17, 23, 42):
            for method in (["CNN", "G1"] if family == "S64" else []) + [f"{family}_A", f"{family}_D"]:
                rows = [row for row in output if row["family"] == family and row["seed"] == seed and row["method"] == method]
                summary.append(dict(layer=layer, family=family, seed=seed, method=method, images=len(rows),
                                    **{metric: float(np.mean([float(row[metric]) for row in rows]))
                                       for metric in ("dice", "iou", "bf1", "hd95", "assd")}))
    path = target / "dev100_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--layer", choices=("archive", "controlled"), required=True)
    main(parser.parse_args().layer)
