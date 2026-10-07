"""Calibrate Phase-3 models on cal50 only; dev100 is inaccessible here."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation import mask_contours, polygon_area, build_prediction_context, compose_prediction_context
from phase3_access import log_access
from phase3_dp import choose_argmax, choose_closed_dp
from phase3_fast_raster import rasterize_exact
from train_local import CACHE, ROOT

ALPHAS = (0.0, .02, .05, .1, .2, .5, 1.0, 2.0)
LAMBDAS = (0.0, .002, .01, .05, .2, .5, 1.0)
FAMILIES = ("S64", "S96", "N0", "T8", "T32", "TG")


def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()


def dice(pred, gt):
    denominator = int(pred.sum()) + int(gt.sum())
    return 2 * int(np.count_nonzero(pred & gt)) / denominator if denominator else 1.0


def records_for_role(role):
    if role != "calibration": raise PermissionError("Calibration process may read cal50 only")
    ids = set((ROOT / "splits/calibration.txt").read_text().splitlines())
    records = json.loads((CACHE / "COMPLETE.json").read_text())["records"]
    selected = [record for record in records if record["image_id"] in ids]
    if len(selected) != len(ids): raise RuntimeError("cal50 role/cache mismatch")
    return selected


def source_scores(layer, family, seed, records):
    if layer == "archive":
        path = ROOT / "archive_scores" / f"{family}_seed{seed}.npy"
        val_records = [r for r in json.loads((CACHE / "COMPLETE.json").read_text())["records"] if r["split"] == "val"]
        index = {r["image_id"]: j for j, r in enumerate(val_records)}
        values = np.load(path, mmap_mode="r")
        return [np.asarray(values[index[r["image_id"]]], dtype=np.float32) for r in records], path
    path = ROOT / "models_phase3" / f"{family}_seed{seed}" / "candidate_scores.npy"
    values = np.load(path, mmap_mode="r")
    return [np.asarray(values[r["index"]], dtype=np.float32) for r in records], path


def main(layer):
    families = tuple(f for f in FAMILIES if layer != "archive" or f != "N0")
    records = records_for_role("calibration")
    log_access(f"calibration_{layer}", "calibration", len(records), True)
    prediction = np.load(CACHE / "prediction.npy", mmap_mode="r")
    gt = np.load(CACHE / "gt.npy", mmap_mode="r")
    candidate_root = CACHE / "candidates_R32"
    points = np.load(candidate_root / "points.npy", mmap_mode="r")
    valid = np.load(candidate_root / "valid.npy", mmap_mode="r")
    has = np.load(candidate_root / "has_contour.npy", mmap_mode="r")
    cases = []
    for record in records:
        index = record["index"]
        pred = np.asarray(prediction[index], bool)
        context = (build_prediction_context(pred, max(mask_contours(pred), key=lambda p: abs(polygon_area(p))))
                   if has[index] else None)
        cases.append(dict(image_id=record["image_id"], pred=pred, context=context,
                          gt=np.asarray(gt[index], bool), points=np.asarray(points[index], np.float32),
                          valid=np.asarray(valid[index], bool), has=bool(has[index])))
    target = ROOT / "results_phase3" / layer
    target.mkdir(parents=True, exist_ok=True)
    for family in families:
        for seed in (17, 23, 42):
            scores, path = source_scores(layer, family, seed, records)
            rows = []
            for mode in ("A", "D"):
                for alpha in ALPHAS:
                    for lam in ((0.0,) if mode == "A" else LAMBDAS):
                        dices = []; offsets = []
                        for case, score in zip(cases, scores):
                            if case["has"]:
                                indices = (choose_argmax(score, case["valid"], alpha) if mode == "A"
                                           else choose_closed_dp(score, case["valid"], alpha, lam))
                                polygon = case["points"][np.arange(len(indices)), indices]
                                mask = compose_prediction_context(rasterize_exact(polygon, case["pred"].shape), case["context"])
                                offsets.append(float(np.mean(np.abs(indices-32))))
                            else:
                                mask = case["pred"]; offsets.append(0.0)
                            dices.append(dice(mask, case["gt"]))
                        rows.append(dict(layer=layer, family=family, seed=seed, mode=mode, alpha=alpha,
                                         smooth_lambda=lam, cal_macro_dice=float(np.mean(dices)),
                                         mean_abs_offset=float(np.mean(offsets)), images=len(cases)))
                    print(f"calibrated {layer} {family} seed{seed} {mode} alpha={alpha}", flush=True)
            grid = target / f"{family}_seed{seed}_calibration_grid.csv"
            with grid.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            chosen = {}
            for mode in ("A", "D"):
                group = [row for row in rows if row["mode"] == mode]
                high = max(row["cal_macro_dice"] for row in group)
                ties = [row for row in group if high-row["cal_macro_dice"] <= 1e-8]
                chosen[mode] = min(ties, key=lambda row: (row["mean_abs_offset"], row["smooth_lambda"],
                                                           -row["alpha"], f"{family}_{seed}"))
            config = dict(layer=layer, family=family, seed=seed, chosen=chosen,
                          calibration_role_sha256=sha(ROOT / "splits/calibration.txt"),
                          scores_sha256=sha(path), calibration_grid_sha256=sha(grid),
                          full_candidate_count=65, dp_exact=True, dev100_opened=False,
                          historical_val150_checkpoint_selection=layer == "archive")
            file = target / f"{family}_seed{seed}_chosen_config.json"
            file.write_text(json.dumps(config, indent=2) + "\n")
            (target / f"{family}_seed{seed}_chosen_config.sha256").write_text(sha(file)+"\n")
            print(f"LOCKED {layer} {family} seed{seed}: {chosen}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--layer", choices=("archive", "controlled"), required=True)
    main(parser.parse_args().layer)
