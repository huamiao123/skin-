"""Independent manifest/role checks and a fixed 10-image result recomputation."""
from __future__ import annotations

import csv
import hashlib
import json

import numpy as np

from evaluation import (full_metrics, mask_contours, polygon_area,
                        build_prediction_context, compose_prediction_context)
from phase3_calibrate import ROOT, CACHE, FAMILIES, ALPHAS, LAMBDAS, sha
from phase3_dp import choose_argmax, choose_closed_dp
from phase3_fast_raster import rasterize_exact


def main():
    target = ROOT / "results_phase3/controlled"
    roles = json.loads((ROOT / "splits/phase3_manifest.json").read_text())["roles"]
    for role, info in roles.items():
        assert sha(ROOT / info["path"]) == info["sha256"], role
    dev_ids = (ROOT / "splits/development_readout.txt").read_text().splitlines()
    assert len(dev_ids) == len(set(dev_ids)) == 100
    model_manifest = json.loads((target / "model_manifest.json").read_text())
    assert len(model_manifest) == 18
    commits = {entry["source_git_commit"] for entry in model_manifest}
    fit_hashes = {entry["fit_ids_hash"] for entry in model_manifest}
    stop_hashes = {entry["stop_ids_hash"] for entry in model_manifest}
    assert len(commits) == len(fit_hashes) == len(stop_hashes) == 1
    assert fit_hashes == {roles["fit"]["sha256"]} and stop_hashes == {roles["stop"]["sha256"]}
    assert all(entry["official_test_images_opened"] == 0 and entry["cal_or_dev_gradient_images"] == 0
               and entry["score_dtype"] == "float32" for entry in model_manifest)
    for entry in model_manifest:
        folder = ROOT / "models_phase3" / f"{entry['family']}_seed{entry['seed']}"
        assert sha(folder / "best.pth") == entry["checkpoint_sha256"]
        assert sha(folder / "candidate_scores.npy") == entry["score_sha256"]
    for seed in (17, 23, 42):
        ref = [entry for entry in model_manifest if entry["seed"] == seed and entry["family"] in ("N0", "T8", "T32", "TG")]
        assert len({entry["upstream_checkpoint_sha256"] for entry in ref}) == 1
        assert len({entry["initial_parameter_sha256"] for entry in ref if entry["family"] != "N0"}) == 1
    configs = {}
    for family in FAMILIES:
        for seed in (17, 23, 42):
            stem = f"{family}_seed{seed}"
            path = target / f"{stem}_chosen_config.json"
            lock = target / f"{stem}_chosen_config.sha256"
            assert sha(path) == lock.read_text().strip()
            config = json.loads(path.read_text())
            grid = target / f"{stem}_calibration_grid.csv"
            assert sha(grid) == config["calibration_grid_sha256"]
            assert config["calibration_role_sha256"] == roles["calibration"]["sha256"]
            assert config["dp_exact"] and config["full_candidate_count"] == 65
            score = ROOT / "models_phase3" / stem / "candidate_scores.npy"
            assert sha(score) == config["scores_sha256"]
            with grid.open() as handle: grid_rows = list(csv.DictReader(handle))
            assert len(grid_rows) == len(ALPHAS) * (1+len(LAMBDAS)) == 64
            configs[(family, seed)] = config
    with (target / "dev100_per_image.csv").open() as handle:
        output_rows = list(csv.DictReader(handle))
    assert len(output_rows) == 4200
    assert {row["image_id"] for row in output_rows} == set(dev_ids)
    keyed = {(row["method"], int(row["seed"]), row["image_id"]): row for row in output_rows}
    assert len(keyed) == 4200
    sample = sorted(dev_ids, key=lambda image_id: hashlib.sha256(f"LC-P3-RECOMPUTE|{image_id}".encode()).hexdigest())[:10]
    records = json.loads((CACHE / "COMPLETE.json").read_text())["records"]
    by_id = {row["image_id"]: row for row in records}
    prediction = np.load(CACHE / "prediction.npy", mmap_mode="r")
    truth = np.load(CACHE / "gt.npy", mmap_mode="r")
    points = np.load(CACHE / "candidates_R32/points.npy", mmap_mode="r")
    valid = np.load(CACHE / "candidates_R32/valid.npy", mmap_mode="r")
    has = np.load(CACHE / "candidates_R32/has_contour.npy", mmap_mode="r")
    recomputed = 0; max_abs = 0.0
    for image_id in sample:
        i = by_id[image_id]["index"]
        pred = np.asarray(prediction[i], bool)
        gt = np.asarray(truth[i], bool)
        if has[i]:
            source = max(mask_contours(pred), key=lambda p: abs(polygon_area(p)))
            context = build_prediction_context(pred, source)
        for method, family, mode in (("N0_A", "N0", "A"), ("TG_A", "TG", "A"),
                                     ("TG_D", "TG", "D"), ("S96_D", "S96", "D")):
            for seed in (17, 23, 42):
                if has[i]:
                    scores = np.load(ROOT / "models_phase3" / f"{family}_seed{seed}" / "candidate_scores.npy", mmap_mode="r")
                    config = configs[(family, seed)]["chosen"][mode]
                    validity = np.asarray(valid[i], bool)
                    selected = (choose_argmax(scores[i], validity, config["alpha"]) if mode == "A"
                                else choose_closed_dp(scores[i], validity, config["alpha"], config["smooth_lambda"]))
                    polygon = np.asarray(points[i], np.float32)[np.arange(256), selected]
                    mask = compose_prediction_context(rasterize_exact(polygon, pred.shape), context)
                else: mask = pred
                current = full_metrics(mask, gt)
                stored = keyed[(method, seed, image_id)]
                for metric in ("dice", "iou", "bf1", "hd95", "assd"):
                    error = abs(current[metric]-float(stored[metric]))
                    max_abs = max(max_abs, error)
                    assert error <= 1e-12, (method, seed, image_id, metric)
                recomputed += 1
    access = [json.loads(line) for line in (ROOT / "models_phase3/access_log.jsonl").read_text().splitlines()]
    assert not any(row["role"] == "test" for row in access)
    result = dict(status="PASS", source_commits=sorted(commits), model_count=len(model_manifest),
                  locked_configs=len(configs), calibration_grid_rows_per_model=64,
                  dev100_per_image_rows=len(output_rows), image_ids=len(dev_ids),
                  recomputation_image_ids=sample, recomputed_method_seed_image_metrics=recomputed,
                  max_metric_abs_error=max_abs, test_access_log_entries=0,
                  official_test_predictions_or_GT_opened=False,
                  caveat="Operational access log and code assertions cannot prove absence of historical upstream validation exposure")
    (target / "audit_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
