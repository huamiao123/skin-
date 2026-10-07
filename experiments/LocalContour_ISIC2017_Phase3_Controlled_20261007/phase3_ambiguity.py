"""Fit-defined separated-peak ambiguity and offline dev100 correction diagnostics."""
from __future__ import annotations

import json

import numpy as np

from phase3_access import log_access
from phase3_calibrate import ROOT, CACHE, sha
from phase3_dp import choose_argmax


def normalized(scores, valid):
    valid = np.asarray(valid, bool)
    scores = np.where(valid, np.asarray(scores, np.float32), 0.0)
    count = np.maximum(valid.sum(axis=-1, keepdims=True), 1)
    mean = scores.sum(axis=-1, keepdims=True)/count
    centred = np.where(valid, scores-mean, 0.0)
    scale = np.maximum(np.sqrt((centred*centred).sum(axis=-1, keepdims=True)/count)+1, 1)
    return np.clip(centred/scale, -5, 5)


def separated_peak_margin(scores, valid):
    value = np.where(valid, normalized(scores, valid), -np.inf)
    left = np.concatenate((value[:, :1], value[:, :-1]), axis=1)
    right = np.concatenate((value[:, 1:], value[:, -1:]), axis=1)
    peaks = valid & (value >= left) & (value >= right) & ((value > left) | (value > right))
    first_score = np.where(peaks, value, -np.inf)
    first = np.argmax(first_score, axis=1)
    far = np.abs(np.arange(65)[None, :] - first[:, None]) >= 4
    second_score = np.where(peaks & far, value, -np.inf).max(axis=1)
    with np.errstate(invalid="ignore"):
        margin = first_score[np.arange(len(value)), first] - second_score
    return np.isfinite(second_score), margin


def main():
    target = ROOT / "results_phase3/controlled"
    records = json.loads((CACHE / "COMPLETE.json").read_text())["records"]
    by_id = {row["image_id"]: row for row in records}
    fit = (ROOT / "splits/fit.txt").read_text().splitlines()
    dev = (ROOT / "splits/development_readout.txt").read_text().splitlines()
    valid = np.load(CACHE / "candidates_R32/valid.npy", mmap_mode="r")
    has = np.load(CACHE / "candidates_R32/has_contour.npy", mmap_mode="r")
    thresholds = {}
    for seed in (17, 23, 42):
        source = ROOT / "models_phase3" / f"S64_seed{seed}" / "candidate_scores.npy"
        scores = np.load(source, mmap_mode="r")
        margins = []
        for image_id in fit:
            index = by_id[image_id]["index"]
            if not has[index]: continue
            two, margin = separated_peak_margin(scores[index], np.asarray(valid[index], bool))
            margins.extend(margin[two].tolist())
        if not margins: raise RuntimeError("No separated peaks in fit")
        thresholds[str(seed)] = dict(low_margin_p25=float(np.quantile(margins, .25)),
                                     fit_peak_nodes=len(margins), fit_ids_sha256=sha(ROOT / "splits/fit.txt"),
                                     s64_scores_sha256=sha(source))
    path = target / "ambiguity_thresholds.json"
    path.write_text(json.dumps(thresholds, indent=2) + "\n")
    (target / "ambiguity_thresholds.sha256").write_text(sha(path)+"\n")
    if sha(path) != (target / "ambiguity_thresholds.sha256").read_text().strip():
        raise RuntimeError("Ambiguity threshold lock mismatch")
    log_access("ambiguity_offline_diagnostic", "development_readout", len(dev), True)
    distance = np.load(CACHE / "candidates_R32/gt_distance.npy", mmap_mode="r")
    rows = []
    for seed in (17, 23, 42):
        s64 = np.load(ROOT / "models_phase3" / f"S64_seed{seed}" / "candidate_scores.npy", mmap_mode="r")
        tg = np.load(ROOT / "models_phase3" / f"TG_seed{seed}" / "candidate_scores.npy", mmap_mode="r")
        a_s64 = json.loads((target / f"S64_seed{seed}_chosen_config.json").read_text())["chosen"]["A"]["alpha"]
        a_tg = json.loads((target / f"TG_seed{seed}_chosen_config.json").read_text())["chosen"]["A"]["alpha"]
        threshold = thresholds[str(seed)]["low_margin_p25"]
        counts = dict(images=len(dev), nodes=0, separated_peaks=0, low_margin=0,
                      candidate_reachable=0, fixable=0, s64_failed=0, tg_corrected=0,
                      low_margin_fixable_s64_failed=0, low_margin_fixable_s64_failed_tg_corrected=0)
        for image_id in dev:
            index = by_id[image_id]["index"]
            if not has[index]: continue
            validity = np.asarray(valid[index], bool)
            distances = np.asarray(distance[index], np.float32)
            two, margin = separated_peak_margin(s64[index], validity)
            low = two & (margin <= threshold)
            d0 = distances[:, 32]
            best = np.where(validity, distances, np.inf).min(axis=1)
            reachable = best <= 2
            fixable = (d0 > 2) & reachable
            s64_index = choose_argmax(s64[index], validity, a_s64)
            tg_index = choose_argmax(tg[index], validity, a_tg)
            row = np.arange(256)
            local_fail = distances[row, s64_index] > 2
            global_correct = distances[row, tg_index] <= 2
            focal = low & fixable & local_fail
            counts["nodes"] += 256
            counts["separated_peaks"] += int(two.sum())
            counts["low_margin"] += int(low.sum())
            counts["candidate_reachable"] += int(reachable.sum())
            counts["fixable"] += int(fixable.sum())
            counts["s64_failed"] += int((fixable & local_fail).sum())
            counts["tg_corrected"] += int((fixable & local_fail & global_correct).sum())
            counts["low_margin_fixable_s64_failed"] += int(focal.sum())
            counts["low_margin_fixable_s64_failed_tg_corrected"] += int((focal & global_correct).sum())
        counts["seed"] = seed
        counts["focal_correction_rate"] = (counts["low_margin_fixable_s64_failed_tg_corrected"] /
                                            counts["low_margin_fixable_s64_failed"]
                                            if counts["low_margin_fixable_s64_failed"] else None)
        rows.append(counts)
    output = dict(definition="S64 normalized 1D local peaks separated by >=4 offsets; fit p25 margin",
                  thresholds=thresholds, dev100=rows,
                  caveat="Nearest 2D GT boundary is a proxy, not semantic contour correspondence")
    (target / "ambiguity_diagnostic.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
