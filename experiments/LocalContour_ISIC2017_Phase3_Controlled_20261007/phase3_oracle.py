"""Same-input GT-distance diagnostic on dev100; not a deployable upper bound."""
from __future__ import annotations

import csv
import json

import numpy as np

from evaluation import (full_metrics, boundary_pixels, mask_contours, polygon_area,
                        build_prediction_context, compose_prediction_context)
from phase3_access import log_access
from phase3_calibrate import ROOT, CACHE
from phase3_dp import choose_closed_dp
from phase3_fast_raster import rasterize_exact

LAMBDA = .05


def main():
    ids = set((ROOT / "splits/development_readout.txt").read_text().splitlines())
    records = [row for row in json.loads((CACHE / "COMPLETE.json").read_text())["records"]
               if row["image_id"] in ids]
    if len(records) != 100: raise RuntimeError("dev100 role mismatch")
    log_access("same_input_oracle", "development_readout", len(records), True)
    prediction = np.load(CACHE / "prediction.npy", mmap_mode="r")
    truth = np.load(CACHE / "gt.npy", mmap_mode="r")
    candidate = CACHE / "candidates_R32"
    points = np.load(candidate / "points.npy", mmap_mode="r")
    source = np.load(candidate / "source_points.npy", mmap_mode="r")
    normals = np.load(candidate / "normals.npy", mmap_mode="r")
    valid = np.load(candidate / "valid.npy", mmap_mode="r")
    distance = np.load(candidate / "gt_distance.npy", mmap_mode="r")
    has = np.load(candidate / "has_contour.npy", mmap_mode="r")
    rows = []
    for record in records:
        i = record["index"]
        pred = np.asarray(prediction[i], bool)
        gt = np.asarray(truth[i], bool)
        gb = boundary_pixels(gt)
        if has[i]:
            validity = np.asarray(valid[i], bool)
            d = np.asarray(distance[i], np.float32)
            choice = choose_closed_dp(-d, validity, 0.0, LAMBDA)
            polygon = np.asarray(points[i], np.float32)[np.arange(256), choice]
            exterior = max(mask_contours(pred), key=lambda p: abs(polygon_area(p)))
            context = build_prediction_context(pred, exterior)
            selected = compose_prediction_context(rasterize_exact(polygon, pred.shape), context)
            best = np.where(validity, d, np.inf).min(axis=1)
            reachable_nodes = int((best <= 2).sum())
            fixable_nodes = int(((d[:, 32] > 2) & (best <= 2)).sum())
            initial_correct_nodes = int((d[:, 32] <= 2).sum())
            lo = np.asarray(source[i], np.float64)-32*np.asarray(normals[i], np.float64)
            hi = np.asarray(source[i], np.float64)+32*np.asarray(normals[i], np.float64)
            band = np.zeros(pred.shape, dtype=bool)
            for j in range(256):
                next_j = (j+1)%256
                quad = np.stack((lo[j], lo[next_j], hi[next_j], hi[j]))
                band |= rasterize_exact(quad, pred.shape)
            band_hit = int((band & gb).sum())
        else:
            selected = pred
            reachable_nodes = fixable_nodes = initial_correct_nodes = band_hit = 0
        metrics = full_metrics(selected, gt)
        rows.append(dict(image_id=record["image_id"], has_contour=bool(has[i]),
                         boundary_pixels=int(gb.sum()), gt_boundary_pixels_in_search_band=band_hit,
                         candidate_nodes=256 if has[i] else 0, reachable_nodes=reachable_nodes,
                         initially_correct_nodes=initial_correct_nodes, fixable_nodes=fixable_nodes,
                         **metrics))
    out = ROOT / "results_phase3/controlled"
    path = out / "oracle_GT_distance_per_image.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = dict(label="oracle_GT_distance_pixel_EDT", lambda_fixed=LAMBDA,
                   status="offline GT-assisted diagnostic, not deployable or a strict Dice upper bound",
                   images=len(rows), no_contour_images=sum(not row["has_contour"] for row in rows),
                   macro_dice=float(np.mean([row["dice"] for row in rows])),
                   macro_bf1=float(np.mean([row["bf1"] for row in rows])),
                   search_band_boundary_pixel_coverage=(sum(row["gt_boundary_pixels_in_search_band"] for row in rows)/
                                                        max(1,sum(row["boundary_pixels"] for row in rows))),
                   candidate_node_reach=(sum(row["reachable_nodes"] for row in rows)/
                                         max(1,sum(row["candidate_nodes"] for row in rows))),
                   initially_correct_nodes=sum(row["initially_correct_nodes"] for row in rows),
                   fixable_nodes=sum(row["fixable_nodes"] for row in rows),
                   denominator_note="Search-band denominator is GT boundary pixels; candidate-reach denominator is predicted nodes")
    (out / "oracle_GT_distance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
