"""Phase-2 fixed-radius candidate ablation on the archived 2017 development set."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from project_paths import PHASE1

import numpy as np
from scipy import signal

SOURCE = PHASE1
sys.path.insert(0, str(SOURCE))
from contour.geometry import (CandidateField, build_prediction_context,
                              closed_dp, compose_prediction_context,
                              distance_to_boundary, extract_geometry,
                              generate_candidates, normal_gt_intersections,
                              rasterize, segmentation_metrics)

ROOT = Path(__file__).resolve().parent
CASES = SOURCE / 'results/cases'
TOLERANCE = 2.0
RADIUS = 16.0
NODES = 256
DP_LAMBDA = 0.05  # fixed historical calibration; no new selection on the 150 images


def candidate_field(score_map, geometry, nonzero_limit: int | None):
    """Zero gets its own slot; peaks use independent nonzero quota; None=dense."""
    old = generate_candidates(score_map, geometry)
    samples, offsets = old.sample_scores, old.sample_offsets
    sampled_valid = old.sample_valid
    rows = len(geometry.points)
    width = 1 + (2 * int(RADIUS) if nonzero_limit is None else nonzero_limit)
    result_offsets = np.zeros((rows, width), np.float64)
    result_scores = np.full((rows, width), -np.inf, np.float64)
    result_valid = np.zeros((rows, width), bool)
    result_scores[:, 0] = samples[:, int(RADIUS)]
    result_valid[:, 0] = True
    for row in range(rows):
        profile = samples[row]
        finite = np.flatnonzero(sampled_valid[row])
        if nonzero_limit is None:
            selected = [int(i) for i in finite if offsets[i] != 0]
        else:
            peaks = list(signal.find_peaks(profile, distance=2)[0])
            if len(finite):
                first, last = int(finite[0]), int(finite[-1])
                if first < last and profile[first] > profile[first + 1]:
                    peaks.append(first)
                if last > first and profile[last] > profile[last - 1]:
                    peaks.append(last)
            ordered = sorted(set(peaks), key=lambda i: (-profile[i], abs(offsets[i]), offsets[i]))
            selected = []
            for index in ordered:
                if offsets[index] == 0 or not np.isfinite(profile[index]):
                    continue
                if any(abs(index - old_index) < 2 for old_index in selected):
                    continue
                selected.append(index)
                if len(selected) >= nonzero_limit:
                    break
        for cursor, index in enumerate(selected, 1):
            result_offsets[row, cursor] = offsets[index]
            result_scores[row, cursor] = profile[index]
            result_valid[row, cursor] = True
    points = geometry.points[:, None, :] + result_offsets[..., None] * geometry.normals[:, None, :]
    return CandidateField(points, result_offsets, result_scores, result_valid,
                          old.sample_offsets, old.sample_scores, old.sample_valid)


def evaluate_one(case_path: Path):
    with np.load(case_path) as z:
        pred, gt, q = z['masks_G0'], z['gt_mask'], z['boundary_probability']
        role = str(z['role'])
        original_g4 = segmentation_metrics(z['masks_G4'], gt)['dice']
    g0 = segmentation_metrics(pred, gt)
    geometry = extract_geometry(pred, NODES, RADIUS)
    base = {'image_id': case_path.stem, 'role': role, 'g0_dice': g0['dice'],
            'g0_bf1': g0['bf1'], 'historical_g4_dice': original_g4}
    if geometry is None:
        return [{**base, 'config': c, 'effective_nodes': 0,
                 'covered_nodes': 0, 'coverage': None, 'oracle_dice': g0['dice'],
                 'oracle_bf1': g0['bf1'], 'status': 'empty_prediction'}
                for c in ('old_3', 'fixed_3', 'fixed_4', 'fixed_8', 'fixed_16', 'dense')], []
    intersections = normal_gt_intersections(geometry, gt)
    valid = intersections['valid']
    targets = intersections['single_offsets']
    context = build_prediction_context(pred, geometry.source_polygon)
    output, node_records = [], []
    configs = [('old_3', None), ('fixed_3', 3), ('fixed_4', 4),
               ('fixed_8', 8), ('fixed_16', 16), ('dense', None)]
    for name, nonzero in configs:
        field = generate_candidates(q, geometry) if name == 'old_3' else candidate_field(q, geometry, nonzero)
        distances = np.where(field.valid, np.abs(field.offsets - targets[:, None]), np.inf)
        nearest = distances.min(axis=1)
        covered = valid & (nearest <= TOLERANCE)
        # The GT-assisted choice uses the same absolute boundary-distance unary
        # and historical closed-DP smoothness for every candidate configuration.
        gt_cost = np.where(field.valid, distance_to_boundary(field.points, gt), np.inf)
        selected, _ = closed_dp(gt_cost, field.offsets, DP_LAMBDA)
        polygon = field.points[np.arange(len(selected)), selected]
        oracle_mask = compose_prediction_context(rasterize(polygon, pred.shape), context)
        oracle = segmentation_metrics(oracle_mask, gt)
        row = {**base, 'config': name, 'effective_nodes': int(valid.sum()),
               'covered_nodes': int(covered.sum()), 'coverage': float(covered.sum()/valid.sum()) if valid.any() else None,
               'oracle_dice': oracle['dice'], 'oracle_bf1': oracle['bf1'],
               'oracle_minus_g0_dice': oracle['dice']-g0['dice'],
               'mean_nearest_gt_distance': float(nearest[valid].mean()) if valid.any() else None,
               'median_nearest_gt_distance': float(np.median(nearest[valid])) if valid.any() else None,
               'p90_nearest_gt_distance': float(np.percentile(nearest[valid],90)) if valid.any() else None,
               'p95_nearest_gt_distance': float(np.percentile(nearest[valid],95)) if valid.any() else None,
               'candidate_count_mean': float(field.valid.sum(axis=1).mean()), 'status': 'ok'}
        for k in (1,2,4,8,16):
            subset = field.valid[:, :min(k, field.valid.shape[1])]
            row[f'recall_at_{k}'] = float((valid & np.any((distances[:, :subset.shape[1]] <= TOLERANCE) & subset, axis=1)).sum()/valid.sum()) if valid.any() else None
        output.append(row)
        for i in np.flatnonzero(valid):
            node_records.append((name, float(nearest[i]), bool(covered[i])))
    return output, node_records


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    paths = sorted(CASES.glob('*.npz'))
    if args.limit: paths = paths[:args.limit]
    out = ROOT/'results/candidates'
    out.mkdir(parents=True, exist_ok=True)
    rows, node_records = [], []
    for n, path in enumerate(paths, 1):
        a,b = evaluate_one(path)
        rows.extend(a); node_records.extend(b)
        if n % 10 == 0: print(f'{n}/{len(paths)}', flush=True)
    write_csv(out/'per_image.csv', rows)
    summary = []
    for name in ('old_3','fixed_3','fixed_4','fixed_8','fixed_16','dense'):
        group=[r for r in rows if r['config']==name]
        distances=np.array([d for c,d,_ in node_records if c==name])
        coverage=sum(r['covered_nodes'] for r in group)/max(1,sum(r['effective_nodes'] for r in group))
        val=lambda key: float(np.mean([r[key] for r in group if r.get(key) is not None]))
        denominator=sum(r['effective_nodes'] for r in group)
        summary.append({'config':name,'n_images':len(group),'effective_nodes':denominator,
                        'candidate_coverage_pooled':coverage,
                        'candidate_coverage_mean_image':val('coverage'),
                        'candidate_coverage_median_image':float(np.median([r['coverage'] for r in group if r['coverage'] is not None])),
                        'oracle_dice_mean':val('oracle_dice'),'oracle_bf1_mean':val('oracle_bf1'),
                        'oracle_minus_g0_dice_mean':float(np.mean([r['oracle_dice']-r['g0_dice'] for r in group])),
                        'mean_nearest_gt_distance':float(np.mean(distances)) if len(distances) else None,
                        'median_nearest_gt_distance':float(np.median(distances)) if len(distances) else None,
                        'p90_nearest_gt_distance':float(np.percentile(distances,90)) if len(distances) else None,
                        'p95_nearest_gt_distance':float(np.percentile(distances,95)) if len(distances) else None,
                        **{f'recall_at_{k}_pooled':sum(r[f'recall_at_{k}']*r['effective_nodes'] for r in group if r[f'recall_at_{k}'] is not None)/denominator for k in (1,2,4,8,16)}})
    write_csv(out/'summary.csv',summary)
    manifest={'scope':'2017 official val150 previously used for development; no independent test claim',
              'source_probe':str(SOURCE),
              'source_cases':len(paths),'nodes':NODES,'radius':RADIUS,'gt_tolerance':TOLERANCE,
              'dp_lambda':DP_LAMBDA,'cnn_frozen':True,'candidate_rule':'zero plus K nonzero peaks; dense all visible integer offsets',
              'stage_a_decision':'candidate_coverage_80_percent_gate',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
