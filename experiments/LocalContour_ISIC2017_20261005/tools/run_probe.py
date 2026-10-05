"""Calibrate the fixed contour interface, lock DP, then diagnose locked100."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contour.data import load_image, load_mask, load_rgb, read_val_split
from contour.geometry import closed_dp, coverage_diagnostics, evaluate_selectors, extract_geometry, generate_candidates, rasterize, segmentation_metrics
from contour.models import FrozenMSGUNet, PUBLIC_WEIGHT_SHA256, file_sha256, segmentation_mask
from contour.reporting import COVERAGE_FIELDS, summarize
from tools.train_boundary import atomic_json, canonical_sha, load_selected_head


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(k for r in rows for k in r))
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in json_clean(row).items()})
    os.replace(temporary, path)


def choose_variant(cal_reconstruction_delta: float, cal_macro_band: float | None):
    """Only the preregistered calibration triggers; never consult locked100."""
    if cal_reconstruction_delta < -0.001:
        return 256, 16.0, 'calibration_reconstruction_drop_below_minus0.001'
    if cal_macro_band is not None and cal_macro_band < 0.9:
        return 128, 32.0, 'calibration_search_band_macro_below0.9'
    return 128, 16.0, 'no_sensitivity_trigger'


def choose_lambda(curve: list[dict]):
    # Ordered grid has smaller lambda first; max() keeps the first exact tie.
    return max(curve, key=lambda row: row['calibration_g3_mean_dice'])['smooth_lambda']


def interface(prediction, boundary_probability, nodes, radius):
    geometry = extract_geometry(prediction, nodes=nodes, search_radius=radius)
    return geometry, None if geometry is None else generate_candidates(boundary_probability, geometry)


def candidates_hash(candidates):
    if candidates is None:
        return 'empty_prediction_no_candidates'
    h = hashlib.sha256()
    for name in ['points', 'offsets', 'scores', 'valid']:
        array = np.ascontiguousarray(getattr(candidates, name))
        h.update(name.encode())
        h.update(array.tobytes())
    return h.hexdigest()


def failure_coverage(prediction, gt, error):
    structure = np.ones((3, 3), bool)
    coverage = {field: 0 for field in COVERAGE_FIELDS}
    coverage.update(search_band_coverage=0.0 if gt.any() else None,
                    prediction_components=int(ndimage.label(prediction, structure=structure)[1]),
                    gt_components=int(ndimage.label(gt, structure=structure)[1]),
                    prediction_holes=int(ndimage.label(ndimage.binary_fill_holes(prediction) & ~prediction, structure=structure)[1]),
                    gt_holes=int(ndimage.label(ndimage.binary_fill_holes(gt) & ~gt, structure=structure)[1]),
                    candidate_coverage_effective=None, local_wrong_rate_given_candidate_coverage=None,
                    geometry_error=str(error))
    return coverage


def calibration_diagnostics(cases, nodes, radius):
    rows = []
    for case in cases:
        try:
            geometry, candidates = interface(case['prediction'], case['q'], nodes, radius)
            zero = case['prediction'] if geometry is None else rasterize(geometry.points, case['prediction'].shape)
            coverage = failure_coverage(case['prediction'], case['gt'], 'empty_prediction') if geometry is None else coverage_diagnostics(geometry, candidates, case['gt'])
            status = 'empty_prediction_no_contour' if geometry is None else 'ok'
            case.update(geometry=geometry, candidates=candidates, geometry_error=None)
        except (ValueError, FloatingPointError) as error:
            case.update(geometry=None, candidates=None, geometry_error=str(error))
            zero = case['prediction']
            coverage, status = failure_coverage(case['prediction'], case['gt'], error), 'geometry_failure_retained_as_G0'
        case['candidate_hash'] = candidates_hash(case['candidates'])
        g0 = segmentation_metrics(case['prediction'], case['gt'])
        g1 = segmentation_metrics(zero, case['gt'])
        row = {'image_id': case['record']['image_id'], 'nodes': nodes, 'search_radius': radius,
               'status': status, 'g0_dice': g0['dice'], 'g1_dice': g1['dice'], 'g1_minus_g0_dice': g1['dice'] - g0['dice']}
        row.update({k: v for k, v in coverage.items() if not k.startswith('node_')})
        rows.append(row)
    reconstruction = float(np.mean([row['g1_minus_g0_dice'] for row in rows]))
    band = [row['search_band_coverage'] for row in rows if row['search_band_coverage'] is not None]
    return rows, reconstruction, float(np.mean(band)) if band else None


def lambda_curve(cases, grid):
    curve = []
    for strength in grid:
        metrics = []
        for case in cases:
            candidates = case['candidates']
            if candidates is None:
                mask = case['prediction']
            else:
                indices, _ = closed_dp(candidates.local_costs, candidates.offsets, strength)
                mask = rasterize(candidates.points[np.arange(len(indices)), indices], case['prediction'].shape)
            metrics.append(segmentation_metrics(mask, case['gt']))
        curve.append({'smooth_lambda': strength, 'calibration_images': len(cases), **{'calibration_g3_mean_' + metric: float(np.mean([row[metric] for row in metrics])) for metric in ['dice', 'iou', 'bf1']}, 'selection_metric': 'g3_mean_dice_only', 'g4_used_for_selection': False})
    return curve


def evaluate_case(case, nodes, radius, strength):
    if case.get('geometry_error'):
        metrics = segmentation_metrics(case['prediction'], case['gt'])
        return {'status': 'geometry_failure_retained_as_G0', 'geometry': None, 'candidates': None,
                'masks': {f'G{i}': case['prediction'].copy() for i in range(5)},
                'metrics': {f'G{i}': dict(metrics) for i in range(5)},
                'coverage': failure_coverage(case['prediction'], case['gt'], case['geometry_error'])}
    try:
        result = evaluate_selectors(case['prediction'], case['q'], case['gt'], strength, nodes=nodes, search_radius=radius)
    except (ValueError, FloatingPointError) as error:
        case['geometry_error'] = str(error)
        return evaluate_case(case, nodes, radius, strength)
    if candidates_hash(result['candidates']) != case['candidate_hash']:
        raise RuntimeError('GT-free candidates changed after GT loading')
    return result


def result_rows(case, result, nodes, radius, strength, signature_sha):
    coverage = {k: v for k, v in result['coverage'].items() if not k.startswith('node_')}
    return [{'image_id': case['record']['image_id'], 'role': case['role'], 'method': method,
             **metrics, **coverage, 'status': result['status'], 'geometry_error': case.get('geometry_error'),
             'nodes': nodes, 'search_radius': radius, 'smooth_lambda': strength,
             'candidate_set_sha256': case['candidate_hash'], 'pipeline_signature_sha256': signature_sha,
             'gt_candidate_generation': False, 'g4_deployable': False, 'g4_dice_upper_bound': False}
            for method, metrics in result['metrics'].items()]


def save_case(case, result, root, nodes, radius, signature_sha):
    artifact = root / 'results/cases' / (case['record']['image_id'] + '.npz')
    artifact.parent.mkdir(parents=True, exist_ok=True)
    geometry, candidates = result['geometry'], result['candidates']
    features = case['features']
    candidate_features = np.zeros((0, 4, 32), np.float16)
    if candidates is not None:
        coordinates = [candidates.points[..., 1].ravel(), candidates.points[..., 0].ravel()]
        sampled = np.stack([ndimage.map_coordinates(channel.astype(np.float32), coordinates, order=1, mode='nearest', prefilter=False).reshape(nodes, 4) for channel in features], axis=-1)
        sampled[~candidates.valid] = 0
        candidate_features = sampled.astype(np.float16)
    arrays = {'image_id': np.array(case['record']['image_id']), 'role': np.array(case['role']),
              'pipeline_signature_sha256': np.array(signature_sha), 'candidate_set_sha256': np.array(case['candidate_hash']),
              'cnn_logits': case['logits'].astype(np.float32), 'cnn_probability': case['probability'].astype(np.float32),
              'boundary_probability': case['q'].astype(np.float32), 'gt_mask': case['gt'].astype(bool),
              'input_rgb': cv2.resize(load_rgb(case['record']['image_path']), (256, 256), interpolation=cv2.INTER_LINEAR),
              'geometry_points': np.empty((0, 2)) if geometry is None else geometry.points,
              'normals': np.empty((0, 2)) if geometry is None else geometry.normals,
              'geometry_search_radius': np.array(radius),
              'candidate_points': np.empty((0, 4, 2)) if candidates is None else candidates.points,
              'candidate_offsets': np.empty((0, 4)) if candidates is None else candidates.offsets,
              'candidate_valid': np.empty((0, 4), bool) if candidates is None else candidates.valid,
              'candidate_scores': np.empty((0, 4)) if candidates is None else candidates.scores,
              'candidate_CNN_features': candidate_features,
              **{'masks_' + key: mask.astype(bool) for key, mask in result['masks'].items()},
              **{'indices_' + key: indices for key, indices in result.get('indices', {}).items()}}
    temporary = artifact.with_name(artifact.name + '.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temporary, artifact)
    coverage = result['coverage']
    node_rows = []
    if geometry is not None:
        for i in range(nodes):
            node_rows.append({'image_id': case['record']['image_id'], 'role': case['role'], 'node': i,
                             'x': float(geometry.points[i, 0]), 'y': float(geometry.points[i, 1]),
                             'normal_x': float(geometry.normals[i, 0]), 'normal_y': float(geometry.normals[i, 1]),
                             'gt_intersection_count': int(coverage['node_gt_intersection_counts'][i]),
                             'effective_single_intersection': bool(coverage['node_valid'][i]),
                             'gt_intersections': coverage['node_gt_intersections'][i].tolist(),
                             'gt_single_offset': coverage['node_gt_offset'][i],
                             'candidate_covered': bool(coverage['node_candidate_covered'][i]),
                             'local_wrong_given_covered': bool(coverage['node_local_wrong'][i]),
                             'candidate_offsets': candidates.offsets[i].tolist(),
                             'candidate_valid': candidates.valid[i].tolist(),
                             'candidate_scores': candidates.scores[i].tolist(),
                             'candidate_set_sha256': case['candidate_hash']})
    return node_rows


def run_probe(*, root: str | Path = ROOT, device='cuda'):
    root = Path(root).resolve()
    output = root / 'results'
    output.mkdir(exist_ok=True)
    prereg_path = root / 'preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    if prereg['DP_lambda_grid'] != [0.0, 0.0001, 0.0005, 0.002, 0.01, 0.05] or file_sha256(root / 'assets/val_split_seed17.csv') != prereg['split_sha256']:
        raise RuntimeError('The fixed split or DP grid differs from preregistration')
    if file_sha256(prereg['taskbook_path']) != prereg['taskbook_sha256']:
        raise RuntimeError('Taskbook changed')
    done = json.loads((root / 'boundary_head/DONE.json').read_text())
    head_config = json.loads((root / 'boundary_head/config.json').read_text())
    if done['status'] != 'PASS' or done['epochs_completed'] != 10 or done['optimizer_updates'] != 2500 or not done['cnn_parameters_and_buffers_unchanged']:
        raise RuntimeError('The authorized score-head training is incomplete')
    for name, expected in head_config['code_sha256'].items():
        if file_sha256(root / name) != expected:
            raise RuntimeError('Frozen head-training core source changed: ' + name)
    for name, expected in done['files_sha256'].items():
        if file_sha256(root / 'boundary_head' / name) != expected:
            raise RuntimeError('Completed score-head artifact changed: ' + name)
    source_paths = sorted([*root.glob('contour/*.py'), *root.glob('tools/*.py'), *root.glob('tests/*.py')])
    signature = {'source_sha256': {str(p.relative_to(root)): file_sha256(p) for p in source_paths},
                 'preregistration_sha256': file_sha256(prereg_path), 'split_sha256': prereg['split_sha256'],
                 'public_weight_sha256': PUBLIC_WEIGHT_SHA256, 'public_source_manifest_sha256': file_sha256(root / 'third_party/msgu_net/source_manifest.json'),
                 'head_best_sha256': done['files_sha256']['best.pth'], 'head_best_epoch': done['best_epoch'],
                 'head_config_sha256': file_sha256(root / 'boundary_head/config.json'),
                 'dataset_manifests_sha256': {name: file_sha256(root / 'assets' / name) for name in ['isic2017_train_asset_manifest.csv', 'isic2017_val_asset_manifest.csv']},
                 'metric_coordinates': 'model_input_256x256; original CNN macro per-image Dice/IoU; BF1 tolerance2', 'device': device}
    signature_sha = canonical_sha(signature)
    complete_path = output / 'RUN_DONE.json'
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text())
        if complete['pipeline_signature_sha256'] != signature_sha:
            raise RuntimeError('Completed results belong to a different source/input signature; preserve them')
        for name, expected in complete['files_sha256'].items():
            if file_sha256(output / name) != expected:
                raise RuntimeError('Completed result file changed: ' + name)
        return json.loads((output / 'summary.json').read_text())
    began = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    provenance = signature | {'pipeline_signature_sha256': signature_sha, 'created_utc': datetime.now(timezone.utc).isoformat(), 'torch_version': torch.__version__, 'cv2_version': cv2.__version__, 'numpy_version': np.__version__, 'full_third_party_source_and_weight_sha256': json.loads((root / 'third_party/msgu_net/source_manifest.json').read_text()), 'boundary_head_training_DONE': done}
    atomic_json(output / 'code_provenance.json', provenance)
    head = load_selected_head(root=root, device=device)
    features = np.load(root / 'boundary_head/feature_cache/features.npy', mmap_mode='r')
    logits = np.load(root / 'boundary_head/feature_cache/cnn_logits.npy', mmap_mode='r')
    masks = np.load(root / 'boundary_head/feature_cache/segmentation_targets.npy', mmap_mode='r')
    cache_index = json.loads((root / 'boundary_head/feature_cache/index.json').read_text())
    if cache_index['signature_sha256'] != done['signature_sha256']:
        raise RuntimeError('Feature cache is not the one used by the completed selected head')
    for name, expected in cache_index['files_sha256'].items():
        if file_sha256(root / 'boundary_head/feature_cache' / name) != expected:
            raise RuntimeError('Head feature/logit/GT cache changed: ' + name)
    calibration_records = read_val_split(root / 'assets/val_split_seed17.csv', 'calibration50')
    cache_by_id = {row['image_id']: index for index, row in enumerate(cache_index['rows']) if row['role'] == 'calibration50'}
    calibration = []
    with torch.no_grad():
        for record in calibration_records:
            index = cache_by_id[record['image_id']]
            feat = np.array(features[index], copy=True)
            z = np.array(logits[index, 0], copy=True)
            tensor = torch.from_numpy(z).to(device)
            q = torch.sigmoid(head(torch.from_numpy(feat)[None].to(device)))[0, 0].cpu().numpy()
            calibration.append({'record': record, 'role': 'cal50', 'features': feat, 'logits': z,
                                'probability': torch.sigmoid(tensor).cpu().numpy(), 'prediction': segmentation_mask(tensor).cpu().numpy(),
                                'q': q, 'gt': np.array(masks[index, 0], dtype=bool, copy=True)})
    base_rows, base_delta, base_band = calibration_diagnostics(calibration, 128, 16.0)
    write_csv(output / 'calibration_base128_radius16.csv', base_rows)
    nodes, radius, reason = choose_variant(base_delta, base_band)
    sensitivity = (nodes, radius) != (128, 16.0)
    selected_rows, selected_delta, selected_band = calibration_diagnostics(calibration, nodes, radius) if sensitivity else (base_rows, base_delta, base_band)
    write_csv(output / 'calibration_selected_variant.csv', selected_rows)
    curve = lambda_curve(calibration, prereg['DP_lambda_grid'])
    write_csv(output / 'calibration_lambda_curve.csv', curve)
    strength = choose_lambda(curve)
    locked = {'status': 'LOCKED_BEFORE_VERIFICATION_GT', 'locked_utc': datetime.now(timezone.utc).isoformat(),
              'pipeline_signature_sha256': signature_sha, 'nodes': nodes, 'search_radius': radius,
              'smooth_lambda': strength, 'sensitivity_performed': sensitivity, 'sensitivity_count': int(sensitivity),
              'sensitivity_reason': reason, 'base_calibration_g1_minus_g0_dice': base_delta,
              'base_calibration_macro_band': base_band, 'selected_calibration_g1_minus_g0_dice': selected_delta,
              'selected_calibration_macro_band': selected_band, 'head_best_epoch': done['best_epoch'],
              'head_best_sha256': done['files_sha256']['best.pth'], 'CNN_weight_sha256': PUBLIC_WEIGHT_SHA256,
              'calibration_candidate_set_sha256': {case['record']['image_id']: case['candidate_hash'] for case in calibration},
              'lambda_selection_metric': 'calibration50 mean G3 Dice; exact tie earlier/smaller lambda',
              'G4_used_for_lambda_or_variant_selection': False, 'locked100_GT_used_for_selection': False,
              'official_test_GT_used': False, 'candidate_generation_uses_GT': False}
    lock_path = output / 'dp_locked.json'
    if lock_path.is_file():
        previous = json.loads(lock_path.read_text())
        for key in ['pipeline_signature_sha256', 'nodes', 'search_radius', 'smooth_lambda', 'calibration_candidate_set_sha256']:
            if previous[key] != locked[key]:
                raise RuntimeError('Previously frozen DP differs; no retuning after locked100')
        locked = previous
    else:
        atomic_json(lock_path, locked)
    print(f'DP frozen before locked100 GT: nodes={nodes}, radius={radius}, lambda={strength}; sensitivity={reason}', flush=True)
    rows, node_rows, errors = [], [], []
    all_records = []
    def deliver(case):
        result = evaluate_case(case, nodes, radius, strength)
        rows.extend(result_rows(case, result, nodes, radius, strength, signature_sha))
        node_rows.extend(save_case(case, result, root, nodes, radius, signature_sha))
        all_records.append(case['record'])
        if case.get('geometry_error'):
            errors.append({'image_id': case['record']['image_id'], 'role': case['role'], 'error': case['geometry_error'], 'action': 'all five masks retain G0; unavailable candidate coverage is zero/undefined'})
    for i, case in enumerate(calibration, 1):
        deliver(case)
        if i % 10 == 0:
            print(f'Five-selector calibration readout {i}/50', flush=True)
    cnn = FrozenMSGUNet(root / 'third_party/msgu_net').to(device)
    before = cnn.state_hash()
    locked_records = read_val_split(root / 'assets/val_split_seed17.csv', 'locked_verification100')
    with torch.no_grad():
        for i, record in enumerate(locked_records, 1):
            details = cnn.forward_with_features(load_image(record['image_path'])[None].to(device))
            feat = details['features'][0].half().cpu().numpy()
            q = torch.sigmoid(head(details['features']))[0, 0].cpu().numpy()
            case = {'record': record, 'role': 'locked100', 'features': feat, 'q': q,
                    'logits': details['logits'][0, 0].cpu().numpy(), 'probability': details['probability'][0, 0].cpu().numpy(),
                    'prediction': details['mask'][0, 0].cpu().numpy(), 'geometry_error': None}
            # Freeze this image's candidates before opening its GT.
            try:
                case['geometry'], case['candidates'] = interface(case['prediction'], q, nodes, radius)
            except (ValueError, FloatingPointError) as error:
                case.update(geometry=None, candidates=None, geometry_error=str(error))
            case['candidate_hash'] = candidates_hash(case['candidates'])
            case['gt'] = load_mask(record, allow_verification=True)[0].numpy().astype(bool)
            deliver(case)
            if i % 10 == 0:
                print(f'Five-selector locked readout {i}/100; seconds={time.perf_counter()-began:.1f}', flush=True)
    if cnn.state_hash() != before or before != done['cnn_state_hash_before']:
        raise RuntimeError('CNN parameters/buffers changed in locked inference')
    for name, expected in signature['source_sha256'].items():
        if file_sha256(root / name) != expected:
            raise RuntimeError('Pipeline source changed during evaluation: ' + name)
    if len(rows) != 750 or len(all_records) != 150:
        raise RuntimeError('Every image must retain all five selectors')
    atomic_json(output / 'geometry_errors.json', {'count': len(errors), 'all_images_retained': True, 'errors': errors})
    write_csv(output / 'node_diagnostics.csv', node_rows)
    metadata = locked | {'candidate_interface_coordinates': 'model_input256', 'metric_original_protocol': 'input256 macro per-image', 'GT_role': 'G4 offline costs and reporting only; never candidate generation/movement', 'all_images_retained': True, 'geometry_error_images': len(errors), 'CNN_parameters_buffers_unchanged': True, 'selected_head_calibration_ability': done['selected_head_calibration'], 'runtime_seconds': time.perf_counter() - began, 'data_selection': 'ISIC2017 only; official test untouched; public historical val exposure -> development diagnosis'}
    payload = summarize(rows, prereg, metadata, output)
    # The reporter writes the full per-image CSV exactly once from complete rows.
    files = ['per_image_results.csv', 'main_table.csv', 'paired_bootstrap.csv', 'summary.json', 'decision.json', 'report.txt', 'node_diagnostics.csv', 'dp_locked.json', 'code_provenance.json', 'geometry_errors.json', 'calibration_lambda_curve.csv', 'calibration_base128_radius16.csv', 'calibration_selected_variant.csv']
    completed = {'status': 'PASS', 'pipeline_signature_sha256': signature_sha, 'images': 150, 'method_rows': 750,
                 'case_npz_files': 150, 'case_sha256': {p.name: file_sha256(p) for p in sorted((output / 'cases').glob('*.npz'))},
                 'files_sha256': {name: file_sha256(output / name) for name in files}, 'runtime_seconds': time.perf_counter() - began,
                 'CNN_training': False, 'Transformer_training': False, 'test_scoring': False}
    atomic_json(complete_path, completed)
    print(json.dumps({'status': completed['status'], 'decision': payload['decision']['status'], 'runtime_seconds': completed['runtime_seconds']}), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    try:
        run_probe(root=args.root, device=args.device)
    except Exception:
        path = Path(args.root) / 'results/last_failure.json'
        atomic_json(path, {'created_utc': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc(), 'no_extra_training_or_retuning_authorized': True})
        raise


if __name__ == '__main__':
    main()
