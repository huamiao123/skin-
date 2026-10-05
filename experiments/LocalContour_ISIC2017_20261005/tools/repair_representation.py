"""CPU-only mechanical compositor repair of the fully archived v1 diagnosis.

No CNN/head is instantiated, no RGB/GT source dataset is read, and no selection
of a checkpoint, node/range sensitivity, or DP strength is performed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contour.geometry import build_prediction_context, compose_prediction_context, mask_contours, polygon_area, rasterize, resample_closed, segmentation_metrics
from contour.reporting import COVERAGE_FIELDS, summarize


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)


def write_csv(path, rows):
    names = list(dict.fromkeys(k for row in rows for k in row))
    temp = Path(path).with_name(Path(path).name + '.tmp')
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in clean(row).items()})
    os.replace(temp, path)


def exact_array(label, old, new):
    old, new = np.asarray(old), np.asarray(new)
    if old.shape != new.shape or old.dtype != new.dtype or old.tobytes() != new.tobytes():
        raise AssertionError('Mechanical repair changed a frozen array: ' + label)


def verify_case_invariants(old, result):
    """Require byte-identical candidates, normals, choices, and old raw fill."""
    candidates, geometry = result['candidates'], result['geometry']
    for stored, attr, empty_shape, dtype in [('candidate_points', 'points', (0, 4, 2), np.float64), ('candidate_offsets', 'offsets', (0, 4), np.float64), ('candidate_scores', 'scores', (0, 4), np.float64), ('candidate_valid', 'valid', (0, 4), bool)]:
        new = np.empty(empty_shape, dtype=dtype) if candidates is None else getattr(candidates, attr)
        exact_array(stored, old[stored], new)
    for stored, attr in [('geometry_points', 'points'), ('normals', 'normals')]:
        new = np.empty((0, 2), np.float64) if geometry is None else getattr(geometry, attr)
        exact_array(stored, old[stored], new)
    exact_array('G0 mask', old['masks_G0'], result['masks']['G0'])
    for method in ['G1', 'G2', 'G3', 'G4']:
        if 'indices_' + method in old:
            exact_array('indices_' + method, old['indices_' + method], result['indices'][method])
        # v1 had exactly this unchanged polygon-centre fill, before composing.
        exact_array('old_raw_mask_' + method, old['masks_' + method], result['raw_masks'][method])
    return {'candidate_points_offsets_scores_valid_byte_identical': True,
            'geometry_points_normals_byte_identical': True,
            'G0_mask_byte_identical': True, 'G1_G2_G3_G4_indices_byte_identical': True,
            'pre_compositor_raw_masks_byte_identical_to_v1': True}


def source_polygon(prediction):
    contours = mask_contours(prediction)
    if not contours:
        return None
    source = max(contours, key=lambda p: abs(polygon_area(p)))
    return source[::-1] if polygon_area(source) < 0 else source


def recompose_saved_case(old, archived_g0_row):
    """Use saved raw masks/choices directly: no candidates or DP are rebuilt."""
    prediction = old['masks_G0']
    source = source_polygon(prediction)
    raw_masks = {method: old['masks_' + method].copy() for method in ['G1', 'G2', 'G3', 'G4']}
    if source is None:
        geometry = candidates = None
        masks = {'G0': prediction.copy(), **raw_masks}
        compositor = {'formula': 'empty prediction unchanged', 'gt_used': False, 'identical_for_G1_G2_G3_G4': True}
    else:
        context = build_prediction_context(prediction, source)
        masks = {'G0': prediction.copy(), **{method: compose_prediction_context(raw, context) for method, raw in raw_masks.items()}}
        geometry = SimpleNamespace(points=old['geometry_points'], normals=old['normals'])
        candidates = SimpleNamespace(points=old['candidate_points'], offsets=old['candidate_offsets'], scores=old['candidate_scores'], valid=old['candidate_valid'])
        compositor = {'formula': '(R & ~H) | C; C=P & ~F; H=F & ~P; F=full original selected source polygon', 'gt_used': False, 'identical_for_G1_G2_G3_G4': True,
                      'preserved_other_foreground_pixels': int(context.foreground_to_preserve.sum()),
                      'preserved_original_background_inside_source_exterior_pixels': int(context.background_to_preserve.sum())}
    return {'geometry': geometry, 'candidates': candidates, 'raw_masks': raw_masks, 'masks': masks,
            'indices': {method: old['indices_' + method] for method in ['G1', 'G2', 'G3', 'G4'] if 'indices_' + method in old},
            'metrics': {method: segmentation_metrics(mask, old['gt_mask']) for method, mask in masks.items()},
            'coverage': {field: archived_g0_row[field] for field in COVERAGE_FIELDS},
            'status': archived_g0_row['status'], 'compositor': compositor}


def parse_csv_row(row):
    result = {}
    for key, value in row.items():
        if value == '':
            result[key] = None
        elif value in ('True', 'False'):
            result[key] = value == 'True'
        else:
            try:
                result[key] = int(value)
            except (ValueError, TypeError):
                try:
                    result[key] = float(value)
                except (ValueError, TypeError):
                    result[key] = value
    return result


def repaired_g1(old, nodes, radius):
    prediction = old['masks_G0']
    source = source_polygon(prediction)
    if source is None:
        return prediction.copy()
    context = build_prediction_context(prediction, source)
    raw = old['masks_G1'] if nodes == 256 else rasterize(resample_closed(source, nodes), prediction.shape)
    return compose_prediction_context(raw, context)


def recompute_calibration_g1(source, cases, nodes, radius):
    with Path(source).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 50:
        raise AssertionError('Archived calibration must have all 50 images')
    output = []
    for row in rows:
        case = cases[row['image_id']]
        g0 = segmentation_metrics(case['masks_G0'], case['gt_mask'])['dice']
        if g0 != float(row['g0_dice']):
            raise AssertionError('Archived G0 calibration metric changed')
        g1 = segmentation_metrics(repaired_g1(case, nodes, radius), case['gt_mask'])['dice']
        output.append(dict(row, archived_v1_g1_dice=float(row['g1_dice']),
                           archived_v1_g1_minus_g0_dice=float(row['g1_minus_g0_dice']),
                           g1_dice=g1, g1_minus_g0_dice=g1 - g0,
                           mechanical_compositor_repair_only=True,
                           new_variant_or_lambda_selection=False))
    return output


def run_repair(*, root=ROOT):
    root = Path(root).resolve()
    archive = root / 'results_v1_representation_loss'
    output = root / 'results'
    output.mkdir(exist_ok=True)
    before = time.perf_counter()
    previous_done = json.loads((archive / 'RUN_DONE.json').read_text())
    previous_provenance = json.loads((archive / 'code_provenance.json').read_text())
    previous_summary = json.loads((archive / 'summary.json').read_text())
    previous_lock = json.loads((archive / 'dp_locked.json').read_text())
    if previous_done['status'] != 'PASS' or (previous_done['images'], previous_done['method_rows']) != (150, 750):
        raise AssertionError('v1 must be fully completed and archived before mechanical correction')
    if (previous_lock['nodes'], previous_lock['search_radius'], previous_lock['smooth_lambda']) != (256, 16.0, .05):
        raise AssertionError('Only the already locked v1 256/16/0.05 interface may be repaired')
    archive_identity = {}
    for name, expected in previous_done['files_sha256'].items():
        if sha(archive / name) != expected:
            raise AssertionError('Archived v1 output changed: ' + name)
        archive_identity[name] = expected
    if len(previous_done['case_sha256']) != 150:
        raise AssertionError('All 150 archived case files are required')
    for name, expected in previous_done['case_sha256'].items():
        if sha(archive / 'cases' / name) != expected:
            raise AssertionError('Archived v1 case changed: ' + name)
    snapshot = root / 'protocols/source_v1_representation_loss'
    for name, expected in previous_provenance['source_sha256'].items():
        if sha(snapshot / name) != expected:
            raise AssertionError('Archived source snapshot differs from executed v1: ' + name)
    head_config = json.loads((root / 'boundary_head/config.json').read_text())
    head_done = json.loads((root / 'boundary_head/DONE.json').read_text())
    for name, expected in head_config['code_sha256'].items():
        if sha(root / name) != expected:
            raise AssertionError('Frozen CNN/head-training core changed: ' + name)
    if sha(root / 'boundary_head/best.pth') != previous_lock['head_best_sha256'] or head_done['best_epoch'] != previous_lock['head_best_epoch']:
        raise AssertionError('Selected head identity changed')
    if sha(root / 'third_party/msgu_net/weights/best_model_isic2017.pth') != previous_lock['CNN_weight_sha256']:
        raise AssertionError('CNN identity changed')
    for name, expected in head_done['files_sha256'].items():
        if sha(root / 'boundary_head' / name) != expected:
            raise AssertionError('Head training artifact changed: ' + name)
    prereg = json.loads((root / 'preregistration.json').read_text())
    if prereg['reporting_rules'] != previous_summary['preregistration']['reporting_rules']:
        raise AssertionError('Post-result repair cannot change any scientific reporting rule')
    if sha(root / 'assets/val_split_seed17.csv') != previous_provenance['split_sha256']:
        raise AssertionError('Official50/100 split changed')
    with (root / 'assets/val_split_seed17.csv').open(newline='') as f:
        roles = {row['image_id']: 'cal50' if row['role'] == 'calibration50' else 'locked100' for row in csv.DictReader(f)}
    paths = sorted((archive / 'cases').glob('*.npz'))
    if {p.stem for p in paths} != set(roles) or list(roles.values()).count('cal50') != 50 or list(roles.values()).count('locked100') != 100:
        raise AssertionError('Every official validation case must keep its existing role')
    with (archive / 'per_image_results.csv').open(newline='') as f:
        archived_rows = {(row['image_id'], row['method']): parse_csv_row(row) for row in csv.DictReader(f)}
    source_sha = {str(p.relative_to(root)): sha(p) for p in sorted([*root.glob('contour/*.py'), *root.glob('tools/*.py'), *root.glob('tests/*.py')])}
    identity = {'repair_scope': 'common GT-free compositor only; exact frozen candidates/paths/raw rasterization',
                'source_sha256': source_sha, 'archive_path': str(archive),
                'archive_RUN_DONE_sha256': sha(archive / 'RUN_DONE.json'),
                'archive_outputs_sha256': archive_identity, 'archive_case_sha256': previous_done['case_sha256'],
                'preregistration_sha256': sha(root / 'preregistration.json'),
                'head_best_sha256': previous_lock['head_best_sha256'], 'CNN_weight_sha256': previous_lock['CNN_weight_sha256'],
                'original_pipeline_signature_sha256': previous_done['pipeline_signature_sha256'],
                'fixed_nodes': 256, 'fixed_search_radius': 16.0, 'fixed_smooth_lambda': .05,
                'scientific_thresholds_unchanged': True, 'new_sensitivity_parameter_selection': False,
                'new_checkpoint_selection': False, 'new_lambda_selection': False,
                'new_CNN_or_head_or_Transformer_training': False, 'new_model_inference': False,
                'source_dataset_image_or_GT_files_opened': 0, 'CPU_only': True}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (output / 'RUN_DONE.json').is_file():
        completed = json.loads((output / 'RUN_DONE.json').read_text())
        if completed['pipeline_signature_sha256'] != signature:
            raise RuntimeError('Preserve already completed results of a different signature')
        for name, expected in completed['files_sha256'].items():
            if sha(output / name) != expected:
                raise AssertionError('Completed repair artifact changed: ' + name)
        return json.loads((output / 'summary.json').read_text())
    if (output / 'per_image_results.csv').is_file():
        raise RuntimeError('An existing incomplete final CSV must be preserved; do not silently rewrite scientific rows')
    atomic_json(output / 'code_provenance.json', identity | {'pipeline_signature_sha256': signature, 'created_utc': datetime.now(timezone.utc).isoformat(), 'historical_v1_code_provenance_sha256': sha(archive / 'code_provenance.json'), 'selected_head_training_DONE': head_done, 'numpy_version': np.__version__})
    rows, checks, cal_cases = [], [], {}
    (output / 'cases').mkdir(exist_ok=True)
    for count, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as stored:
            old = {key: np.array(stored[key], copy=True) for key in stored.files}
        image_id, role = str(old['image_id'].item()), str(old['role'].item())
        if role != roles[image_id]:
            raise AssertionError('Archived case role changed')
        result = recompose_saved_case(old, archived_rows[image_id, 'G0'])
        invariants = verify_case_invariants(old, result)
        for field in COVERAGE_FIELDS:
            current = result['coverage'][field]
            prior = archived_rows[image_id, 'G0'][field]
            if current is None:
                if prior not in (None, '', 'None', 'null'):
                    raise AssertionError('Coverage availability changed: ' + field)
            elif float(current) != float(prior):
                raise AssertionError('Frozen candidate diagnostic changed: ' + field)
        for metric in ['dice', 'iou', 'bf1']:
            if result['metrics']['G0'][metric] != float(archived_rows[image_id, 'G0'][metric]):
                raise AssertionError('Original CNN metric changed: ' + metric)
        for method, metrics in result['metrics'].items():
            rows.append({**archived_rows[image_id, method], 'image_id': image_id, 'role': role, 'method': method, **metrics,
                         'status': result['status'], 'geometry_error': None, 'nodes': 256,
                         'search_radius': 16.0, 'smooth_lambda': .05,
                         'candidate_set_sha256': old['candidate_set_sha256'].item(),
                         'pipeline_signature_sha256': signature,
                         'source_v1_pipeline_signature_sha256': previous_done['pipeline_signature_sha256'],
                         'mechanical_representation_repair': True, 'gt_candidate_generation': False,
                         'g4_deployable': False, 'g4_dice_upper_bound': False})
        fixed_keys = set(old) - {'masks_G1', 'masks_G2', 'masks_G3', 'masks_G4'}
        repaired = dict(old)
        repaired.update({'masks_' + method: mask for method, mask in result['masks'].items()})
        repaired.update({'raw_masks_' + method: mask for method, mask in result['raw_masks'].items()})
        repaired['representation_repair_signature_sha256'] = np.array(signature)
        for key in fixed_keys:
            exact_array('unchanged stored ' + key, old[key], repaired[key])
        destination = output / 'cases' / path.name
        temp = destination.with_name(destination.name + '.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, **repaired)
        os.replace(temp, destination)
        checks.append({'image_id': image_id, 'role': role, **invariants,
                       'compositor': result['compositor'], 'changed_mask_pixels': {m: int(np.count_nonzero(old['masks_' + m] != result['masks'][m])) for m in ['G1', 'G2', 'G3', 'G4']}})
        if role == 'cal50':
            cal_cases[image_id] = old
        if count % 10 == 0:
            print(f'CPU representation repair {count}/150; candidates, choices, G0 and raw fill exact; seconds={time.perf_counter()-before:.1f}', flush=True)
    if len(rows) != 750 or len(checks) != 150 or len(cal_cases) != 50:
        raise AssertionError('The repaired readout must retain all150 images and all5 methods')
    base_cal = recompute_calibration_g1(archive / 'calibration_base128_radius16.csv', cal_cases, 128, 16.0)
    selected_cal = recompute_calibration_g1(archive / 'calibration_selected_variant.csv', cal_cases, 256, 16.0)
    write_csv(output / 'calibration_base128_radius16.csv', base_cal)
    write_csv(output / 'calibration_selected_variant.csv', selected_cal)
    for name in ['node_diagnostics.csv', 'calibration_lambda_curve.csv', 'geometry_errors.json']:
        shutil.copyfile(archive / name, output / name)
        if sha(archive / name) != sha(output / name):
            raise AssertionError('Historical unchanged file did not copy exactly: ' + name)
    atomic_json(output / 'lambda_curve_provenance.json', {'status': 'HISTORICAL_V1_SELECTION_ONLY', 'curve_copied_byte_identically': True, 'original_curve_sha256': sha(archive / 'calibration_lambda_curve.csv'), 'note': 'Curve records the v1 naive rasterization selection; it was not recalculated or optimized after compositor repair.', 'fixed_selected_lambda': .05, 'lambda_reselected': False})
    lock = previous_lock | {'mechanical_representation_repair': True, 'representation_repair_signature_sha256': signature,
                            'archived_v1_path': str(archive), 'archived_v1_dp_locked_sha256': sha(archive / 'dp_locked.json'),
                            'repair_scope': identity['repair_scope'], 'source_input_archive_identity': identity,
                            'all150_candidates_and_DP_choices_byte_identical': True,
                            'new_sensitivity_runs': 0, 'new_parameter_or_checkpoint_selection': False,
                            'lambda_curve_is_historical_v1': True,
                            'repaired_base_calibration_g1_minus_g0_dice': float(np.mean([r['g1_minus_g0_dice'] for r in base_cal])),
                            'repaired_selected_calibration_g1_minus_g0_dice': float(np.mean([r['g1_minus_g0_dice'] for r in selected_cal]))}
    atomic_json(output / 'dp_locked.json', lock)
    atomic_json(output / 'representation_repair_acceptance.json', {'status': 'PASS', 'image_count': 150, 'all_cases_retained': True, 'all_invariants_passed': True, 'rows': checks, 'scope': identity['repair_scope'], 'new_inference': False, 'new_training': False, 'new_tuning': False})
    metadata = previous_summary['configuration_metadata'] | {'mechanical_representation_repair': True,
                'archived_v1_path': str(archive), 'representation_repair_signature_sha256': signature,
                'all150_candidate_and_DP_choices_identical': True,
                'sensitivity_performed': True, 'sensitivity_count': 1,
                'new_sensitivity_runs': 0, 'new_training': False, 'new_inference': False, 'new_lambda_selection': False,
                'nodes': 256, 'search_radius': 16.0, 'smooth_lambda': .05,
                'current_calibration_g1_minus_g0_dice': lock['repaired_selected_calibration_g1_minus_g0_dice'],
                'comparison_of_repair_to_v1_is_representation_only': True,
                'compositor': '(R & ~H) | C; C=P & ~F, H=F & ~P, F=full original selected source polygon fill; no GT',
                'original_lambda_curve_historical': True, 'runtime_seconds': time.perf_counter() - before}
    for name, expected in source_sha.items():
        if sha(root / name) != expected:
            raise AssertionError('Corrector source changed during readout: ' + name)
    payload = summarize(rows, prereg, metadata, output)
    names = ['per_image_results.csv', 'main_table.csv', 'paired_bootstrap.csv', 'summary.json', 'decision.json', 'report.txt',
             'node_diagnostics.csv', 'dp_locked.json', 'code_provenance.json', 'geometry_errors.json', 'calibration_lambda_curve.csv',
             'calibration_base128_radius16.csv', 'calibration_selected_variant.csv', 'lambda_curve_provenance.json', 'representation_repair_acceptance.json']
    completed = {'status': 'PASS', 'pipeline_signature_sha256': signature, 'mechanical_representation_repair': True,
                 'images': 150, 'method_rows': 750, 'case_npz_files': 150,
                 'files_sha256': {name: sha(output / name) for name in names},
                 'case_sha256': {p.name: sha(p) for p in sorted((output / 'cases').glob('*.npz'))},
                 'all150_candidates_choices_G0_and_raw_rasterization_byte_identical_to_v1': True,
                 'original_v1_archive_path': str(archive), 'source_v1_RUN_DONE_sha256': sha(archive / 'RUN_DONE.json'),
                 'CNN_training': False, 'boundary_head_training': False, 'Transformer_training': False,
                 'new_model_inference': False, 'new_parameter_selection': False, 'test_scoring': False,
                 'runtime_seconds': time.perf_counter() - before}
    atomic_json(output / 'RUN_DONE.json', completed)
    print(json.dumps({'status': 'PASS', 'decision': payload['decision']['status'], 'mechanical_repair_only': True, 'runtime_seconds': completed['runtime_seconds']}), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    args = parser.parse_args()
    try:
        run_repair(root=args.root)
    except Exception:
        atomic_json(Path(args.root) / 'results/repair_failure.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc(), 'new_training_or_inference_performed': False})
        raise


if __name__ == '__main__':
    main()
