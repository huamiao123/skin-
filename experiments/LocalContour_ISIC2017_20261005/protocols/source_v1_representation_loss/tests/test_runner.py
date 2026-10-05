from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contour.geometry import extract_geometry, generate_candidates
from tools.run_probe import candidates_hash, choose_lambda, choose_variant, evaluate_case


def test_sensitivity_priority_and_single_change():
    assert choose_variant(-0.002, 0.1)[:2] == (256, 16.0)
    assert choose_variant(0.0, 0.8)[:2] == (128, 32.0)
    assert choose_variant(-0.001, .9)[:2] == (128, 16.0)
    assert choose_variant(0.0, None)[:2] == (128, 16.0)


def test_lambda_selection_only_g3_and_smaller_exact_tie():
    curve = [{'smooth_lambda': 0.0, 'calibration_g3_mean_dice': .8, 'g4_dice': 1.0},
             {'smooth_lambda': .0001, 'calibration_g3_mean_dice': .81, 'g4_dice': .1},
             {'smooth_lambda': .05, 'calibration_g3_mean_dice': .81, 'g4_dice': 1.0}]
    assert choose_lambda(curve) == .0001


def test_gt_cannot_change_fixed_candidate_interface():
    prediction = np.zeros((32, 32), bool)
    prediction[7:25, 8:24] = True
    q = np.full((32, 32), .5)
    geometry = extract_geometry(prediction)
    candidates = generate_candidates(q, geometry)
    case = {'prediction': prediction, 'q': q, 'candidate_hash': candidates_hash(candidates), 'geometry_error': None, 'gt': prediction.copy()}
    first = evaluate_case(case, 128, 16.0, .002)
    case['gt'] = np.zeros_like(prediction)
    case['gt'][3:29, 4:28] = True
    second = evaluate_case(case, 128, 16.0, .002)
    assert candidates_hash(first['candidates']) == candidates_hash(second['candidates'])
    assert all(np.array_equal(first['masks'][key], second['masks'][key]) for key in ['G0', 'G1', 'G2', 'G3'])


def test_geometry_failure_keeps_all_methods_without_fake_coverage():
    prediction = np.zeros((16, 16), bool)
    prediction[4:12, 4:12] = True
    case = {'prediction': prediction, 'gt': prediction.copy(), 'geometry_error': 'degenerate normal'}
    result = evaluate_case(case, 128, 16.0, .002)
    assert set(result['masks']) == {'G0', 'G1', 'G2', 'G3', 'G4'}
    assert all(np.array_equal(mask, prediction) for mask in result['masks'].values())
    assert result['coverage']['search_band_coverage'] == 0.0
    assert result['coverage']['effective_normal_nodes'] == 0
