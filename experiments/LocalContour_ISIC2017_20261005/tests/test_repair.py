from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contour.geometry import evaluate_selectors
from contour.reporting import COVERAGE_FIELDS
from tools.repair_representation import exact_array, parse_csv_row, recompose_saved_case, verify_case_invariants


def archive_case(result, prediction, gt):
    geometry, candidates = result['geometry'], result['candidates']
    arrays = {'masks_G0': prediction, 'gt_mask': gt, 'geometry_points': geometry.points,
              'normals': geometry.normals, 'candidate_points': candidates.points,
              'candidate_offsets': candidates.offsets, 'candidate_scores': candidates.scores,
              'candidate_valid': candidates.valid}
    arrays.update({'masks_' + method: mask for method, mask in result['raw_masks'].items()})
    arrays.update({'indices_' + method: choices for method, choices in result['indices'].items()})
    return arrays


def test_saved_raw_repair_preserves_choices_and_recovers_other_foreground_and_hole():
    prediction = np.zeros((32, 32), bool)
    prediction[8:25, 8:25] = True
    prediction[13:16, 13:16] = False
    prediction[2:5, 2:5] = True
    original = evaluate_selectors(prediction, np.full(prediction.shape, .5), prediction, .05, nodes=256, search_radius=16)
    old = archive_case(original, prediction, prediction)
    row = {'status': 'ok', **{key: original['coverage'][key] for key in COVERAGE_FIELDS}}
    repaired = recompose_saved_case(old, row)
    assert all(verify_case_invariants(old, repaired).values())
    for method in ['G1', 'G2', 'G3', 'G4']:
        assert repaired['masks'][method][3, 3]
        assert not repaired['masks'][method][14, 14]
    assert repaired['compositor']['gt_used'] is False


def test_candidate_change_is_rejected_bytewise():
    with pytest.raises(AssertionError, match='changed'):
        exact_array('candidate', np.array([0.0]), np.array([-0.0]))
    with pytest.raises(AssertionError, match='changed'):
        exact_array('candidate', np.array([1.0]), np.array([1.00001]))


def test_csv_conversion_retains_undefined_counts_and_flags():
    row = parse_csv_row({'n': '256', 'coverage': '0.8', 'empty': '', 'flag': 'False', 'id': 'ISIC_001'})
    assert row == {'n': 256, 'coverage': .8, 'empty': None, 'flag': False, 'id': 'ISIC_001'}
