"""CPU equality tests: exact KD pixel-boundary queries versus original EDT."""
import math

import numpy as np
import pytest

from rsi.metrics import binary_mask_metrics as original_metrics
from tools.fast_exact_metrics import binary_mask_metrics as fast_metrics


@pytest.mark.parametrize("shape", [(1, 1), (1, 19), (17, 1), (9, 13), (31, 47), (97, 128)])
@pytest.mark.parametrize("density", [0., .05, .25, .5, .95, 1.])
def test_fixed_random_and_empty_masks_match_all_fields_exactly(shape, density):
    rng = np.random.default_rng(17)
    prediction = rng.random(shape) < density
    target = rng.random(shape) < density
    assert fast_metrics(prediction, target) == original_metrics(prediction, target)


@pytest.mark.parametrize("pred_coordinate,target_coordinate", [((0, 0), (0, 0)), ((0, 0), (12, 17)),
                                                            ((0, 17), (12, 0)), ((6, 8), (7, 9))])
def test_single_pixels_and_image_edges_match_exactly(pred_coordinate, target_coordinate):
    prediction = np.zeros((13, 18), dtype=bool)
    target = np.zeros_like(prediction)
    prediction[pred_coordinate] = True
    target[target_coordinate] = True
    assert fast_metrics(prediction, target) == original_metrics(prediction, target)


@pytest.mark.parametrize("empty_side", ["prediction", "target", "both"])
def test_one_or_both_empty_preserve_every_empty_convention(empty_side):
    prediction = np.ones((7, 11), dtype=bool)
    target = prediction.copy()
    if empty_side in ("prediction", "both"):
        prediction[:] = False
    if empty_side in ("target", "both"):
        target[:] = False
    fast = fast_metrics(prediction, target)
    assert fast == original_metrics(prediction, target)
    assert fast["hd95"] == (0. if empty_side == "both" else math.hypot(7, 11))
    assert fast["dice"] == fast["iou"] == fast["bf1"] == (1. if empty_side == "both" else 0.)


def test_combined_HD95_is_not_max_of_directed_quantiles():
    prediction = np.ones((1, 20), dtype=bool)
    target = np.zeros_like(prediction)
    target[0, 0] = True
    result = fast_metrics(prediction, target)
    assert result == original_metrics(prediction, target)
    assert result["hd95"] == 18.  # percentile95 of [0,...,19] concatenated with [0]
    assert np.percentile(np.arange(20), 95) != result["hd95"]


def test_three_boundary_tolerances_and_noncontiguous_readonly_inputs():
    prediction = np.zeros((320, 240), dtype=np.uint8)
    target = np.zeros_like(prediction)
    prediction[43:237, 40:160] = 1
    target[40:234, 40:160] = 1
    prediction = prediction[:, ::-1]
    target = target[:, ::-1]
    prediction.flags.writeable = target.flags.writeable = False
    before = prediction.tobytes(), target.tobytes()
    result = fast_metrics(prediction, target)
    assert result == original_metrics(prediction, target)
    assert result["bf1_025"] < result["bf1"] < result["bf1_1pct"]
    assert (prediction.tobytes(), target.tobytes()) == before


@pytest.mark.parametrize("prediction,target", [
    (np.ones((2, 3, 4)), np.ones((3, 4))),
    (np.ones((2, 3)), np.ones((3, 2))),
    (np.ones((0, 4)), np.ones((0, 4))),
    (np.full((3, 4), np.nan), np.ones((3, 4))),
    (np.full((3, 4), .5), np.ones((3, 4))),
])
def test_input_validation_and_error_messages_match_original(prediction, target):
    with pytest.raises(ValueError) as original:
        original_metrics(prediction, target)
    with pytest.raises(ValueError) as fast:
        fast_metrics(prediction, target)
    assert str(fast.value) == str(original.value)


def test_singleton_leading_dimensions_preserve_original_shape_handling():
    prediction = np.zeros((1, 1, 11, 17), dtype=np.float32)
    target = np.zeros_like(prediction)
    prediction[..., 0:6, 0:5] = 1
    target[..., 3:10, 4:16] = 1
    assert fast_metrics(prediction, target) == original_metrics(prediction, target)
