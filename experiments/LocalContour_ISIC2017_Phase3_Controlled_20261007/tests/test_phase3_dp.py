import itertools

import numpy as np

from phase3_dp import choose_argmax, choose_closed_dp
from phase3_fast_raster import rasterize_exact
from phase3_movement import node_move_threshold
from evaluation import rasterize


def test_closed_dp_matches_bruteforce_and_closing_edge():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(3, 65))
    valid = np.zeros_like(scores, dtype=bool)
    valid[:, [30, 32, 34]] = True
    for alpha, lam in ((0.0, 0.05), (0.1, 0.5)):
        actual = choose_closed_dp(scores, valid, alpha, lam, device="cpu")
        possibilities = itertools.product((30, 32, 34), repeat=3)
        best = min(possibilities, key=lambda p: sum(-scores[i, j]+alpha*abs(j-32) for i, j in enumerate(p))
                   +lam*sum((p[i]-p[(i+1)%3])**2 for i in range(3)))
        assert np.array_equal(actual, best)


def test_zero_lambda_and_keep_tie():
    scores = np.zeros((4, 65))
    valid = np.ones_like(scores, dtype=bool)
    assert np.all(choose_argmax(scores, valid) == 32)
    assert np.all(choose_closed_dp(scores, valid, 0.0, 0.0, device="cpu") == 32)
    valid[:, 32] = False
    assert np.all(choose_argmax(scores, valid) == 31)


def test_fast_raster_matches_original_pixel_centre_rule():
    polygons = [np.array([[1, 1], [9, 1], [9, 9], [1, 9]], float),
                np.array([[0.3, 0.4], [15.2, 4.1], [7.2, 14.9], [3.5, 5.1]], float),
                np.array([[-2, 4], [20, 2], [8, 18]], float)]
    for polygon in polygons:
        assert np.array_equal(rasterize(polygon, (20, 20)), rasterize_exact(polygon, (20, 20)))


def test_movement_threshold_matches_argmax_for_all_alpha():
    rng = np.random.default_rng(21)
    scores = rng.normal(size=(17, 65))
    valid = rng.random(size=scores.shape) > .15
    valid[:, 32] = True
    threshold = node_move_threshold(scores, valid)
    for alpha in (0.0, .02, .1, .5, 2.0):
        assert np.array_equal(threshold > alpha, choose_argmax(scores, valid, alpha) != 32)
