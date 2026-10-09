import numpy as np
import pytest
from sklearn.metrics import confusion_matrix
from engine import binary_confusion_counts


@pytest.mark.parametrize('threshold', [0., .5, 1.])
def test_batch_counts_equal_pooled_sklearn(threshold):
    rng = np.random.default_rng(42)
    pred = rng.choice([0., .3, .5, .8, 1.], (7, 16, 16))
    gt = rng.choice([0., .49, .5, 1.], pred.shape)
    actual = sum(binary_confusion_counts(pred[i:i+3], gt[i:i+3], threshold)
                 for i in range(0, len(pred), 3))
    expected = confusion_matrix((gt >= .5).ravel(), (pred >= threshold).ravel(), labels=[False, True])
    np.testing.assert_array_equal(actual, expected)


def test_single_class_and_shape_check():
    a = np.zeros((1, 4, 4))
    np.testing.assert_array_equal(binary_confusion_counts(a, a), [[16, 0], [0, 0]])
    np.testing.assert_array_equal(binary_confusion_counts(a+1, a), [[0, 16], [0, 0]])
    with pytest.raises(ValueError):
        binary_confusion_counts(a, a[0])
