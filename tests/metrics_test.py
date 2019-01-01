import pytest

from src.validation.metrics import (precision, precision_thresholded,
                                    recall, recall_thresholded)


@pytest.mark.parametrize("recommendations, ground_truth, expected", [
    ([1, 2, 3], [1, 2, 3], 1),
    ([1, 2, 3, 4], [1, 2, 5, 6], 0.5),
    ([1, 2, 3, 4], [5, 1, 2, 6], 0.5),
    ([], [1, 2], 0),
    ([1, 2, 3, 4], [1, 2], 0.5)
])
def test_precision(recommendations, ground_truth, expected):
    result = precision(recommendations, ground_truth)
    assert result == expected


@pytest.mark.parametrize("recommendations, ground_truth, expected", [
    ([1, 2, 3], [1, 2, 3], 1),
    ([1, 2], [1, 2, 3, 4], 0.5),
    ([1, 2], [4, 2, 1, 3], 0.5),
    ([1, 2, 3, 4], [], 0),
    ([1, 2], [1, 2], 1)
])
def test_recall(recommendations, ground_truth, expected):
    result = recall(recommendations, ground_truth)
    assert result == expected


@pytest.mark.parametrize("recommendations, ground_truth, threshold, expected", [
    ([(11, 5), (22, 4)], [11, 22], 4, 1),
    ([(11, 5), (22, 4)], [11, 22], 5, 1/2),
    ([(11, 4), (33, 4)], [11, 22], 4, 1/2),
    ([(11, 2), (33, 1)], [11, 22], 5, 0),
])
def test_precision_with_threshold(recommendations, ground_truth,
                                  threshold, expected):
    result = precision_thresholded(recommendations, ground_truth, threshold)
    assert result == expected


@pytest.mark.parametrize("recommendations, ground_truth, threshold, expected", [
    ([(11, 5), (22, 4), (33, 5)], [11, 22, 33], 4, 1),
    ([(11, 5), (22, 4), (33, 5)], [11, 22, 33], 5, 2/3),
    ([(11, 4), (33, 4), (44, 5)], [11, 22, 33], 4, 2/3),
    ([(11, 2), (33, 1), (44, 5)], [11, 22, 33], 5, 0),
])
def test_recall__thresholded(recommendations, ground_truth,
                             threshold, expected):
    result = recall_thresholded(recommendations, ground_truth, threshold)
    assert result == expected
