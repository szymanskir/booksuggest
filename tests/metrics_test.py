import pytest

from src.validation.metrics import precision, recall


@pytest.mark.parametrize("recommendations, ground_truth, expected", [
    ([1, 2, 3], [1, 2, 3], 1),
    ([1, 2, 3, 4], [1, 2, 5, 6], 0.5),
    ([1, 2, 3, 4], [5, 1, 2, 6], 0.5),
    ([], [1, 2], 0)
])
def test_precision(recommendations, ground_truth, expected):
    result = precision(recommendations, ground_truth)
    assert result == expected


@pytest.mark.parametrize("recommendations, ground_truth, expected", [
    ([1, 2, 3], [1, 2, 3], 1),
    ([1, 2], [1, 2, 3, 4], 0.5),
    ([1, 2], [4, 2, 1, 3], 0.5),
    ([1, 2, 3, 4], [], 0)
])
def test_recall(recommendations, ground_truth, expected):
    result = recall(recommendations, ground_truth)
    assert result == expected
