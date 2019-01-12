import pytest
from os.path import dirname, join, realpath
import pandas as pd

from booksuggest.evaluation.cf_effectiveness_evaluation import (
    evaluate_binary_truth, evaluate_scaled_truth,
    _get_binary_relevance_lists, _get_scaled_relevance_lists)
from booksuggest.evaluation.metrics import ndcg

test_case_dir = join(dirname(realpath(__file__)), 'data')


@pytest.mark.parametrize("predictions_filepath, to_read_filepath, expected", [
    (join(test_case_dir, "predictions-simple.csv"),
     join(test_case_dir, "to_read-simple.csv"),
     ((1 + 0.5 + 0.5)/3, (1 + 0.5 + 0.25)/3, 0.7420981285103055))
])
def test_to_read_evaluation(predictions_filepath, to_read_filepath, expected):
    predictions_df = pd.read_csv(predictions_filepath)
    to_read_df = pd.read_csv(to_read_filepath)
    result = evaluate_binary_truth(predictions_df, to_read_df, 4, 20)
    assert result == expected


@pytest.mark.parametrize("predictions_filepath, testset_filepath, expected", [
    (join(test_case_dir, "predictions-simple.csv"),
     join(test_case_dir, "testset-simple.csv"),
     ((1 + 0.5 + 0.5)/3, (1 + 0.5 + 0.25)/3, 0.7212211872125157))
])
def test_testset_evaluation(predictions_filepath, testset_filepath, expected):
    predictions_df = pd.read_csv(predictions_filepath)
    testset_df = pd.read_csv(testset_filepath)
    result = evaluate_scaled_truth(predictions_df, testset_df, 4, 20)
    assert result == expected


@pytest.mark.parametrize("recommendations, ground_truth, expected", [
    ([11, 22], [11, 22], 1),
    ([11, 22], [11, 22, 33], 1),
    ([11, 55], [11, 22], 0.6131471927654584),
    ([11, 22], [55, 66], 0),
])
def test_relevance_binary(recommendations, ground_truth, expected):
    rec, user = _get_binary_relevance_lists(recommendations, ground_truth)
    assert ndcg(rec, user) == expected


@pytest.mark.parametrize("recommendations, ground_truth, expected", [
    ([(11, 5), (11, 4)], [(11, 5), (11, 4)], 1),
    ([(11, 5), (11, 4)], [(11, 5), (11, 4), (33, 5)], 0.9226294385530918),
])
def test_relevance(recommendations, ground_truth, expected):
    rec, user = _get_scaled_relevance_lists(recommendations, ground_truth)
    assert ndcg(rec, user) == expected
