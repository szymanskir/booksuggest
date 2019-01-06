import pytest

from os.path import dirname, join, realpath

from src.evaluation.cb_evaluation import (
    calculate_single_score,
    read_similar_books
)


test_case_dir = join(dirname(realpath(__file__)), 'data')


@pytest.mark.parametrize("similar_books_filepath, expected", [
    (join(test_case_dir, 'read_similar_books-test_case-1.csv'),
        {1: [2, 3, 5], 10: [5, 3, 20]})
])
def test_read_similar_books(similar_books_filepath, expected):
    result = read_similar_books(similar_books_filepath)
    assert result == expected


@pytest.mark.parametrize("prediction_file, test_cases, expected", [
    (join(test_case_dir, 'calculate_single_score-prediction_file.csv'),
     {1: [2, 3, 5], 2: [10, 30], 3: [40, 41]},
     {'model': 'calculate_single_score-prediction_file.csv',
      'precision': 1,
      'recall': 1,
      'correct_hits': 5})
])
def test_calculate_single_score(
        prediction_file,
        test_cases,
        expected
):
    result = calculate_single_score(prediction_file, test_cases)
    assert result == expected
