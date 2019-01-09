import pytest
from os.path import dirname, join, realpath
import pandas as pd

from booksuggest.evaluation.cf_effectiveness_evaluation import evaluate_on_predictions

test_case_dir = join(dirname(realpath(__file__)), 'data')


@pytest.mark.parametrize("predictions_filepath, to_read_filepath, expected", [
    (join(test_case_dir, "predictions-simple.csv"),
     join(test_case_dir, "to_read-simple.csv"),
     ((1 + 0.5 + 0.5)/3, (1 + 0.5 + 0.25)/3))
])
def test_to_read_evaluation(predictions_filepath, to_read_filepath, expected):
    predictions_df = pd.read_csv(predictions_filepath)
    to_read_df = pd.read_csv(to_read_filepath)
    result = evaluate_on_predictions(predictions_df, to_read_df, 4, 20)
    assert result == expected
