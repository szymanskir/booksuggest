import pytest
from os.path import dirname, join, realpath
import pandas as pd

from src.validation.cf_to_read_evaluation import evaluate_to_read

test_case_dir = join(dirname(realpath(__file__)), 'data')


@pytest.mark.parametrize("predictions_filepath, to_read_filepath, expected", [
    (join(test_case_dir, "predictions-simple.csv"),
     join(test_case_dir, "to_read-simple.csv"), 0.5)
])
def test_to_read_evaluation(predictions_filepath, to_read_filepath, expected):
    predictions_df = pd.read_csv(predictions_filepath)
    to_read_df = pd.read_csv(to_read_filepath)
    result = evaluate_to_read(predictions_df, to_read_df)
    assert result == expected
