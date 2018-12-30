import pandas as pd
import pytest
from scipy.sparse import coo_matrix
from numpy.testing import assert_array_equal

from os.path import dirname, join, realpath

from src.features.build_user_features import (
    build_interaction_matrix, InvalidUserDataError
)


CURRENT_PATH = dirname(realpath(__file__))


def test_build_interaction_matrix_exception():
    user_data = join(CURRENT_PATH,
                     'data/build_interaction_matrix-user_data-exception.csv')
    with pytest.raises(InvalidUserDataError):
        build_interaction_matrix(pd.read_csv(user_data))


@pytest.mark.parametrize("user_data, expected", [
    (join(CURRENT_PATH, 'data/build_interaction_matrix-user_data-1.csv'),
     coo_matrix([[1, 1, 0], [1, 0, 1]]))
])
def test_build_interaction_matrix(user_data, expected):
    result = build_interaction_matrix(pd.read_csv(user_data))
    assert_array_equal(result.toarray(), expected.toarray())
