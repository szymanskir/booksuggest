import pytest

from os.path import dirname, join, realpath

from src.validation.cb_evaluation import read_similar_books


test_case_dir = join(dirname(realpath(__file__)), 'data')


@pytest.mark.parametrize("similar_books_filepath, expected", [
    (join(test_case_dir, 'read_similar_books-test_case-1.csv'),
        {1: [2, 3, 5], 10: [5, 3, 20]})
])
def test_read_similar_books(similar_books_filepath, expected):
    result = read_similar_books(similar_books_filepath)
    assert result == expected
