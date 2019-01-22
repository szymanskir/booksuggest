import pandas as pd
import pytest

from os.path import dirname, join, realpath

import booksuggest.features.build_tag_features as btf

test_case_dir = join(dirname(realpath(__file__)), 'data', 'build_tag_features')


@pytest.mark.parametrize("book_tags, tags, expected", [(
    join(test_case_dir, "build_all_tag_features-book_tags.csv"),
    join(test_case_dir, "build_all_tag_features-tags.csv"),
    [[0.8, 0.2, 0], [0.5, 0, 0.5]]
)])
def test_build_all_tag_features(book_tags, tags, expected):
    tag_features_list = btf.build_all_tag_features(pd.read_csv(book_tags),
                                                   pd.read_csv(tags))
    assert tag_features_list.values.tolist() == expected
    assert tag_features_list.index.tolist() == [2, 3]


@pytest.mark.parametrize("book_tags, tags, expected", [(
    join(test_case_dir, "build_tag_features-book_tags-simple.csv"),
    join(test_case_dir, "build_tag_features-tags-simple.csv"),
    [0.8, 0.1, 0.1, 0, 0]
)])
def test_build_tag_features(book_tags, tags, expected):
    tag_features = btf.build_tag_features(pd.read_csv(book_tags),
                                          pd.read_csv(tags))
    assert tag_features == expected


validate_book_tags_simple = join(
    test_case_dir, "build_tag_features-book_tags-simple.csv")

validate_book_tags_non_unique_book = join(
    test_case_dir, "validate_book_tags_data-non_unique_book.csv")

validate_book_tags_invalid_book = join(
    test_case_dir, "validate_book_tags_data-invalid_book.csv")


@pytest.mark.parametrize("book_tags, expected", [
    (validate_book_tags_simple, True),
    (validate_book_tags_non_unique_book, False),
    (validate_book_tags_invalid_book, False)
])
def test_validate_book_tags_data(book_tags, expected):
    result = btf.validate_book_tags_data(pd.read_csv(book_tags))
    assert result == expected


validate_tags_correct = join(
    test_case_dir, 'validate_tags_data-correct.csv')

validate_tags_incorrect = join(
    test_case_dir, 'validate_tags_data-incorrect.csv')


@pytest.mark.parametrize("tags, expected", [
    (validate_tags_correct, True),
    (validate_tags_incorrect, False),
])
def test_validate_tags_data(tags, expected):
    result = btf.validate_tags_data(pd.read_csv(tags))
    assert result == expected
