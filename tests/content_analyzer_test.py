import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest
from os.path import dirname, join, realpath
from unittest.mock import MagicMock, Mock
from scipy.sparse import coo_matrix
from src.models.content_analyzer import (
    TextBasedContentAnalyzer,
    TextAndTagBasedContentAnalyzer
)

current_path = dirname(realpath(__file__))
book_data = join(
    current_path, "data/text_and_tag_based_content_analyzer-book_data-1.csv"
)
tag_features = join(
    current_path, "data/text_and_tag_based_content_analyzer-tag_features-1.csv"
)


@pytest.mark.parametrize(
    "book_data, tag_features, expected_features, expected_vector", [
        (book_data,
         tag_features,
         np.array([[1, 2, 3, 0.1, 0.3, 0.5], [4, 5, 6, 0.3, 0.1, 0.2]]),
         np.array([[1, 2, 3, 0.1, 0.3, 0.5]]))
    ])
def test_text_and_tag_based_content_analyzer(
        book_data,
        tag_features,
        expected_features,
        expected_vector
):
    text_content_analyzer = TextBasedContentAnalyzer(
        pd.read_csv(book_data, index_col='book_id'),
        Mock()
    )
    text_content_analyzer.build_features = MagicMock(
        return_value=coo_matrix([[1, 2, 3], [4, 5, 6]])
    )
    text_content_analyzer.get_feature_vector = MagicMock(
        return_value=coo_matrix([1, 2, 3])
    )

    text_and_tag_based_content_analyzer = TextAndTagBasedContentAnalyzer(
        text_content_analyzer=text_content_analyzer,
        tag_features=pd.read_csv(tag_features, index_col='book_id')
    )

    features = text_and_tag_based_content_analyzer.build_features()
    feature_vec = text_and_tag_based_content_analyzer.get_feature_vector(1)

    assert_array_equal(features.toarray(), expected_features)
    assert_array_equal(feature_vec.toarray(), expected_vector)
