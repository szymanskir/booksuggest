import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest
from os.path import dirname, join, realpath
from unittest.mock import MagicMock, Mock
from src.models.content_analyzer import (
    ContentAnalyzerBuilder,
    EnsembledContentAnalyzer,
    InvalidBuilderConfigError,
    TagBasedContentAnalyzer,
    TextBasedContentAnalyzer,
    TextAndTagBasedContentAnalyzer
)

current_path = dirname(realpath(__file__))
book_data_file = join(
    current_path, "data/text_and_tag_based_content_analyzer-book_data-1.csv"
)
tag_features_file = join(
    current_path, "data/text_and_tag_based_content_analyzer-tag_features-1.csv"
)
book_data = pd.read_csv(book_data_file, index_col='book_id')
tag_features = pd.read_csv(tag_features_file, index_col='book_id')


def create_mock_text_feature_extractor():
    text_feature_extractor = Mock()
    text_feature_extractor.fit_transform = MagicMock(
        return_value=[[0.1, 0.3, 0.5], [0.3, 0.1, 0.2]]
    )
    text_feature_extractor.transform = MagicMock(
        return_value=[[0.1, 0.3, 0.5]]
    )

    return text_feature_extractor


text_based_test_case = (
    book_data,
    TextBasedContentAnalyzer(create_mock_text_feature_extractor()),
    np.array([[0.1, 0.3, 0.5], [0.3, 0.1, 0.2]]),
    np.array([[0.1, 0.3, 0.5]])
)

tag_based_test_case = (
    book_data,
    TagBasedContentAnalyzer(tag_features),
    np.array([[0.1, 0.3, 0.5], [0.3, 0.1, 0.2]]),
    np.array([[0.1, 0.3, 0.5]])
)

ensemble_test_case = (
    book_data,
    EnsembledContentAnalyzer([
        TextBasedContentAnalyzer(create_mock_text_feature_extractor()),
        TagBasedContentAnalyzer(tag_features),
    ]),
    np.array([[0.1, 0.3, 0.5, 0.1, 0.3, 0.5], [0.3, 0.1, 0.2, 0.3, 0.1, 0.2]]),
    np.array([[0.1, 0.3, 0.5, 1, 0.3, 0.5]])
)


@pytest.mark.parametrize(
    "book_data, content_analyzer, expected_features, expected_vector", [
        text_based_test_case,
        tag_based_test_case
    ])
def test_content_analyzers(
        book_data,
        content_analyzer,
        expected_features,
        expected_vector
):
    features = content_analyzer.build_features(book_data)
    feature_vec = content_analyzer.get_feature_vector(1)

    assert_array_equal(features, expected_features)
    assert_array_equal(feature_vec, expected_vector)


@pytest.mark.parametrize(
    "name, ngrams, tag_features, expected", [
        ('blank_model', 3, None, 'Invalid model name blank_model'),
        ('tf-idf', None, None, ''),
        ('tf-idf', -1,  None, ''),
        ('tag', -1,  None, ''),
        ('tf-idf-tag', -1, None, ''),
        ('tf-idf-tag', -1, tag_features, ''),
    ])
def test_content_analyzer_builder_config_validation(
        name,
        ngrams,
        tag_features,
        expected
):
    with pytest.raises(InvalidBuilderConfigError) as excinfo:
        ContentAnalyzerBuilder(
            name, ngrams, tag_features
        )
    assert str(excinfo.value) == expected


@pytest.mark.parametrize(
    "name, ngrams, tag_features, expected", [
        ('tf-idf', 3, None, TextBasedContentAnalyzer),
        ('count', 3, None, TextBasedContentAnalyzer),
        ('tag', None, tag_features, TagBasedContentAnalyzer),
        ('tf-idf-tag', 20, tag_features, TextAndTagBasedContentAnalyzer),
        ('count-tag', 20, tag_features, TextAndTagBasedContentAnalyzer),
    ])
def test_content_analyzer_builder(
        name,
        ngrams,
        tag_features,
        expected
):
    builder = ContentAnalyzerBuilder(
        name, ngrams, tag_features
    )
    content_analyzer = builder.build_content_analyzer()
    assert isinstance(content_analyzer, expected)
