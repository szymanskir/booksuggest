import pytest
import numpy as np

from numpy.testing import assert_array_equal
from os.path import dirname, join, realpath

from booksuggest.features.feature_aggregator import (
    MeanFeatureAggregator,
    MedianFeatureAggregator,
    MinFeatureAggregator,
    MaxFeatureAggregator,
    MinMaxFeatureAggregator
)


test_case_dir = join(dirname(realpath(__file__)), 'data')


@pytest.mark.parametrize("features, expected", [
    (np.array([[1, 2, 3], [3, 4, 3]],), np.array([2, 3, 3]))
])
def test_mean_feature_aggregation(features, expected):
    aggregator = MeanFeatureAggregator()
    result = aggregator.aggregate_features(features)
    assert_array_equal(result, expected)



@pytest.mark.parametrize("features, expected", [
    (np.array([[1, 2, 3], [3, 4, 3],[4, 2, 8]]), np.array([3, 2, 3]))
])
def test_median_feature_aggregation(features, expected):
    aggregator = MedianFeatureAggregator()
    result = aggregator.aggregate_features(features)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("features, expected", [
    (np.array([[1, 2, 3], [3, 4, 3],[4, 2, 8]]), np.array([1, 2, 3]))
])
def test_min_feature_aggregation(features, expected):
    aggregator = MinFeatureAggregator()
    result = aggregator.aggregate_features(features)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("features, expected", [
    (np.array([[1, 2, 3], [3, 4, 3],[4, 2, 8]]), np.array([4, 4, 8]))
])
def test_max_feature_aggregation(features, expected):
    aggregator = MaxFeatureAggregator()
    result = aggregator.aggregate_features(features)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("features, expected", [
    (np.array([[1, 2, 3], [3, 4, 3],[4, 2, 8]]), np.array([1, 2, 3, 4, 4, 8]))
])
def test_minmax_feature_aggregation(features, expected):
    aggregator = MinMaxFeatureAggregator()
    result = aggregator.aggregate_features(features)
    assert_array_equal(result, expected)