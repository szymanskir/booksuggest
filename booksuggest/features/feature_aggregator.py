from abc import ABCMeta, abstractmethod
import numpy as np


class IFeatureAggregator(metaclass=ABCMeta):
    """Interface for feature aggregators responsible
    for aggregating feature vectors into one.
    """
    @abstractmethod
    def aggregate_features(self, features: np.array):
        pass


class MeanFeatureAggregator(IFeatureAggregator):
    def aggregate_features(self, features: np.array):
        return np.mean(features, axis=0)


class MedianFeatureAggregator(IFeatureAggregator):
    def aggregate_features(self, features: np.array):
        return np.median(features, axis=0)


class MinFeatureAggregator(IFeatureAggregator):
    def aggregate_features(self, features: np.array):
        return np.min(features, axis=0)


class MaxFeatureAggregator(IFeatureAggregator):
    def aggregate_features(self, features: np.array):
        return np.max(features, axis=0)


class MinMaxFeatureAggregator(IFeatureAggregator):
    def aggregate_features(self, features: np.array):
        return np.hstack((np.min(features, axis=0), np.max(features, axis=0)))


class InvalidFeatureAggregatorError(Exception):
    pass

class FeatureAggregatorFactory():
    @staticmethod
    def create(feature_aggregator_type : str):
        type_to_constructor_mapping = {
            'mean' : MeanFeatureAggregator,
            'median' : MedianFeatureAggregator,
            'min' : MinFeatureAggregator,
            'max' : MaxFeatureAggregator,
            'minmax' : MinMaxFeatureAggregator
        }

        if feature_aggregator_type not in list(type_to_constructor_mapping.keys()):
            raise InvalidFeatureAggregatorError


        return type_to_constructor_mapping[feature_aggregator_type]()