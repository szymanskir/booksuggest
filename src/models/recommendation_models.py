import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Callable, Dict, List, Tuple
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from statistics import mean

from ..validation.metrics import precision, recall


class IRecommendationModel(metaclass=ABCMeta):

    @abstractmethod
    def recommend(self, user_ratings: Dict[int, int]) -> List[int]:
        """ Returns recommended books based on the users ratings.

        The user_ratings have a form of <book_id>: <rating>. The
        ratings are based on a scale from 1 to 5.
        """
        pass


class IContentBasedRecommendationModelValidation(IRecommendationModel):
    def _score_for_given_metric(
            self,
            results: Dict[int, List[int]],
            test_cases: Dict[int, List[int]],
            metric: Callable[[List[int], List[int]], int]
    ) -> int:
        """Calculates the average score for a given metric.
        """
        score = [metric(results[key], test_cases[key])
                 for key in test_cases.keys()]

        return mean(score)

    def score(self, test_cases: Dict[int, List[int]]) -> Tuple[int, int]:
        """Calculates the precision and recall score of the model
        """
        results = {key: list(self.recommend({key: None}).keys())
                   for key in test_cases.keys()}
        precision_score = self._score_for_given_metric(
            results, test_cases, precision
        )
        recall_score = self._score_for_given_metric(
            results, test_cases, recall
        )

        return (precision_score, recall_score)


class DummyModel(IRecommendationModel):
    def recommend(self, user_ratings: Dict[int, int]) -> List[int]:
        books: List[int] = list(user_ratings.keys())
        return books[0:6]


class TfIdfRecommendationModel(IContentBasedRecommendationModelValidation):
    """Recommendation model using the tf-idf method for
    feature extraction. Later uses the cosine similarity
    in order to select the most similar books
    """

    def __init__(self, input_filepath: str, recommendation_count):
        self.data = pd.read_csv(input_filepath, index_col='book_id')
        self.content_analyzer = TfidfVectorizer()
        self.filtering_component = NearestNeighbors(
            n_neighbors = recommendation_count + 1,
            metric = 'cosine'
        )

    def train(self):
        """Prepares tf_idf feature vectors
        """

        descriptions = self.data['description']
        result = self.content_analyzer.fit_transform(descriptions)
        self.filtering_component.fit(result)

    def recommend(self, user_ratings: Dict[int, int]) -> Dict[int, float]:
        """ Based on the user input in form a dictionary containing
        book ids and their ratings recommendations are determined.

        The model makes use of tf-idf features calculated using book
        descriptions. The cosine metric is used in order to determine
        which books are similar.
        """
        selected_book_id = next(iter(user_ratings))

        try:
            selected_book_description = self.data['description'].loc[selected_book_id]
        except KeyError:
            return dict()

        feature_vec = self.content_analyzer.transform([selected_book_description])
        distances, ids = self.filtering_component.kneighbors(feature_vec)
        recommendations = self.data['description'].index[ids.flatten()[1:]]

        return dict(zip(recommendations, distances.flatten()))
