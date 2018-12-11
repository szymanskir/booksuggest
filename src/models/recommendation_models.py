import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List
from scipy.spatial import distance


class IRecommendationModel(metaclass=ABCMeta):

    @abstractmethod
    def recommend(self, user_ratings: Dict[int, int]) -> List[int]:
        """ Returns recommended books based on the users ratings.

        The user_ratings have a form of <book_id>: <rating>. The
        ratings are based on a scale from 1 to 5.
        """
        pass


class DummyModel(IRecommendationModel):
    def recommend(self, user_ratings: Dict[int, int]) -> List[int]:
        books: List[int] = list(user_ratings.keys())
        return books[0:6]


class TfIdfRecommendationModel(IRecommendationModel):
    """Recommendation model using the tf-idf method for
    feature extraction. Later uses the cosine similarity
    in order to select the most similar books
    """

    def __init__(self, input_filepath: str, recommendation_count):
        self.data = pd.read_csv(input_filepath, index_col='book_id')
        self.recommendation_count = recommendation_count

    def train(self):
        """Prepares tf_idf feature vectors
        """

        # TODO fix initial data cleaning so no NAs are present
        descriptions = self.data['description'].dropna()
        vectorizer = TfidfVectorizer()
        result = vectorizer.fit_transform(descriptions)
        feature_vectors = [result[i, :] for i in range(result.shape[0])]
        self.features = pd.DataFrame(
            data={'feature_vector': feature_vectors},
            index=descriptions.index
        )

    def recommend(self, user_ratings: Dict[int, int]) -> List[int]:
        """ Based on the user input in form a dictionary containing
        book ids and their ratings recommendations are determined.

        The model makes use of tf-idf features calculated using book
        descriptions. The cosine metric is used in order to determine
        which books are similar.
        """
        selected_book_id = next(iter(user_ratings))
        selected_book_features = self.features.loc[
            selected_book_id, 'feature_vector'
        ].toarray()

        remove_selected_book = self.features.index.isin([selected_book_id])
        candidate_features = self.features[~remove_selected_book]
        distances = candidate_features.apply(
            lambda x: distance.cosine(
                x.loc['feature_vector'].toarray(),
                selected_book_features
            ),
            axis=1
        )

        recommendation_ids = distances.sort_values().index.tolist()
        recommendation_ids = recommendation_ids[:self.recommendation_count]
        return dict(zip(
            recommendation_ids, distances.loc[recommendation_ids].tolist()
        ))
