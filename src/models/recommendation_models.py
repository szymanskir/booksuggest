import pandas as pd

from abc import ABCMeta, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List
from sklearn.neighbors import NearestNeighbors
from os import environ


class IRecommendationModel(metaclass=ABCMeta):

    @abstractmethod
    def recommend(self, user_ratings: Dict[int, int]) -> List[int]:
        """Returns recommended books based on the users ratings.

        The user_ratings have a form of <book_id>: <rating>. The
        ratings are based on a scale from 1 to 5.

        Args:
            user_ratings: dictionary containg book ratings.

        Returns:
            list of recommended books.
        """
        pass


class DummyModel(IRecommendationModel):
    """Dummy recommendation model used for web app integration purposes.
    """

    def recommend(self, user_ratings: Dict[int, int]) -> List[int]:
        books: List[int] = list(user_ratings.keys())
        return books[0:6]


class ContentBasedRecommendationModel(IRecommendationModel):
    """Recommendation model using the text features.
    Later uses the cosine similarity in order to select
    the most similar books.

    Attributes:
        data: Data frame containing book data.
        content_analyzer: Component used for feature extraction from the data.
        filtering_component: Component used for calculating most similar books
        based on the features calculated by the content_analyzer.
    """

    def __init__(
            self,
            input_filepath: str,
            recommendation_count: int,
            content_analyzer
    ):
        """Initializes an instance of the TfIdfRecommendationModel class.

        Args:
            input_filepath: Filepath containing book data.
            recommendation_count: How many recommendations should.
            be returned for a single book.
        """
        self.data = pd.read_csv(input_filepath, index_col='book_id').dropna()
        self.content_analyzer = content_analyzer
        self.filtering_component = NearestNeighbors(
            n_neighbors=recommendation_count + 1,
            metric='cosine'
        )
        if environ['TEST_RUN'] == '1':
            self.data = self.data.head(100)

    def train(self):
        """Prepares tf_idf feature vectors.
        """

        descriptions = self.data['description']
        result = self.content_analyzer.fit_transform(descriptions)
        self.filtering_component.fit(result)

    def recommend(self, user_ratings: Dict[int, int]) -> Dict[int, float]:
        """ Based on the user input in form a dictionary containing


        The model makes use of tf-idf features calculated using book
        descriptions. The cosine metric is used in order to determine
        which books are similar.
        """
        selected_book_id = next(iter(user_ratings))

        try:
            descriptions = self.data['description']
            selected_book_description = descriptions.loc[selected_book_id]
        except KeyError:
            return dict()

        feature_vec = self.content_analyzer.transform(
            [selected_book_description]
        )
        distances, ids = self.filtering_component.kneighbors(feature_vec)
        recommendations = self.data['description'].index[ids.flatten()[1:]]

        return dict(zip(recommendations, distances.flatten()))
