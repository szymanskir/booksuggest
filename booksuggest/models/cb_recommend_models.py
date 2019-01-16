"""Recommendation models using content based methods.
"""

from abc import ABCMeta, abstractmethod
from typing import Dict
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from .content_analyzer import IContentAnalyzer
from .model_exceptions import UntrainedModelError


class ICbRecommendationModel(metaclass=ABCMeta):
    """Interface for content based recommendation models.
    """

    def __init__(self):
        self._book_data = None

    def _is_trained(self):
        if self._book_data is None:
            raise UntrainedModelError()

    @abstractmethod
    def train(self, book_data: pd.DataFrame):
        """Trains the content based recommendation model.

        Args:
            book_data:
                Data frame containing book data, composed of the following
                columns: book_id, authors, original_publication_year,
                original_title, title, isbn13, description.

        """

    @abstractmethod
    def recommend(self, book_id: int, rec_count: int) -> Dict[int, float]:
        """Recommends books similar to the given book.

        Args:
            book_id (int):
                Id of the book for which recommendations would be given.
            rec_int(int):
                How many recomendations to return.

        Returns:
            Dict[int, float]:
                Dictionary composed of book ids and their distances
                from the original book key value pairs.
        """


class ContentBasedRecommendationModel(ICbRecommendationModel):
    """Recommendation model using text features.
    Later uses the cosine similarity in order to select
    the most similar books.

    Attributes:
        content_analyzer: Component used for feature extraction from the data.
        filtering_component: Component used for calculating most similar books
            based on the features calculated by the content_analyzer.
    """

    def __init__(
            self,
            content_analyzer: IContentAnalyzer,
            recommendation_count: int,
    ):
        """Initializes an instance of the ContentBasedRecommendationModel class.

        Args:
            input_filepath: Filepath containing book data.
            recommendation_count:
                How many recommendations should be returned for a single book.
        """
        super().__init__()
        self.content_analyzer = content_analyzer
        self.filtering_component = NearestNeighbors(
            n_neighbors=recommendation_count + 1,
            metric='cosine'
        )

    def train(self, book_data: pd.DataFrame):
        """Prepares feature vectors.
        """
        self._book_data = book_data
        result = self.content_analyzer.build_features(self._book_data)
        self.filtering_component.fit(result)

    def recommend(
            self,
            book_id: int,
            rec_count: int = None
    ) -> Dict[int, float]:
        """ Based on the user input in form a dictionary containing

        The model makes use of tf-idf features calculated using book
        descriptions. The cosine metric is used in order to determine
        which books are similar.
        """
        self._is_trained()
        rec_count = rec_count if rec_count else self.recommendation_count
        try:
            feature_vec = self.content_analyzer.get_feature_vector(book_id)
        except KeyError:
            return dict()

        distances, ids = self.filtering_component.kneighbors(
            feature_vec, rec_count)
        recommendations = self._book_data.index[
            ids.flatten()[1:]]

        return dict(zip(recommendations, distances.flatten()[1:]))
