import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class ICbRecommendationModel(metaclass=ABCMeta):
    @abstractmethod
    def recommend(self, book_id: int) -> Dict[int, float]:
        """Recommends books similar to given book.

        Args:
            book_id (int): Id of the book for which recommendation would be given.

        Returns:
            Dict[int, float]: Dictionary of ``similar_book_id: distance_between_book_and_similar_book`` key-value pairs.
        """
        pass


class DummyModel(ICbRecommendationModel):
    """Dummy recommendation model used for web app integration purposes.
    """

    def recommend(self, book_id: int) -> Dict[int, float]:
        return {book_id: 0.00}


class ContentBasedRecommendationModel(ICbRecommendationModel):
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
            recommendation_count: How many recommendations should be returned for a single book.
        """
        self.data = pd.read_csv(input_filepath, index_col='book_id').dropna()
        self.content_analyzer = content_analyzer
        self.filtering_component = NearestNeighbors(
            n_neighbors=recommendation_count + 1,
            metric='cosine'
        )

    def train(self):
        """Prepares tf_idf feature vectors.
        """

        descriptions = self.data['description']
        result = self.content_analyzer.fit_transform(descriptions)
        self.filtering_component.fit(result)

    def recommend(self, book_id: int) -> Dict[int, float]:
        """ Based on the user input in form a dictionary containing

        The model makes use of tf-idf features calculated using book
        descriptions. The cosine metric is used in order to determine
        which books are similar.
        """
        try:
            descriptions = self.data['description']
            selected_book_description = descriptions.loc[book_id]
        except KeyError:
            return dict()

        feature_vec = self.content_analyzer.transform(
            [selected_book_description]
        )
        distances, ids = self.filtering_component.kneighbors(feature_vec)
        recommendations = self.data['description'].index[ids.flatten()[1:]]

        return dict(zip(recommendations, distances.flatten()))
