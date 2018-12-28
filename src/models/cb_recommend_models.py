from abc import ABCMeta, abstractmethod
from typing import Dict

from sklearn.neighbors import NearestNeighbors

from .content_analyzer import IContentAnalyzer


class ICbRecommendationModel(metaclass=ABCMeta):
    @abstractmethod
    def recommend(self, book_id: int) -> Dict[int, float]:
        """Recommends books similar to the given book.

        Args:
            book_id (int): Id of the book for which recommendations would be given.

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
            recommendation_count: How many recommendations should be returned for a single book.
        """
        self.content_analyzer = content_analyzer
        self.filtering_component = NearestNeighbors(
            n_neighbors=recommendation_count + 1,
            metric='cosine'
        )

    def train(self):
        """Prepares feature vectors.
        """
        result = self.content_analyzer.build_features()
        self.filtering_component.fit(result)

    def recommend(self, book_id: int) -> Dict[int, float]:
        """ Based on the user input in form a dictionary containing

        The model makes use of tf-idf features calculated using book
        descriptions. The cosine metric is used in order to determine
        which books are similar.
        """
        try:
            feature_vec = self.content_analyzer.get_feature_vector(book_id)
        except KeyError:
            return dict()

        distances, ids = self.filtering_component.kneighbors(feature_vec)
        recommendations = self.content_analyzer.book_data.index[
            ids.flatten()[1:]]

        return dict(zip(recommendations, distances.flatten()[1:]))
