import scipy.sparse
import pandas as pd

from abc import ABCMeta, abstractmethod
from typing import Dict, List

from sklearn.neighbors import NearestNeighbors


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
        data: Data frame containing book data.
        content_analyzer: Component used for feature extraction from the data.
        filtering_component: Component used for calculating most similar books
            based on the features calculated by the content_analyzer.
    """

    def __init__(
            self,
            input_filepath: str,
            recommendation_count: int,
            content_analyzer,
            tag_features=None
    ):
        """Initializes an instance of the ContentBasedRecommendationModel class.

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
        self.tag_features = pd.read_csv(tag_features) if tag_features is not None else None

    def train(self):
        """Prepares feature vectors.
        """

        descriptions = self.data['description']
        result = self.content_analyzer.fit_transform(descriptions)

        if self.tag_features is not None:
            result = scipy.sparse.hstack((
                result,
                self.tag_features.loc[descriptions.index].iloc[:, 1:]
            ))

        self.filtering_component.fit(result)

    def _get_feature_vector(self, book_id: int) -> List[float]:
        descriptions = self.data['description']
        book_description = descriptions.loc[book_id]

        text_features = self.content_analyzer.transform(
            [book_description])
        tag_features = self.tag_features.loc[
            book_id] if self.tag_features is not None else None

        if self.tag_features is None:
            feature_vec = text_features
        else:
            feature_vec = scipy.sparse.hstack((
                text_features, tag_features.values[1:]
            ))

        return feature_vec

    def recommend(self, book_id: int) -> Dict[int, float]:
        """ Based on the user input in form a dictionary containing

        The model makes use of tf-idf features calculated using book
        descriptions. The cosine metric is used in order to determine
        which books are similar.
        """
        try:
            feature_vec = self._get_feature_vector(book_id)
        except KeyError:
            return dict()

        distances, ids = self.filtering_component.kneighbors(feature_vec)
        recommendations = self.data['description'].index[ids.flatten()[1:]]

        return dict(zip(recommendations, distances.flatten()))
