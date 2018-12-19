import pandas as pd

from abc import ABCMeta, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple
from sklearn.neighbors import NearestNeighbors

from surprise.model_selection import KFold
from surprise import SVD
from surprise import Reader
from surprise import Dataset


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


class TfIdfRecommendationModel(IRecommendationModel):
    """Recommendation model using the tf-idf method for
    feature extraction. Later uses the cosine similarity
    in order to select the most similar books.

    Attributes:
        data: data frame containing book data.
        content_analyzer: component used for feature extraction from the data.
        filtering_component: component used for calculating most similar books
        based on the features calculated by the content_analyzer.
    """

    def __init__(self, input_filepath: str, recommendation_count):
        """Initializes an instance of the TfIdfRecommendationModel class.

        Args:
            input_filepath: filepath containing book data.
            recommendation_count: how many recommendations should.
            be returned for a single book.
        """
        self.data = pd.read_csv(input_filepath, index_col='book_id')
        self.content_analyzer = TfidfVectorizer()
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

    def recommend(self, user_ratings: Dict[int, int]) -> Dict[int, float]:
        """ Based on the user input in form a dictionary containing
        book ids and their ratings recommendations are determined.

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


class SvdRecommendationModel(IRecommendationModel):
    """Recommendation model using the Singular Value Decomposition operation.

    Attributes:
        data: Data frame containing ratings data.
    """

    def __init__(self, input_filepath: str, recommendation_count: int):
        """Initializes an instance of the SvdRecommendationModel class.

        Args:
            input_filepath: filepath containing ratings data.
            recommendation_count: how many recommendations should be returned for a single user.
        """
        self.recommendation_count = recommendation_count

        ratings_df = pd.read_csv(input_filepath)
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(
            ratings_df[['user_id', 'book_id', 'rating']], reader)
        self.trainset = dataset.build_full_trainset()

    def train(self):
        """Prepares user and items vectors.
        """
        algo = SVD()
        algo.fit(self.trainset)
        self.model = algo

    def recommend(self, user_id: int) -> List[Tuple[int, float]]:
        """Recommends top n books for given user.
        """
        try:
            user_inner_id = self.trainset.to_inner_uid(user_id)
        except ValueError:
            return list()

        read_book_ids = set(iid for iid, _ in self.trainset.ur[user_inner_id])
        unread_books_ids = set(self.trainset.all_items()) - read_book_ids

        to_predict = list()
        for item_inner_id in unread_books_ids:
            to_predict.append((user_inner_id, item_inner_id, 0))

        predictions = self.model.test(to_predict)
        top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[
            :self.recommendation_count]

        rec_books = dict()
        for _, iid, _, est, _ in top_n:
            rec_books[self.trainset.to_raw_iid(iid)] = est

        return rec_books
