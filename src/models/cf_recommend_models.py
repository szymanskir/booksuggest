import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Dict

from surprise import SVD
from surprise import Reader
from surprise import Dataset


class ICfRecommendationModel(metaclass=ABCMeta):
    @abstractmethod
    def recommend(self, user_id: int) -> Dict[int, float]:
        """Recommends books for given user present in the training dataset.

        Args:
            user_id (int): Id of the user for who recommendation would be given.

        Returns:
            Dict[int, float]: Dictionary of ``book_id: predicted_rating`` key-value pairs.
        """
        pass


class DummyModel(ICfRecommendationModel):
    """Dummy recommendation model used for web app integration purposes.
    """

    def recommend(self, user_id: int) -> Dict[int, float]:
        return {book_id: 5.00 for book_id in range(1, 5)}


class SvdRecommendationModel(ICfRecommendationModel):
    """Recommendation model using the Singular Value Decomposition operation.

    Attributes:
        recommendation_count: How many recommendations should be returned for a single user.
        trainset: Dataset containing (user_id, book_id, rating) tuples.
    """

    def __init__(self, input_filepath: str, recommendation_count: int):
        """Initializes an instance of the SvdRecommendationModel class.

        Args:
            input_filepath: Filepath containing ratings data.
            recommendation_count: How many recommendations should be returned for a single user.
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

    def recommend(self, user_id: int) -> Dict[int, float]:
        """Recommends top n books for given user.
        """
        try:
            user_inner_id = self.trainset.to_inner_uid(user_id)
        except ValueError:
            return dict()

        read_book_ids = set(iid for iid, _ in self.trainset.ur[user_inner_id])
        unread_books_ids = set(self.trainset.all_items()) - read_book_ids

        to_predict = [(user_inner_id, item_inner_id, 0)
                      for item_inner_id in unread_books_ids]

        predictions = self.model.test(to_predict)
        top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[
            :self.recommendation_count]

        rec_books = {self.trainset.to_raw_iid(
            iid): est for _, iid, _, est, _ in top_n}

        return rec_books
