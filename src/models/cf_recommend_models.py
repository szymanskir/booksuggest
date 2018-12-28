import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple
import logging

from surprise import SVD
from surprise import Reader, Dataset, Prediction


class ICfRecommendationModel(metaclass=ABCMeta):
    @abstractmethod
    def recommend(self, user_id: int) -> Dict[int, float]:
        """Recommends books for given user present in the training dataset.

        Args:
            user_id (int): Id of the user for who recommendations would be given.

        Returns:
            Dict[int, float]: Dictionary of ``book_id: predicted_rating`` key-value pairs.
        """
        pass

    @abstractmethod
    def test(self, ratings: List[Tuple[int, int, float]]) -> List[Prediction]:
        """Tests model on the given dataset

        Args:
            ratings (List[Tuple[int, int, float]]): list of (user_id, item_id, rating) tuples as a ground-truth

        Returns:
            List[Prediction]: List of (user_id, item_id, true_rating, estimated_rating) for the given dataset
        """
        pass


class DummyModel(ICfRecommendationModel):
    """Dummy recommendation model used for web app integration purposes.
    """

    def recommend(self, user_id: int) -> Dict[int, float]:
        return {book_id: 5.00 for book_id in range(1, 5)}

    def test(self, ratings: List[Tuple[int, int, float]]) -> List[Prediction]:
        ui, ii, rating = ratings[0]
        return [Prediction(ui, ii, rating, 0, {})]


class SurpriseBasedModel(ICfRecommendationModel):
    def __init__(self, input_filepath: str, recommendation_count: int):
        """Initializes an instance of the SurpriseBasedModel class.

        Args:
            input_filepath: Filepath containing ratings data.
            recommendation_count: How many recommendations should be returned for a single user.
        """
        self.recommendation_count = recommendation_count
        self.trainset = self._read_trainset(input_filepath)

    @staticmethod
    def _read_trainset(input_filepath: str):
        ratings_df = pd.read_csv(input_filepath)
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(
            ratings_df[['user_id', 'book_id', 'rating']], reader)
        return dataset.build_full_trainset()

    def test(self, ratings: List[Tuple[int, int, float]]) -> List[Prediction]:
        ratings_inner = list()
        for ui, ii, rating in ratings:
            try:
                ratings_inner.append((self.trainset.to_inner_uid(
                    ui), self.trainset.to_inner_iid(ii), rating))
            except ValueError:
                logging.debug(f"Unknown user {ui} or item {ii}")

        return self.model.test(ratings_inner)


class SvdRecommendationModel(SurpriseBasedModel):
    """Recommendation model using the Singular Value Decomposition operation.
    """

    def train(self):
        """Prepares user and items vectors.
        """
        algo = SVD()
        algo.fit(self.trainset)
        self.model = algo

    def recommend(self, user_id: int) -> Dict[int, float]:
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
