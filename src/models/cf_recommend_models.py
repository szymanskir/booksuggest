import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Iterable
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
        """Tests _algo on the given dataset

        Args:
            ratings (List[Tuple[int, int, float]]): list of (user_id, item_id, rating) tuples as a ground-truth

        Returns:
            List[Prediction]: List of (user_id, item_id, true_rating, estimated_rating) for the given dataset
        """
        pass


class DummyModel(ICfRecommendationModel):
    """Dummy recommendation _algo used for web app integration purposes.
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
        self._recommendation_count = recommendation_count
        self._trainset = self._read_trainset(input_filepath)

    @staticmethod
    def _read_trainset(input_filepath: str):
        ratings_df = pd.read_csv(input_filepath)
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(
            ratings_df[['user_id', 'book_id', 'rating']], reader)
        return dataset.build_full_trainset()

    @property
    def recommendation_count(self):
        return self._recommendation_count

    def test(self, ratings: List[Tuple[int, int, float]]) -> List[Prediction]:
        """Tests the model on the given ground-truth dataset.

        Args:
            ratings (List[Tuple[int, int, float]]): `(user_id, book_id, true_rating)` tuples

        Returns:
            List[Prediction]: List of predictions with true and estimated ratings. 
        """
        return self._algo.test(ratings)

    def recommend(self, user_id: int) -> Dict[int, float]:
        """Returns a top `recommendation_count` recommendations for specific user.

        Args:
            user_id (int): Id of the user.

        Returns:
            Dict[int, float]: `book_id: estimated_rating` pairs
        """
        try:
            user_inner_id = self._trainset.to_inner_uid(user_id)
        except ValueError:
            return dict()

        to_predict = [x for x in self._generate_antitest(user_inner_id)]
        predictions = self._algo.test(to_predict)

        top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[
            :self._recommendation_count]
        rec_books = {iid: est for _, iid, _, est, _ in top_n}

        return rec_books

    def generate_antitest_set(self) -> Iterable[Tuple[int, int, float]]:
        """Yields a list of ratings which are not already present in the trainset.

        All the ratings where user is known and item is know, but rating for (user, item) is not present in the trainset.

        Yields:
            Iterable[Tuple[int, int, float]]: A list of tuples (uid, iid, global_mean)
        """
        for uiid in self._trainset.all_users():
            yield from self._generate_antitest(uiid)

    def _generate_antitest(self, user_inner_id: int):
        fill = self._trainset.global_mean
        user_id = self._trainset.to_raw_uid(user_inner_id)
        user_items = set([j for (j, _) in self._trainset.ur[user_inner_id]])
        yield from [(user_id, self._trainset.to_raw_iid(i), fill) for i in self._trainset.all_items() if i not in user_items]


class SvdRecommendationModel(SurpriseBasedModel):
    """Recommendation _algo using the Singular Value Decomposition operation.
    """

    def train(self):
        """Prepares user and items vectors.
        """
        algo = SVD()
        algo.fit(self._trainset)
        self._algo = algo
