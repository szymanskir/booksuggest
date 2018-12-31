from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Iterable

import pandas as pd

from surprise import SVD, SlopeOne, KNNBaseline
from surprise import Reader, Dataset, Prediction


class UntrainedModelError(Exception):
    pass


class ICfRecommendationModel(metaclass=ABCMeta):
    @abstractmethod
    def recommend(
            self,
            user_id: int,
            recommendations_count: int = 10
    ) -> Dict[int, float]:
        """Returns a top `recommendation_count` recommendations for specific user.

        Args:
            user_id (int): Id of the user.
            recommendations_count (int, optional):
                Defaults to 10. Specifies how many recommendations to return.

        Raises:
            UntrainedModelError:
                Raised when method is used before model is trained.

        Returns:
            Dict[int, float]: `book_id: estimated_rating` pairs
        """

    @abstractmethod
    def test(self, ratings: List[Tuple[int, int, float]]) -> List[Prediction]:
        """Tests the model on the given ground-truth dataset.

        Args:
            ratings (List[Tuple[int, int, float]]):
                `(user_id, book_id, true_rating)` tuples

        Raises:
            UntrainedModelError:
                Raised when method is used before model is trained.

        Returns:
            List[Prediction]:
                List of predictions with true and estimated ratings.
        """

    def generate_antitest_set(self) -> Iterable[Tuple[int, int, float]]:
        """Yields a list of ratings which are not already present in the trainset.

        All the ratings where user is known and item is know, but rating for
        (user, item) is not present in the trainset.

        Yields:
            Iterable[Tuple[int, int, float]]:
                A list of tuples (uid, iid, global_mean)
        """


class SurpriseBasedModel(ICfRecommendationModel):
    """Base class for models, which use algorithms from Surprise package.

    Args:
        input_filepath: Filepath containing ratings data.

    Attributes:
        _trainset (Trainset): Dataset used for model training.
        _algorithm (AlgoBase):
            Algorithm(defined in Surprise package) used by model.
    """

    def __init__(self, input_filepath: str):
        self._trainset = self._read_trainset(input_filepath)
        self._algorithm = None

    @staticmethod
    def _read_trainset(input_filepath: str):
        ratings_df = pd.read_csv(input_filepath)
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(
            ratings_df[['user_id', 'book_id', 'rating']], reader)
        return dataset.build_full_trainset()

    def test(self, ratings: List[Tuple[int, int, float]]) -> List[Prediction]:
        if not self._algorithm:
            raise UntrainedModelError

        return self._algorithm.test(ratings)

    def recommend(
            self,
            user_id: int,
            recommendations_count: int = 10
    ) -> Dict[int, float]:
        try:
            user_inner_id = self._trainset.to_inner_uid(user_id)
        except ValueError:
            return dict()

        to_predict = [x for x in self._generate_antitest(user_inner_id)]
        if not self._algorithm:
            raise UntrainedModelError
        predictions = self._algorithm.test(to_predict)

        top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[
            :self._recommendation_count]
        rec_books = {iid: est for _, iid, _, est, _ in top_n}

        return rec_books

    def generate_antitest_set(self) -> Iterable[Tuple[int, int, float]]:
        for uiid in self._trainset.all_users():
            yield from self._generate_antitest(uiid)

    def _generate_antitest(self, user_inner_id: int):
        fill = self._trainset.global_mean
        user_id = self._trainset.to_raw_uid(user_inner_id)
        user_items = set([j for (j, _) in self._trainset.ur[user_inner_id]])
        yield from [(user_id, self._trainset.to_raw_iid(i), fill)
                    for i in self._trainset.all_items() if i not in user_items]


class SlopeOneRecommendationModel(SurpriseBasedModel):
    """Recommendation algorithm using the SlopeOne algorithm.
    """

    def train(self):
        """Computes users average ratings based on common items.
        """
        self._algorithm = SlopeOne().fit(self._trainset)


class SvdRecommendationModel(SurpriseBasedModel):
    """Recommendation algorithm using the Singular Value Decomposition operation.
    """

    def train(self):
        """Prepares user and items vectors.
        """
        self._algorithm = SVD().fit(self._trainset)


class KNNRecommendationModel(SurpriseBasedModel):
    """Recommendation algorithm using the neighbor similarity.
    """

    def train(self):
        """Computes user and items similarities.
        """
        bsl_options = {'method': 'als',
                       'n_epochs': 10,
                       'reg_u': 15,
                       'reg_i': 10}
        sim_options = {'name': 'pearson_baseline',
                       'user_based': False,
                       'min_support': 1}
        algo = KNNBaseline(bsl_options=bsl_options,
                           sim_options=sim_options, verbose=False)
        self._algorithm = algo.fit(self._trainset)
