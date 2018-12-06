from abc import ABCMeta, abstractmethod
from typing import Dict, List


class IRecommendationModel(metaclass=ABCMeta):

    @abstractmethod
    def recommend(self, user_ratings: Dict[int, int]) -> List[int]:
        """ Returns recommended books based on the users ratings.

        The user_ratings have a form of <book_id>: <rating>. The
        ratings are based on a scale from 1 to 5.
        """
        pass


class DummyModel(IRecommendationModel):
    def recommend(self, user_ratings: Dict[int, int]) -> List[int]:
        books: List[int] = list(user_ratings.keys())
        return books[0:6]
