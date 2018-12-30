import numpy as np
import pandas as pd
from lightfm import LightFM
from scipy.sparse import coo_matrix

from .cf_recommend_models import ICfRecommendationModel
from .content_analyzer import IContentAnalyzer

from ..features.build_user_features import build_interaction_matrix


class LightFMBasedModel(ICfRecommendationModel):
    """Basic hybrid recommendation model using LightFM.

    Attributes:
        _content_analyzer: Content analyzer used for extracting item features.
        _user_data: Data frame containing user ratings.
    """

    def __init__(
            self,
            content_analyzer: IContentAnalyzer,
            user_data: pd.DataFrame,
            recommendation_count: int
    ):
        """Initializes an instance of the LightFMBasedModel class

        Args:
            content_analyzer:
                Content analyzer used for extracting item features.
            user_data: Data frame containg user ratings.
        """
        self._content_analyzer = content_analyzer
        self._user_data = user_data[user_data['book_id'].isin(
            self._content_analyzer.book_data.index)]
        self._algorithm = LightFM()
        self._recommendation_count = recommendation_count

    def train(self):
        item_features = self._content_analyzer.build_features()
        user_features = build_interaction_matrix(self._user_data)
        self._algorithm.fit(
            user_features,
            item_features=item_features,
        )

        user_data_to_predict = coo_matrix(
            np.logical_not(user_features.toarray()).astype('int'))
        book_ranks = self._algorithm.predict_rank(
            user_data_to_predict,
            item_features=item_features
        )

        self._book_rank = pd.DataFrame(
            data=book_ranks.toarray(),
            index=self._user_data.user_id.unique()
        )

    def recommend(self, user_id):
        return(
            self._book_rank.loc[
                user_id, :].sort_values(
                ).where(lambda x: x > 0).dropna()[:self._user_data]
        )

    def test(self):
        pass
