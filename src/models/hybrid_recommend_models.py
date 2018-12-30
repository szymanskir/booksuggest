import pandas as pd
from lightfm import LightFM

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
            user_data: pd.DataFrame
    ):
        """Initializes an instance of the LightfmBasedModel class

        Args:
            content_analyzer:
                Content analyzer used for extracting item features.
            user_data: Data frame containg user ratings.
        """
        self._content_analyzer = content_analyzer
        self._user_data = user_data[user_data['book_id'].isin(
            self._content_analyzer.book_data.index)]
        self._algorithm = LightFM()

    def train(self):
        item_features = self._content_analyzer.build_features()
        user_features = build_interaction_matrix(self._user_data)
        self._algorithm.fit(
            user_features,
            item_features=item_features,
        )

    def recommend(self):
        pass

    def test(self):
        pass
