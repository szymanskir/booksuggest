"""Content analyzers used by content based recommendation models.

Content analyzers are objects that extract features from the items
of interest. They play a main part in content based recommendation
systems.
"""
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    VectorizerMixin
)

from scipy.sparse import hstack


class IContentAnalyzer(metaclass=ABCMeta):
    """Interface for content analyzers responsible
    for creating feature vectors for books.
    """

    @abstractproperty
    def book_data(self) -> pd.DataFrame:
        """Returns data frame containing book data.
        """

    @abstractmethod
    def build_features(self) -> np.ndarray:
        """Builds feature matrix for the book_data data frame.
        """

    @abstractmethod
    def get_feature_vector(self, book_id: int) -> np.ndarray:
        """Returns the feature vector of a specific book.

        Args:
            book_id:
                Specifies the book for which the feature vector
                will be returned.

        Returns:
            Feature vector of the given book.
        """


class TextBasedContentAnalyzer(IContentAnalyzer):
    """Content analyzer that extracts tf id text features
    from book descriptions.

    Attributes:
        book_data: Data frame containing book data
        text_feature_extractor: object responsible for
                                extracting text based features
    """

    def __init__(
            self,
            book_data: pd.DataFrame,
            text_feature_extractor: VectorizerMixin
    ):
        self._book_data = book_data
        self.text_feature_extractor = text_feature_extractor

    @property
    def book_data(self):
        return self._book_data

    def build_features(self) -> np.ndarray:
        descriptions = self._book_data['description']
        features = self.text_feature_extractor.fit_transform(descriptions)
        return features

    def get_feature_vector(self, book_id):
        descriptions = self._book_data['description']
        book_description = descriptions[book_id]
        feature_vector = self.text_feature_extractor.transform(
            [book_description])

        return feature_vector


class TextAndTagBasedContentAnalyzer(IContentAnalyzer):
    """Content analyzer that creates feature vectors composed of
    both text features and tag features.

    Attributes:
        text_content_analyzer: content analyzer repsonsible
                               for extracting text features.
        tag_features: Path to tag based features.
    """

    def __init__(
            self,
            text_content_analyzer: TextBasedContentAnalyzer,
            tag_features: pd.DataFrame
    ):
        self.text_content_analyzer = text_content_analyzer

        self.tag_features = tag_features.loc[self.book_data.index]

    @property
    def book_data(self):
        return self.text_content_analyzer.book_data

    def build_features(self) -> np.ndarray:
        text_features = self.text_content_analyzer.build_features()
        features = hstack((text_features, self.tag_features.iloc[:, 1:]))

        return features

    def get_feature_vector(self, book_id):
        text_feature_vector = self.text_content_analyzer.get_feature_vector(
            book_id)
        feature_vector = hstack((
            text_feature_vector,
            self.tag_features.loc[book_id].values[1:]
        ))

        return feature_vector


def build_content_analyzer(
        book_data: str,
        text_feature_extractor: VectorizerMixin,
        tag_features: np.ndarray = None
) -> IContentAnalyzer:
    """Builds a content analyzer based on the input arguments.
    """
    text_content_analyzer = TextBasedContentAnalyzer(
        book_data=book_data,
        text_feature_extractor=text_feature_extractor
    )

    if tag_features is None:
        return text_content_analyzer

    return TextAndTagBasedContentAnalyzer(
        text_content_analyzer=text_content_analyzer,
        tag_features=tag_features
    )
