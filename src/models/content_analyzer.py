"""Content analyzers used by content based recommendation models.

Content analyzers are objects that extract features from the items
of interest. They play a main part in content based recommendation
systems.
"""
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    VectorizerMixin
)

from scipy.sparse import hstack


class UnbuiltFeaturesError(Exception):
    """Error is thrown when the content analyzer
    are used before feature_building.
    """


class IContentAnalyzer(metaclass=ABCMeta):
    """Interface for content analyzers responsible
    for creating feature vectors for books.
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


class ContentAnalyzer(IContentAnalyzer):
    """Base class representing a content analyzer.
    """

    def __init__(self):
        self._book_data = None

    def _has_built_features(self):
        if self._book_data is None:
            raise UnbuiltFeaturesError()


class TextBasedContentAnalyzer(ContentAnalyzer):
    """Content analyzer that extracts tf idf text features
    from book descriptions.

    Attributes:
        _text_feature_extractor:
            Object responsible for extracting text based features.
    """

    def __init__(
            self,
            text_feature_extractor: VectorizerMixin
    ):
        super().__init__()
        self._text_feature_extractor = text_feature_extractor

    def build_features(self, book_data: pd.DataFrame) -> np.ndarray:
        self._book_data = book_data
        descriptions = book_data['description']
        features = self._text_feature_extractor.fit_transform(descriptions)
        return features

    def get_feature_vector(self, book_id: int):
        self._has_built_features()
        descriptions = self._book_data['description']
        book_description = descriptions[book_id]
        feature_vector = self._text_feature_extractor.transform(
            [book_description])

        return feature_vector


class TagBasedContentAnalyzer(ContentAnalyzer):
    """Content analyzer that uses book tags to construct
    feature vectors.
    """

    def __init__(
            self,
            tag_features: pd.DataFrame
    ):
        super().__init__()
        self.tag_features = tag_features

    def build_features(self, book_data) -> np.ndarray:
        self._book_data = book_data
        return self.tag_features.loc[book_data.index].values

    def get_feature_vector(self, book_id):
        self._has_built_features()
        return self.tag_features.loc[book_id].values


class EnsembledContentAnalyzer(ContentAnalyzer):
    """Content analyzer that creates feature vectors composed of
    both text features and tag features.

    Attributes:
        text_content_analyzer:
            Content analyzer repsonsible for extracting text features.
        tag_features: Path to tag based features.
    """

    def __init__(
            self,
            content_analyzers: List[IContentAnalyzer],
    ):
        super().__init__()
        self._content_analyzers = content_analyzers

    def build_features(self, book_data) -> np.ndarray:
        return hstack(
            tuple(content_analyzer.build_features(book_data)
                  for content_analyzer in self._content_analyzers)
        )

    def get_feature_vector(self, book_id):
        return hstack(
            tuple(content_analyzer.get_feature_vector(book_id)
                  for content_analyzer in self._content_analyzers)
        )


class TextAndTagBasedContentAnalyzer(EnsembledContentAnalyzer):
    def __init__(
            self,
            text_feature_extractor: VectorizerMixin,
            tag_features: pd.DataFrame
    ):
        super().__init__([
            TextBasedContentAnalyzer(text_feature_extractor),
            TagBasedContentAnalyzer(tag_features)
        ])


class InvalidBuilderConfigError(Exception):
    """Content analyzer building configuration error.
    """


class ContentAnalyzerBuilder():
    def __init__(
            self,
            name: str,
            ngrams: int = None,
            recommendation_count: int = 20,
            tag_features: pd.DataFrame = None
    ):
        self._name = name
        self._ngrams = ngrams
        self._recommendation_count = recommendation_count
        self._tag_features = tag_features
        self._validate_config()

    def _validate_config(self):
        valid_ngram = isinstance(self._ngrams, int) and self._ngrams > 0
        validation_rules = {
            'tf-idf': valid_ngram,
            'count': valid_ngram,
            'tag': self._tag_features is not None,
            'tf-idf-tag': all([
                valid_ngram,
                self._tag_features is not None
            ]),
            'count-tag': all([
                valid_ngram,
                self._tag_features is not None
            ])
        }
        valid_model_name = self._name in validation_rules.keys()

        if (not isinstance(self._recommendation_count, int) or
                self._recommendation_count < 0):
            raise InvalidBuilderConfigError(
                f'Invalid recommendation count: {self._recommendation_count}'
            )

        if not valid_model_name:
            raise InvalidBuilderConfigError(f'Invalid model name {self._name}')

        if not validation_rules[self._name]:
            raise InvalidBuilderConfigError()

    def build_content_analyzer(self) -> IContentAnalyzer:
        building_rules = {
            'tf-idf': partial(
                TextBasedContentAnalyzer,
                TfidfVectorizer(ngram_range=(1, self._ngrams))
            ),
            'count': partial(
                TextBasedContentAnalyzer,
                CountVectorizer(ngram_range=(1, self._ngrams))
            ),
            'tag': partial(TagBasedContentAnalyzer, self._tag_features),
            'tf-idf-tag': partial(
                TextAndTagBasedContentAnalyzer,
                TfidfVectorizer(ngram_range=(1, self._ngrams)),
                self._tag_features
            ),
            'count-tag': partial(
                TextAndTagBasedContentAnalyzer,
                CountVectorizer(ngram_range=(1, self._ngrams)),
                self._tag_features
            )
        }

        constructor = building_rules[self._name]

        return constructor()
