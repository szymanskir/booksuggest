"""Content analyzers used by content based recommendation models.

Content analyzers are objects that extract features from the items
of interest. They play a main part in content based recommendation
systems.
"""
from abc import ABCMeta, abstractmethod
from functools import partial
from gensim.models import Word2Vec
from typing import Callable, Dict, List
import configparser
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    VectorizerMixin
)

from scipy.sparse import hstack

from .model_exceptions import UnbuiltFeaturesError


class IContentAnalyzer(metaclass=ABCMeta):
    """Interface for content analyzers responsible
    for creating feature vectors for books.
    """

    def __init__(self):
        self._book_data = None

    def _has_built_features(self):
        if self._book_data is None:
            raise UnbuiltFeaturesError()

    @abstractmethod
    def build_features(self, book_data: pd.DataFrame) -> np.ndarray:
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


class Word2VecContentAnalyzer(IContentAnalyzer):
    """Content analyzer that uses word2vec embeddings to
    construct feature vectors
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._feature_size = kwargs.get('feature_size')
        self._window_size = kwargs.get('window_size')
        self._iter_num = kwargs.get('iter_num')

    def _tokenize_description(self, description: str) -> List[List[str]]:
        sentences = sent_tokenize(description)
        sentences_by_words = [word_tokenize(sentence)for sentence in sentences]
        return sentences_by_words

    def _train_model(self, book_data: pd.DataFrame):
        descriptions = book_data['description']
        sentences_by_words = sum([self._tokenize_description(
            description) for description in descriptions], [])
        self._model = Word2Vec(sentences_by_words, size=self._feature_size,
                               window=self._window_size, min_count=1, workers=4, iter=self._iter_num)

    def _build_single_feature(self, description: str):
        words = word_tokenize(description)
        word_vectors = [self._model.wv[word] for word in words]
        feature_vector = np.mean(word_vectors, axis=0)
        return feature_vector

    def build_features(self, book_data: pd.DataFrame) -> np.ndarray:
        self._book_data = book_data
        self._train_model(book_data=book_data)
        descriptions = book_data['description']
        features = [self._build_single_feature(
            description) for description in descriptions]
        return np.array(features)

    def get_feature_vector(self, book_id: int):
        descriptions = self._book_data['description']
        book_description = descriptions[book_id]
        return np.array([self._build_single_feature(book_description)])

    @classmethod
    def create_from_config(cls, config):
        return cls(
            feature_size = config.getint("feature_size"),
            window_size = config.getint("window_size"),
            iter_num = config.getint("iter_num")
        )


class TagBasedContentAnalyzer(IContentAnalyzer):
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
        return self.tag_features.loc[book_id].values.reshape(1, -1)


class EnsembledContentAnalyzer(IContentAnalyzer):
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
    """Content analyzer combining text and tag based features.
    """

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
    """Builder class used for creating content analyzers
    based on the given configuration.

    Args:
        name: Type of the content analyzer.
        ngram: Maximal number of words in a single feature.
        tag_features: Data frame containing calculated tag features.
    """

    def __init__(
            self,
            config: configparser.ConfigParser 
    ):
        self._config = config

    def build_content_analyzer(self) -> IContentAnalyzer:
        """Build a content analyzer based on the object
        configuration.
        """
        building_rules: Dict[str, Callable] = {
            # 'tf-idf': partial(
            #     TextBasedContentAnalyzer,
            #     TfidfVectorizer(ngram_range=(1, self._ngrams))
            # ),
            # 'count': partial(
            #     TextBasedContentAnalyzer,
            #     CountVectorizer(ngram_range=(1, self._ngrams))
            # ),
            # 'tag': partial(TagBasedContentAnalyzer, self._tag_features),
            # 'tf-idf-tag': partial(
            #     TextAndTagBasedContentAnalyzer,
            #     TfidfVectorizer(ngram_range=(1, self._ngrams)),
            #     self._tag_features
            # ),
            # 'count-tag': partial(
            #     TextAndTagBasedContentAnalyzer,
            #     CountVectorizer(ngram_range=(1, self._ngrams)),
            #     self._tag_features
            # ),
            'word2vec':  Word2VecContentAnalyzer.create_from_config
        }

        constructor = building_rules[self._config.get('BASE', 'model_type')]

        return constructor(self._config['PARAMETERS'])
