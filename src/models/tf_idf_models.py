import click
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List
from scipy.spatial import distance

from .recommendation_models import IRecommendationModel


class TfIdfRecommendationModel(IRecommendationModel):
    """Recommendation model using the tf-idf method for
    feature extraction. Later uses the cosine similarity
    in order to select the most similar books
    """

    def __init__(self, input_filepath: str, recommendation_count):
        self.data = pd.read_csv(input_filepath, index_col='book_id')
        self.recommendation_count = recommendation_count

    def train(self):
        """Prepares tf_idf feature vectors
        """

        # TODO fix initial data cleaning so no NAs are present
        descriptions = self.data['description'].dropna()
        vectorizer = TfidfVectorizer()
        result = vectorizer.fit_transform(descriptions).toarray()
        feature_vectors = [x.flatten()
                           for x in np.vsplit(result, result.shape[0])]
        self.features = pd.DataFrame(
            data={'feature_vector': feature_vectors},
            index=descriptions.index
        )

    def recommend(self, user_ratings: Dict[int, int]) -> List[int]:
        selected_book_id = next(iter(user_ratings))
        selected_book_features = self.features.loc[
            selected_book_id, 'feature_vector'
        ]

        candidate_features = self.features[~self.features.index.isin([selected_book_id])]
        distances = candidate_features.apply(
            lambda x: distance.cosine(
                x.loc['feature_vector'],
                selected_book_features
            ),
            axis=1
        )

        recommendation_ids = distances.sort_values().index.tolist()[:self.recommendation_count]
        return dict(zip(
            recommendation_ids, distances.loc[recommendation_ids].tolist()
        ))


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, output_filepath: str):
    pass


if __name__ == '__main__':
    pass
