"""Command used for creating different types of content based recommendation
models.
"""

import logging
import click
import pandas as pd

from .cb_recommend_models import ContentBasedRecommendationModel
from .content_analyzer import ContentAnalyzerBuilder
from ..utils.serialization import save_object


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--ngrams', default=1,
              help='Length of n-grams to be considered')
@click.option('--rec_count', default=1,
              help='How many recommendations are returned by the model')
@click.option('--name')
@click.option('--tag_features_filepath', type=click.Path())
def main(
        input_filepath: str,
        output_filepath: str,
        rec_count: int,
        ngrams: int,
        name: str,
        tag_features_filepath: str
):
    """Main script used for training content based recommendation models.

    Args:
        input_filepath:
            Path to file containg training data.
        output_filepath:
            Path to file in which the trained model should be saved.
        rec_count:
            Specifies how many recommendations the model should return.
        ngrams:
            Specifies the ngram range used when extracting text features.
        name:
            Type of the model to train.
        tag_features_filepath:
        Path to file containing precalculated tag features.
    """
    logger = logging.getLogger(__name__)

    logger.info('Reading data...')
    book_data = pd.read_csv(input_filepath, index_col='book_id')
    tag_features = (pd.read_csv(tag_features_filepath, index_col='book_id')
                    if tag_features_filepath else None)

    logger.info('Training %s model...', name)
    content_analyzer_builder = ContentAnalyzerBuilder(
        name, ngrams, tag_features
    )

    content_analyzer = content_analyzer_builder.build_content_analyzer()
    cb_model = ContentBasedRecommendationModel(
        content_analyzer,
        rec_count
    )
    cb_model.train(book_data[~book_data['description'].isna()])

    logger.info('Saving model to %s...', output_filepath)
    save_object(cb_model, output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
