"""Command used for creating different types of content based recommendation
models.
"""

import logging
import click
import configparser
import pandas as pd

from .cb_recommend_models import ContentBasedRecommendationModel
from .content_analyzer import ContentAnalyzerBuilder
from ..utils.serialization import save_object


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--rec_count', default=1,
              help='How many recommendations are returned by the model')
@click.option('--config_filepath', type=click.Path(exists=True))
def main(
        input_filepath: str,
        output_filepath: str,
        rec_count: int,
        config_filepath: str
):
    """Main script used for training content based recommendation models.

    Args:
        input_filepath:
            Path to file containg training data.
        output_filepath:
            Path to file in which the trained model should be saved.
        configfilepath:
            Path to config file.
    """
    logger = logging.getLogger(__name__)
    config = configparser.ConfigParser()
    config.read(config_filepath)

    logger.info('Reading data...')
    book_data = pd.read_csv(input_filepath, index_col='book_id')

    logger.info('Training %s model...', config['BASE']['model_type'])
    content_analyzer_builder = ContentAnalyzerBuilder(config)

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
