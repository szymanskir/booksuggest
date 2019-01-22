import logging

import click

from .cf_recommend_models import KNNRecommendationModel
from ..utils.serialization import save_object


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, output_filepath: str):
    logger = logging.getLogger(__name__)

    logger.info('Training KNN model...')
    knn_model = KNNRecommendationModel(input_filepath)
    knn_model.train()

    logger.info('Saving KNN model to %s...', output_filepath)
    save_object(knn_model, output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
