import logging

import click

from .cf_recommend_models import SlopeOneRecommendationModel
from ..utils.serialization import save_object


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, output_filepath: str):
    logger = logging.getLogger(__name__)

    logger.info('Training SlopeOne model...')
    slopeone_model = SlopeOneRecommendationModel(input_filepath)
    slopeone_model.train()

    logger.info('Saving SlopeOne model to %s...', output_filepath)
    save_object(slopeone_model, output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
