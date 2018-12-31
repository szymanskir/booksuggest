import logging

import click

from .cf_recommend_models import SvdRecommendationModel
from ..utils.serialization import save_object


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, output_filepath: str):
    logger = logging.getLogger(__name__)

    logger.info('Training svd model...')
    svd_model = SvdRecommendationModel(input_filepath)
    svd_model.train()

    logger.info('Saving svd model to %s...', output_filepath)
    save_object(svd_model, output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
