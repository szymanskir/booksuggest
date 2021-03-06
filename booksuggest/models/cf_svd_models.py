import logging

import click

from .cf_recommend_models import SvdRecommendationModel
from ..utils.serialization import save_object


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.option('--random-state', type=int)
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, random_state: int, output_filepath: str):
    logger = logging.getLogger(__name__)

    logger.info('Training SVD model...')
    svd_model = SvdRecommendationModel(input_filepath)
    svd_model.train(random_state=random_state)

    logger.info('Saving SVD model to %s...', output_filepath)
    save_object(svd_model, output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
