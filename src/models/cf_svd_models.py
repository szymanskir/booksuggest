import click
import logging

from .cf_recommend_models import SvdRecommendationModel
from ..utils.serialization import save_object


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--n', default=5,
              help='How many recommendations are returned by the model')
def main(input_filepath: str, output_filepath: str, n: int):
    logger = logging.getLogger(__name__)

    logger.info('Training svd model...')
    svd_model = SvdRecommendationModel(input_filepath, n)
    svd_model.train()

    logger.info(f'Saving svd model to {output_filepath}...')
    save_object(svd_model, output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
