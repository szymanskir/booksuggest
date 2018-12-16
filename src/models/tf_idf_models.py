import click
import logging

from .recommendation_models import TfIdfRecommendationModel
from ..utils.serialization import save_object


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--n', default=1,
              help='How many recommendations are returned by the model')
def main(input_filepath: str, output_filepath: str, n: int):
    logger = logging.getLogger(__name__)

    logger.info('Training tf-idf model...')
    tf_idf_model = TfIdfRecommendationModel(input_filepath, n)
    tf_idf_model.train()

    logger.info(f'Saving tf-idf model to {output_filepath}...')
    save_object(tf_idf_model, output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
