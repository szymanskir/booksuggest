import click
import logging

from .recommendation_models import DummyModel

from ..utils.serialization import save_object


def train_dummy_model():
    """Trains a dumy model object
    """
    dummy_model = DummyModel()
    return dummy_model


@click.command()
@click.argument('filename')
def main(filename):
    logger = logging.getLogger(__name__)

    logger.info('Training dummy model...')
    dummy_model = train_dummy_model()

    logger.info(f'Saving dummy model to {filename}...')
    save_object(dummy_model, filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
