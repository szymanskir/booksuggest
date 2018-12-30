import click
import logging

from .cf_recommend_models import DummyModel
from ..utils.serialization import save_object


def train_dummy_model():
    """Trains a dumy model object.
    """
    dummy_model = DummyModel()
    return dummy_model


@click.command()
@click.argument('filename')
def main(filename):
    """Trains a CF dummy model and saves it to the specified
    file.

    Args:
        filename: file in which in the model should be saved.
    """
    logger = logging.getLogger(__name__)

    logger.info('Training CF dummy model...')
    dummy_model = train_dummy_model()

    logger.info(f'Saving CF dummy model to {filename}...')
    save_object(dummy_model, filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
