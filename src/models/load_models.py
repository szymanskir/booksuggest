from os import listdir
from os.path import join, dirname
from typing import List

from ..utils.serialization import read_object
from .recommendation_models import IRecommendationModel


class InvalidModelException(Exception):
    """Custom exception for handling model retrieval errors.
    """
    pass


def _get_serialized_models_dir() -> str:
    """Returns the internal directory containing serialized models.
    """
    return join(dirname(__file__), 'serialized_models')


def _is_model_available(model: str) -> bool:
    """Checks if the given model is available.

    Args:
        model: name of the model to look for.

    Returns:
        True if the model is available in the serialized_models directory,
        False otherwise.
    """
    available_models: List[str] = listdir(_get_serialized_models_dir())
    return model in available_models


def load_model(model_name: str) -> IRecommendationModel:
    """Loads the model specified by the model_name argument

    Currently available models:
    'dummy_model' - dummy model used for integration testing purposes.
    'basic-tf-idf-model' - basic content based model using only book
                           descriptions as features.

    Args:
        model_name: name of the model to load.

    Returns:
        recommendation model object.
    """
    model: str = f'{model_name}.pkl'
    if _is_model_available(model):
        return read_object(join(_get_serialized_models_dir(), model))

    raise InvalidModelException()
