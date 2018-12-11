from os import listdir
from os.path import join, dirname
from typing import List

from ..utils.serialization import read_object


class InvalidModelException(Exception):
    pass


def _get_serialized_models_dir():
    return join(dirname(__file__), 'serialized_models')


def _is_model_available(model: str):
    available_models: List[str] = listdir(_get_serialized_models_dir())
    return model in available_models


def load_model(model_name: str):
    """Loads the model specified by the model_name argument

    Currently available models:
    'dummy_model' - dummy model used for integration testing purposes
    'basic-tf-idf-model' - basic content based model using only book
                           descriptions as features
    """
    model: str = f'{model_name}.pkl'
    if _is_model_available(model):
        return read_object(join(_get_serialized_models_dir(), model))

    raise InvalidModelException()
