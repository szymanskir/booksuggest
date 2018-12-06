from os.path import join, dirname

from ..utils.serialization import read_object


def _get_serialized_models_dir():
    return join(dirname(__file__), 'serialized_models')


def load_dummy_model():
    """Returns an instance of a dummy model
    """

    return read_object(join(_get_serialized_models_dir(), 'dummy_model.pkl'))
