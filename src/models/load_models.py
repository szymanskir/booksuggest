import os
import errno
from typing import TypeVar

from ..utils.serialization import read_object
from .cb_recommend_models import ICbRecommendationModel
from .cf_recommend_models import ICfRecommendationModel

T = TypeVar('T', 'ICbRecommendationModel', 'ICfRecommendationModel')


class InvalidModelException(Exception):
    """Custom exception for handling model retrieval errors.
    """


def load_model(model_file_path: str) -> T:
    """Loads the model specified stored in model_file_path

    Args:
        model_file_path (str): Path to a file containing recommendation model.

    Raises:
        InvalidModelException:
            Raised when object does not implement ICbRecommendationModel
            or ICfRecommendationModel interface.

    Returns:
        T: Recommendation model object.
    """
    if not os.path.isfile(model_file_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), model_file_path)

    model = read_object(model_file_path)
    if isinstance(model, (ICbRecommendationModel, ICfRecommendationModel)):
        return model

    raise InvalidModelException()
