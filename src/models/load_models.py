import os
import errno
from typing import List

from ..utils.serialization import read_object
from .recommendation_models import IRecommendationModel


class InvalidModelException(Exception):
    """Custom exception for handling model retrieval errors.
    """
    pass


def load_model(model_file_path: str) -> IRecommendationModel:
    """Loads the model specified stored in model_file_path

    Args:
        model_file_path (str): Path to a file containing recommendation model.

    Raises:
        InvalidModelException: Raised when object does not implement IRecommendationModel interface.

    Returns:
        IRecommendationModel: Recommendation model object.
    """
    if not os.path.isfile(model_file_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), model_file_path)

    model = read_object(model_file_path)
    if isinstance(model, IRecommendationModel):
        return model

    raise InvalidModelException()
