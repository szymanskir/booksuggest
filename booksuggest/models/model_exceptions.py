"""Exceptions regarding recommendation models.
"""


class UntrainedModelError(Exception):
    """Error is raised when a method is used
    using an untrained model.
    """


class UnbuiltFeaturesError(Exception):
    """Error is thrown when the content analyzer
    are used before feature_building.
    """
