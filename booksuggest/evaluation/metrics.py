"""Functions calculating metric scores used for recommendation models
evaluation.
"""
from typing import List, Tuple


def precision(recommendations: List[int], ground_truth: List[int]) -> float:
    """Calculates the precision of the recommendations based on
    the given sets of book ids.

    TODO add latex equation
    """
    relevant_retrieved = len(set(recommendations) & set(ground_truth))
    retrieved = len(recommendations)
    if retrieved == 0:
        return 0

    precision_score = relevant_retrieved / retrieved
    return precision_score


def recall(recommendations: List[int], ground_truth: List[int]) -> float:
    """Calculates the recall of the recommendations based on
    the given sets of book ids.

    TODO add latex equation
    """
    relevant_retrieved = len(set(recommendations) & set(ground_truth))
    relevant = len(ground_truth)
    if relevant == 0:
        return 0

    accuracy_score = relevant_retrieved / relevant
    return accuracy_score


def precision_thresholded(recommendations: List[Tuple[int, float]],
                          ground_truth: List[int], threshold: float) -> float:
    """Calculates the precision of the recommendations based on
    the given sets of book ids.
    Filters out recommendations with rating below threshold.

    TODO add latex equation
    """
    valid_recommendations = [x[0] for x in recommendations
                             if x[1] >= threshold]
    relevant_retrieved = len(set(valid_recommendations) & set(ground_truth))
    retrieved = len(recommendations)
    if retrieved == 0:
        return 0

    precision_score = relevant_retrieved / retrieved
    return precision_score


def recall_thresholded(recommendations: List[Tuple[int, float]],
                       ground_truth: List[int], threshold: float) -> float:
    """Calculates the recall of the recommendations based on
    the given sets of book ids.
    Filters out recommendations with rating below threshold.

    TODO add latex equation
    """
    valid_recommendations = [x[0] for x in recommendations
                             if x[1] >= threshold]
    return recall(valid_recommendations, ground_truth)
