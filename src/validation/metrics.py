from typing import List


def precision(recommendations: List[int], ground_truth: List[int]) -> int:
    """Calculates the precision of the recommendations based on
    the given sets of book ids.

    TODO add latex equation
    """
    relevant_retrieved = len(set(recommendations) & set(ground_truth))
    retrieved = len(recommendations)
    precision_score = relevant_retrieved / retrieved
    return precision_score


def recall(recommendations: List[int], ground_truth: List[int]) -> int:
    """Calculates the recall of the recommendations based on
    the given sets of book ids.

    TODO add latex equation
    """
    relevant_retrieved = len(set(recommendations) & set(ground_truth))
    relevant = len(ground_truth)
    accuracy_score = relevant_retrieved / relevant
    return accuracy_score