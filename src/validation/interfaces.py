from statistics import mean
from typing import Callable, Dict, List, Tuple

from .metrics import precision, recall


class IContentBasedRecommendationModelValidation():
    def _score_for_given_metric(
            self,
            results: Dict[int, List[int]],
            test_cases: Dict[int, List[int]],
            metric: Callable[[List[int], List[int]], int]
    ) -> int:
        """Calculates the average score for a given metric.
        """
        score = [metric(results[key], test_cases[key])
                 for key in test_cases.keys()]

        return mean(score)

    def score(self, test_cases: Dict[int, List[int]]) -> Tuple[int, int]:
        """Calculates the precision and recall score of the model
        """
        results = {key: list(self.recommend({key: None}).keys())
                   for key in test_cases.keys()}
        precision_score = self._score_for_given_metric(
            results, test_cases, precision
        )
        recall_score = self._score_for_given_metric(
            results, test_cases, recall
        )

        return (precision_score, recall_score)
