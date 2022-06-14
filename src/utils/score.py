from typing import Callable
from numpy import ndarray, mean, round as approximate, array
from scipy.stats import mode
from sklearn.base import ClassifierMixin


class MergeScorer:
    def __init__(
        self, scorer: Callable, merge_strategy: Callable, time_length: int = 60
    ):
        self.scorer = scorer
        self.time_length = time_length
        if callable(merge_strategy):
            self.merge_strategy = merge_strategy
        elif isinstance(merge_strategy, str):
            self.merge_strategy = self._get_merge_strategy(strategy_name=merge_strategy)

    @staticmethod
    def _get_merge_strategy(strategy_name: str) -> Callable:
        if strategy_name == "average" or strategy_name == "mean":
            return lambda x: approximate(mean(x, axis=1))
        elif strategy_name == "majority_voting":
            return lambda x: mode(x, axis=1)[0].reshape(
                -1,
            )
        else:
            raise ValueError(
                f"Unknown merge strategy {strategy_name}. Accepted are 'average' and 'majority_voting'"
            )

    @staticmethod
    def check_truth(y_true: ndarray) -> ndarray:
        # TODO: implement some check. The main problem is that the check
        # has to be time effetive
        return y_true[:, 0]

    def score(
        self,
        estimator: ClassifierMixin,
        X_test: ndarray,
        y_true: ndarray,
        sample_weight: ndarray = None,
    ) -> float:
        y_pred = estimator.predict(X_test)
        y_true = array(y_true).reshape(-1, self.time_length)
        y_pred = array(y_pred).reshape(-1, self.time_length)
        # the predictions have to be merged w/ the given method
        y_pred = self.merge_strategy(y_pred)
        # the ground truth, I expect them to be all the same
        y_true = self.check_truth(y_true)
        return self.scorer(y_true, y_pred, sample_weight=sample_weight)
