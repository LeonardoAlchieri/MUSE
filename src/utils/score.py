from typing import Callable, Iterable
from logging import getLogger
from numpy import array, mean, ndarray
from numpy import round as approximate
from scipy.stats import mode
from sklearn.base import ClassifierMixin
from joblib import Parallel, delayed
from copy import deepcopy

logger = getLogger(__name__)


class Merger:
    def __init__(
        self, scorer: Callable, merge_strategy: Callable, time_length: int = 60
    ):
        self.scorer = scorer
        self.time_length = time_length
        if callable(merge_strategy):
            self.merge_strategy = merge_strategy
        elif isinstance(merge_strategy, str):
            self.merge_strategy = self._get_merge_strategy(strategy_name=merge_strategy)
        elif isinstance(merge_strategy, None):
            raise ValueError(
                "Must provide a merge_strategy: since the value received is None, you probably forgot to fill in the field in the config file."
            )
        else:
            raise ValueError(
                f"merge_strategy must be a callable or a string. Received {type(merge_strategy)} instead."
            )

    @staticmethod
    def _get_merge_strategy(strategy_name: str) -> Callable[..., ndarray]:
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

    @staticmethod
    def ravel_back(y: ndarray, time_length: int) -> ndarray:
        return array(y).reshape(-1, time_length)

    def score(
        self,
        estimator: ClassifierMixin,
        X_test: ndarray,
        y_true: ndarray,
        sample_weight: ndarray = None,
    ) -> float:
        y_pred = estimator.predict(X_test)
        # here we go from (N*T) to (N, T)
        y_true = self.ravel_back(y_true, time_length=self.time_length)
        y_pred = self.ravel_back(y_pred, time_length=self.time_length)

        # the predictions have to be merged w/ the given method
        y_pred = self.merge_strategy(y_pred)
        # the ground truth, I expect them to be all the same
        y_true = self.check_truth(y_true)
        return self.scorer(y_true, y_pred, sample_weight=sample_weight)


def fit_and_score(
    estimator: ClassifierMixin,
    x: dict[str, ndarray] | ndarray,
    y: ndarray,
    train_idx: ndarray,
    test_idx: ndarray,
    cm: bool,
    **kwargs,
) -> tuple[float, tuple[dict[str, ndarray], ndarray | None], ndarray] | tuple[
    float, tuple[dict[str, ndarray], ndarray | None]
]:
    if isinstance(x, dict):
        logger.info(f"x Input is dictionary")
        x_train: dict[str, ndarray] = {
            data_name: x[data_name][train_idx] for data_name in x.keys()
        }
        x_test: dict[str, ndarray] = {
            data_name: x[data_name][test_idx] for data_name in x.keys()
        }
    else:
        logger.info(f"x Input is not dictionary. Assuming it is ndarray")
        x_train = x[train_idx]
        x_test = x[test_idx]

    y_train: ndarray = y[train_idx]
    y_test: ndarray = y[test_idx]

    estimator.fit(deepcopy(x_train), deepcopy(y_train))

    if "n_repeats" in kwargs.keys():
        n_repeats: int = kwargs["n_repeats"]
    else:
        n_repeats: int = 1

    if "n_jobs" in kwargs.keys():
        n_jobs: int = kwargs["n_jobs"]
    else:
        n_jobs: int = 1

    if n_repeats != 0:
        feature_importances: tuple[
            dict[str, ndarray], ndarray | None
        ] = estimator.feature_importance(
            x=x_test, y=y_test, n_repeats=n_repeats, n_jobs=n_jobs
        )
    else:
        feature_importances: tuple[dict[str, ndarray], ndarray | None] = (None, None)

    if cm:
        confusion_matrix = estimator.confusion_matrix(x=x_test, y=y_test)
        return (
            estimator.score(deepcopy(x_test), deepcopy(y_test)),
            feature_importances,
            confusion_matrix,
        )
    else:
        return (
            estimator.score(deepcopy(x_test), deepcopy(y_test)),
            feature_importances,
        )


def cross_validation(
    x: dict[str, ndarray] | ndarray,
    y: ndarray,
    estimator: ClassifierMixin,
    cv: Iterable[tuple[ndarray, ndarray]],
    cm: bool = False,
    n_jobs: int | None = None,
    n_repeats: int = 1,
    **kwargs,
) -> list[tuple[float, tuple[dict[str, ndarray], ndarray | None], ndarray]]:

    if "parallel" in kwargs:
        parallel_args: dict = kwargs["parallel"]
    else:
        logger.debug(
            "No args for parallel given. Setting verbose to 0 and pre_dispatch to all"
        )
        parallel_args: dict = dict(verbose=0, pre_dispatch="all")

    parallel = Parallel(
        n_jobs=n_jobs,
        verbose=parallel_args["verbose"],
        pre_dispatch=parallel_args["pre_dispatch"],
        backend="loky",
    )
    if not n_jobs == 1:
        results = parallel(
            delayed(fit_and_score)(
                estimator=deepcopy(estimator),
                x=deepcopy(x),
                y=deepcopy(y),
                train_idx=train_idx,
                test_idx=test_idx,
                n_repeats=n_repeats,
                n_jobs=n_jobs,
                cm=cm,
            )
            for train_idx, test_idx in cv
        )
    else:
        results = [
            fit_and_score(
                estimator=deepcopy(estimator),
                x=x,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                n_repeats=n_repeats,
                n_jobs=n_jobs,
                cm=cm,
            )
            for train_idx, test_idx in cv
        ]
    return list(results)
