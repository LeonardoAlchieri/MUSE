from sklearn.metrics import accuracy_score
from sklearn.base import ClassifierMixin
from numpy import ndarray
from typing import Callable
from numpy import stack
from warnings import warn
from logging import getLogger

from src.utils.score import Merger

logger = getLogger(__name__)


class MultiModalClassifier(ClassifierMixin):
    def __init__(
        self,
        models: dict[str, ClassifierMixin],
        fusion_method: Callable | ClassifierMixin | str,
        time_length: int,
        probability: bool = False,
    ):
        """Classifier to train a multi-modal approach. The classifier will train
        a model (at minute-level) for

        Parameters
        ----------
        models : dict[str, ClassifierMixin]
            _description_
        fusion_method : Callable | ClassifierMixin | str
            _description_
        time_length : int
            _description_
        probability : bool, optional
            _description_, by default False
        """
        self.time_length = time_length
        self.data_names: list[str] = list(models.keys())
        self.models = models
        self.probability = probability
        if isinstance(fusion_method, str):
            self.fusion_method = Merger._get_merge_strategy(fusion_method)
        else:
            self.fusion_method = fusion_method

    def fit(self, x: dict[str, ndarray], y: ndarray):
        if not isinstance(x, dict):
            raise TypeError(f"x must be a dict. Got {type(x)} instead")

        y_preds: dict[str, ndarray] = {}
        for data_name, model in self.models.items():
            model.fit(x[data_name], y)
            y_preds[data_name] = (
                model.predict(x[data_name])
                if not self.probability
                else model.predict_proba(x[data_name])[:, 1]
            )
            self.models[data_name] = model

        if not isinstance(self.fusion_method, str) and not callable(self.fusion_method):
            warn(f"Assuming fusion method to be ML based.")
            # logger.warning('Assuming fusion method to be ML based')
            y_pred = self._ravel_back_results(y_preds=y_preds)

            y = Merger.check_truth(Merger.ravel_back(y=y, time_length=self.time_length))
            self.fusion_method.fit(y_pred, y)

    def _ravel_back_results(self, y_preds: ndarray) -> ndarray:
        y_pred: ndarray = stack(list(y_preds.values()), axis=1)
        y_pred: ndarray = y_pred.reshape(-1, self.time_length * y_pred.shape[-1])
        return y_pred

    def predict(self, x: dict[str, ndarray]) -> ndarray:
        y_preds: dict[str, ndarray] = {}
        for data_name, model in self.models.items():
            y_preds[data_name] = (
                model.predict(x[data_name])
                if not self.probability
                else model.predict_proba(x[data_name])[:, 1]
            )
        logger.debug(f"y_preds: {y_preds}")

        y_pred = self._ravel_back_results(y_preds=y_preds)
        if not isinstance(self.fusion_method, str) and not callable(self.fusion_method):
            return self.fusion_method.predict(y_pred)
        else:
            return self.fusion_method(y_pred)

    def score(self, x: dict[str, ndarray], y: ndarray) -> float:
        y_pred = self.predict(x)
        y = Merger.check_truth(Merger.ravel_back(y=y, time_length=self.time_length))
        return accuracy_score(y_pred, y)
