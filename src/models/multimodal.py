from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import ClassifierMixin
from sklearn.inspection import permutation_importance
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

    def confusion_matrix(self, x: dict[str, ndarray], y: ndarray) -> ndarray:
        """Method to compute the confusion matrix.

        Parameters
        ----------
        x : dict[str, ndarray]
            x input data, given as a dictionary of ndarrays, where the keys are
            the different features
        y : ndarray
            array of true labels

        Returns
        -------
        ndarray
            the method returns the confusion matrix, as evaluated using the
            `confusion_matrix` method in `sklearn`.
        """
        y_preds = {
            data_name: model.predict(x[data_name])
            if not self.probability
            else model.predict_proba(x[data_name])[:, 1]
            for data_name, model in self.models.items()
        }
        y_pred = self._ravel_back_results(y_preds=y_preds)
        y = Merger.check_truth(Merger.ravel_back(y=y, time_length=self.time_length))

        return confusion_matrix(y_true=y, y_pred=self.fusion_method.predict(y_pred))

    def feature_importance(
        self, x: dict[str, ndarray], y: ndarray, n_repeats: int = 10, n_jobs: int = -1
    ) -> tuple[dict[str, ndarray], ndarray | None]:
        """Get the feature importance of the models, both the first (minute-level)
        and the second (fusion-level) level models.

        Parameters
        ----------
        x : dict[str, ndarray]
            x array of input, of size (n_samples, n_features)
        y : ndarray
            array of ground truth, of size (n_samples,)
        n_repeats : int, optional
            number of repetitions for the `permutation_importance` method, by default 10
        n_jobs : int, optional
            number of jobs of jobs for the `permutation_importance` method, by default -1

        Returns
        -------
        tuple[dict[str, ndarray], ndarray | None]
            the method returns a tuple, where the first element is the feature
            importance for the minute-level models (as a dictionary, where the keys are
            the model names), and the second element is the feature importance for
            the fusion-level model.
        """
        permutation_importance_scores: dict[str, ndarray] = {}

        for data_name, model in self.models.items():
            permutation_importance_scores[data_name] = permutation_importance(
                model,
                x[data_name],
                y,
                scoring="accuracy",
                n_repeats=n_repeats,
                n_jobs=n_jobs,
            )["importances_mean"]
            logger.debug(f"x: {x[data_name].shape}")
            logger.debug(f"y: {y.shape}")
            logger.debug(
                f"permutation_importance_scores: {permutation_importance_scores[data_name].shape}"
            )

        fusion_importance: ndarray | None
        if not isinstance(self.fusion_method, str) and not callable(self.fusion_method):
            if hasattr(self.fusion_method, "feature_importances_"):
                fusion_importance: ndarray = self.fusion_method.feature_importances_
            else:
                warn(
                    "Fusion ML method does not have feature_importances_. Using permutation importance."
                )
                logger.warning(
                    "Fusion ML method does not have feature_importances_. Using permutation importance."
                )
                y_preds = {
                    data_name: model.predict(x[data_name])
                    if not self.probability
                    else model.predict_proba(x[data_name])[:, 1]
                    for data_name, model in self.models.items()
                }
                y_pred = self._ravel_back_results(y_preds=y_preds)
                y = Merger.check_truth(
                    Merger.ravel_back(y=y, time_length=self.time_length)
                )
                fusion_importance: ndarray = permutation_importance(
                    self.fusion_method, y_pred, y
                )["importances_mean"]
                logger.debug(f"fusion_importance: {fusion_importance.shape}")
        else:
            logger.warning(
                "Can only compute importance at fusion for ML models. Skipping."
            )
            warn("Can only compute importance at fusion for ML models. Skipping.")
            fusion_importance: None = None

        return permutation_importance_scores, fusion_importance

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
        logger.info(f"y_pred: {y_pred}")
        logger.info(f"y: {y}")

        return accuracy_score(y_pred, y)
