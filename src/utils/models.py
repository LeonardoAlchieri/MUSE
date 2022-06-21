from typing import Callable
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern


def get_fusion_method(
    fusion_method: str,
    gaussian_process_kernel: str | None,
    n_jobs: int = -1,
    probability: bool = False,
) -> Callable | ClassifierMixin:
    """Simple auxiliary function to select the fusion method based on a config string.

    Parameters
    ----------
    fusion_method : str
        the name of the fusion method to use.
        Currently accepted:
        - majority_voting
        - max
        - average
        - one of following models:
            - GaussianProcess
            - AdaBoost
            - QDA
            - SVM

    n_jobs : int
        number of jobs for the ml models, if the model accepts the parameter

    gaussian_process_kernel : str | None, optional
        string to identify the kernel of the GaussianProcessClassifier, by default None
        Currently accepted are:
        - "matern" for Matern kernel
        - None, for RBF
        For all other input values, the choice will be RBF.

    probability: bool
        if True, the model will be trained with probability estimates.

    Returns
    -------
    Callable | ClassifierMixin
        the selected fusion method

    Raises
    ------
    ValueError
        if an ml model string not accepted is given, an error is thrown.
    """
    if fusion_method == "majority_voting":
        return "majority_voting"
    elif fusion_method == "max":
        return max
    elif fusion_method == "average":
        return "average"
    elif fusion_method == "GaussianProcess":
        return get_ml_model(
            "GaussianProcess",
            n_jobs=n_jobs,
            gaussian_process_kernel=gaussian_process_kernel,
            probability=probability,
        )
    elif fusion_method == "AdaBoost":
        return get_ml_model("AdaBoost", n_jobs=n_jobs)
    elif fusion_method == "QDA":
        return get_ml_model("QDA", n_jobs=n_jobs)
    elif fusion_method == "SVM":
        return get_ml_model("SVM", n_jobs=n_jobs)
    else:
        raise ValueError(
            f"Unknown fusion method: {fusion_method}.\
                Accepted values are: {['majority_voting', 'max', 'average', 'GaussianProcess', 'AdaBoost', 'QDA', 'SVM']}"
        )


def get_ml_model(
    model_name: str,
    n_jobs: int,
    gaussian_process_kernel: str | None = None,
    probability: bool = False,
) -> ClassifierMixin:
    """Simple auxiliary function to select the ml model based on a config string.

    Parameters
    ----------
    model_name : str
        model name. Accepted are:
        - "AdaBoost"
        - "GaussianProcess"
        - "QDA"
        - "SMV"

    n_jobs : int
        number of jobs for the ml models, if the model accepts the parameter

    gaussian_process_kernel : str | None, optional
        string to identify the kernel of the GaussianProcessClassifier, by default None
        Currently accepted are:
        - "matern" for Matern kernel
        - None, for RBF
        For all other input values, the choice will be RBF.

    probability: bool
        if True, the model will be trained with probability estimates.

    Returns
    -------
    ClassifierMixin
        the model returns a classifier object, already initialized.

    Raises
    ------
    ValueError
        if an ml model string not accepted is given, an error is thrown.
    """
    # TODO: do some optimization with regard the hyperparameter
    ml_models: list[ClassifierMixin] = [
        GaussianProcessClassifier(
            n_jobs=n_jobs,
            copy_X_train=False,
            kernel=Matern(length_scale=1.0, nu=0.5)
            if gaussian_process_kernel == "matern"
            else None,
        ),  # this is O(m^3) in memory!!!
        AdaBoostClassifier(),
        QuadraticDiscriminantAnalysis(),
        SVC(probability=probability),
    ]
    if model_name == "GaussianProcess":
        return ml_models[0]
    elif model_name == "AdaBoost":
        return ml_models[1]
    elif model_name == "QDA":
        return ml_models[2]
    elif model_name == "SVM":
        return ml_models[3]
    else:
        raise ValueError(
            f"Unknown model name: {model_name}.\
                Accepted values are: {['GaussianProcess', 'AdaBoost', 'QDA', 'SVM']}"
        )
