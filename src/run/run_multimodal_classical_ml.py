from gc import collect as picking_trash_up
from logging import DEBUG, INFO, WARNING, basicConfig, getLogger
from os.path import basename
from os.path import join as join_paths
from sys import path
from typing import Callable

from numpy import ndarray
from numpy.random import seed as set_seed
from pandas import DataFrame
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

path.append(".")
from src.data.smile import SmileData
from src.models import MultiModalClassifier
from src.utils import make_binary, all_subsets
from src.utils.cv import make_unravelled_folds
from src.utils.score import cross_validation
from src.utils.io import (
    create_output_folder,
    delete_output_folder_exception,
    load_config,
)
from src.utils.score import Merger


_filename: str = basename(__file__).split(".")[0][4:]
basicConfig(filename=f"logs/run/{_filename}.log", level=INFO)
logger = getLogger(_filename)


def get_fusion_method(
    fusion_method: str, gaussian_process_kernel: str | None, n_jobs: int = -1
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
    model_name: str, n_jobs: int, gaussian_process_kernel: str | None = None
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
        SVC(),
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


def main(random_state: int):
    set_seed(random_state)

    path_to_config: str = f"src/run/config_{_filename}.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    cv_num: int = configs["cross_validation_folds"]
    n_jobs: int = configs["n_jobs"]
    time_length: int = configs["time_length"]
    path_to_data: str = configs["path_to_data"]
    gaussian_process_kernel: str = configs["gaussian_process_kernel"]
    fusion_method: str = configs["fusion_method"]
    binary: bool = configs["binary"]
    unravelled: bool = configs["unravelled"]
    debug_mode: bool = configs["debug_mode"]
    st_feat: bool = configs["st_feat"]
    cp_all_config: bool = configs["cp_all_config"]
    feature_selection: bool = configs["feature_selection"]
    feature_selection_configs: dict = configs["feature_selection_configs"]
    models_config: dict = configs["models_config"]

    if not debug_mode:
        current_session_path = create_output_folder(
            path_to_config=path_to_config, task=_filename, cp_all_config=cp_all_config
        )
    else:
        print("DEBUG MODE ACTIVATED!")

    data = SmileData(path_to_data=path_to_data, test=False, debug_mode=debug_mode)

    if feature_selection:
        data.feature_selection(**feature_selection_configs)

    if st_feat:
        features: list[tuple[str, str]] = [
            ("hand_crafted_features", "ECG_features"),
            ("hand_crafted_features", "GSR_features"),
            ("hand_crafted_features", "ST_features"),
        ]
    else:
        features: list[tuple[str, str]] = [
            ("hand_crafted_features", "ECG_features"),
            ("hand_crafted_features", "GSR_features"),
        ]
    # 2070, 60, M
    if not unravelled:
        raise NotImplementedError(
            "This code is not implemented correctly for not-unravelled data."
        )

    results = DataFrame(
        index=[i for i in range(cv_num)],
    )

    features: list[tuple[tuple[str, str], ...]] = [
        subset for subset in all_subsets(features) if len(subset) != 0
    ]

    x: dict[str, ndarray] = data.get_handcrafted_features()
    y: ndarray = data.get_labels()
    if binary:
        y: ndarray = make_binary(y)

    cv: list[tuple[ndarray, ndarray]] = make_unravelled_folds(
        t=time_length, n_folds=cv_num, n_data=2070
    )

    for feature_subset in features:
        logger.info(f"Starting feature subset: {feature_subset}")
        current_feature_names: list[str] = [feature[1] for feature in feature_subset]
        models: list[ClassifierMixin] = {
            feature_name: get_ml_model(
                model_name=models_config[feature_name],
                gaussian_process_kernel=gaussian_process_kernel,
                n_jobs=n_jobs,
            )
            for feature_name in current_feature_names
        }
        multimodal_classifier = MultiModalClassifier(
            models=models,
            fusion_method=get_fusion_method(
                fusion_method,
                gaussian_process_kernel=gaussian_process_kernel,
                n_jobs=n_jobs,
            ),
            time_length=time_length,
            probability=True,
        )

        scores = cross_validation(
            x=x, y=y, estimator=multimodal_classifier, cv=cv, n_jobs=3
        )

        results[
            f"{feature_subset}_{[model.__class__.__name__ for model in models]}"
        ] = scores

    if not debug_mode:
        results.to_csv(join_paths(current_session_path, "cross_val.csv"))
    else:
        print(results)
        print("SUCCESS")


if __name__ == "__main__":
    random_state: int = 42
    try:
        main(random_state=random_state)
    except:
        logger.warning("Process terminated early. Removing save directory.")
        print("!!!!! Process terminated early. Removing save directory. !!!!!")
        res = delete_output_folder_exception(task=_filename)
        if res:
            print("Dir removed successfully")
            logger.info("Dir removed successfully")
        else:
            print("Dir not removed")
            logger.info("Dir not removed")
        raise
