from re import I
from sys import path
from logging import getLogger, basicConfig, DEBUG, INFO, WARNING
from typing import Callable
from os.path import basename, join as join_paths
from pandas import DataFrame
from numpy.random import seed as set_seed
from numpy import ndarray
from gc import collect as picking_trash_up
from tqdm import tqdm

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

path.append(".")
from src.data.smile import SmileData
from src.utils.io import load_config, create_output_folder
from src.utils import make_binary
from src.utils.inputation import (
    filling_mean,
    filling_prev,
    filling_median,
    filling_mode,
    filling_max,
)
from src.utils.cv import make_unravelled_folds
from src.utils.score import MergeScorer


_filename: str = basename(__file__).split(".")[0][4:]
basicConfig(filename=f"logs/run/{_filename}.log", level=INFO)
logger = getLogger(_filename)


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
    missing_values_inputation: str = configs["missing_values_inputation"]
    time_merge_strategy: str = configs["time_merge_strategy"]
    binary: bool = configs["binary"]
    unravelled: bool = configs["unravelled"]
    debug_mode: bool = configs["debug_mode"]

    if not debug_mode:
        current_session_path = create_output_folder(
            path_to_config=path_to_config, task=_filename
        )
    else:
        print("DEBUG MODE ACTIVATED!")

    data = SmileData(path_to_data=path_to_data, test=False, debug_mode=debug_mode)

    features: list[tuple[str, str]] = [
        ("hand_crafted_features", "ECG_features"),
        ("hand_crafted_features", "GSR_features"),
        ("deep_features", "ECG_features_C"),
        ("deep_features", "ECG_features_T"),
    ]
    # 2070, 60, M
    if not unravelled:
        join_types: list[str] = [
            "feature_average",  # (2070, 60, 1)
            "concat_feature_level",  # (2070, 60*M), where M = 8, 12, 256, 64
            "window_average",  # (2070, 1, 8)
        ]
    else:
        join_types: list[str] = [
            "concat_label_level"
        ]  # (2070*60, M), where M = 8, 12, 256, 512

    # NOTE: in the notebook on this repo we show that this is the only feature w/ missing values
    feature_missing_values: tuple[str, str] = ("hand_crafted_features", "ECG_features")

    # TODO: do some optimization with regard the hyperparameter
    ml_models: list[ClassifierMixin] = [
        KNeighborsClassifier(n_jobs=n_jobs),
        # SVC(verbose=1,),
        GaussianProcessClassifier(
            n_jobs=n_jobs, copy_X_train=False
        ),  # this is O(m^3) in memory!!!
        DecisionTreeClassifier(),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        RandomForestClassifier(n_jobs=n_jobs),
        XGBClassifier(verbose=1, n_jobs=n_jobs),
    ]

    # "average" # or "remove_user", "previous_val", "mediam", "most_frequent"
    missing_methods_dict: dict[Callable] = dict(
        average=filling_mean,
        previous_val=filling_prev,
        median=filling_median,
        most_frequent=filling_mode,
        max_val=filling_max,
        none=None,
        # remove_user=filling_remove_user
    )

    results = DataFrame(
        index=[i for i in range(cv_num)],
    )
    for feature_tuple in features:
        # NOTE: the shape will be: (2070, 60, N_FEATURES)
        if feature_tuple == feature_missing_values:
            logger.info(f"Applying missing values inputation to {feature_tuple}")
            data.fill_missing_values(
                features=feature_tuple,
                filling_method=missing_methods_dict[missing_values_inputation],
            )

        for join_type in join_types:

            x, y = data.data_feature_level_joined(
                join_type=join_type, features=feature_tuple, get_labels=True
            )
            if binary:
                y = make_binary(y)

            for ml_model in tqdm(
                ml_models,
                desc=f"feature tuple: {feature_tuple}\tjoin type: {join_type}",
            ):
                logger.info(f"Current model {ml_model}")

                if unravelled:
                    # if the data is unravelled, we need a custom way to give
                    # folds to teh cross_val_score method
                    cv: list[tuple[ndarray, ndarray]] = make_unravelled_folds(
                        t=time_length, n_folds=cv_num, n_data=2070
                    )
                else:
                    cv: int = cv_num

                scores = cross_val_score(
                    estimator=ml_model,
                    X=x,
                    y=y,
                    cv=cv,
                    n_jobs=2,
                    scoring=MergeScorer(
                        scorer=accuracy_score,
                        merge_strategy=time_merge_strategy,
                        time_length=time_length,
                    ).score,
                    error_score="raise",
                )
                results[
                    f"{join_type}_{feature_tuple}_{ml_model.__class__.__name__}"
                ] = scores

        del x
        picking_trash_up()

    if not debug_mode:
        results.to_csv(join_paths(current_session_path, "cross_val.csv"))
    else:
        print(results)
        print("SUCCESS")


if __name__ == "__main__":
    random_state: int = 42
    main(random_state=random_state)
