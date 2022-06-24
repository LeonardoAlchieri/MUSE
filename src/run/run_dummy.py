from gc import collect as picking_trash_up
from logging import DEBUG, INFO, WARNING, basicConfig, getLogger
from os.path import basename
from os.path import join as join_paths
from sys import path
from typing import Callable

from tqdm import tqdm
from numpy import ndarray
from numpy.random import seed as set_seed
from pandas import DataFrame
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

path.append(".")
from src.data.smile import SmileData
from src.utils import make_binary
from src.utils.cv import make_unravelled_folds
from src.utils.inputation import (
    filling_max,
    filling_mean,
    filling_median,
    filling_mode,
    filling_prev,
)
from src.utils.io import (
    create_output_folder,
    delete_output_folder_exception,
    load_config,
)
from src.utils.score import Merger


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
    n_jobs_cv: int = configs["n_jobs_cv"]
    time_length: int = configs["time_length"]
    path_to_data: str = configs["path_to_data"]
    missing_values_inputation: str = configs["missing_values_inputation"]
    time_merge_strategy: str = configs["time_merge_strategy"]
    gaussian_process_kernel: str = configs["gaussian_process_kernel"]
    binary: bool = configs["binary"]
    unravelled: bool = configs["unravelled"]
    debug_mode: bool = configs["debug_mode"]
    st_feat: bool = configs["st_feat"]
    cp_all_config: bool = configs["cp_all_config"]
    feature_selection: bool = configs["feature_selection"]
    feature_selection_configs: dict = configs["feature_selection_configs"]

    if not debug_mode:
        current_session_path = create_output_folder(
            path_to_config=path_to_config, task=_filename, cp_all_config=cp_all_config
        )
    else:
        print("DEBUG MODE ACTIVATED!")

    data = SmileData(path_to_data=path_to_data, test=False, debug_mode=debug_mode)

    if feature_selection:
        data.feature_selection(**feature_selection_configs)

    if not unravelled:
        data.separate_skin_temperature()

    if st_feat:
        features: list[tuple[str, str]] = [
            ("hand_crafted_features", "ECG_features"),
            ("hand_crafted_features", "GSR_features"),
            ("hand_crafted_features", "ST_features"),
            # ("deep_features", "ECG_features_C"),
            # ("deep_features", "ECG_features_T"),
        ]
    else:
        features: list[tuple[str, str]] = [
            ("hand_crafted_features", "ECG_features"),
            ("hand_crafted_features", "GSR_features"),
            # ("deep_features", "ECG_features_C"),
            # ("deep_features", "ECG_features_T"),
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
        DummyClassifier(strategy="most_frequent"),
        DummyClassifier(strategy="uniform"),
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
                    n_data: int = x.shape[0]
                    if n_data != y.shape[0]:
                        raise ValueError(
                            f"The number of data points in x and y are not the same: {n_data} != {y.shape[0]}"
                        )
                    cv: list[tuple[ndarray, ndarray]] = make_unravelled_folds(
                        t=time_length, n_folds=cv_num, n_data=n_data // time_length
                    )
                else:
                    cv: int = cv_num

                scores = cross_val_score(
                    estimator=ml_model,
                    X=x,
                    y=y,
                    cv=cv,
                    n_jobs=n_jobs_cv,
                    scoring=Merger(
                        scorer=accuracy_score,
                        merge_strategy=time_merge_strategy,
                        time_length=time_length,
                    ).score
                    if unravelled
                    else None,
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
