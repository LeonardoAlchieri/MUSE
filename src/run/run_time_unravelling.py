# In this script, we load the data, of shape (2070, `TIME`, `N_FEATURES`)
# and change it to shape (2070*`TIME`, `N_FEATURES`).
from sys import path
from logging import basicConfig, getLogger, INFO
from os.path import basename, join as join_paths
from warnings import warn
from numpy.random import seed as set_seed
from typing import Callable

from joblib import Parallel, delayed

path.append(".")
from src.utils.io import load_config
from src.data.smile import SmileData
from src.utils.inputation import (
    filling_mean,
    filling_prev,
    filling_median,
    filling_mode,
    filling_max,
)


_filename: str = basename(__file__).split(".")[0][4:]
basicConfig(filename=f"logs/run/{_filename}.log", level=INFO)
logger = getLogger(_filename)


def main(random_state: int):
    set_seed(random_state)
    path_to_config: str = f"src/run/config_{_filename}.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    timestep_length: int = configs["timestep_length"]
    n_jobs: int = configs["n_jobs"]
    path_to_data: str = configs["path_to_data"]
    save_format: str = configs["save_format"]
    missing_values_inputation: str = configs["missing_values_inputation"]
    output_path: str = configs["output_path"]
    debug_mode: bool = configs["debug_mode"]
    test: bool = configs["test"]
    make_st_feat: bool = configs["make_st_feat"]
    remove_flatlines: bool = configs["remove_flatlines"]

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
    if not make_st_feat:
        features: list[tuple[str, str]] = [
            ("hand_crafted_features", "ECG_features"),
            ("hand_crafted_features", "GSR_features"),
            ("deep_features", "ECG_features_C"),
            ("deep_features", "ECG_features_T"),
        ]
    else:
        features: list[tuple[str, str]] = [
            ("hand_crafted_features", "ECG_features"),
            ("hand_crafted_features", "GSR_features"),
            ("hand_crafted_features", "ST_features"),
            ("deep_features", "ECG_features_C"),
            ("deep_features", "ECG_features_T"),
        ]

    data = SmileData(path_to_data=path_to_data, test=test, debug_mode=debug_mode)
    if make_st_feat:
        data.separate_skin_temperature()

    if n_jobs > 1:
        # FIXME: an error w/ assignment destination is given at the moment
        Parallel(n_jobs=n_jobs)(
            delayed(data.fill_missing_values)(
                features=feature_tuple,
                filling_method=missing_methods_dict[missing_values_inputation],
            )
            for feature_tuple in features
        )
    else:
        for feature_tuple in features:
            logger.info(f"Missing for {feature_tuple}")
            data.fill_missing_values(
                features=feature_tuple,
                filling_method=missing_methods_dict[missing_values_inputation],
            )
    if remove_flatlines:
        data.remove_flatlines()
    data.timecut(timestep_length=timestep_length)
    data.unravel(inplace=True)

    if not make_st_feat:
        output_filename: str = (
            f"dataset_smile_challenge_unravelled_train_cut{timestep_length}"
            if not test
            else f"dataset_smile_challenge_unravelled_test_cut{timestep_length}"
        )
    else:
        output_filename: str = (
            f"dataset_smile_challenge_unravelled_train_cut{timestep_length}_stadd"
            if not test
            else f"dataset_smile_challenge_unravelled_test_cut{timestep_length}_stadd"
        )
    if save_format == "json":
        warn("Saving to json is slow.")
    data.save(path=join_paths(output_path, output_filename), format=save_format)


if __name__ == "__main__":
    random_state: int = 42
    main(random_state=random_state)
