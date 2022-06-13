# In this script, we load the data, of shape (2070, `TIME`, `N_FEATURES`)
# and change it to shape (2070*`TIME`, `N_FEATURES`).
from sys import path
from logging import basicConfig, getLogger, INFO
from os.path import basename, join as join_paths
from warnings import warn
from numpy.random import seed as set_seed
from typing import Callable

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

    path_to_data: str = configs["path_to_data"]
    save_format: str = configs["save_format"]
    missing_values_inputation: str = configs["missing_values_inputation"]
    debug_mode: bool = configs["debug_mode"]
    test: bool = configs["test"]
    output_path: str = configs["output_path"]

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
    features: list[tuple[str, str]] = [
        ("hand_crafted_features", "ECG_features"),
        ("hand_crafted_features", "GSR_features"),
        ("deep_features", "ECG_features_C"),
        ("deep_features", "ECG_features_T"),
    ]

    data = SmileData(path_to_data=path_to_data, test=test, debug_mode=debug_mode)
    for feature_tuple in features:
        logger.info(f"Missing for {feature_tuple}")
        data.fill_missing_values(
            features=feature_tuple,
            filling_method=missing_methods_dict[missing_values_inputation],
        )

    data.unravel(inplace=True)

    output_filename: str = (
        "dataset_smile_challenge_unravelled_train"
        if not test
        else "dataset_smile_challenge_unravelled_test"
    )
    if save_format == "json":
        warn("Saving to json is slow.")
    data.save(path=join_paths(output_path, output_filename), format=save_format)


if __name__ == "__main__":
    random_state: int = 42
    main(random_state=random_state)
