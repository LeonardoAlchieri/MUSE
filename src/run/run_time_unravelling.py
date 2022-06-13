# In this script, we load the data, of shape (2070, `TIME`, `N_FEATURES`)
# and change it to shape (2070*`TIME`, `N_FEATURES`).
from sys import path
from logging import basicConfig, getLogger, INFO
from os.path import basename, join as join_paths
from warnings import warn
from numpy.random import seed as set_seed

path.append(".")
from src.utils.io import load_config
from src.data.smile import SmileData

basicConfig(filename="logs/run/classical_ml.log", level=INFO)
_filename: str = basename(__file__).split(".")[0][4:]
logger = getLogger(_filename)


def main(random_state: int):
    set_seed(random_state)
    path_to_config: str = f"src/run/config_{_filename}.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_data: str = configs["path_to_data"]
    save_format: str = configs["save_format"]
    debug_mode: bool = configs["debug_mode"]
    test: bool = configs["test"]
    output_path: str = configs["output_path"]

    data = SmileData(path_to_data=path_to_data, test=test, debug_mode=debug_mode)

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
