from sys import path
from os.path import basename

path.append(".")
from logging import getLogger, basicConfig, DEBUG, INFO, WARNING
from sklearn.model_selection import cross_val_score
from numpy import ndarray

from src.data.smile import SmileData
from src.utils.io import load_config


basicConfig(filename="logs/run/classical_ml.log", level=INFO)
_filename: str = basename(__file__).split(".")[0]
logger = getLogger(_filename)


def main():
    path_to_config: str = f"src/run/config_{_filename}.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    cv: int = configs["cross_validation_folds"]
    n_jobs: int = configs["n_jobs"]
    path_to_data: str = configs["path_to_data"]

    data = SmileData(path_to_data=path_to_data)

    # for feature in features:
    #     for join_type in join_types:
    #         for ml_model in ml_models:
    #             X: ndarray = ...
    #             y: ndarray = ...
    #             cross_val_score(estimator=ml_model, X=X, y=y, cv=cv, n_jobs=n_jobs)


if __name__ == "__main__":
    main()
