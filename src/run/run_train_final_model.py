from gc import collect as picking_trash_up
from logging import DEBUG, INFO, WARNING, basicConfig, getLogger
from os.path import basename
from os.path import join as join_paths
from sys import path

from numpy import ndarray, savetxt, sqrt
from numpy.random import seed as set_seed
from sklearn.base import ClassifierMixin

path.append(".")
from src.data.smile import SmileData
from src.models import MultiModalClassifier
from src.utils import make_binary
from src.utils.models import get_ml_model, get_fusion_method
from src.utils.score import cross_validation
from src.utils.io import (
    create_output_folder,
    delete_output_folder_exception,
    load_config,
)
from src.utils.score import Merger


_filename: str = basename(__file__).split(".")[0][4:]
basicConfig(filename=f"logs/run/{_filename}.log", level=DEBUG)
logger = getLogger(_filename)


def main(random_state: int):
    set_seed(random_state)

    path_to_config: str = f"src/run/config_{_filename}.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    n_jobs: int = configs["n_jobs"]
    time_length: int = configs["time_length"]
    path_to_data_train: str = configs["path_to_data_train"]
    path_to_data_test: str = configs["path_to_data_test"]
    gaussian_process_kernel: str = configs["gaussian_process_kernel"]
    fusion_method: str = configs["fusion_method"]
    features_selected: str | list[str] = configs["features_selected"]
    binary: bool = configs["binary"]
    unravelled: bool = configs["unravelled"]
    debug_mode: bool = configs["debug_mode"]
    cp_all_config: bool = configs["cp_all_config"]
    feature_selection: bool = configs["feature_selection"]
    probability: bool = configs["probability"]
    feature_selection_configs: dict = configs["feature_selection_configs"]
    models_config: dict = configs["models_config"]

    if not debug_mode:
        current_session_path = create_output_folder(
            path_to_config=path_to_config, task=_filename, cp_all_config=cp_all_config
        )
    else:
        print("DEBUG MODE ACTIVATED!")

    data = SmileData(path_to_data=path_to_data_train, test=False, debug_mode=debug_mode)
    data_test = SmileData(
        path_to_data=path_to_data_test, test=True, debug_mode=debug_mode
    )

    if feature_selection:
        data.feature_selection(**feature_selection_configs)
        data_test.feature_selection(**feature_selection_configs)

    if not unravelled:
        raise NotImplementedError(
            "This code is not implemented correctly for not-unravelled data."
        )

    x: dict[str, ndarray] = data.get_handcrafted_features()
    y: ndarray = data.get_labels()
    if binary:
        y: ndarray = make_binary(y)

    if not isinstance(features_selected, list):
        features_selected: list[str] = [features_selected]

    models: list[ClassifierMixin] = {
        feature_name: get_ml_model(
            model_name=models_config[feature_name],
            gaussian_process_kernel=gaussian_process_kernel,
            n_jobs=n_jobs,
            probability=probability,
        )
        for feature_name in features_selected
    }

    multimodal_classifier = MultiModalClassifier(
        models=models,
        fusion_method=get_fusion_method(
            fusion_method,
            gaussian_process_kernel=gaussian_process_kernel,
            n_jobs=n_jobs,
            probability=probability,
        ),
        time_length=time_length,
        probability=probability,
    )

    multimodal_classifier.fit(x, y)

    x_test: dict[str, ndarray] = data_test.get_handcrafted_features()

    y_pred: ndarray = multimodal_classifier.predict(x_test)

    savetxt(fname=join_paths(current_session_path, "answer.txt"), X=y_pred, fmt="%i")


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
