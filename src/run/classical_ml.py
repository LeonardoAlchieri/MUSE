from sys import path
from logging import getLogger, basicConfig, DEBUG, INFO, WARNING
from warnings import warn
from numpy import ndarray
from os.path import basename, join as join_paths
from pandas import DataFrame
from numpy.random import seed as set_seed
from random import randint
from gc import collect as picking_trash_up
from tqdm import tqdm

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.base import ClassifierMixin

path.append(".")
from src.data.smile import SmileData
from src.utils.io import load_config, create_output_folder


basicConfig(filename="logs/run/classical_ml.log", level=INFO)
_filename: str = basename(__file__).split(".")[0]
logger = getLogger(_filename)


def main(random_state: int):
    set_seed(random_state)

    path_to_config: str = f"src/run/config_{_filename}.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    cv: int = configs["cross_validation_folds"]
    n_jobs: int = configs["n_jobs"]
    path_to_data: str = configs["path_to_data"]

    current_session_path = create_output_folder(
        path_to_config=path_to_config, task=_filename
    )

    data = SmileData(path_to_data=path_to_data)

    features: list[tuple[str, str]] = [
        ("hand_crafted_features", "ECG_features"),
        ("hand_crafted_features", "GSR_features"),
        ("deep_features", "ECG_features_C"),
        ("deep_features", "ECG_features_T"),
    ]

    join_types: list[str] = [
        "average",
        "concat_feature_level",
    ]  # "concat_label_level"]

    # TODO: do some optimization with regard the hyperparameter
    ml_models: list[ClassifierMixin] = [
        KNeighborsClassifier(),
        SVC(),
        GaussianProcessClassifier(),
        DecisionTreeClassifier(),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        RandomForestClassifier(),
    ]

    results = DataFrame(
        columns=[model.__class__.__name__ for model in ml_models],
        index=[i for i in range(cv)],
    )
    for feature_tuple in features:
        # NOTE: the shape will be: (2070, 60, N_FEATURES)

        for join_type in join_types:

            x, y = data.data_feature_level_joined(
                join_type=join_type, features=feature_tuple, get_labels=True, test=False
            )
            try:
                for ml_model in tqdm(
                    ml_models,
                    desc=f"feature tuple: {feature_tuple}\tjoin type: {join_type}",
                ):
                    logger.info(f"Current model {ml_model}")
                    scores = cross_val_score(
                        estimator=ml_model, X=x, y=y, cv=cv, n_jobs=n_jobs
                    )
                    results[ml_model.__class__.__name__] = scores
            except Exception as e:
                warn(
                    f"Some error occurd during the training. Skipping to next session. -> {e}"
                )
                continue

        del x
        picking_trash_up()

    results.to_csv(join_paths(current_session_path, "cross_val.csv"))


if __name__ == "__main__":
    random_state: int = 42
    main(random_state=random_state)
