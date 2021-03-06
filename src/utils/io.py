from glob import glob
from json import JSONEncoder
from logging import getLogger
from os import getlogin, makedirs, mkdir
from os.path import isdir, isfile
from os.path import join as join_paths
from shutil import copyfile, rmtree
from typing import Any, Dict, List
from warnings import warn

from execution_time_wrapper import get_execution_time_log
from numpy import load, ndarray

from yaml import safe_load as load_yaml

logger = getLogger(__name__)


def load_smile_data(path_to_data: str) -> dict:
    """Load the Smile data from the numpy file.

    Parameters
    ----------
    path_to_data : str
        path to the `.npy` file containing the data

    Returns
    -------
    dict
        returns the dictionary, as given by the authors (see https://compwell.rice.edu/workshops/embc2022/challenge)
    """
    data: dict = load(file=path_to_data, allow_pickle=True).item()
    return data


@get_execution_time_log
def load_config(path: str) -> dict[str, Any]:
    """Simple method to load yaml configuration for a given script.

    Parameters
    ----------
    path: str
        path to the yaml file

    Returns
    -------
    Dict[str, Any]
        the method returns a dictionary with the loaded configurations
    """
    with open(path, "r") as file:
        config_params = load_yaml(file)
    return config_params


@get_execution_time_log
def check_create_folder(path: str) -> None:
    """This method can be used to check if a folder exists and create it if it doesn't.

    Args:
        path (str): path to the folder to be checked and created
    """
    if isdir(path):
        logger.debug(f"Path exists. Not creating it.")
    elif isfile(path):
        raise RuntimeError(f"The output path {path} exists but is not a folder.")
    else:
        logger.warning(f"Creating output folder {path}.")
        makedirs(path)


@get_execution_time_log
def create_output_folder(
    path_to_config: str, task: str, cp_all_config: bool = False
) -> str:
    """This method can be used to create the output folder for the current task.

    Parameters
    ----------
    path_to_config : str
        path to the model configuration file, which will be copied with
        the rest of the information
    task : str
        task name
    cp_all_config : bool
        if True, all configuration files will be copied to the output folder,
        by default False

    Returns
    -------
    str
        the model returns the relative session output path

    Raises
    ------
    RuntimeError
        if the path for the model exists as a file, an error in thrown
    RuntimeError
        if the current identified session exists, an error is thrown
    """
    namespace: str = getlogin()
    base_output_path: str = "./results/train/"
    given_output_path: str = join_paths(base_output_path, task)
    logger.info(f"General output path: {given_output_path}")
    check_create_folder(path=given_output_path)

    current_sessions_ids: List[int] = [
        int("".join([char for char in session.split("/")[-1] if char.isdigit()]))
        for session in glob(join_paths(given_output_path, f"{namespace}session*"))
    ]
    if len(current_sessions_ids) == 0:
        current_session: int = 0
    else:
        last_session: int = max(current_sessions_ids)
        logger.debug(f"Last session: {last_session}")
        current_session: int = last_session + 1
    logger.debug(f"Current session: {current_session}")
    current_session_path: str = join_paths(
        given_output_path, f"{namespace}session{current_session}"
    )
    logger.info(f"Saving current session to {current_session_path}")
    if isdir(current_session_path):
        raise RuntimeError(f"The output path {current_session_path} already exists.")
    elif isfile(current_session_path):
        raise RuntimeError(
            f"The output path {current_session_path} exists but is not a folder."
        )
    else:
        mkdir(current_session_path)

    if not cp_all_config:
        copyfile(path_to_config, join_paths(current_session_path, "config.yaml"))
    else:
        current_path: str = join_paths(*path_to_config.split("/")[:-1])
        logger.info(f"Copying all configuration files in {current_session_path}")
        for file in glob(join_paths(current_path, "config*")):
            config_filename: str = file.split("/")[-1]
            copyfile(file, join_paths(current_session_path, config_filename))
    return current_session_path


@get_execution_time_log
def delete_output_folder_exception(task: str) -> bool:
    namespace: str = getlogin()
    base_output_path: str = "./results/train/"
    given_output_path: str = join_paths(base_output_path, task)
    logger.info(f"General output path: {given_output_path}")
    check_create_folder(path=given_output_path)

    current_sessions_ids: List[int] = [
        int("".join([char for char in session.split("/")[-1] if char.isdigit()]))
        for session in glob(join_paths(given_output_path, f"{namespace}session*"))
    ]
    if len(current_sessions_ids) == 0:
        current_session: int = 0
    else:
        last_session: int = max(current_sessions_ids)
        logger.debug(f"Last session: {last_session}")
        current_session: int = last_session

    current_session_path: str = join_paths(
        given_output_path, f"{namespace}session{current_session}"
    )

    logger.debug(f"Current session to be deleted: {current_session_path}")
    csv_files_in_current_session: list[str] = glob(
        join_paths(current_session_path, "*.csv")
    )
    if len(csv_files_in_current_session) > 0:
        warn(
            f"Found {len(csv_files_in_current_session)} csv files in {current_session_path}. Stopping delete"
        )
        logger.warning(
            f"Found {len(csv_files_in_current_session)} csv files in {current_session_path}. Stopping delete"
        )
        return False
    else:
        rmtree(current_session_path)
        return True


class NumpyEncoder(JSONEncoder):
    """Simple method to json-ify dictionary or other objects w/ numpy ndarrays
    in them.

    Credits to David Hempy on StackOverflow: https://stackoverflow.com/a/47626762
    """

    def default(self, obj: Any) -> JSONEncoder | list:
        """Simple method to json-ify dictionary or other objects w/ numpy ndarrays
        in them.

        Credits to David Hempy on StackOverflow: https://stackoverflow.com/a/47626762

        Parameters
        ----------
        obj : Any
            object to save

        Returns
        -------
        JSONEncoder
            returns the default json encoder if no ndarray is present, otherwise the object converted to list
        """
        if isinstance(obj, ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
