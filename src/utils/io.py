from numpy import load
from execution_time_wrapper import get_execution_time_log
from typing import Any
from yaml import safe_load as load_yaml


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
