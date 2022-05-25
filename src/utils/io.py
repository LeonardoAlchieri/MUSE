from numpy import load


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
