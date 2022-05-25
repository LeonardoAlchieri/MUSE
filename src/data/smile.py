from src.utils.io import load_smile_data
from numpy import concatenate, ndarray


class SmileData(object):
    """Class used to load and get the different features of the Smile dataset. Indeed,
    the data, as provided by the authors (see https://compwell.rice.edu/workshops/embc2022/challenge),
    is structured as a nested dictionary.
    This class allows to access the different features of the dataset using some simple methods.

    Parameters
    ----------
    path_to_data : str
        path to the `.npy` file containing the data
    """

    def __init__(self, path_to_data: str):
        """Class used to load and get the different features of the Smile dataset. Indeed,
        the data, as provided by the authors (see https://compwell.rice.edu/workshops/embc2022/challenge),
        is structured as a nested dictionary.
        This class allows to access the different features of the dataset using some simple methods.

        Parameters
        ----------
        path_to_data : str
            path to the `.npy` file containing the data
        """

        data = load_smile_data(path_to_data)
        self.train = data["train"]
        self.test = data["test"]

    def get_handcrafted_features(
        self, test: bool = False, joined: bool = False, **kwargs
    ) -> dict[str, ndarray] | ndarray:
        """Get the hand crafted features of the dataset, either as a dicitonary or as
        a single array (obtained from the contatenation of the 2 different sets of features)

        Parameters
        ----------
        test : bool, optional
            if True, the test set will be selected, by default False
        joined : bool, optional
            if True, the return will be a concatenated dictionary, by default False

        Returns
        -------
        dict[str, ndarray] | ndarray
            the method returns the required data, either as a dictionary or as a single array
        """
        if test:
            data: dict = self.test["handcrafted_features"]
        else:
            data: dict = self.train["hand_crafted_features"]

        if "concat_axis" in kwargs:
            concat_axis: int = kwargs["concat_axis"]
        else:
            concat_axis: int = 2

        if joined:
            return concatenate(
                [data["ECG_features"], data["GSR_features"]], axis=concat_axis
            )
        else:
            return data

    def get_deep_features(
        self, test: bool = False, joined: bool = False, **kwargs
    ) -> dict[str, ndarray] | ndarray:
        """Get the deep features of the dataset, either as a dicitonary or as
        a single array (obtained from the contatenation of the 2 different sets of features)

        Parameters
        ----------
        test : bool, optional
            if True, the test set will be selected, by default False
        joined : bool, optional
            if True, the return will be a concatenated dictionary, by default False

        Returns
        -------
        dict[str, ndarray] | ndarray
            the method returns the required data, either as a dictionary or as a single array
        """
        if test:
            data: dict = self.test["deep_features"]
        else:
            data: dict = self.train["deep_features"]

        if "concat_axis" in kwargs:
            concat_axis: int = kwargs["concat_axis"]
        else:
            concat_axis: int = 2

        if joined:
            return concatenate(
                [data["ECG_features_C"], data["ECG_features_T"]], axis=concat_axis
            )
        else:
            return data

    def get_labels(self, test: bool = False, **kwargs) -> ndarray:
        """This method returns the labels of the dataset.

        Parameters
        ----------
        test : bool, optional
            if True, the test labels for the test set will be given, by default False

        Returns
        -------
        ndarray
            the method returns the array with the labels
        """
        if test:
            data: dict = self.test["labels"]
        else:
            data: dict = self.train["labels"]

        return data
