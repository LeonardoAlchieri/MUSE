from numpy import concatenate, ndarray
from logging import getLogger

from src.utils.io import load_smile_data

logger = getLogger(__name__)


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

    hand_crafted_features: list[str] = ["ECG_features", "GSR_features"]
    deep_features: list[str] = ["ECG_features_C", "ECG_features_T"]

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

    def _get_feature_type_from_feature_name(self, feature_name: str) -> str:
        if feature_name in self.hand_crafted_features:
            return "hand_crafted_features"
        elif feature_name in self.deep_features:
            return "deep_features"
        else:
            raise ValueError(f'Feature name "{feature_name}" not found in the dataset')

    def data_feature_level_joined(
        self,
        join_type: str,
        features: tuple[str, str] | str,
        get_labels: bool = True,
        test: bool = False,
    ) -> ndarray | tuple[ndarray, ndarray]:
        if isinstance(features, tuple):
            logger.debug(
                f"Got features as tuple: {features}. First position is type of features, while second is name of feature."
            )
            feature_type: str = features[0]
            feature_name: str = features[1]
        elif isinstance(features, str):
            logger.debug(
                f"Got features as string: {features}. Assuming the string is referred to the feature name"
            )
            feature_name: str = features
            feature_type = self._get_feature_type_from_feature_name(
                feature_name=feature_name
            )
        else:
            raise TypeError(
                f"Variable features can be either tuple of string or string. Got {type(features)} instead."
            )

        if feature_type == "hand_crafted_features":
            data: ndarray = self.get_handcrafted_features(joined=False)[feature_name]
        elif feature_type == "deep_features":
            data: ndarray = self.get_deep_features(joined=False)[feature_name]
        else:
            raise ValueError(f'Feature type "{feature_type}" not found in the dataset')

        if join_type == "average":
            return data.mean(axis=1) if get_labels else data.mean(
                axis=1
            ), self.get_labels(test=test)
        elif join_type == "concat_feature_level":
            return data.reshape(data.shape[0], -1) if get_labels else data.reshape(
                data.shape[0], -1
            ), self.get_labels(test=test)
        elif join_type == "concat_label_level":
            # TODO: implement where for each label, we have to spawn 60 more!
            # return data.reshape(-1, data.shape[-1]) if get_labels else tuple(data.reshape(-1, data.shape[-1]),
            ...
            raise NotImplementedError(
                "Leo was lazy and has not implemented this join_type yet! 😴"
            )
        else:
            raise ValueError(
                f'Join type "{join_type}" not recognized. Accepted values are "average", "concat_feature_level" and "concat_label_level"'
            )
