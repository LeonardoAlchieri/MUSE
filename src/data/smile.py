from typing import Callable
from numpy import concatenate, isin, ndarray
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
    test : bool
        if True, the dataset will be loaded from the test set, otherwise from the train set
    path_to_data : str
        path to the `.npy` file containing the data
    """

    hand_crafted_features: list[str] = ["ECG_features", "GSR_features"]
    deep_features: list[str] = ["ECG_features_C", "ECG_features_T"]
    unravelled: bool = False

    def __init__(
        self,
        path_to_data: str,
        test: bool = False,
        debug_mode: bool = False,
        unravelled: bool = False,
    ):
        """Class used to load and get the different features of the Smile dataset. Indeed,
        the data, as provided by the authors (see https://compwell.rice.edu/workshops/embc2022/challenge),
        is structured as a nested dictionary.
        This class allows to access the different features of the dataset using some simple methods.

        Parameters
        ----------
        test : bool
            if True, the dataset will be loaded from the test set, otherwise from the train set
        path_to_data : str
            path to the `.npy` file containing the data
        debug_mode : bool
            if True, only a portion of the dataset will be loaded, by default False
        """

        data = load_smile_data(path_to_data)
        self.data = data["train"] if not test else data["test"]
        self.unravelled = unravelled
        if debug_mode:
            logger.warning(
                "Debug mode activated, only a portion of the dataset will be loaded"
            )
            for feature_type in self.data.keys():
                if isinstance(self.data[feature_type], dict):
                    for feature in self.data[feature_type].keys():
                        self.data[feature_type][feature] = self.data[feature_type][
                            feature
                        ][:50]
                else:
                    self.data[feature_type] = self.data[feature_type][:50]

    def get_handcrafted_features(
        self, joined: bool = False, **kwargs
    ) -> dict[str, ndarray] | ndarray:
        """Get the hand crafted features of the dataset, either as a dicitonary or as
        a single array (obtained from the contatenation of the 2 different sets of features)

        Parameters
        ----------
        joined : bool, optional
            if True, the return will be a concatenated dictionary, by default False

        Returns
        -------
        dict[str, ndarray] | ndarray
            the method returns the required data, either as a dictionary or as a single array
        """
        data: dict = self.data["hand_crafted_features"]

        if "concat_axis" in kwargs:
            concat_axis: int = kwargs["concat_axis"]
        else:
            concat_axis: int = 2

        if joined:
            return concatenate(
                [
                    data[self.hand_crafted_features[0]],
                    data[self.hand_crafted_features[1]],
                ],
                axis=concat_axis,
            )
        else:
            return data

    def get_deep_features(
        self, joined: bool = False, **kwargs
    ) -> dict[str, ndarray] | ndarray:
        """Get the deep features of the dataset, either as a dicitonary or as
        a single array (obtained from the contatenation of the 2 different sets of features)

        Parameters
        ----------
        joined : bool, optional
            if True, the return will be a concatenated dictionary, by default False

        Returns
        -------
        dict[str, ndarray] | ndarray
            the method returns the required data, either as a dictionary or as a single array
        """
        data: dict = self.data["deep_features"]

        if "concat_axis" in kwargs:
            concat_axis: int = kwargs["concat_axis"]
        else:
            concat_axis: int = 2

        if joined:
            return concatenate(
                [data[self.deep_features[0]], data[self.deep_features[1]]],
                axis=concat_axis,
            )
        else:
            return data

    def set_handcrafted_feature(self, feature: str, data: ndarray) -> None:
        """Method to set the hand crafted feature of the dataset.

        Parameters
        ----------
        feature : str
            feature of the handcrafter features to set
        data : ndarray
            array to substitute the current data
        """
        self.data["hand_crafted_features"][feature] = data

    def set_deep_feature(self, feature: str, data: ndarray) -> None:
        """Method to set the deep feature of the dataset.

        Parameters
        ----------
        feature : str
            feature of the deep features to set
        data : ndarray
            array to substitute the current data
        """
        self.data["deep_features"][feature] = data

    def set_labels(self, labels: ndarray) -> None:
        """Method to set the labels of the dataset.

        Parameters
        ----------
        labels : ndarray
            array to substitute the current data
        """
        self.data["labels"] = labels

    def get_labels(self) -> ndarray:
        """This method returns the labels of the dataset.

        Returns
        -------
        ndarray
            the method returns the array with the labels
        """

        return self.data["labels"]

    def _get_feature_type_from_feature_name(self, feature_name: str) -> str:
        # TODO: add docstring
        if feature_name in self.hand_crafted_features:
            return "hand_crafted_features"
        elif feature_name in self.deep_features:
            return "deep_features"
        else:
            raise ValueError(f'Feature name "{feature_name}" not found in the dataset')

    def check_features_input(self, features: tuple[str, str] | str) -> tuple[str, str]:
        """Method to check if the input of the variable `features` is a tuple or a string, and give
        the correct `feature_name` and `feature_type` accordingly.

        Parameters
        ----------
        features : tuple[str, str] | str
            features as tuple or string with just the feature name.

        Returns
        -------
        tuple[str, str]
            the  method returns `feature_name`, `feature_type`

        Raises
        ------
        TypeError
            if the type of `features` is not tuple or str, the method fails
        """
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

        return feature_name, feature_type

    def data_feature_level_joined(
        self,
        join_type: str,
        features: tuple[str, str] | str,
        get_labels: bool = True,
    ) -> ndarray | tuple[ndarray, ndarray]:
        # TODO: add docstring

        feature_name, feature_type = self.check_features_input(features=features)

        # FIXME: very bad implementation -> should have a single method at this point
        if feature_type == "hand_crafted_features":
            data: ndarray = self.get_handcrafted_features(joined=False)[feature_name]
        elif feature_type == "deep_features":
            data: ndarray = self.get_deep_features(joined=False)[feature_name]
        else:
            raise ValueError(f'Feature type "{feature_type}" not found in the dataset')

        if join_type == "window_average":
            return (
                data.mean(axis=1) if get_labels else data.mean(axis=1),
                self.get_labels(),
            )
        elif join_type == "feature_average":
            return (
                data.mean(axis=2) if get_labels else data.mean(axis=2),
                self.get_labels(),
            )
        elif join_type == "concat_feature_level":
            return (
                data.reshape(data.shape[0], -1)
                if get_labels
                else data.reshape(data.shape[0], -1),
                self.get_labels(),
            )
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

    def fill_missing_values(
        self, features: tuple[str, str] | str, filling_method: Callable
    ):
        feature_name, feature_type = self.check_features_input(features=features)

        if feature_type == "hand_crafted_features":
            data: ndarray = self.get_handcrafted_features(joined=False)[feature_name]
        elif feature_type == "deep_features":
            data: ndarray = self.get_deep_features(joined=False)[feature_name]
        else:
            raise ValueError(f'Feature type "{feature_type}" not found in the dataset')
        # shape (2070, 60, M). I have to fill, for each M, over the 60 other
        for user in range(data.shape[0]):
            for feat in range(data.shape[2]):
                data[user, :, feat] = filling_method(data[user, :, feat])

        if feature_type == "hand_crafted_features":
            self.set_handcrafted_feature(feature_name, data)
        elif feature_type == "deep_features":
            self.set_deep_feature(feature_name, data)
        else:
            raise ValueError(f'Feature type "{feature_type}" not found in the dataset')

    def _get_time_duration(self):
        """Method to get the time duration of the dataset.

        Will fail the data has been unravelled.
        """
        hand_crafted_data: dict[str, ndarray] = self.get_handcrafted_features(
            joined=False
        )
        deep_data: dict[str, ndarray] = self.get_deep_features(joined=False)
        durations = list()
        durations.append(
            [hand_crafted_data[feat].shape[1] for feat in self.hand_crafted_features]
        )
        durations.append([deep_data[feat].shape[1] for feat in self.deep_features])
        duration: list = list(set(durations))
        if len(duration) > 1:
            raise ValueError(
                f"The dataset has different durations for handcrafted and deep features. "
                f"Got {duration}."
            )
        else:
            return duration[0]

    def unravel(self):
        """Method to unravel the dataset.

        The method unravels the dataset by concatenating all the features in the dataset.
        """
        hand_crafted_data: dict[str, ndarray] = self.get_handcrafted_features(
            joined=False
        )
        deep_data: dict[str, ndarray] = self.get_deep_features(joined=False)
        for feat in self.hand_crafted_features:
            self.set_handcrafted_feature(
                data=hand_crafted_data[feat].reshape(
                    -1, hand_crafted_data[feat].shape[-1]
                ),
                feature=feat,
            )
        for feat in self.deep_features:
            self.set_deep_feature(
                data=deep_data[feat].reshape(-1, deep_data[feat].shape[-1]),
                feature=feat,
            )

        self.set_labels(self.get_labels(), self._get_time_duration())
        self.unravelled: bool = True
