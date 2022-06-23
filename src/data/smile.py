from json import dump as jspn_dump
from logging import getLogger
from pickle import HIGHEST_PROTOCOL as pickle_protocol_high
from pickle import dump as pickle_dump
from typing import Callable, Union

from numpy import concatenate, delete, ndarray, repeat
from numpy import save as numpy_save
from numpy import swapaxes
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection._univariate_selection import (
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    _BaseFilter,
)

from src.utils.io import NumpyEncoder, load_smile_data

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
        st_feat: bool = True,
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
        if "train" in data.keys() or "test" in data.keys():
            self.data = data["train"] if not test else data["test"]
        else:
            self.data = data

        self.test = test
        self.unravelled = unravelled
        if st_feat:
            self.hand_crafted_features: list[str] = [
                "ECG_features",
                "GSR_features",
                "ST_features",
            ]
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

    def separate_skin_temperature(self) -> None:
        """Method to separate the skin temperature from the deep features."""

        gsr_data: ndarray = self.get_handcrafted_features(joined=False)["GSR_features"]
        self.hand_crafted_features: list[str] = [
            "ECG_features",
            "GSR_features",
            "ST_features",
        ]

        st_data: ndarray = gsr_data[:, :, -4:]
        gsr_data: ndarray = gsr_data[:, :, :-4]
        logger.info(f"ST feature shape: {st_data.shape}")
        logger.info(f"GSR feature shape: {gsr_data.shape}")
        self.set_handcrafted_feature(feature="GSR_features", data=gsr_data)
        self.set_handcrafted_feature(feature="ST_features", data=st_data)

    @staticmethod
    def remove_masking(data: dict[str, ndarray]) -> dict[str, ndarray]:
        """Method to remove the masking arrays from the dataset.

        Parameters
        ----------
        data : dict[str, ndarray]
        """
        return {key: val for key, val in data.items() if not key.endswith("_masking")}

    def get_handcrafted_features(
        self,
        joined: bool = False,
        masking: bool = False,
        **kwargs,
    ) -> dict[str, ndarray] | ndarray:
        """Get the hand crafted features of the dataset, either as a dicitonary or as
        a single array (obtained from the contatenation of the 2 different sets of features)

        Parameters
        ----------
        joined : bool, optional
            if True, the return will be a concatenated dictionary, by default False
        masking : bool, optional
            if True, the masking arrays will be given, otherwise not, by default True

        Returns
        -------
        dict[str, ndarray] | ndarray
            the method returns the required data, either as a dictionary or as a single array
        """
        data: dict = self.data["hand_crafted_features"]
        if not masking:
            data = self.remove_masking(data)

        if "concat_axis" in kwargs:
            concat_axis: int = kwargs["concat_axis"]
        else:
            concat_axis: int = 2 if not self.unravelled else 1

        if joined:
            return concatenate(
                [
                    data[self.hand_crafted_features[0]],
                    data[self.hand_crafted_features[1]],
                    data[self.hand_crafted_features[2]],
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

        Raises
        ------
        RuntimeError
            if the test set is selected, no labels are present
        """
        if not self.test:
            return self.data["labels"]
        else:
            raise RuntimeError(
                "You asked the labels for the test set: they are not available!"
            )

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
                swapaxes(data, 1, 2).reshape(data.shape[0], -1)
                if get_labels
                else swapaxes(data, 1, 2).reshape(data.shape[0], -1),
                self.get_labels(),
            )
        elif join_type == "concat_label_level":
            # TODO: implement where for each label, we have to spawn 60 more!
            # return data.reshape(-1, data.shape[-1]) if get_labels else tuple(data.reshape(-1, data.shape[-1]),
            return data, self.get_labels()
        else:
            raise ValueError(
                f'Join type "{join_type}" not recognized. Accepted values are "average", "concat_feature_level" and "concat_label_level"'
            )

    def fill_missing_values(
        self, features: tuple[str, str] | str, filling_method: Callable | None
    ):
        if filling_method is None:
            return None
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
        if self.unravelled:
            raise ValueError(
                "The dataset has been unravelled. You can't get the time duration of the dataset."
            )
        hand_crafted_data: dict[str, ndarray] = self.get_handcrafted_features(
            joined=False
        )
        deep_data: dict[str, ndarray] = self.get_deep_features(joined=False)
        durations = list()
        durations.extend(
            [hand_crafted_data[feat].shape[1] for feat in self.hand_crafted_features]
        )
        durations.extend([deep_data[feat].shape[1] for feat in self.deep_features])
        duration: list = list(set(durations))
        if len(duration) > 1:
            raise ValueError(
                f"The dataset has different durations for handcrafted and deep features. "
                f"Got {duration}."
            )
        else:
            return duration[0]

    def unravel(self, inplace: bool = True) -> Union[None, "SmileData"]:
        """Method to unravel the dataset.

        The method unravels the dataset by concatenating all the features in the dataset.

        Parameters
        ----------
        inplace : bool, optional
            if True, the method will modify the smile data inplace. The default is True.

        Returns
        -------
        None | 'SmileData'
            if inplace is True, the method returns None. Otherwise, it returns a new SmileData object.
        """
        if self.test:
            # FIXME: this is just a gargabe workaround to get the method not to fail
            self.test = False
            self.set_labels(repeat(self.get_labels(), self._get_time_duration()))
            self.test = True
        else:
            self.set_labels(repeat(self.get_labels(), self._get_time_duration()))

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

        self.unravelled: bool = True

    def save(self, path: str, format: str = "dict"):
        """Save the dataset to a file. The format can be either 'dict' or 'pickle'.

        Parameters
        ----------
        path : str
            path for saving
        format : str, optional
            format for the file to be saved with. Accepted are:
            - 'dict', which follows the initial Smile dataset structure of a nested dictionary
            - 'pickle', which saves the dataset as a SmileDataset sereliazed object
            - 'json', which uses a json
            by default 'dict'

        Raises
        ------
        ValueError
            if a wrong format is given, the method will fail
        """
        if format == "dict":
            numpy_save(f"{path}.npy", self.data)
        elif format == "pickle":
            with open(f"{path}.pkl", "wb") as f:
                pickle_dump(self, f, pickle_protocol_high)
        elif format == "json":
            with open(f"{path}.json", "w") as f:
                jspn_dump(self.data, f, cls=NumpyEncoder)
        else:
            raise ValueError(
                f'Format "{format}" not recognized. Accepted values are "dict", "pickle" and "json"'
            )

    def timecut(self, timestep_length: int = 60) -> None:
        """This method allows to reduce the timestep "back", w/ respect to the stress label.

        Parameters
        ----------
        timestep_length : int, optional
            number of steps to be considered out of 60 (max), by default 60
        """
        if timestep_length == 60:
            logger.warning("The timestep length is set to 60. Nothing to do.")
            return None

        hand_crafted_data: dict[str, ndarray] = self.get_handcrafted_features(
            joined=False
        )
        deep_data: dict[str, ndarray] = self.get_deep_features(joined=False)

        for feat in self.hand_crafted_features:
            self.set_handcrafted_feature(
                data=hand_crafted_data[feat][:, -timestep_length:, :],
                feature=feat,
            )

        for feat in self.deep_features:
            self.set_deep_feature(
                data=deep_data[feat][:, -timestep_length:, :],
                feature=feat,
            )

    def remove_flatlines(self) -> None:
        """The dataset contains, for some labels and some features,
        some timeseries which are completely 0, which are referred to as
        "flatlines".

        From descriptive analysis, the data without any flatlines is about
        1500 samples, out of 2070, or about 70%.

        This method removes the flatlines from the dataset, in order to
        have data which is only "clean".
        """
        hand_crafted_data: dict[str, ndarray] = self.get_handcrafted_features(
            joined=True
        )
        deep_data: dict[str, ndarray] = self.get_deep_features(joined=True)
        labels: ndarray = self.get_labels()

        def get_non_flatline_indexes(x: ndarray) -> list[int]:
            return [idx for idx, row in enumerate(x) for feat in row if sum(feat) == 0]

        hand_crafted_data = swapaxes(hand_crafted_data, 1, 2)
        idxs_to_remove = get_non_flatline_indexes(x=hand_crafted_data)
        hand_crafted_data_clean: ndarray = delete(
            hand_crafted_data, idxs_to_remove, axis=0
        )
        # swap back the axes, in order to have (N, 60, 20)
        hand_crafted_data_clean = swapaxes(hand_crafted_data_clean, 1, 2)
        hand_crafted_data_clean: dict[ndarray] = (
            {
                self.hand_crafted_features[0]: hand_crafted_data_clean[:, :, :8],
                self.hand_crafted_features[1]: hand_crafted_data_clean[:, :, 8:16],
                self.hand_crafted_features[2]: hand_crafted_data_clean[:, :, 16:],
            }
            if len(self.hand_crafted_features) == 3
            else {
                self.hand_crafted_features[0]: hand_crafted_data_clean[:, :, :8],
                self.hand_crafted_features[1]: hand_crafted_data_clean[:, :, 8:],
            }
        )
        for feature_name, feature_data in hand_crafted_data_clean.items():
            self.set_handcrafted_feature(data=feature_data, feature=feature_name)

        deep_data = swapaxes(deep_data, 1, 2)
        deep_data_clean: ndarray = delete(deep_data, idxs_to_remove, axis=0)
        deep_data_clean = swapaxes(deep_data_clean, 1, 2)
        deep_data_clean: dict[ndarray] = {
            self.deep_features[0]: deep_data_clean[:, :, :256],
            self.deep_features[1]: deep_data_clean[:, :, 256:],
        }
        for feature_name, feature_data in deep_data_clean.items():
            self.set_deep_feature(data=feature_data, feature=feature_name)

        labels_clean: ndarray = delete(labels, idxs_to_remove, axis=0)
        self.set_labels(labels_clean)

    @staticmethod
    def _get_feature_selection_criterion(criterion: str) -> Callable:
        if criterion == "correlation":
            return pearsonr
        elif criterion == "mutual information":
            return mutual_info_classif
        elif criterion == "chi-square":
            return chi2
        elif criterion == "f-score":
            return f_classif
        else:
            raise ValueError(
                f'Criterion "{criterion}" not recognized. Accepted values are "correlation", "mutual information", "chi-square" and "f-score"'
            )

    @staticmethod
    def _get_feature_selection_method(method: str) -> _BaseFilter:
        if method == "percentage":
            return SelectPercentile
        elif method == "fixed number":
            return SelectKBest
        elif method == "p value":
            return SelectFwe
        else:
            raise ValueError(
                f'Method "{method}" not recognized. Accepted values are "percentage", "fixed number" and "p value"'
            )

    @staticmethod
    def _get_method_attribute_name(method: str) -> _BaseFilter:
        if method == "percentage":
            return "percentile"
        elif method == "fixed number":
            return "k"
        elif method == "p value":
            return "alpha"
        else:
            raise ValueError(
                f'Method "{method}" not recognized. Accepted values are "percentage", "fixed number" and "p value"'
            )

    def feature_selection(
        self,
        criterion: str,
        method: str,
        method_attribute: int | float,
        joined: bool = False,
        deep_features: bool = False,
    ) -> dict[str, ndarray] | dict[str, dict[str, ndarray]]:
        """This method allows to select the features to be used in the model.

        Parameters
        ----------
        criterion : str
            criterion to be used for the feature selection. Accepted are:
            - 'correlation', which uses the correlation coefficient
            - 'mutual information' which uses the mutual information
            - 'chi-square' which uses the chi-square
            - 'f-score' which uses the f-score

        method : str
            method to be used for the feature selection. Accepted are:
            - 'percentage' which uses the percentage of the features to be kept
            - 'fixed number' which uses a fixed number of features to be kept
            - ' p value' which uses the p value of the features to be kept

        method_attribute : int | float
            attribute related to the method used, e.g. 10 for `method='percentage'`

        joined : bool, optional
            if True, the data is joined before the feature selection, otherwise not

        deep_features : bool, optional
            if True, the feature selection is applied to the deep features as well

        Returns
        -------
        dict[str, ndarray] | dict[str, dict[str, ndarray]]
            if deep_features is False, a dict with the selected features as keys and the data as values;
            otherwise, one more level for the dict, distinguishing between the handcrafted and deep features,
            will be available.
        """
        criterion = self._get_feature_selection_criterion(criterion=criterion)
        method_attribute: dict[str, int | float] = {
            self._get_method_attribute_name(method=method): method_attribute
        }
        method: _BaseFilter = self._get_feature_selection_method(method=method)

        hand_crafted_data: dict[str, ndarray] | ndarray = self.get_handcrafted_features(
            joined=joined
        )
        labels: ndarray = self.get_labels()

        if deep_features:
            deep_data: dict[str, ndarray] | ndarray = self.get_deep_features(
                joined=joined
            )

        if joined:
            # TODO: implement feature selection for joined data.
            raise NotImplementedError(
                "I still have not implemented feature selection for when the data is joined together."
            )
        else:
            result_idx: dict[str, ndarray] = {}
            for feature_name, feature_data in hand_crafted_data.items():
                x: ndarray = feature_data
                y: ndarray = labels
                feature_selector: _BaseFilter = method(
                    score_func=criterion, **method_attribute
                )
                feature_data_trimmed: ndarray = feature_selector.fit_transform(X=x, y=y)
                self.set_handcrafted_feature(
                    feature=feature_name, data=feature_data_trimmed
                )
                result_idx[feature_name] = feature_selector.get_support(indices=True)
                del feature_selector

            if deep_features:
                result_idx: dict[str, dict[str, ndarray]] = dict(
                    handcrafted_features=result_idx
                )
                for feature_name, feature_data in deep_data.items():
                    x: ndarray = swapaxes(feature_data, 1, 2).reshape(
                        feature_data.shape[0], -1
                    )
                    y: ndarray = labels
                    feature_selector: _BaseFilter = method(criterion, method_attribute)
                    feature_data_trimmed: ndarray = feature_selector.fit_transform(
                        X=x, y=y
                    )
                    self.set_deep_feature(
                        feature=feature_name, data=feature_data_trimmed
                    )
                    result_idx["deep_features"][
                        feature_name
                    ] = feature_selector.get_support(indices=True)
                    del feature_selector
                return result_idx
            else:
                return result_idx

    def trim_features_selected(
        self,
        idxs: dict[str, ndarray] | dict[str, dict[str, ndarray]],
        deep_features: bool = False,
        joined: bool = False,
        **kwargs,
    ) -> None:
        """This method allows to select a subset of features from a given array of indeces.

        Parameters
        ----------
        idxs : dict[str, ndarray] | dict[str, dict[str, ndarray]]
            if deep_features is False, a dict with the selected features as keys and the indces as values;
            otherwise, one more level for the dict, distinguishing between the handcrafted and deep features.

        deep_features : bool, optional
            if True, the feature selection is applied to the deep features as well

        joined : bool, optional
            if True, the data is joined before the feature selection, otherwise not
        """
        hand_crafted_data: dict[str, ndarray] | ndarray = self.get_handcrafted_features(
            joined=joined
        )
        if deep_features:
            deep_data: dict[str, ndarray] | ndarray = self.get_deep_features(
                joined=joined
            )
            hand_crafted_idxs: dict[str, ndarray] = idxs["handcrafted_features"]
            deep_idxs: dict[str, ndarray] = idxs["deep_features"]

            for feature_name, idx in hand_crafted_idxs.items():
                self.set_handcrafted_feature(
                    feature=feature_name,
                    data=hand_crafted_data[feature_name][:, idx],
                )

            for feature_name, idx in deep_idxs.items():
                self.set_deep_feature(
                    feature=feature_name,
                    data=deep_data[feature_name][:, idx],
                )
        else:
            for feature_name, idx in idxs.items():
                self.set_handcrafted_feature(
                    feature=feature_name,
                    data=hand_crafted_data[feature_name][:, idx],
                )
