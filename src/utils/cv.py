from numpy import ndarray, concatenate, arange
from warnings import warn


def make_unravelled_folds(
    t: int, n_folds: int, n_data: int = 2070
) -> list[tuple[ndarray, ndarray]]:
    """Method to create, for the unravelled data, i.e. of shape ``(2070*t, N_FEATURES)``,
    the folds for the cross-validation. This is needed since otherwise some problems,
    with some timeseries being cut in one fold and not present in another, might arise.``

    Parameters
    ----------
    t : int
        length of the timestep considered, e.g. `t=60` for a 60-minute-long time series
    n_folds : int
        number of folds for the crossvalidation
    n_data : int, optional
        lenght of the input array, by default 2070

    Returns
    -------
    list[tuple[ndarray, ndarray]]
        the method returns a list of tuples, where each tuple contains is of type (train,test).
        This is the structure required by sklearn for the ``cross_val_score`` method.
    """

    def make_unravelled_idx(idx_arr: ndarray, t: int) -> ndarray:
        idx_arr = (idx_arr + 1) * t
        return concatenate(list(arange(el - t, el) for el in idx_arr))

    # this array represent the original n_data labels
    idx_arr: ndarray = arange(0, n_data)
    len_fold = len(idx_arr) // n_folds
    if len_fold * n_folds != len(idx_arr):
        warn(
            f"The array length is not nicely dividible into folds of same length. Some data discarded."
        )
        # TODO: add a way to handle this phenomenon

    separate_idx_arr: list[ndarray] = [
        idx_arr[i * len_fold : (i + 1) * len_fold] for i in range(n_folds)
    ]

    cv: list[tuple[ndarray, ndarray]] = [
        (
            concatenate(
                [
                    make_unravelled_idx(separate_idx_arr[j], t=t)
                    for j in range(len(separate_idx_arr))
                    if j != i
                ]
            ),
            make_unravelled_idx(separate_idx_arr[i], t=t),
        )
        for i in range(n_folds)
    ]
    return cv
