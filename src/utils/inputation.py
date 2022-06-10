from numpy import nan_to_num, ndarray, nanmean, isnan, array, where, nanmedian
from scipy.stats import mode


def filling_mean(arr: ndarray) -> ndarray:
    mean = nanmean(arr)
    return nan_to_num(arr, nan=mean)


def filling_prev(arr: ndarray) -> ndarray:
    mask = isnan(arr)
    idxs = array(where(mask))[0]
    for idx in idxs:
        arr[idx] = arr[idx - 1] if idx > 0 else nanmean(arr)
    return arr


def filling_median(arr: ndarray) -> ndarray:
    median = nanmedian(arr)
    return nan_to_num(arr, nan=median)


def filling_mode(arr: ndarray) -> ndarray:
    return nan_to_num(arr, nan=mode(arr)[0][0])


def filling_max(arr: ndarray) -> ndarray:
    return nan_to_num(arr, nan=max(arr))
