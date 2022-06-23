from itertools import chain, combinations
from typing import Iterable
from numpy import ndarray


def make_binary(y: ndarray) -> ndarray:
    """Simple method to make an array from categorical to binary. The method implements
    the fastest way to perform the operation in numpy.

    Parameters
    ----------
    y : ndarray
        input array, expected of type int (but the method will work if other types nonetheless,
        so beware)

    Returns
    -------
    ndarray
        the input array binarized

    Examples
    --------
    >>> a
    array([1, 0, 0, 0, 3, 2, 4, 1])
    >>> make_binary(a)
    array([1, 0, 0, 0, 1, 1, 1, 1])
    """
    return (y > 0).astype(int)


def all_subsets(ss: Iterable) -> chain:
    """Simple method to create all possible subsets of a list.
    Credits to Dan H: https://stackoverflow.com/a/5898031

    Parameters
    ----------
    ss : iterable
        iterable object over which to create the subsets

    Returns
    -------
    chain
        a chain object (iterable) of tuples, where each tuple is a
        subset of the input list. The subset empty is also generated
    """
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss) + 1)))
