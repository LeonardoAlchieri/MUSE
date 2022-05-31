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