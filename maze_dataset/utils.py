import math

import numpy as np
from jaxtyping import Bool


def bool_array_from_string(
    string: str, shape: list[int], true_symbol: str = "T"
) -> Bool[np.ndarray, "*shape"]:
    """Transform a string into an ndarray of bools.

    Parameters
    ----------
    string: str
        The string representation of the array
    shape: list[int]
        The shape of the resulting array
    true_symbol:
        The character to parse as True. Whitespace will be removed. All other characters will be parsed as False.

    Returns
    -------
    np.ndarray
        A ndarray with dtype bool of shape `shape`

    Examples
    --------
    >>> bool_array_from_string(
    ...     "TT TF", shape=[2,2]
    ... )
    array([[ True,  True],
        [ True, False]])
    """
    stripped = "".join(string.split())

    expected_symbol_count = math.prod(shape)
    symbol_count = len(stripped)
    if len(stripped) != expected_symbol_count:
        raise ValueError(
            f"Connection List contains the wrong number of symbols. Expected {expected_symbol_count}. Found {symbol_count} in {stripped}."
        )

    bools = [True if symbol == true_symbol else False for symbol in stripped]
    return np.array(bools).reshape(*shape)


def corner_first_ndindex(n: int, ndim: int = 2) -> list[tuple]:
    """returns an array of indices, sorted by distance from the corner

    this gives the property that `np.ndindex((n,n))` is equal to
    the first n^2 elements of `np.ndindex((n+1, n+1))`

    ```
    >>> corner_first_ndindex(1)
    [(0, 0)]
    >>> corner_first_ndindex(2)
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    >>> corner_first_ndindex(3)
    [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2)]
    ```
    """

    unsorted: list = list(np.ndindex(tuple([n for _ in range(ndim)])))
    return sorted(
        unsorted, 
        key=lambda x: (max(x), x if x[0] % 2 == 0 else x[::-1])
    )

    # alternate numpy version from GPT-4:
    """
    # Create all index combinations
    indices = np.indices([n]*ndim).reshape(ndim, -1).T
    # Find the max value for each index
    max_indices = np.max(indices, axis=1)
    # Identify the odd max values
    odd_mask = max_indices % 2 != 0
    # Make a copy of indices to avoid changing the original one
    indices_copy = indices.copy()
    # Reverse the order of the coordinates for indices with odd max value
    indices_copy[odd_mask] = indices_copy[odd_mask, ::-1]
    # Sort by max index value, then by coordinates
    sorted_order = np.lexsort((*indices_copy.T, max_indices))
    return indices[sorted_order]
    """


def adj_list_to_nested_set(adj_list: list) -> set:
    """Used for comparison of adj_lists

    Adj_list looks like [[[0, 1], [1, 1]], [[0, 0], [0, 1]], ...]
    We don't care about order of coordinate pairs within
    the adj_list or coordinates within each coordinate pair."""
    return {
        frozenset([tuple(start_coord), tuple(end_coord)])
        for start_coord, end_coord in adj_list
    }
