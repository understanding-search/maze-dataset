"""utilities required by MazeTokenizer"""

import re
import typing
from typing import Callable, Iterable, Generator
from collections import Counter
from jaxtyping import Float, Int8

import numpy as np
from muutils.misc import list_join

from maze_dataset.constants import CoordTup, ConnectionList
from maze_dataset.utils import WhenMissing

# coordinate to strings
# ==================================================


def _coord_to_strings_UT(coord: typing.Sequence[int]) -> list[str]:
    """convert a coordinate to a string: `(i,j)`->"(i,j)"
    always returns a list of length 1"""
    return [f"({','.join(str(c) for c in coord)})"]


def _coord_to_strings_indexed(coord: typing.Sequence[int]) -> list[str]:
    """convert a coordinate to a list of indexed strings: `(i,j)`->"(", "i", ",", "j", ")" """
    return [
        "(",
        *list_join([str(c) for c in coord], lambda: ","),
        ")",
    ]


# string to coordinate representation
# ==================================================


def str_is_coord(coord_str: str, allow_whitespace: bool = True) -> bool:
    """return True if the string represents a coordinate, False otherwise"""

    strip_func: Callable[[str], str] = lambda x: x.strip() if allow_whitespace else x

    coord_str = strip_func(coord_str)

    return all(
        [
            coord_str.startswith("("),
            coord_str.endswith(")"),
            "," in coord_str,
            all(
                [
                    strip_func(x).isdigit()
                    for x in strip_func(coord_str.lstrip("(").rstrip(")")).split(",")
                ]
            ),
        ]
    )


def coord_str_to_tuple(
    coord_str: str, allow_whitespace: bool = True
) -> tuple[int, ...]:
    """convert a coordinate string to a tuple"""
    strip_func: Callable[[str], str] = lambda x: x.strip() if allow_whitespace else x
    coord_str = strip_func(coord_str)
    stripped: str = strip_func(coord_str.lstrip("(").rstrip(")"))
    return tuple(int(strip_func(x)) for x in stripped.split(","))


def coord_str_to_coord_np(coord_str: str, allow_whitespace: bool = True) -> np.ndarray:
    """convert a coordinate string to a numpy array"""
    return np.array(coord_str_to_tuple(coord_str, allow_whitespace=allow_whitespace))


def coord_str_to_tuple_noneable(coord_str: str) -> CoordTup | None:
    """convert a coordinate string to a tuple, or None if the string is not a coordinate string"""
    if not str_is_coord(coord_str):
        return None
    return coord_str_to_tuple(coord_str)


def coords_string_split_UT(coords: str) -> list[str]:
    """Splits a string of tokens into a list containing the UT tokens for each coordinate.

    Not capable of producing indexed tokens ("(", "1", ",", "2", ")"), only unique tokens ("(1,2)").
    Non-whitespace portions of the input string not matched are preserved in the same list:
    "(1,2) <SPECIAL_TOKEN> (5,6)" -> ["(1,2)", "<SPECIAL_TOKEN>", "(5,6)"]
    """
    # ty gpt4
    return re.findall(r"\([^)]*\)|\S+", coords)


# back and forth in wrapped form
# ==================================================
def strings_to_coords(
    text: str | list[str],
    when_noncoord: WhenMissing = "skip",
) -> list[str | CoordTup]:
    """converts a list of tokens to a list of coordinates

    returns list[CoordTup] if `when_noncoord` is "skip" or "error"
    returns list[str | CoordTup] if `when_noncoord` is "include"
    """
    tokens_joined: str = text if isinstance(text, str) else " ".join(text)
    tokens_processed: list[str] = coords_string_split_UT(tokens_joined)
    result: list[str] = list()
    for token in tokens_processed:
        coord: CoordTup | None = coord_str_to_tuple_noneable(token)
        if coord is None:
            if when_noncoord == "skip":
                continue
            elif when_noncoord == "error":
                raise ValueError(
                    f"Invalid non-coordinate token '{token}' in text: '{text}'"
                )
            elif when_noncoord == "include":
                result.append(token)
            else:
                raise ValueError(f"Invalid when_noncoord value '{when_noncoord}'")
        else:
            result.append(coord)
    return result


def coords_to_strings(
    coords: list[str | CoordTup],
    coord_to_strings_func: Callable[[CoordTup], list[str]],
    when_noncoord: WhenMissing = "skip",
) -> list[str]:
    """converts a list of coordinates to a list of strings (tokens)

    expects list[CoordTup] if `when_noncoord` is "error"
    expects list[str | CoordTup] if `when_noncoord` is "include" or "skip"
    """
    result: list[str] = list()
    for coord in coords:
        if isinstance(coord, str):
            if when_noncoord == "skip":
                continue
            elif when_noncoord == "error":
                raise ValueError(
                    f"Invalid non-coordinate '{coord}' in list of coords: '{coords}'"
                )
            elif when_noncoord == "include":
                result.append(coord)
            else:
                raise ValueError(f"Invalid when_noncoord value '{when_noncoord}'")
        else:
            result.extend(coord_to_strings_func(coord))
    return result


def connection_list_to_adj_list(
    conn_list: ConnectionList,
    shuffle_d0: bool = True, 
    shuffle_d1: bool = True
    ) ->  Int8[np.ndarray, "conn start_end coord"]:
    n_connections = conn_list.sum()
    adj_list: Int8[np.ndarray, "conn start_end coord"] = np.full(
            (n_connections, 2, 2),
            -1,
        )

    if shuffle_d1:
        flip_d1: Float[np.array, "conn"] = np.random.rand(n_connections)

    # loop over all nonzero elements of the connection list
    i: int = 0
    for d, x, y in np.ndindex(conn_list.shape):
        if conn_list[d, x, y]:
            c_start: CoordTup = (x, y)
            c_end: CoordTup = (
                x + (1 if d == 0 else 0),
                y + (1 if d == 1 else 0),
            )
            adj_list[i, 0] = np.array(c_start)
            adj_list[i, 1] = np.array(c_end)

            # flip if shuffling
            if shuffle_d1 and (flip_d1[i] > 0.5):
                c_s, c_e = adj_list[i, 0].copy(), adj_list[i, 1].copy()
                adj_list[i, 0] = c_e
                adj_list[i, 1] = c_s

            i += 1

    if shuffle_d0:
        np.random.shuffle(adj_list)

    return adj_list


def equal_except_adj_list_sequence(rollout1: list[str], rollout2: list[str]) -> bool:
    """Returns if the rollout strings are equal, allowing for differently sequenced adjacency lists.
    <ADJLIST_START> and <ADJLIST_END> tokens must be in the rollouts.
    Intended ONLY for determining if two tokenization schemes are the same for rollouts generated from the same maze.
    This function should NOT be used to determine if two rollouts encode the same `LatticeMaze` object.
    
    # Warning: CTT False Positives
    This function is not robustly correct for some corner cases using `CoordTokenizers.CTT`.
    If rollouts are passed for identical tokenizers processing two slightly different mazes, a false positive is possible.
    More specifically, some cases of zero-sum adding and removing of connections in a maze within square regions along the diagonal will produce a false positive.    
    """
    def get_token_regions(toks: list[str]) -> tuple[list[str], list[str]]:
        adj_list_start, adj_list_end = toks.index("<ADJLIST_START>") + 1, toks.index(
            "<ADJLIST_END>"
        )
        adj_list = toks[adj_list_start:adj_list_end]
        non_adj_list = toks[:adj_list_start] + toks[adj_list_end:]
        return adj_list, non_adj_list
    
    if len(rollout1) != len(rollout2): return False
    if ("<ADJLIST_START>" in rollout1) ^ ("<ADJLIST_START>" in rollout2): return False
    if ("<ADJLIST_END>" in rollout1) ^ ("<ADJLIST_END>" in rollout2): return False
    
    adj_list1, non_adj_list1 = get_token_regions(rollout1)
    adj_list2, non_adj_list2 = get_token_regions(rollout2)
    if non_adj_list1 != non_adj_list2:
        return False
    counter1: Counter = Counter(adj_list1)
    counter2: Counter = Counter(adj_list2)
    return counter1 == counter2

def flatten(it: Iterable[any], levels_to_flatten: int | None = None) -> Generator:
    """
    Flattens an arbitrarily nested iterable.
    Flattens all iterable data types except for `str` and `bytes`.
    
    # Returns
    Generator over the flattened sequence.
    
    # Parameters
    - `it`: Any arbitrarily nested iterable.
    - `levels_to_flatten`: Number of levels to flatten by. If `None`, performs full flattening.
    """
    for x in it:
        # TODO: swap type check with more general check for __iter__() or __next__() or whatever
        if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)) and (levels_to_flatten is None or levels_to_flatten > 0):
            yield from flatten(x, None if levels_to_flatten == None else levels_to_flatten-1)
        else:
            yield x
