"""a whole bunch of utilities for tokenization"""

import re
import typing
from typing import Callable

import numpy as np
from muutils.misc import list_join

from maze_dataset.constants import SPECIAL_TOKENS, CoordTup
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


def coords_string_split(coords: str) -> list[str]:
    """Splits a list of tokens into a list containing the tokens for each coordinate

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
    tokens_processed: list[str] = coords_string_split(tokens_joined)
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


# filtering things from a prompt or generated text
# ==================================================


def remove_padding_from_token_str(token_str: str) -> str:
    token_str = token_str.replace(f"{SPECIAL_TOKENS.PADDING} ", "")
    token_str = token_str.replace(f"{SPECIAL_TOKENS.PADDING}", "")
    return token_str


def tokens_between(
    tokens: list[str],
    start_value: str,
    end_value: str,
    include_start: bool = False,
    include_end: bool = False,
    except_when_tokens_not_unique: bool = False,
) -> list[str]:
    if start_value == end_value:
        raise ValueError(
            f"start_value and end_value cannot be the same: {start_value = } {end_value = }"
        )
    if except_when_tokens_not_unique:
        if (tokens.count(start_value) != 1) or (tokens.count(end_value) != 1):
            raise ValueError(
                "start_value or end_value is not unique in the input tokens:",
                f"{tokens.count(start_value) = } {tokens.count(end_value) = }"
                f"{start_value = } {end_value = }",
                f"{tokens = }",
            )
    else:
        if (tokens.count(start_value) < 1) or (tokens.count(end_value) < 1):
            raise ValueError(
                "start_value or end_value is not present in the input tokens:",
                f"{tokens.count(start_value) = } {tokens.count(end_value) = }",
                f"{start_value = } {end_value = }",
                f"{tokens = }",
            )

    start_idx: int = tokens.index(start_value) + int(not include_start)
    end_idx: int = tokens.index(end_value) + int(include_end)

    assert start_idx < end_idx, "Start must come before end"

    return tokens[start_idx:end_idx]


def get_adj_list_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens, SPECIAL_TOKENS.ADJLIST_START, SPECIAL_TOKENS.ADJLIST_END
    )


def get_path_tokens(tokens: list[str], trim_end: bool = False) -> list[str]:
    """The path is considered everything from the first path coord to the path_end token, if it exists."""
    if SPECIAL_TOKENS.PATH_START not in tokens:
        raise ValueError(
            f"Path start token {SPECIAL_TOKENS.PATH_START} not found in tokens:\n{tokens}"
        )
    start_idx: int = tokens.index(SPECIAL_TOKENS.PATH_START) + int(trim_end)
    end_idx: int | None = None
    if trim_end and (SPECIAL_TOKENS.PATH_END in tokens):
        end_idx = tokens.index(SPECIAL_TOKENS.PATH_END)
    return tokens[start_idx:end_idx]


def get_context_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens,
        SPECIAL_TOKENS.ADJLIST_START,
        SPECIAL_TOKENS.PATH_START,
        include_start=True,
        include_end=True,
    )


def get_origin_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens,
        SPECIAL_TOKENS.ORIGIN_START,
        SPECIAL_TOKENS.ORIGIN_END,
        include_start=False,
        include_end=False,
    )


def get_target_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens,
        SPECIAL_TOKENS.TARGET_START,
        SPECIAL_TOKENS.TARGET_END,
        include_start=False,
        include_end=False,
    )


def get_tokens_up_to_path_start(
    tokens: list[str], include_start_coord: bool = True
) -> list[str]:
    path_start_idx: int = tokens.index(SPECIAL_TOKENS.PATH_START) + 1
    if include_start_coord:
        return tokens[: path_start_idx + 1]
    else:
        return tokens[:path_start_idx]
