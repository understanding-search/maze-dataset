"""a whole bunch of utilities for tokenization"""

import typing
from typing import Any, Iterable, Literal, Callable

import numpy as np

from maze_dataset.constants import SPECIAL_TOKENS, Coord, CoordTup
from maze_dataset.utils import WhenMissing, apply_mapping


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
            all([
                strip_func(x).isdigit() 
                for x in strip_func(
                    coord_str.lstrip("(").rstrip(")")
                ).split(",")
            ]),
        ]
    )

def coord_str_to_tuple(coord_str: str, allow_whitespace: bool = True) -> tuple[int, ...]:
    """convert a coordinate string to a tuple"""
    strip_func: Callable[[str], str] = lambda x: x.strip() if allow_whitespace else x
    coord_str = strip_func(coord_str)
    stripped: str = strip_func(coord_str.lstrip("(").rstrip(")"))
    return tuple(
        int(strip_func(x)) 
        for x in stripped.split(",")
    )

def coord_str_to_coord_np(coord_str: str, allow_whitespace: bool = True) -> np.ndarray:
    """convert a coordinate string to a numpy array"""
    return np.array(coord_str_to_tuple(coord_str, allow_whitespace=allow_whitespace))

def coord_str_to_tuple_noneable(coord_str: str) -> CoordTup | None:
    """convert a coordinate string to a tuple, or None if the string is not a coordinate string"""
    if not str_is_coord(coord_str):
        return None
    return coord_str_to_tuple(coord_str)



# coordinate to tokens
# ==================================================

def _coord_to_tokens_UT(coord: typing.Sequence[int]) -> list[str]:
    """convert a coordinate to a string: `(i,j)`->"(i,j)" """
    return f"({','.join(str(c) for c in coord)})"


def _coord_to_tokens_indexed(coord: typing.Sequence[int]) -> list[str]:
    """convert a coordinate to a list of indexed strings: `(i,j)`->"(", "i", ",", "j", ")" """
    return [
        "(",
        *[str(c) for c in coord],
        ")",
    ]


# filtering things from a prompt or generated text
# ==================================================

def remove_padding_from_token_str(token_str: str) -> str:
    token_str = token_str.replace(f"{SPECIAL_TOKENS['padding']} ", "")
    token_str = token_str.replace(f"{SPECIAL_TOKENS['padding']}", "")
    return token_str


def tokens_between(
    tokens: list[str],
    start_value: str,
    end_value: str,
    include_start: bool = False,
    include_end: bool = False,
) -> list[str]:
    start_idx = tokens.index(start_value) + int(not include_start)
    end_idx = tokens.index(end_value) + int(include_end)

    assert start_idx < end_idx, "Start must come before end"

    return tokens[start_idx:end_idx]


def get_adj_list_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens, SPECIAL_TOKENS["adj_list_start"], SPECIAL_TOKENS["adj_list_end"]
    )


def get_path_tokens(tokens: list[str], trim_end: bool = False) -> list[str]:
    """The path is considered everything from the first path coord to the path_end token, if it exists."""
    if SPECIAL_TOKENS["path_start"] not in tokens:
        raise ValueError(
            f"Path start token {SPECIAL_TOKENS['path_start']} not found in tokens:\n{tokens}"
        )
    start_idx: int = tokens.index(SPECIAL_TOKENS["path_start"]) + int(trim_end)
    end_idx: int | None = None
    if trim_end and (SPECIAL_TOKENS["path_end"] in tokens):
        end_idx = tokens.index(SPECIAL_TOKENS["path_end"])
    return tokens[start_idx:end_idx]


def get_context_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens,
        SPECIAL_TOKENS["adj_list_start"],
        SPECIAL_TOKENS["path_start"],
        include_start=True,
        include_end=True,
    )


def get_origin_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens, SPECIAL_TOKENS["origin_start"], SPECIAL_TOKENS["origin_end"]
    )


def get_target_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens, SPECIAL_TOKENS["target_start"], SPECIAL_TOKENS["target_end"]
    )


def get_tokens_up_to_path_start(
    tokens: list[str], include_start_coord: bool = True
) -> list[str]:
    path_start_idx: int = tokens.index(SPECIAL_TOKENS["path_start"]) + 1
    if include_start_coord:
        return tokens[: path_start_idx + 1]
    else:
        return tokens[:path_start_idx]
