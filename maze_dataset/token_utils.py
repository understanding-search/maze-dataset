"""a whole bunch of utilities for tokenization"""

import numpy as np
from jaxtyping import Int

from maze_dataset.constants import CARDINAL_MAP, SPECIAL_TOKENS, VOCAB

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


def get_cardinal_direction(coords: Int[np.ndarray, "start_end=2 axis=2"]) -> str:
    """Returns the cardinal direction token corresponding to traveling from `coords[0]` to `coords[1]`."""
    return CARDINAL_MAP[tuple(coords[1] - coords[0])]


def get_relative_direction(coords: Int[np.ndarray, "prev_cur_next=3 axis=2"]) -> str:
    """Returns the relative first-person direction token corresponding to traveling from `coords[1]` to `coords[2]`.
    # Parameters
    - `coords`: Contains 3 Coords, each of which must neighbor the previous Coord.
      - `coords[0]`: The previous location, used to determine the current absolute direction that the "agent" is facing.
      - `coords[1]`: The current location
      - `coords[2]`: The next location. May be equal to the current location.
    """
    if coords.shape != (3, 2):
        raise ValueError(f"`coords` must have shape (3,2). Got {coords.shape} instead.")
    directions = coords[1:] - coords[:-1]
    if not np.all(np.linalg.norm(directions, axis=1) <= np.array([1.1, 1.1])):
        # Use floats as constant since `np.linalg.norm` returns float array
        raise ValueError(
            f"Adjacent `coords` must be neighboring or equivalent. Got {coords} instead."
        )
    if np.array_equal(coords[1], coords[2]):
        return VOCAB.PATH_STAY
    if np.array_equal(coords[0], coords[2]):
        return VOCAB.PATH_BACKWARD
    if np.array_equal(coords[0], coords[1]):
        raise ValueError(
            f"Previous first-person direction indeterminate from {coords=}."
        )
    if np.array_equal(directions[0], directions[1]):
        return VOCAB.PATH_FORWARD
    directions = np.append(
        directions, [[0], [0]], axis=1
    )  # Augment to represent unit basis vectors in 3D
    match np.cross(directions[0], directions[1])[-1]:
        case 1:
            return VOCAB.PATH_LEFT
        case -1:
            return VOCAB.PATH_RIGHT


a = 1
