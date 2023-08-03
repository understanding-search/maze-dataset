import warnings
from dataclasses import dataclass

import numpy as np
from jaxtyping import Int8

Coord = Int8[np.ndarray, "x y"]
CoordTup = tuple[int, int]
CoordArray = Int8[np.ndarray, "coord x y"]
CoordList = list[CoordTup]


class SpecialTokensError(Exception):
    pass


_SPECIAL_TOKENS_ABBREVIATIONS: dict[str, str] = {
    "<ADJLIST_START>": "<A_S>",
    "<ADJLIST_END>": "<A_E>",
    "<TARGET_START>": "<T_S>",
    "<TARGET_END>": "<T_E>",
    "<ORIGIN_START>": "<O_S>",
    "<ORIGIN_END>": "<O_E>",
    "<PATH_START>": "<P_S>",
    "<PATH_END>": "<P_E>",
    "<-->": "<-->",
    ";": ";",
    "<PADDING>": "<PAD>",
}


@dataclass(frozen=True)
class _SPECIAL_TOKENS_BASE:
    ADJLIST_START: str = "<ADJLIST_START>"
    ADJLIST_END: str = "<ADJLIST_END>"
    TARGET_START: str = "<TARGET_START>"
    TARGET_END: str = "<TARGET_END>"
    ORIGIN_START: str = "<ORIGIN_START>"
    ORIGIN_END: str = "<ORIGIN_END>"
    PATH_START: str = "<PATH_START>"
    PATH_END: str = "<PATH_END>"
    CONNECTOR: str = "<-->"
    ADJACENCY_ENDLINE: str = ";"
    PADDING: str = "<PADDING>"

    def __getitem__(self, key: str) -> str:
        key_upper: str = key.upper()

        if not isinstance(key, str):
            raise TypeError(f"key must be str, not {type(key)}")

        # error checking for old lowercase format
        if not key == key_upper:
            warnings.warn(
                f"Accessing special token '{key}' without uppercase. this is deprecated and will be removed in the future.",
                DeprecationWarning,
            )
            key = key_upper

        # `ADJLIST` used to be `adj_list`, changed to match actual token content
        if not key_upper in self.keys():
            key_upper_modified: str = key_upper.replace("ADJ_LIST", "ADJLIST")
            if key_upper_modified in self.keys():
                warnings.warn(
                    f"Accessing '{key}' in old format, should use {key_upper_modified}. this is deprecated and will be removed in the future.",
                    DeprecationWarning,
                )
                return getattr(self, key_upper_modified)
            else:
                raise KeyError(f"invalid special token '{key}'")

        # normal return
        return getattr(self, key.upper())

    def get_abbrev(self, key: str) -> str:
        return _SPECIAL_TOKENS_ABBREVIATIONS[self[key]]

    def __iter__(self):
        return iter(self.__dict__.keys())

    def __len__(self):
        return len(self.__dict__.keys())

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()


SPECIAL_TOKENS: _SPECIAL_TOKENS_BASE = _SPECIAL_TOKENS_BASE()


DIRECTIONS_MAP: Int8[np.ndarray, "direction axes"] = np.array(
    [
        [0, 1],  # down
        [0, -1],  # up
        [1, 1],  # right
        [1, -1],  # left
    ]
)


NEIGHBORS_MASK: Int8[np.ndarray, "coord point"] = np.array(
    [
        [0, 1],  # down
        [0, -1],  # up
        [1, 0],  # right
        [-1, 0],  # left
    ]
)
