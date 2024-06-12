import warnings
from dataclasses import dataclass, field, make_dataclass

import numpy as np
from jaxtyping import Bool, Int8

from maze_dataset.utils import corner_first_ndindex

Coord = Int8[np.ndarray, "row_col"]
CoordTup = tuple[int, int]
CoordArray = Int8[np.ndarray, "coord row_col"]
CoordList = list[CoordTup]
Connection = Int8[np.ndarray, "coord=2 row_col=2"]
ConnectionList = Bool[np.ndarray, "lattice_dim=2 row col"]
ConnectionArray = Int8[np.ndarray, "edges leading_trailing_coord=2 row_col=2"]


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


_VOCAB_FIELDS: list = [
    # *[(k, str, field(default=v)) for k, v in SPECIAL_TOKENS.items()],
    ("COORD_PRE", str, field(default="(")),
    ("COORD_INTRA", str, field(default=",")),
    ("COORD_POST", str, field(default=")")),
    ("TARGET_INTRA", str, field(default="=")),
    ("TARGET_POST", str, field(default="||")),
    ("PATH_INTRA", str, field(default=":")),
    ("PATH_POST", str, field(default="THEN")),
    ("NEGATIVE", str, field(default="-")),
    ("UNKNOWN", str, field(default="<UNK>")),
    *[
        (f"TARGET_{a}", str, field(default=f"TARGET_{a}"))
        for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ],
    ("TARGET_NORTH", str, field(default="TARGET_NORTH")),
    ("TARGET_SOUTH", str, field(default="TARGET_SOUTH")),
    ("TARGET_EAST", str, field(default="TARGET_EAST")),
    ("TARGET_WEST", str, field(default="TARGET_WEST")),
    ("TARGET_NORTHEAST", str, field(default="TARGET_NORTHEAST")),
    ("TARGET_NORTHWEST", str, field(default="TARGET_NORTHWEST")),
    ("TARGET_SOUTHEAST", str, field(default="TARGET_SOUTHEAST")),
    ("TARGET_SOUTHWEST", str, field(default="TARGET_SOUTHWEST")),
    ("TARGET_CENTER", str, field(default="TARGET_CENTER")),
    ("PATH_NORTH", str, field(default="NORTH")),
    ("PATH_SOUTH", str, field(default="SOUTH")),
    ("PATH_EAST", str, field(default="EAST")),
    ("PATH_WEST", str, field(default="WEST")),
    ("PATH_FORWARD", str, field(default="FORWARD")),
    ("PATH_BACKWARD", str, field(default="BACKWARD")),
    ("PATH_LEFT", str, field(default="LEFT")),
    ("PATH_RIGHT", str, field(default="RIGHT")),
    ("PATH_STAY", str, field(default="STAY")),
    *[
        (f"I_{i:03}", str, field(default=f"+{i}")) for i in range(256)
    ],  # General purpose positive int tokens. Used by `StepTokenizers.Distance`.
    *[
        (f"CTT_{i}", str, field(default=f"{i}")) for i in range(128)
    ],  # Coord tuple tokens
    *[
        (f"I_N{-i:03}", str, field(default=f"{i}")) for i in range(-256, 0)
    ],  # General purpose negative int tokens
    ("PATH_PRE", str, field(default="STEP")),
    ("ADJLIST_PRE", str, field(default="ADJ_GROUP")),
    ("ADJLIST_INTRA", str, field(default="&")),
    ("ADJLIST_WALL", str, field(default="<XX>")),
    *[(f"RESERVE_{i}", str, field(default=f"<RESERVE_{i}>")) for i in range(708, 1596)],
    *[
        (f"UT_{x:02}_{y:02}", str, field(default=f"({x},{y})"))
        for x, y in corner_first_ndindex(50)
    ],
]


_VOCAB_BASE: type = make_dataclass(
    "_VOCAB_BASE", fields=_VOCAB_FIELDS, bases=(_SPECIAL_TOKENS_BASE,), frozen=True
)
# TODO: edit __getitem__ to add warning for accessing a RESERVE token

VOCAB: _VOCAB_BASE = _VOCAB_BASE()
VOCAB_LIST: list[str] = list(VOCAB.values())
VOCAB_TOKEN_TO_INDEX: dict[str, int] = {token: i for i, token in enumerate(VOCAB_LIST)}

# CARDINAL_MAP: Maps tuple(coord1 - coord0) : cardinal direction
CARDINAL_MAP: dict[tuple[int, int], str] = {
    (-1,0): VOCAB.PATH_NORTH,
    (1,0): VOCAB.PATH_SOUTH,
    (0,-1): VOCAB.PATH_WEST,
    (0,1): VOCAB.PATH_EAST,
}