import numpy as np
from jaxtyping import Int8

Coord = Int8[np.ndarray, "x y"]
CoordTup = tuple[int, int]
CoordArray = Int8[np.ndarray, "coord x y"]
CoordList = list[CoordTup]

SPECIAL_TOKENS: dict[str, str] = dict(
    adj_list_start="<ADJLIST_START>",
    adj_list_end="<ADJLIST_END>",
    target_start="<TARGET_START>",
    target_end="<TARGET_END>",
    origin_start="<ORIGIN_START>",
    origin_end="<ORIGIN_END>",
    path_start="<PATH_START>",
    path_end="<PATH_END>",
    connector="<-->",
    adjacency_endline=";",
    padding="<PADDING>",
)

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
