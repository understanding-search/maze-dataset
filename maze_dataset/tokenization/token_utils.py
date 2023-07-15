from typing import Any, Iterable, Literal, Callable
from enum import Enum
from functools import cached_property
import warnings

import numpy as np
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field

from maze_dataset.constants import SPECIAL_TOKENS, CoordTup
from maze_dataset.utils import corner_first_ndindex

WhenMissing = Literal["except", "skip", "include"]

def str_is_coord(coord_str: str) -> bool:
    """return True if the string is a coordinate string, False otherwise"""
    return all(
        [
            coord_str.startswith("("),
            coord_str.endswith(")"),
            "," in coord_str,
            all([x.isdigit() for x in coord_str.lstrip("(").rstrip(")").split(",")]),
        ]
    )


def coord_str_to_tuple(coord_str: str) -> tuple[int, ...]:
    """convert a coordinate string to a tuple"""

    stripped: str = coord_str.lstrip("(").rstrip(")")
    return tuple(int(x) for x in stripped.split(","))


def coord_str_to_tuple_noneable(coord_str: str) -> CoordTup | None:
    """convert a coordinate string to a tuple, or None if the string is not a coordinate string"""
    if not str_is_coord(coord_str):
        return None
    return coord_str_to_tuple(coord_str)


def coord_to_str(coord: typing.Sequence[int]) -> str:
    """convert a coordinate to a string: `(i,j)`->"(i,j)""""
    return f"({','.join(str(c) for c in coord)})"

def coord_to_indexed_string(coord: typing.Sequence[int]) -> list[str]:
    """convert a coordinate to a list of indexed strings: `(i,j)`->"(", "i", ",", "j", ")" """
    return [
        "(", 
        *[str(c) for c in coord],
        ")",
    ]

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


def get_origin_token(tokens: list[str]) -> str:
    return tokens_between(
        tokens, SPECIAL_TOKENS["origin_start"], SPECIAL_TOKENS["origin_end"]
    )[0]


def get_target_token(tokens: list[str]) -> str:
    return tokens_between(
        tokens, SPECIAL_TOKENS["target_start"], SPECIAL_TOKENS["target_end"]
    )[0]


def get_tokens_up_to_path_start(
    tokens: list[str], include_start_coord: bool = True
) -> list[str]:
    path_start_idx: int = tokens.index(SPECIAL_TOKENS["path_start"]) + 1
    if include_start_coord:
        return tokens[: path_start_idx + 1]
    else:
        return tokens[:path_start_idx]


def apply_mapping(
    iter: Iterable[Any],
    mapping: dict[Any, Any],
    when_missing: WhenMissing = "skip",
) -> list[Any]:
    """Given a list and a mapping, apply the mapping to the list"""
    output: list = list()
    for item in iter:
        if item in mapping:
            output.append(mapping[item])
            continue
        match when_missing:
            case "skip":
                continue
            case "include":
                output.append(item)
            case "except":
                raise ValueError(f"item {item} is missing from mapping {mapping}")
            case _:
                raise ValueError(f"invalid value for {when_missing = }")
    return output


def tokens_to_coords(
    tokens: list[str],
    maze_data_cfg,  # TODO: cannot type this right now because importing MazeDatasetConfig causes a circular import
    when_noncoord: WhenMissing = "skip",
) -> list[str | CoordTup]:
    return apply_mapping(tokens, maze_data_cfg.token_node_map, when_noncoord)


def coords_to_tokens(
    coords: list[str | CoordTup],
    maze_data_cfg,  # TODO: cannot type this right now because importing MazeDatasetConfig causes a circular import
    when_noncoord: WhenMissing = "skip",
) -> list[str]:
    return apply_mapping(coords, maze_data_cfg.node_token_map, when_noncoord)


def remove_padding_from_token_str(token_str: str) -> str:
    token_str = token_str.replace(f"{SPECIAL_TOKENS['padding']} ", "")
    token_str = token_str.replace(f"{SPECIAL_TOKENS['padding']}", "")
    return token_str

def _str_to_coord(coord_str: str) -> Coord:
    return np.array(tuple(int(x) for x in coord_str.strip("() \t").split(",")))

class TokenizationMode(Enum):
    """mode of tokenization

    # Abbreviations:
    - `AOTP`: Ajacency list, Origin, Target, Path
    - `UT`: Unique Token (for each coordiate)
    - `CTT`: Coordinate Tuple Tokens (each coordinate is tokenized as a tuple of integers)

    # Modes:
    - `AOTP_UT_rasterized`: the "classic" mode: assigning tokens to each coordinate is done via rasterization
        example: for a 3x3 maze, token order is `(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)`
    - `AOTP_UT_uniform`: new mode, where a 3x3 tokenization scheme and 5x5 tokenizations scheme are compatible
        uses `corner_first_ndindex` function to order the tokens
    - `AOTP_indexed`: each coordinate is a tuple of integers (not implemented)
    """

    AOTP_UT_rasterized = "AOTP_UT_rasterized"
    AOTP_UT_uniform = "AOTP_UT_uniform"
    AOTP_indexed = "AOTP_indexed"

_NDINDEX_FUNC_MAP: dict[
     TokenizationMode, 
     Callable[[int], Iterable[tuple[int, ...]]]
] = {
    TokenizationMode.AOTP_UT_rasterized: np.ndindex,
    TokenizationMode.AOTP_UT_uniform: corner_first_ndindex,
}

_MAZETOKENIZER_PROPERTIES_TO_SERIALIZE: list[str] = [
    "name"
    "grid_size",
    "padding_token_index",
    "token_arr",
    "tokenizer_map",
    "token_node_map",
    "n_tokens",
    # "node_token_map", # doesn't work by default due to keys being tuples
]

# TODO: re-add later, depends on a feature coming in muutils 0.3.2
# __MAZEDATASET_PROPERTIES_TO_VALIDATE: list[str] = [
#     "token_arr",
#     "padding_token_index",
#     "tokenizer_map",
#     "grid_shape",
#     "token_node_map",
#     "n_tokens",
# ]

@serializable_dataclass(properties_to_serialize=["name"])
class MazeTokenizer(SerializableDataclass):
    tokenization_mode: TokenizationMode = serializable_field(
        default=TokenizationMode.AOTP_UT_uniform,
        serialization_fn=lambda x: x.value,
        loading_fn=lambda x: TokenizationMode[x["tokenization_mode"]],
    )

    # TODO: there could in principle be a way to avoid having to specify this,
    # since it shouldn't matter for the `AOTP_UT_uniform` mode or the `AOTP_indexed` mode
    # but, this adds a lot of complexity. Just set it to a big value if you're not sure
    max_grid_size: int = serializable_field()

    @property
    def name(self) -> str:
        return f"maze_tokenizer-{self.tokenization_mode.value}-n{self.max_grid_size}"
        
    @cached_property
    def node_token_map(self) -> dict[CoordTup, str]:
        """map from node to token"""
        if self.tokenization_mode in (
            TokenizationMode.AOTP_UT_rasterized, 
            TokenizationMode.AOTP_UT_uniform,
        ):
            # if rasterized, use np.ndindex, if uniform use corner_first_ndindex
            return {
                tuple(coord): coord_to_str(coord)
                for coord in 
                _NDINDEX_FUNC_MAP[self.tokenization_mode](self.max_grid_size)
            }
        elif self.tokenization_mode == TokenizationMode.AOTP_indexed:
            raise NotImplementedError("AOTP_indexed mode not compatible with node_token_map")
        else:
            raise ValueError(
                f"Invalid tokenization mode {self.tokenization_mode}",
                f"expected one of {TokenizationMode.__members__}",
            )

    def map_coord_to_tokens(self, coord: CoordTup) -> list[str]:
        """map a coordinate to a token"""
        if self.tokenization_mode in (
            TokenizationMode.AOTP_UT_rasterized, 
            TokenizationMode.AOTP_UT_uniform,
        ):
            return [self.node_token_map[coord]]
        elif self.tokenization_mode == TokenizationMode.AOTP_indexed:
            return coord_to_indexed_string(coord)
    
    @cached_property
    def token_node_map(self) -> dict[str, CoordTup]:
        """map from token to node"""
        raise DeprecationWarning("this isn't used anywhere??")
        return {v: k for k, v in self.node_token_map.items()}

    @cached_property
    def token_arr(self) -> list[str]:
        """map from index to token"""
        output: list[str] = list(SPECIAL_TOKENS.values())

        if self.tokenization_mode in (
            TokenizationMode.AOTP_UT_rasterized, 
            TokenizationMode.AOTP_UT_uniform,
        ):
            output.extend(self.node_token_map.values())
        elif self.tokenization_mode == TokenizationMode.AOTP_indexed:
            # TODO: this is hacky, but we don't want to modify the original SPECIAL_TOKENS since that will break old models
            output.extend([
                "(", ",", ")", # new special chars
                *map(str, range(self.max_grid_size)), # numbers
            ])
        else:
            raise ValueError(
                f"Invalid tokenization mode {self.tokenization_mode}",
                f"expected one of {TokenizationMode.__members__}",
            )
    
    @property
    def vocab_size(self) -> int:
        return len(self.token_arr)

    @property
    def n_tokens(self) -> int:
        # TODO: deprecate
        return self.vocab_size
    
    @cached_property
    def padding_token_index(self) -> int:
        return self.tokenizer_map[SPECIAL_TOKENS["padding"]]
