"""TokenizationMode enum and the MazeTokenizer class"""
from enum import Enum
from functools import cached_property
from typing import Callable, Iterable

import numpy as np
from muutils.json_serialize import (
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)

from maze_dataset.constants import SPECIAL_TOKENS, CoordTup
from maze_dataset.tokenization.token_utils import (
    _coord_to_strings_indexed,
    _coord_to_strings_UT,
    # coord_to_indexed_string,
    # coord_to_str,
    coords_to_strings,
    strings_to_coords,
)
from maze_dataset.utils import WhenMissing, corner_first_ndindex


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
    TokenizationMode, Callable[[int], Iterable[tuple[int, ...]]]
] = {
    TokenizationMode.AOTP_UT_rasterized: np.ndindex,
    TokenizationMode.AOTP_UT_uniform: corner_first_ndindex,
}

_MAZETOKENIZER_PROPERTIES_TO_SERIALIZE: list[str] = [
    "name" "grid_size",
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


@serializable_dataclass(properties_to_serialize=["name"], kw_only=True)
class MazeTokenizer(SerializableDataclass):
    tokenization_mode: TokenizationMode = serializable_field(
        default=TokenizationMode.AOTP_UT_uniform,
        serialization_fn=lambda x: x.value,
        loading_fn=lambda x: TokenizationMode[x["tokenization_mode"]],
    )

    # TODO: there could in principle be a way to avoid having to specify this,
    # since it shouldn't matter for the `AOTP_UT_uniform` mode or the `AOTP_indexed` mode
    # but, this adds a lot of complexity. Just set it to a big value if you're not sure
    max_grid_size: int|None = serializable_field(default=None)

    @property
    def name(self) -> str:
        max_grid_size_str: str = f"-g{self.max_grid_size}" if self.max_grid_size is not None else ""
        return f"maze_tokenizer-{self.tokenization_mode.value}{max_grid_size_str}"

    @cached_property
    def node_token_map(self) -> dict[CoordTup, str]:
        """map from node to token"""
        if self.max_grid_size is None:
            raise ValueError(
                "max_grid_size must be specified to use node_token_map property"
            )

        if self.tokenization_mode in (
            TokenizationMode.AOTP_UT_rasterized,
            TokenizationMode.AOTP_UT_uniform,
        ):
            # if rasterized, use np.ndindex, if uniform use corner_first_ndindex
            return {
                tuple(coord): _coord_to_strings_UT(coord)
                for coord in _NDINDEX_FUNC_MAP[self.tokenization_mode](
                    self.max_grid_size
                )
            }
        elif self.tokenization_mode == TokenizationMode.AOTP_indexed:
            raise NotImplementedError(
                "AOTP_indexed mode not compatible with node_token_map"
            )
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
            return _coord_to_strings_UT(coord)
        elif self.tokenization_mode == TokenizationMode.AOTP_indexed:
            return _coord_to_strings_indexed(coord)

    @cached_property
    def token_node_map(self) -> dict[str, CoordTup]:
        """map from token to node"""
        raise DeprecationWarning("this isn't used anywhere??")
        return {v: k for k, v in self.node_token_map.items()}

    @cached_property
    def token_arr(self) -> list[str]:
        """map from index to token"""
        if self.max_grid_size is None:
            raise ValueError(
                "max_grid_size must be specified to use node_token_map property"
            )

        output: list[str] = list(SPECIAL_TOKENS.values())

        if self.tokenization_mode in (
            TokenizationMode.AOTP_UT_rasterized,
            TokenizationMode.AOTP_UT_uniform,
        ):
            output.extend(self.node_token_map.values())
        elif self.tokenization_mode == TokenizationMode.AOTP_indexed:
            # TODO: this is hacky, but we don't want to modify the original SPECIAL_TOKENS since that will break old models
            output.extend(
                [
                    "(",
                    ",",
                    ")",  # new special chars
                    *map(str, range(self.max_grid_size)),  # numbers
                ]
            )
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
        return self.tokenizer_map[SPECIAL_TOKENS.PADDING]

    def coords_to_strings(
        self,
        coords: list[CoordTup],
        when_noncoord: WhenMissing = "skip",
    ) -> list[str]:
        if self.tokenization_mode in (
            TokenizationMode.AOTP_UT_rasterized,
            TokenizationMode.AOTP_UT_uniform,
        ):
            return coords_to_strings(
                coords=coords,
                coord_to_strings_func=_coord_to_strings_UT,
                when_noncoord=when_noncoord,
            )
        elif self.tokenization_mode == TokenizationMode.AOTP_indexed:
            return coords_to_strings(
                coords=coords,
                coord_to_strings_func=_coord_to_strings_indexed,
                when_noncoord=when_noncoord,
            )
        else:
            raise ValueError(
                f"Invalid tokenization mode {self.tokenization_mode}",
                f"expected one of {TokenizationMode.__members__}",
            )

    @staticmethod
    def strings_to_coords(
        text: str,
        when_noncoord: WhenMissing = "skip",
    ) -> list[str | CoordTup]:
        return strings_to_coords(text=text, when_noncoord=when_noncoord)


    def is_AOTP(self) -> bool:
        """returns true if a tokenization mode is Adjacency list, Origin, Target, Path"""
        return self.tokenization_mode in (
            TokenizationMode.AOTP_UT_rasterized,
            TokenizationMode.AOTP_UT_uniform,
            TokenizationMode.AOTP_indexed,
        )