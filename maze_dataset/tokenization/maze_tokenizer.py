"""TokenizationMode enum and the MazeTokenizer class"""

import abc
import itertools
import warnings
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from muutils.json_serialize import (
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)
from muutils.kappa import Kappa
from zanj import ZANJ
from zanj.loading import load_item_recursive

from maze_dataset.constants import (
    SPECIAL_TOKENS,
    VOCAB,
    VOCAB_LIST,
    VOCAB_TOKEN_TO_INDEX,
    ConnectionList,
    Coord,
    CoordArray,
    CoordTup,
    Int8,
)
from maze_dataset.tokenization.token_utils import tokens_between
from maze_dataset.tokenization.util import (
    TokenizerPendingDeprecationWarning,
    TokenizerDeprecationWarning,
    _coord_to_strings_indexed,
    _coord_to_strings_UT,
    connection_list_to_adj_list,
    coords_to_strings,
    strings_to_coords,
)
from maze_dataset.utils import (
    WhenMissing,
    corner_first_ndindex,
    flatten,
    unpackable_if_true_attribute,
)

if TYPE_CHECKING:
    from maze_dataset import LatticeMaze


class TokenError(ValueError):
    """error for tokenization"""

    pass


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
    - `AOTP_CTT_indexed`: each coordinate is a tuple of integers
    """

    AOTP_UT_rasterized = "AOTP_UT_rasterized"
    AOTP_UT_uniform = "AOTP_UT_uniform"
    AOTP_CTT_indexed = "AOTP_CTT_indexed"


_NDINDEX_FUNC_MAP: dict[
    TokenizationMode, Callable[[int], Iterable[tuple[int, ...]]]
] = {
    TokenizationMode.AOTP_UT_rasterized: lambda n: list(np.ndindex(n, n)),
    TokenizationMode.AOTP_UT_uniform: lambda n: corner_first_ndindex(n, 2),
}


def is_UT(tokenization_mode: TokenizationMode) -> bool:
    return tokenization_mode in (
        TokenizationMode.AOTP_UT_rasterized,
        TokenizationMode.AOTP_UT_uniform,
    )


def get_tokens_up_to_path_start(
    tokens: list[str],
    include_start_coord: bool = True,
    tokenization_mode: TokenizationMode = TokenizationMode.AOTP_UT_uniform,
) -> list[str]:
    warnings.warn(
        "`maze_tokenizer.get_tokens_up_to_path_start` will be deprecated for a `MazeTokenizer2`-compatible function in a future release.",
        TokenizerPendingDeprecationWarning,
    )
    path_start_idx: int = tokens.index(SPECIAL_TOKENS.PATH_START) + 1
    if include_start_coord:
        if is_UT(tokenization_mode):
            return tokens[: path_start_idx + 1]
        elif tokenization_mode == TokenizationMode.AOTP_CTT_indexed:
            return tokens[: path_start_idx + 5]
        else:
            raise ValueError(f"Invalid tokenization mode: {tokenization_mode}")
    else:
        return tokens[:path_start_idx]


_MAZETOKENIZER_PROPERTIES_TO_SERIALIZE: list[str] = [
    "name",
    "grid_size",
    "token_arr",
    "tokenizer_map",
    "vocab_size",
    "padding_token_index",
]


@serializable_dataclass(
    properties_to_serialize=_MAZETOKENIZER_PROPERTIES_TO_SERIALIZE, kw_only=True
)
class MazeTokenizer(SerializableDataclass):
    """Tokenizer for mazes

    # Parameters:
     - `tokenization_mode: TokenizationMode`
        mode of tokenization. required.
    - `max_grid_size: int | None`
        maximum grid size. required for actually turning text tokens to numerical tokens, but not for moving between coordinates/mazes and text

    # Properties
    - `name: str`
        auto-generated name of the tokenizer from mode and size

    ## Conditional Properties

    - `node_strings_map: Mapping[CoordTup, str]`
        map from node to string. This returns a `muutils.kappa.Kappa` object which you can use like a dictionary. returns `None` if not a `UT` mode

    these all return `None` if `max_grid_size` is `None`.
    Prepend `_` to the name to get a guaranteed type, and cause an exception if `max_grid_size` is `None`

    - `token_arr: list[str]`
        list of tokens, in order of their indices in the vocabulary
    - `tokenizer_map: Mapping[str, int]`
        map from token to index
    - `vocab_size: int`
        size of the vocabulary
    - `padding_token_index: int`
        index of the padding token

    # Methods
    - `coords_to_strings(coords: list[CoordTup]) -> list[str]`
        convert a list of coordinates to a list of tokens. Optionally except, skip, or ignore non-coordinates
    - `strings_to_coords(strings: list[str]) -> list[CoordTup]`
        convert a list of tokens to a list of coordinates. Optionally except, skip, or ignore non-coordinates

    """

    # parameters
    # ============================================================

    tokenization_mode: TokenizationMode = serializable_field(
        default=TokenizationMode.AOTP_UT_uniform,
        serialization_fn=lambda x: x.value,
        loading_fn=lambda x: TokenizationMode[x["tokenization_mode"]],
    )

    max_grid_size: int | None = serializable_field(default=None)

    # properties
    # ============================================================

    @property
    def name(self) -> str:
        max_grid_size_str: str = (
            f"-g{self.max_grid_size}" if self.max_grid_size is not None else ""
        )
        return f"maze_tokenizer-{self.tokenization_mode.value}{max_grid_size_str}"

    @cached_property
    def _node_strings_map(self) -> Mapping[CoordTup, list[str]]:
        """map a coordinate to a token"""
        if self.tokenization_mode in (
            TokenizationMode.AOTP_UT_rasterized,
            TokenizationMode.AOTP_UT_uniform,
        ):
            return Kappa(_coord_to_strings_UT)
        elif self.tokenization_mode == TokenizationMode.AOTP_CTT_indexed:
            return Kappa(_coord_to_strings_indexed)
        else:
            raise ValueError(
                f"Invalid tokenization mode {self.tokenization_mode}",
                f"expected one of {TokenizationMode.__members__}",
            )

    @cached_property
    def node_strings_map(self) -> Mapping[CoordTup, list[str]] | None:
        """map a coordinate to a token"""
        if self.tokenization_mode in (
            TokenizationMode.AOTP_UT_rasterized,
            TokenizationMode.AOTP_UT_uniform,
        ):
            return None
        else:
            return self._node_strings_map

    # conditional properties (on max_grid_size existing)
    # ------------------------------------------------------------

    @cached_property
    def _token_arr(self) -> list[str]:
        """map from index to token"""
        if self.max_grid_size is None:
            raise ValueError(
                f"max_grid_size must be specified to use token_arr property: {self.max_grid_size = }"
            )

        output: list[str] = list(SPECIAL_TOKENS.values())

        if self.tokenization_mode in (
            TokenizationMode.AOTP_UT_rasterized,
            TokenizationMode.AOTP_UT_uniform,
        ):
            output.extend(
                [
                    self._node_strings_map[coord][0]
                    for coord in _NDINDEX_FUNC_MAP[self.tokenization_mode](
                        self.max_grid_size
                    )
                ]
            )
        elif self.tokenization_mode == TokenizationMode.AOTP_CTT_indexed:
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

        return output

    @cached_property
    def token_arr(self) -> list[str] | None:
        if self.max_grid_size is None:
            return None
        return self._token_arr

    @cached_property
    def _tokenizer_map(self) -> dict[str, int]:
        """map from token to index"""
        return {token: i for i, token in enumerate(self._token_arr)}

    @cached_property
    def tokenizer_map(self) -> dict[str, int] | None:
        if self.max_grid_size is None:
            return None
        return self._tokenizer_map

    @property
    def _vocab_size(self) -> int:
        return len(self._token_arr)

    @property
    def vocab_size(self) -> int | None:
        if self.max_grid_size is None:
            return None
        return self._vocab_size

    @property
    def _n_tokens(self) -> int:
        # TODO: deprecate
        return self._vocab_size

    @property
    def n_tokens(self) -> int | None:
        if self.max_grid_size is None:
            return None
        return self._n_tokens

    @cached_property
    def _padding_token_index(self) -> int:
        return self.tokenizer_map[SPECIAL_TOKENS.PADDING]

    @cached_property
    def padding_token_index(self) -> int | None:
        if self.max_grid_size is None:
            return None
        return self._padding_token_index

    # conversion functions
    # ============================================================

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
        elif self.tokenization_mode == TokenizationMode.AOTP_CTT_indexed:
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

    def encode(self, text: str | list[str]) -> list[int]:
        """encode a string or list of strings into a list of tokens"""
        try:
            if isinstance(text, str):
                text = text.split()
            return [self.tokenizer_map[token] for token in text]
        except KeyError as e:
            raise TokenError(
                f"Token {e} not found",
                f"in vocabulary of {self}:",
                f"{self.token_arr}",
            ) from e

    def decode(
        self, tokens: Sequence[int], joined_tokens: bool = False
    ) -> list[str] | str:
        """decode a list of tokens into a string or list of strings"""
        try:
            output: list[str] = [self.token_arr[token] for token in tokens]
        except IndexError as e:
            raise TokenError(
                f"Token index '{e}' not found in vocabulary of length {self.vocab_size}"
            ) from e
        if joined_tokens:
            return " ".join(output)
        else:
            return output

    # UT-only coordinate stuff
    # ============================================================

    @cached_property
    def coordinate_tokens_coords(self) -> dict[CoordTup, int]:
        print(f"{self.tokenization_mode = }")
        if not self.is_UT():
            raise ValueError(
                f"coordinate_tokens_coords is only valid for UT tokenization modes, got {self.tokenization_mode = }"
            )
        if self.max_grid_size is None:
            raise ValueError(
                f"max_grid_size must be specified to use coordinate_tokens: {self.max_grid_size = }"
            )

        raw_converted: list[CoordTup | str] = self.strings_to_coords(
            self.token_arr, when_noncoord="include"
        )

        # filter out non-coordinates
        return {
            coord: i
            for i, coord in enumerate(raw_converted)
            if not isinstance(coord, str)
        }

    @cached_property
    def coordinate_tokens_ids(self) -> dict[str, int]:
        # checks performed in call
        output: dict[str, int] = dict()

        for coord, index in self.coordinate_tokens_coords.items():
            _for_key: list[str] = self.coords_to_strings([coord])
            assert len(_for_key) == 1
            output[_for_key[0]] = index

        return output

    # other
    # ============================================================

    def summary(self) -> dict:
        """returns a summary of the tokenization mode"""
        return {
            "tokenization_mode": self.tokenization_mode.value,
            "max_grid_size": self.max_grid_size,
            "vocab_size": self.vocab_size,
        }

    def is_AOTP(self) -> bool:
        """returns true if a tokenization mode is Adjacency list, Origin, Target, Path"""
        return self.tokenization_mode in (
            TokenizationMode.AOTP_UT_rasterized,
            TokenizationMode.AOTP_UT_uniform,
            TokenizationMode.AOTP_CTT_indexed,
        )

    def is_UT(self) -> bool:
        return is_UT(self.tokenization_mode)

    def clear_cache(self):
        """clears all cached properties"""
        # delete the properties only if they exist
        for name, prop in self.__class__.__dict__.items():
            if isinstance(prop, cached_property):
                # if the property exists, delete it
                try:
                    delattr(self, name)
                except AttributeError as e:
                    pass


@serializable_dataclass(frozen=True, kw_only=True)
class TokenizerElement(SerializableDataclass, abc.ABC):
    """Superclass for tokenizer elements.
    
    # Development
    Due to the functionality of `ALL_TOKENIZERS`, `TokenizerElement` subclasses may only contain fields of type `utils.FiniteValued`.
    Implementing a subclass with an `int` or `float`-typed field, for example, is not supported.
    In the event that adding such fields is deemed necessary, `ALL_TOKENIZERS` must be updated.
    """

    # def __repr__(self) -> str:
    #     members_str: str = ",".join(
    #         [f"{k}={repr(v)}" for k, v in self.__dict__.items()]
    #     )
    #     return f"{type(self).__name__}({members_str})"

    @property
    def name(self) -> str:
        def _stringify(k: str, v: Any):
            if isinstance(v, bool):
                return f'{k}={str(v)[0]}'
            if isinstance(v, TokenizerElement):
                return v.name
            return f'{k}={v}'
        members_str: str = ", ".join(
            [_stringify(k, v) for k, v in self.__dict__.items()]
        )
        r = f"{type(self).__name__}({members_str})"
        if "." in r:
            return "".join(r.split(".")[1:])
        else:
            return r
    
    @classmethod
    def _level_one_subclass(cls) -> type['TokenizerElement']:
        """Returns the immediate subclass of `TokenizerElement` of which `cls` is an instance.
        """
        return set(cls.__mro__).intersection(set(TokenizerElement.__subclasses__())).pop()
    
    def _tokenizer_elements(self) -> list['TokenizerElement']:
        """Returns a list of all `TokenizerElement` instances contained in the subtree.
        """
        return list(flatten([[el] + el._tokenizer_elements() for el in self.__dict__.values() if isinstance(el, TokenizerElement)]))
    
    @classmethod
    @abc.abstractmethod
    def attribute_key(cls) -> str:
        """Returns the binding used in `MazeTokenizer2` for that type of `TokenizerElement`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_tokens(self, *args, **kwargs) -> list[str]:
        """Converts a maze element into a list of tokens."""
        raise NotImplementedError

    @classmethod
    def from_tokens(self, toks: list[str]):
        """Converts a list of tokens into a maze element."""
        raise NotImplementedError(
            f"Conversion from tokens to {type(self)} not yet supported."
        )


class _TokenizerElementNamespace(abc.ABC):
    """ABC for namespaces

    # Properties
    - key: The binding used in `MazeTokenizer2` for instances of the classes contained within that `_TokenizerElementNamespace`.
    """

    key: str = NotImplementedError


class CoordTokenizers(_TokenizerElementNamespace):
    key = "coord_tokenizer"

    @serializable_dataclass(frozen=True, kw_only=True)
    class CoordTokenizer(TokenizerElement, abc.ABC):
        @abc.abstractmethod
        def to_tokens(self, coord: Coord | CoordTup) -> list[str]:
            pass
        
        @classmethod
        def attribute_key(cls) -> str:
            return CoordTokenizers.key

        # Define some (abstract) methods

    @serializable_dataclass(frozen=True, kw_only=True)
    class UT(CoordTokenizer, abc.ABC):
        def to_tokens(self, coord: Coord | CoordTup) -> list[str]:
            return ["".join(["(", str(coord[0]), ",", str(coord[1]), ")"])]

    @serializable_dataclass(frozen=True, kw_only=True)
    class CTT(CoordTokenizer):
        """Coordinate tuple tokenizer

        # Parameters
        - `pre`: Whether all coords include an integral preceding delimiter token
        - `intra`: Whether all coords include a delimiter token between coordinates
        - `post`: Whether all coords include an integral following delimiter token
        """

        pre: bool = serializable_field(default=True)
        intra: bool = serializable_field(default=True)
        post: bool = serializable_field(default=True)
        # Implement methods

        def to_tokens(self, coord: Coord | CoordTup) -> list[str]:
            return [
                *unpackable_if_true_attribute([VOCAB.COORD_PRE], self, "pre"),
                str(coord[0]),
                *unpackable_if_true_attribute([VOCAB.COORD_INTRA], self, "intra"),
                str(coord[1]),
                *unpackable_if_true_attribute([VOCAB.COORD_POST], self, "post"),
            ]


class AdjListTokenizers(_TokenizerElementNamespace):
    key = "adj_list_tokenizer"

    @serializable_dataclass(frozen=True, kw_only=True)
    class AdjListTokenizer(TokenizerElement, abc.ABC):
        """
        Specifies how the adjacency list is tokenized.
        """

        @abc.abstractmethod
        def to_tokens(self, conn_list: ConnectionList) -> list[str]:
            pass
        
        @classmethod
        def attribute_key(cls) -> str:
            return AdjListTokenizers.key
        

    # TODO: figure out properties_to_serialize
    @serializable_dataclass(frozen=True, kw_only=True)
    class Coords(AdjListTokenizer):
        """
        Represents a connection as the tokens of 2 coords with optional delimiters.

        # Parameters
        - `intra`: Whether all coords include a delimiter token between coordinates
        - `post`: Whether all coords include an integral following delimiter token
        - `walls`: Whether the tokenized adjacency list should list the walls in the maze rather than the connections.
        """

        intra: bool = serializable_field(default=True)
        post: bool = serializable_field(default=True)
        walls: bool = serializable_field(default=False)

        def _single_connection_tokens(
            self,
            coord1: Coord,
            coord2: Coord,
            coord_tokenizer: CoordTokenizers.CoordTokenizer,
        ) -> list[str]:
            return [
                *coord_tokenizer.to_tokens(coord1),
                *([VOCAB.CONNECTOR] if self.intra else ()),
                *coord_tokenizer.to_tokens(coord2),
                *([VOCAB.ADJACENCY_ENDLINE] if self.post else ()),
            ]

        def to_tokens(
            self,
            conn_list: ConnectionList,
            coord_tokenizer: CoordTokenizers.CoordTokenizer,
        ) -> list[str]:
            if self.walls:
                conn_list = np.logical_not(conn_list)
                conn_list[0,-1,:] = False
                conn_list[1,:,-1] = False
            adj_list = connection_list_to_adj_list(conn_list)
            return itertools.chain.from_iterable(
                [
                    self._single_connection_tokens(c_s, c_e, coord_tokenizer)
                    for c_s, c_e in adj_list
                ]
            )


class TargetTokenizers(_TokenizerElementNamespace):
    key = "target_tokenizer"

    @serializable_dataclass(frozen=True, kw_only=True)
    class TargetTokenizer(TokenizerElement, abc.ABC):
        """Superclass of tokenizers for maze targets."""

        @abc.abstractmethod
        def to_tokens(
            self,
            targets: Iterable[Coord],
            coord_tokenizer: CoordTokenizers.CoordTokenizer,
        ) -> list[str]:
            """Returns tokens representing the target."""
            pass
        
        @classmethod
        def attribute_key(cls) -> str:
            return TargetTokenizers.key
        

    @serializable_dataclass(frozen=True, kw_only=True)
    class Unlabeled(TargetTokenizer):
        """Targets are simply listed as coord tokens.
        - `post`: Whether all coords include an integral following delimiter token
        """

        post: bool = serializable_field(default=False)

        def to_tokens(
            self,
            targets: Iterable[Coord],
            coord_tokenizer: CoordTokenizers.CoordTokenizer,
        ) -> list[str]:
            return list(
                flatten(
                    [
                        [
                            *coord_tokenizer.to_tokens(target),
                            *unpackable_if_true_attribute(
                                [VOCAB.TARGET_POST], self, "post"
                            ),
                        ]
                        for target in targets
                    ]
                )
            )


class PathTokenizers(_TokenizerElementNamespace):
    key = "path_tokenizer"

    @serializable_dataclass(frozen=True, kw_only=True)
    class PathTokenizer(TokenizerElement, abc.ABC):
        """Superclass of tokenizers for maze solution paths."""

        @abc.abstractmethod
        def to_tokens(
            self, path: list[Coord], coord_tokenizer: CoordTokenizers.CoordTokenizer
        ) -> list[str]:
            """Returns tokens representing the solution path."""
            pass
        
        @classmethod
        def attribute_key(cls) -> str:
            return PathTokenizers.key
        

    @serializable_dataclass(frozen=True, kw_only=True)
    class StepSequence(PathTokenizer, abc.ABC):
        """Any `PathTokenizer` where the tokenization may be assembled from token subsequences, each of which represents a step along the path.

        Steps may be of any length.
        Allows for a sequence of leading and trailing tokens which don't fit the step pattern.
        """

        def to_tokens(
            self, path: list[Coord], coord_tokenizer: CoordTokenizers.CoordTokenizer
        ) -> list[str]:
            return [
                *self._leading_tokens(path[0], coord_tokenizer),
                *itertools.chain.from_iterable(
                    [
                        self._single_step_tokens(c0, c1, coord_tokenizer)
                        for c0, c1 in zip(path[:-1], path[1:])
                    ]
                ),
                *self._trailing_tokens(path[-1], coord_tokenizer),
            ]

        @abc.abstractmethod
        def _single_step_tokens(
            self, c0: Coord, c1: Coord, coord_tokenizer: CoordTokenizers.CoordTokenizer
        ) -> list[str]:
            """Returns the token sequence representing a single step along the path."""
            pass

        @abc.abstractmethod
        def _leading_tokens(
            self, c: Coord, coord_tokenizer: CoordTokenizers.CoordTokenizer
        ) -> list[str]:
            """Returns tokens preceding those from the sequence from `_single_step_tokens`.
            Since the for loop in `to_tokens` iterates `len(path)-1` times, there may be a fencepost problem.
            There may be information about the start of the path that it misses that needs to be appended.
            <PATH_START> should NOT be included.
            """
            raise NotImplementedError(
                "Subclasses must implement `_leading_tokens`."
                "If no leading tokens are desired then implement to return an empty tuple."
            )

        @abc.abstractmethod
        def _trailing_tokens(
            self, c: Coord, coord_tokenizer: CoordTokenizers.CoordTokenizer
        ) -> list[str]:
            """Returns tokens following those from the sequence from `_single_step_tokens`.
            <PATH_END> should NOT be included.
            """
            raise NotImplementedError(
                "Subclasses must implement `_trailing_tokens`."
                "If no leading tokens are desired then implement to return an empty tuple."
            )

    @serializable_dataclass(frozen=True, kw_only=True)
    class PathCoords(StepSequence):
        """Represents a path as a sequence of tokens for all the coords traversed.

        # Parameters
        - `post`: Whether all coords include an integral following delimiter token
        """

        post: bool = serializable_field(default=False)

        def _single_step_tokens(
            self, c0: Coord, c1: Coord, coord_tokenizer: CoordTokenizers.CoordTokenizer
        ) -> list[str]:
            return [
                *coord_tokenizer.to_tokens(c1),
                *unpackable_if_true_attribute([VOCAB.PATH_POST], self, "post"),
            ]

        def _leading_tokens(
            self, c: CoordArray, coord_tokenizer: CoordTokenizers.CoordTokenizer
        ) -> list[str]:
            return coord_tokenizer.to_tokens(c)

        def _trailing_tokens(
            self, c: CoordArray, coord_tokenizer: CoordTokenizers.CoordTokenizer
        ) -> list[str]:
            return ()


class PromptSequencers(_TokenizerElementNamespace):
    key = "prompt_sequencer"

    @serializable_dataclass(frozen=True, kw_only=True)
    class PromptSequencer(TokenizerElement, abc.ABC):
        """
        Sequences regions into a complete tokenization.
        
        # Parameters
        - `coord_tokenizer`: Tokenizer element which tokenizes a single `Coord` aka maze position.
        - `adj_list_tokenizer`: Tokenizer element which tokenizes the adjacency list of a `LatticeMaze`.
        Uses `coord_tokenizer` to tokenize coords if that is part of the design of that `AdjListTokenizer`.
        """
        coord_tokenizer: CoordTokenizers.CoordTokenizer = serializable_field(
        default=CoordTokenizers.UT(),
        loading_fn=lambda x: _load_tokenizer_element(x, CoordTokenizers),
        )
        adj_list_tokenizer: AdjListTokenizers.AdjListTokenizer = serializable_field(
        default=AdjListTokenizers.Coords(),
        loading_fn=lambda x: _load_tokenizer_element(x, AdjListTokenizers),
        )
        
        @classmethod
        def attribute_key(cls) -> str:
            return PromptSequencers.key
        
        @staticmethod
        def _trim_if_unsolved_maze(
            untrimmed: list[str], is_untargeted: bool = False, is_unsolved: bool = False
        ):
            """Trims a full `SolvedMaze` prompt if the maze data reflects an unsolved or untargeted maze.

            # Development
            This implementation should function for `AOTP`, `AOP`, and other concrete classes using any subsequence of AOTP.
            It is not located in `token_utils.py` because it may need to be overridden in more exotic `PromptSequencer` subclasses.
            """
            if is_untargeted:
                return tokens_between(
                    untrimmed,
                    VOCAB.ADJLIST_START,
                    VOCAB.ADJLIST_END,
                    include_start=True,
                    include_end=True,
                )
            if is_unsolved:
                if VOCAB.TARGET_END in untrimmed:
                    return tokens_between(
                        untrimmed,
                        VOCAB.ADJLIST_START,
                        VOCAB.TARGET_END,
                        include_start=True,
                        include_end=True,
                    )
                else:
                    return tokens_between(
                        untrimmed,
                        VOCAB.ADJLIST_START,
                        VOCAB.ORIGIN_END,
                        include_start=True,
                        include_end=True,
                    )
            return untrimmed

        def to_tokens(
            self,
            adj_list: Int8[np.ndarray, "conn start_end coord"],
            origin: Coord | None,
            target: Iterable[Coord] | None,
            path: CoordArray | None,
            *args,
            **kwargs,
        ) -> list[str]:
            """Returns a complete list of tokens for a given set of maze elements."""
            untrimmed: list[str] = self._sequence_tokens(
                *self._get_prompt_regions(
                    adj_list,
                    origin,
                    target,
                    path,
                )
            )
            return self._trim_if_unsolved_maze(untrimmed, origin is None, path is None)

        def _get_prompt_regions(
            self,
            adj_list: Int8[np.ndarray, "conn start_end coord"],
            origin: Coord | None,
            target: Iterable[Coord] | None,
            path: CoordArray | None,
            *args,
            **kwargs,
        ) -> list[list[str]]:
            """Gets the prompt regions of a maze in a fixed sequence.

            This method is NOT responsible for including/excluding any prompt regions.
            Always return according to the API described under Returns.
            This implementation is expected to be suitable for most `PromptSequencer` subclasses.
            Subclasses may override this method if needed for special behavior.

            # Returns
            - [0]: list[str] Adjacency list tokens
            - [1]: list[str] Origin tokens
            - [2]: list[str] Target tokens
            - [3]: list[str] Path tokens

            # `None`-valued Args
            If one or more of `origin`, `target`, or `path` are `None`, that indicates that an unsolved or untargeted maze is being tokenized.
            To ensure unpackability in `_sequence_tokens`, these `None` values are substituted for empty iterables.
            """
            return [
                self.adj_list_tokenizer.to_tokens(adj_list, coord_tokenizer=self.coord_tokenizer) if hasattr(self, "adj_list_tokenizer") else [],
                self.coord_tokenizer.to_tokens(origin) if origin is not None else [],
                (
                    self.target_tokenizer.to_tokens(target, coord_tokenizer=self.coord_tokenizer)
                    if target[0] is not None and hasattr(self, "target_tokenizer")
                    else []
                ),
                (
                    self.path_tokenizer.to_tokens(path, coord_tokenizer=self.coord_tokenizer)
                    if path is not None and hasattr(self, "path_tokenizer")
                    else []
                ),
            ]

        @abc.abstractmethod
        def _sequence_tokens(
            self,
            adj_list: list[str],
            origin: list[str] | None,
            target: list[str] | None,
            path: list[str] | None,
        ) -> list[str]:
            """Sequences token regions into a complete prompt.
            Includes any boundary tokens in `constants.SPECIAL_TOKENS` such as <ADJLIST_START>, <ORIGIN_END>, etc.
            # Parameters
            - `adj_list`: Tokens representing the adjacency list
            - `origin`: Tokens representing the origin
            - `target`: Tokens representing the target
            - `path`: Tokens representing the path
            """
            pass

    @serializable_dataclass(frozen=True, kw_only=True)
    class AOTP(PromptSequencer):
        """
        Sequences a prompt as [adjacency list, origin, target, path].
        
        # Parameters
        - `target_tokenizer`: Tokenizer element which tokenizes the target(s) of a `TargetedLatticeMaze`.
        Uses `coord_tokenizer` to tokenize coords if that is part of the design of that `TargetTokenizer`.
        - `path_tokenizer`: Tokenizer element which tokenizes the solution path of a `SolvedMaze`.
        Uses `coord_tokenizer` to tokenize coords if that is part of the design of that `PathTokenizer`.

        """
        target_tokenizer: TargetTokenizers.TargetTokenizer = serializable_field(
            default=TargetTokenizers.Unlabeled(),
            loading_fn=lambda x: _load_tokenizer_element(x, TargetTokenizers),
        )
        path_tokenizer: PathTokenizers.PathTokenizer = serializable_field(
            default=PathTokenizers.PathCoords(),
            loading_fn=lambda x: _load_tokenizer_element(x, PathTokenizers),
        )
        
        def _sequence_tokens(
            self,
            adj_list: list[str],
            origin: list[str],
            target: list[str],
            path: list[str],
        ) -> list[str]:
            return [
                VOCAB.ADJLIST_START,
                *adj_list,
                VOCAB.ADJLIST_END,
                VOCAB.ORIGIN_START,
                *origin,
                VOCAB.ORIGIN_END,
                VOCAB.TARGET_START,
                *target,
                VOCAB.TARGET_END,
                VOCAB.PATH_START,
                *path,
                VOCAB.PATH_END,
            ]

    @serializable_dataclass(frozen=True, kw_only=True)
    class AOP(PromptSequencer):
        """Sequences a prompt as [adjacency list, origin, path].

        # Parameters
        - `path_tokenizer`: Tokenizer element which tokenizes the solution path of a `SolvedMaze`.
        Uses `coord_tokenizer` to tokenize coords if that is part of the design of that `PathTokenizer`.
        """
        path_tokenizer: PathTokenizers.PathTokenizer = serializable_field(
            default=PathTokenizers.PathCoords(),
            loading_fn=lambda x: _load_tokenizer_element(x, PathTokenizers),
        )

        def _sequence_tokens(
            self,
            adj_list: list[str],
            origin: list[str],
            target: list[str],
            path: list[str],
        ) -> list[str]:
            return [
                VOCAB.ADJLIST_START,
                *adj_list,
                VOCAB.ADJLIST_END,
                VOCAB.ORIGIN_START,
                *origin,
                VOCAB.ORIGIN_END,
                VOCAB.TARGET_START,
                VOCAB.TARGET_END,
                VOCAB.PATH_START,
                *path,
                VOCAB.PATH_END,
            ]


def _load_tokenizer_element(
    data: dict[str, Any], namespace: type[_TokenizerElementNamespace]
) -> TokenizerElement:
    key: str = namespace.key
    format: str = data[key]["__format__"]
    cls_name: str = format.split("(")[0]
    # return getattr(namespace, cls_name).deserialize(data[key])
    cls: type[TokenizerElement] = getattr(namespace, cls_name)
    # TODO: Refactor line below with `zanj.loading.load_item_recursive` to properly leverage zanj
    # zanj = ZANJ()
    kwargs: dict[str, Any] = {k: load_item_recursive(data[key][k], tuple()) for k, v in data[key].items()}
    if "__format__" in kwargs:
        kwargs.pop("__format__")
    return cls(**kwargs)


@serializable_dataclass(frozen=True, kw_only=True)
class MazeTokenizer2(SerializableDataclass):
    """Tokenizer for mazes

    # Parameters
    - `prompt_sequencer`: Tokenizer element which assembles token regions (adjacency list, origin, target, path) into a complete prompt.
    
    # Development
    - To ensure backwards compatibility, the default constructor must always return a tokenizer equivalent to the legacy `TokenizationMode.AOTP_UT_Uniform`.
    - Furthermore, the mapping reflected in `from_legacy` must also be maintained.
    - Updates to `MazeTokenizer2` or the `TokenizerElement` hierarchy must maintain that behavior.
    """

    prompt_sequencer: PromptSequencers.PromptSequencer = serializable_field(
        default=PromptSequencers.AOTP(),
        loading_fn=lambda x: _load_tokenizer_element(x, PromptSequencers),
    )

    # def __repr__(self) -> str:
    #     return "-".join(
    #         [type(self).__name__, repr(self.prompt_sequencer)]
    #     )

    # Information Querying Methods

    @cached_property
    def _tokenizer_elements(self):
        return [
            self.prompt_sequencer,
            *self.prompt_sequencer._tokenizer_elements()
        ]

    @property
    def name(self) -> str:
        """Serializes MazeTokenizer into a key for encoding in zanj"""
        return "-".join(
            [type(self).__name__, self.prompt_sequencer.name]
        )

    def summary(self) -> dict[str, TokenizerElement]:
        return {
            # "prompt_sequencer": self.prompt_sequencer.name,
            **{elem.attribute_key(): elem for elem in self._tokenizer_elements}
        }
        
    def has_element(
        self,
        elements: type[TokenizerElement] | TokenizerElement | Iterable[type[TokenizerElement] | TokenizerElement]
        ) -> bool:
        """Returns True if the `MazeTokenizer2` instance contains ALL of the items specified in `elements`.
        
        Querying with a partial subset of `TokenizerElement` fields is not currently supported.
        To do such a query, assemble multiple calls to `has_elements`.
        
        # Parameters
        - `elements`: Singleton or iterable of `TokenizerElement` instances or classes.
        If an instance is provided, then comparison is done via equality.
        If a class is provided, then comparison isdone via `isinstance`. I.e., any instance of that class is accepted.
        """
        def type_check(obj: any) -> None:
            if not (isinstance(obj, TokenizerElement) or (isinstance(obj, type) and issubclass(obj, TokenizerElement))):
                raise TypeError(f"{elements} is not a `TokenizerElement` instance or subclass.")
        
        def has_element_singular(el: type[TokenizerElement] | TokenizerElement):
            type_check(el)
            if isinstance(el, type):
                return any([isinstance(e, el) for e in self._tokenizer_elements])
            else:
                return el in self._tokenizer_elements
        
        if not isinstance(elements, Iterable):
            return has_element_singular(elements)
        else:
            return all([has_element_singular(e) for e in elements])

    # Alternate Constructors
    # ======================

    @classmethod
    def from_name(cls, key: str) -> "MazeTokenizer2":
        """Builds a MazeTokenizer from the output of `MazeTokenizer2.name`"""
        raise NotImplementedError

    @classmethod
    def from_legacy(
        cls, legacy_maze_tokenizer: MazeTokenizer | TokenizationMode
    ) -> "MazeTokenizer2":
        """Maps a legacy `MazeTokenizer` to its equivalent `MazeTokenizer2` instance."""
        if isinstance(legacy_maze_tokenizer, MazeTokenizer):
            legacy_maze_tokenizer = legacy_maze_tokenizer.tokenization_mode
        return {
            TokenizationMode.AOTP_UT_uniform: MazeTokenizer2(),
            TokenizationMode.AOTP_UT_rasterized: MazeTokenizer2(),
            TokenizationMode.AOTP_CTT_indexed: MazeTokenizer2(
                prompt_sequencer=PromptSequencers.AOTP(
                    coord_tokenizer=CoordTokenizers.CTT()
                )
            ),
        }[legacy_maze_tokenizer]

    # Simple properties
    # =================

    @property
    def token_arr(self) -> list[str] | None:
        return VOCAB_LIST

    @property
    def tokenizer_map(self) -> dict[str, int]:
        """map from token to index"""
        return VOCAB_TOKEN_TO_INDEX

    @property
    def vocab_size(self) -> int:
        return len(VOCAB_LIST)

    @property
    def n_tokens(self) -> int:
        raise NameError(
            "`MazeTokenizer2.n_tokens` has been removed. Use `len(maze_dataset.VOCAB_LIST)` instead."
        )

    @property
    def padding_token_index(self) -> int:
        return VOCAB_TOKEN_TO_INDEX[VOCAB.PADDING]

    # conversion functions
    # ============================================================

    def to_tokens(
        self,
        maze: "LatticeMaze",
    ) -> list[str]:
        """Converts maze into a list of tokens."""
        return self.prompt_sequencer.to_tokens(
            maze.connection_list,
            getattr(maze, "start_pos", None),
            [
                getattr(maze, "end_pos", None)
            ],  # TargetTokenizer requires target: Iterable[Coord]
            getattr(maze, "solution", None),
        )

    def coords_to_strings(self, coords: list[CoordTup | Coord]) -> list[str]:
        return list(flatten([self.prompt_sequencer.coord_tokenizer.to_tokens(c) for c in coords]))

    @staticmethod
    def strings_to_coords(
        text: str,
        when_noncoord: WhenMissing = "skip",
    ) -> list[str | CoordTup]:
        warnings.warn(
            "`MazeTokenizer2.strings_to_coords` only supports legacy UT strings.",
            TokenizerPendingDeprecationWarning,
        )
        return strings_to_coords(text=text, when_noncoord=when_noncoord)

    @staticmethod
    def encode(text: str | list[str]) -> list[int]:
        """encode a string or list of strings into a list of tokens"""
        try:
            if isinstance(text, str):
                text = text.split()
            return [VOCAB_TOKEN_TO_INDEX[token] for token in text]
        except KeyError as e:
            raise TokenError(
                f"Token {e} not found",
                f"in `VOCAB`.",
            ) from e

    @staticmethod
    def decode(
        token_ids: Sequence[int], joined_tokens: bool = False
    ) -> list[str] | str:
        """decode a list of tokens into a string or list of strings"""
        try:
            output: list[str] = [VOCAB_LIST[token_id] for token_id in token_ids]
        except IndexError as e:
            raise TokenError(f"Token index '{e}' not found in `VOCAB`.") from e
        if joined_tokens:
            return " ".join(output)
        else:
            return output

    # utils
    # =============

    def is_AOTP(self) -> bool:
        warnings.warn(
            "`MazeTokenizer2.is_AOTP` is deprecated. Use `MazeTokenizer2.has_element(PromptSequencers.AOTP)` instead.",
            TokenizerDeprecationWarning,
        )
        return self.has_element(PromptSequencers.AOTP)

    def is_UT(self) -> bool:
        warnings.warn(
            "`MazeTokenizer2.is_UT` is deprecated. Use `MazeTokenizer2.has_element(CoordTokenizers.UT)` instead.",
            TokenizerDeprecationWarning,
        )
        return self.has_element(CoordTokenizers.UT)

