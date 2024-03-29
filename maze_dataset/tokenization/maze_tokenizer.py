"""TokenizationMode enum and the MazeTokenizer class"""

from enum import Enum
from functools import cached_property
import itertools
from typing import Callable, Iterable, Mapping, Sequence, Any

import numpy as np
import abc
from muutils.json_serialize import (
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)
from muutils.kappa import Kappa
from numpy.core.multiarray import array as array

from maze_dataset.constants import SPECIAL_TOKENS, Int8, CoordTup, CoordArray, ConnectionList
from maze_dataset.tokenization.util import (
    _coord_to_strings_indexed,
    _coord_to_strings_UT,
    coords_to_strings,
    strings_to_coords,
    connection_list_to_adj_list
)
from maze_dataset.utils import WhenMissing, corner_first_ndindex, unpackable_if_true_attribute


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


class _DELIMITERS:
    """For all `TokenizerElement`s, the tokens to be used for optional delimiters.
    """
    COORD_PRE = "("
    COORD_INTRA = ","
    COORD_POST = ")"
    ADJ_LIST_INTRA = SPECIAL_TOKENS.CONNECTOR
    ADJ_LIST_POST = SPECIAL_TOKENS.ADJACENCY_ENDLINE
    PATH_INTRA = ","
    PATH_POST = SPECIAL_TOKENS.ADJACENCY_ENDLINE
    

# TODO: figure out properties_to_serialize
@serializable_dataclass(frozen=True, kw_only=True)
class TokenizerElement(SerializableDataclass, abc.ABC):
    """Superclass for tokenizer elements."""
    def __repr__(self) -> str:
        members_str: str = ','.join([f'{k}={repr(v)}' for k, v in self.__dict__.items()])
        return f"{type(self).__name__}({members_str})"
    
    @property
    def name(self) -> str:
        return repr(self)
    
    def serialize(self) -> dict[str, Any]:
        return self.__dict__
    
    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> 'cls':
        return cls(data)
    
    @abc.abstractmethod
    def to_tokens(self, *args, **kwargs) -> list[str]:
        """Converts a maze element into a list of tokens."""
        pass
    
    @classmethod
    def from_tokens(self, toks: list[str]):
        """Converts a list of tokens into a maze element."""
        raise NotImplementedError(f'Conversion from tokens to ')
    
    
class CoordTokenizers:
    class CoordTokenizer(TokenizerElement, abc.ABC):
        @abc.abstractmethod
        def to_tokens(coord: CoordTup) -> list[str]: pass
        # Define some (abstract) methods

    # Intermediate abstract tokenizer elements
    class UT(CoordTokenizer, abc.ABC):
        def to_tokens(coord: CoordTup) -> list[str]:
            return [''.join(['(', coord[0], ',', coord[1], ')'])]

    class UTRasterized(UT): pass
    # Implement methods


    class UTUniform(UT): pass
    # Implement methods
    
    
    class CTT(CoordTokenizer):
        """Coordinate tuple tokenizer
        
        # Parameters
        - `pre`: Whether all coords include an integral preceding delimiter token
        - `intra`: Whether all coords include a delimiter token between coordinates
        - `post`: Whether all coords include an integral following delimiter token
        """
        pre: bool = True
        intra: bool = True
        post: bool = True
        # Implement methods


class AdjListTokenizers:
    # TODO: figure out properties_to_serialize
    @serializable_dataclass(frozen=True, kw_only=True)
    class AdjListTokenizer(TokenizerElement, abc.ABC):
        """
        Specifies how the adjacency list is tokenized.
        """
        @abc.abstractmethod
        def to_tokens(conn_list: ConnectionList) -> list[str]: pass
        # Define some (abstract) methods


    # TODO: figure out properties_to_serialize
    @serializable_dataclass(frozen=True, kw_only=True)
    class Coords(AdjListTokenizer):
        """
        A connection is represented as the tokens of 2 coords with optional delimiters.
        """
        intra: bool = serializable_field(default=True, compare=False)
        post: bool = serializable_field(default=True, compare=False)
        walls: bool = serializable_field(default=False, compare=False)
        
        def _single_connection_tokens(
            self, 
            coord1: CoordTup, coord2: CoordTup, 
            coord_tokenizer: CoordTokenizers.CoordTokenizer
            ) -> list[str]:
            return [
                *coord_tokenizer.to_tokens(coord1),
                *([_DELIMITERS.ADJ_LIST_INTRA] if self.intra else ()),
                *coord_tokenizer.to_tokens(coord2),
                *([_DELIMITERS.ADJ_LIST_POST] if self.post else ())
            ]
        
        def to_tokens(
            self,
            conn_list: ConnectionList, 
            coord_tokenizer: CoordTokenizers.CoordTokenizer
            ) -> list[str]:
            if self.walls:
                conn_list = np.logical_not(conn_list)
            adj_list = connection_list_to_adj_list(conn_list)
            return itertools.chain.from_iterable(
                [
                    self._single_connection_tokens(c_s, c_e, coord_tokenizer)
                    for c_s, c_e in adj_list
                ]
            )
        

class PathTokenizers:
    class PathTokenizer(TokenizerElement, abc.ABC):
        """Superclass of tokenizers for maze solution paths.
        """
        @abc.abstractmethod
        def to_tokens(self, path: list[CoordTup], coord_tokenizer: CoordTokenizers.CoordTokenizer) -> list[str]:
            """Returns tokens representing the solution path.
            """
        
        
    class StepSequence(PathTokenizer, abc.ABC):
        """Any `PathTokenizer` where the tokenization may be assembled from token subsequences, each of which represents a step along the path.
        
        Steps may be of any length.
        """
        def to_tokens(self, path: list[CoordTup], coord_tokenizer: CoordTokenizers.CoordTokenizer) -> list[str]:
            return itertools.chain.from_iterable(
                [
                    self._single_step_tokens(c0, c1, coord_tokenizer)
                    for c0, c1 in self.as_adj_list()
                ]
            )
            
        @abc.abstractmethod
        def _single_step_tokens(self, c0: CoordTup, c1: CoordTup, coord_tokenizer: CoordTokenizers.CoordTokenizer) -> list[str]:
            pass

    
    class Coords(StepSequence):
        post: bool = False
        
        def _single_step_tokens(self, c0: CoordTup, c1: CoordTup, coord_tokenizer: CoordTokenizers.CoordTokenizer) -> list[str]:
            return [
                coord_tokenizer(c1),
                *unpackable_if_true_attribute([_DELIMITERS.PATH_POST], self, 'post')
            ]

class PromptSequencers:
    """Namespace for `PromptSequencer` subclass hierarchy."""
    class PromptSequencer(TokenizerElement, abc.ABC):
        def to_tokens(
            self, 
            adj_list: Int8[np.ndarray, "conn start_end coord"],
            origin: CoordTup,
            target: CoordTup,
            path: CoordArray,
            coord_tokenizer: CoordTokenizers.CoordTokenizer,
            adj_list_tokenizer: AdjListTokenizers.AdjListTokenizer,
            path_tokenizer: PathTokenizers.PathTokenizer,
            *args,
            **kwargs
            ) -> list[str]:
            """Returns a complete list of tokens for a given set of maze elements."""
            return self._sequence_tokens(
                self._get_prompt_regions(
                    adj_list,
                    origin,
                    target,
                    path,
                    coord_tokenizer,
                    adj_list_tokenizer,
                    path_tokenizer,
                )
            )
        
        def _get_prompt_regions(
            self,
            adj_list: Int8[np.ndarray, "conn start_end coord"],
            origin: CoordTup,
            target: CoordTup,
            path: CoordArray,
            coord_tokenizer: CoordTokenizers.CoordTokenizer,
            adj_list_tokenizer: AdjListTokenizers.AdjListTokenizer,
            path_tokenizer: PathTokenizers.PathTokenizer,
            *args,
            **kwargs
            ) -> list[list[str]]:
            """Gets the prompt regions of a maze in a fixed sequence.
            
            This implementation is expected to be suitable for most `PromptSequencer` subclasses.
            Subclasses may override this method if needed for special behavior.
                        
            # Returns
            - [0]: Adjacency list tokens
            - [1]: Origin tokens
            - [2]: Target tokens
            - [3]: Path tokens
            """
            # adj_list_tokens: list[str] = adj_list_tokenizer.to_tokens(adj_list, coord_tokenizer=coord_tokenizer)
            # origin_tokens: list[str] = coord_tokenizer.to_tokens(origin)
            # target_tokens: list[str] = coord_tokenizer.to_tokens(target)
            # path_tokens: list[str] = path_tokenizer.to_tokens(path, coord_tokenizer=coord_tokenizer)
            
            return [
                adj_list_tokenizer.to_tokens(adj_list, coord_tokenizer=coord_tokenizer),
                coord_tokenizer.to_tokens(origin),
                coord_tokenizer.to_tokens(target),
                path_tokenizer.to_tokens(path, coord_tokenizer=coord_tokenizer)
            ]
        
        @abc.abstractmethod
        def _sequence_tokens(self, adj_list: list[str], origin: list[str], target: list[str], path: list[str]) -> list[str]:
            """Sequences token regions into a complete prompt.
            Includes any boundary tokens in `constatns.SPECIAL_TOKENS` such as <ADJLIST_START>, <ORIGIN_END>, etc.
            """
            pass
        

    class AOTP(PromptSequencer):
        def _sequence_tokens(self, adj_list: list[str], origin: list[str], target: list[str], path: list[str]) -> list[str]:
            return [
                SPECIAL_TOKENS.ADJLIST_START,
                *adj_list,
                SPECIAL_TOKENS.ADJLIST_END,
                SPECIAL_TOKENS.ORIGIN_START,
                *origin,
                SPECIAL_TOKENS.ORIGIN_END,
                SPECIAL_TOKENS.TARGET_START,
                *target,
                SPECIAL_TOKENS.TARGET_END,
                SPECIAL_TOKENS.PATH_START,
                *path,
                SPECIAL_TOKENS.PATH_END
            ]
            
    class AOP(PromptSequencer):
        include_target_special_tokens: bool = False
        
        def _sequence_tokens(self, adj_list: list[str], origin: list[str], target: list[str], path: list[str]) -> list[str]:
            return [
                SPECIAL_TOKENS.ADJLIST_START,
                *adj_list,
                SPECIAL_TOKENS.ADJLIST_END,
                SPECIAL_TOKENS.ORIGIN_START,
                *origin,
                SPECIAL_TOKENS.ORIGIN_END,
                *([SPECIAL_TOKENS.TARGET_START, SPECIAL_TOKENS.TARGET_END] if self.include_target_special_tokens else ()),
                SPECIAL_TOKENS.PATH_START,
                *path,
                SPECIAL_TOKENS.PATH_END
            ]


@serializable_dataclass(frozen=True, kw_only=True)
class MazeTokenizer2(SerializableDataclass):
    """Tokenizer for mazes
    
    # TODO: write docstring
    """
    prompt_sequencer: PromptSequencers.PromptSequencer = serializable_field(
        default=PromptSequencers.AOTP(),
        serialization_fn=lambda x: x.serialize(),
        loading_fn=lambda x: x.TokenizationElement.from_name(x)
    )
    coord_tokenizer: CoordTokenizers.CoordTokenizer = serializable_field(
        default=CoordTokenizers.UTUniform(),
        serialization_fn=lambda x: x.serialize(),
        loading_fn=lambda x: x.TokenizationElement.from_name(x)
    )
    adj_list_tokenizer: AdjListTokenizers.AdjListTokenizer = serializable_field(
        default=AdjListTokenizers.Coords(),
        serialization_fn=lambda x: x.serialize(),
        loading_fn=lambda x: x.TokenizationElement.from_name(x)
    )
    path_tokenizer: PathTokenizers.PathTokenizer = serializable_field(
        default=PathTokenizers.Coords(),
        serialization_fn=lambda x: x.serialize(),
        loading_fn=lambda x: x.TokenizationElement.from_name(x)
    )
    max_grid_size: int | None = serializable_field(default=None)
	    
    def __repr__(self) -> str:
        return "-".join([type(self).__name__, *(repr(el) for el in self._tokenizer_elements)])
   
    @cached_property
    def _tokenizer_elements(self):
        return [self.prompt_sequencer, self.coord_tokenizer, self.adj_list_tokenizer, self.path_tokenizer]
    
    @property
    def name(self) -> str:
        """ Serializes MazeTokenizer into a key for encoding in zanj """
        return '-'.join(['maze_tokenizer'] + [el.name for el in [self._tokenizer_elements]])
    
    @classmethod
    def from_name(cls, key: str) -> 'MazeTokenizer2':
        """ Builds a MazeTokenizer from the output of `MazeTokenizer2.name`"""
        pass
        
    def to_tokens(
        self,
        conn_list: ConnectionList,
        origin: CoordTup,
        target: CoordTup,
        path: CoordArray
        ) -> list[str]:
        return self.prompt_sequencer.to_tokens(conn_list, origin, target, path, self.coord_tokenizer, self.adj_list_tokenizer, self.path_tokenizer)
        
    @classmethod
    def from_tokens(
        cls, 
        tokens: str | list[str], 
        max_grid_size: int | None = None,
        rasterization_mode: type[CoordTokenizers.CoordTokenizer] = CoordTokenizers.UTUniform,
        ) -> 'MazeTokenizer2':
        """
        Infers most MazeTokenizer parameters from a full set of tokens.
        Could be useful for adapting old code to new `MazeTokenizer`.
        Would probably need a couple of other pieces of info besides just tokens.
        - max_grid_size
        - rasterization_mode: Only needed if UT tokens
        - Anything else?
        """
        # Don't need directly, but something similar needed for LatticeMaze.from_tokens
        
    @cached_property
    def _token_arr(self) -> list[str]:
        """map from index to token"""
        if self.max_grid_size is None:
            raise ValueError(
                f"max_grid_size must be specified to use token_arr property: {self.max_grid_size = }"
            )
        output: list[str] = list(SPECIAL_TOKENS.values())
        # output.extend(self._
            
        
    @cached_property
    def token_arr(self) -> list[str] | None:
        if self.max_grid_size is None:
            return None
        return self._token_arr
    
    
_line_for_debugging_breakpoint = 1