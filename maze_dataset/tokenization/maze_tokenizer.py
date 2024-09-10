"""turning a maze into text: `MazeTokenizerModular` and the legacy `TokenizationMode` enum and `MazeTokenizer` class"""

import abc
import base64
import hashlib
import random
import warnings
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    TypedDict,
    TypeVar,
)

import numpy as np
from jaxtyping import Bool, Int, Int64
from muutils.json_serialize import (
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)
from muutils.kappa import Kappa
from muutils.misc import empty_sequence_if_attr_false, flatten
from muutils.misc.sequence import WhenMissing
from zanj.loading import load_item_recursive

# from maze_dataset import SolvedMaze
from maze_dataset.constants import (
    SPECIAL_TOKENS,
    VOCAB,
    VOCAB_LIST,
    VOCAB_TOKEN_TO_INDEX,
    ConnectionArray,
    ConnectionList,
    Coord,
    CoordTup,
)
from maze_dataset.generation import numpy_rng
from maze_dataset.maze.lattice_maze import LatticeMaze, SolvedMaze
from maze_dataset.token_utils import (
    TokenizerPendingDeprecationWarning,
    _coord_to_strings_indexed,
    _coord_to_strings_UT,
    connection_list_to_adj_list,
    coords_to_strings,
    get_cardinal_direction,
    get_relative_direction,
    is_connection,
    strings_to_coords,
    tokens_between,
)
from maze_dataset.utils import corner_first_ndindex, lattice_connection_array


class TokenError(ValueError):
    """error for tokenization"""

    pass


class TokenizationMode(Enum):
    """legacy tokenization modes

    > [!CAUTION]
    > Legacy mode of tokenization. will still be around in future releases, but is no longer recommended for use.
    > Use `MazeTokenizerModular` instead.

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

    def to_legacy_tokenizer(self, max_grid_size: int | None = None):
        return MazeTokenizer(tokenization_mode=self, max_grid_size=max_grid_size)


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
        "`maze_tokenizer.get_tokens_up_to_path_start` will be deprecated for a `MazeTokenizerModular`-compatible function in a future release.",
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
    "max_grid_size",
    "token_arr",
    "tokenizer_map",
    "vocab_size",
    "padding_token_index",
]


@serializable_dataclass(
    properties_to_serialize=_MAZETOKENIZER_PROPERTIES_TO_SERIALIZE, kw_only=True
)
class MazeTokenizer(SerializableDataclass):
    """LEGACY Tokenizer for mazes

    > [!CAUTION]
    > `MazeTokenizerModular` is the new standard for tokenization. This class is no longer recommended
    > for use, but will remain for compatibility with existing code.

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
class _TokenizerElement(SerializableDataclass, abc.ABC):
    """Superclass for tokenizer elements.
    Subclasses contain modular functionality for maze tokenization.

    # Development
    > [!TIP]
    > Due to the functionality of `get_all_tokenizers()`, `_TokenizerElement` subclasses
    > may only contain fields of type `utils.FiniteValued`.
    > Implementing a subclass with an `int` or `float`-typed field, for example, is not supported.
    > In the event that adding such fields is deemed necessary, `get_all_tokenizers()` must be updated.

    """

    @staticmethod
    def _stringify(k: str, v: Any):
        if isinstance(v, bool):
            return f"{k}={str(v)[0]}"
        if isinstance(v, _TokenizerElement):
            return v.name
        if isinstance(v, tuple):
            return f"{k}={''.join(['(', *[str(x)+', ' for x in v], ')'])}"
        else:
            return f"{k}={v}"

    @property
    def name(self) -> str:
        members_str: str = ", ".join(
            [self._stringify(k, v) for k, v in self.__dict__.items() if k != "_type_"]
        )
        output: str = f"{type(self).__name__}({members_str})"
        if "." in output and output.index("(") > output.index("."):
            return "".join(output.split(".")[1:])
        else:
            return output

    def __str__(self):
        return self.name

    def __init_subclass__(cls, **kwargs):
        """
        Hack: dataclass hashes don't include the class itself in the hash function inputs.
        This causes dataclasses with identical fields but different types to hash identically.
        This hack circumvents this by adding a slightly hidden field to every subclass with a value of `repr(cls)`.
        To maintain compatibility with `all_instances`, the static type of the new field can only have 1 possible value.
        So we type it as a singleton `Literal` type.
        muutils 0.6.1 doesn't support `Literal` type validation, so `assert_type=False`.
        Ignore Pylance complaining about the arg to `Literal` being an expression.
        """
        super().__init_subclass__(**kwargs)
        cls._type_ = serializable_field(
            init=True, repr=False, default=repr(cls), assert_type=False
        )
        cls.__annotations__["_type_"] = Literal[repr(cls)]  # type: ignore

    def __hash__(self):
        "Stable hash to identify unique `MazeTokenizerModular` instances. uses name"
        return int.from_bytes(
            hashlib.blake2b(self.name.encode("utf-8")).digest(),
            byteorder="big",
        )

    @classmethod
    def _level_one_subclass(cls) -> type["_TokenizerElement"]:
        """Returns the immediate subclass of `_TokenizerElement` of which `cls` is an instance."""
        return (
            set(cls.__mro__).intersection(set(_TokenizerElement.__subclasses__())).pop()
        )

    def tokenizer_elements(self, deep: bool = True) -> list["_TokenizerElement"]:
        """
        Returns a list of all `_TokenizerElement` instances contained in the subtree.
        Currently only detects `_TokenizerElement` instances which are either direct attributes of another instance or
        which sit inside a `tuple` without further nesting.

        # Parameters
        - `deep: bool`: Whether to return elements nested arbitrarily deeply or just a single layer.
        """
        if not any(type(el) == tuple for el in self.__dict__.values()):
            return list(
                flatten(
                    [
                        [el] + el.tokenizer_elements()
                        for el in self.__dict__.values()
                        if isinstance(el, _TokenizerElement)
                    ]
                )
                if deep
                else filter(
                    lambda x: isinstance(x, _TokenizerElement), self.__dict__.values()
                )
            )
        else:
            non_tuple_elems: list[_TokenizerElement] = list(
                flatten(
                    [
                        [el] + el.tokenizer_elements()
                        for el in self.__dict__.values()
                        if isinstance(el, _TokenizerElement)
                    ]
                    if deep
                    else filter(
                        lambda x: isinstance(x, _TokenizerElement),
                        self.__dict__.values(),
                    )
                )
            )
            tuple_elems: list[_TokenizerElement] = list(
                flatten(
                    [
                        (
                            [
                                [tup_el] + tup_el.tokenizer_elements()
                                for tup_el in el
                                if isinstance(tup_el, _TokenizerElement)
                            ]
                            if deep
                            else filter(lambda x: isinstance(x, _TokenizerElement), el)
                        )
                        for el in self.__dict__.values()
                        if isinstance(el, tuple)
                    ]
                )
            )
            non_tuple_elems.extend(tuple_elems)
            return non_tuple_elems

    def tokenizer_element_tree(self, depth: int = 0, abstract: bool = False) -> str:
        """
        Returns a string representation of the tree of tokenizer elements contained in `self`.

        # Parameters
        - `depth: int`: Current depth in the tree. Used internally for recursion, no need to specify.
        - `abstract: bool`: Whether to print the name of the abstract base class or the concrete class for each `_TokenizerElement` instance.
        """
        name: str = "\t" * depth + (
            type(self).__name__
            if not abstract
            else type(self)._level_one_subclass().__name__
        )
        return (
            name
            + "\n"
            + "".join(
                el.tokenizer_element_tree(depth + 1, abstract)
                for el in self.tokenizer_elements(deep=False)
            )
        )

    def tokenizer_element_dict(self) -> dict:
        """
        Returns a dictionary representation of the tree of tokenizer elements contained in `self`.
        """
        return {
            type(self).__name__: {
                key: (
                    val.tokenizer_element_dict()
                    if isinstance(val, _TokenizerElement)
                    else (
                        val
                        if not isinstance(val, tuple)
                        else [
                            (
                                el.tokenizer_element_dict()
                                if isinstance(el, _TokenizerElement)
                                else el
                            )
                            for el in val
                        ]
                    )
                )
                for key, val in self.__dict__.items()
                if key != "_type_"
            }
        }

    @classmethod
    @abc.abstractmethod
    def attribute_key(cls) -> str:
        """Returns the binding used in `MazeTokenizerModular` for that type of `_TokenizerElement`."""
        raise NotImplementedError

    def to_tokens(self, *args, **kwargs) -> list[str]:
        """Converts a maze element into a list of tokens.
        Not all `_TokenizerElement` subclasses produce tokens, so this is not an abstract method.
        Those subclasses which do produce tokens should override this method.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_valid(self) -> bool:
        """Returns if `self` contains data members capable of producing an overall valid `MazeTokenizerModular`.
        Some `_TokenizerElement` instances may be created which are not useful despite obeying data member type hints.
        `is_valid` allows for more precise detection of invalid `_TokenizerElement`s beyond type hinting alone.
        If type hints are sufficient to constrain the possible instances of some subclass, then this method may simply `return True` for that subclass.

        # Types of Invalidity
        In nontrivial implementations of this method, each conditional clause should contain a comment classifying the reason for invalidity and one of the types below.
        Invalidity types, in ascending order of invalidity:
        - Uninteresting: These tokenizers might be used to train functional models, but the schemes are not interesting to study.
        E.g., `_TokenizerElement`s which are strictly worse than some alternative.
        - Duplicate: These tokenizers have identical tokenization behavior as some other valid tokenizers.
        - Untrainable: Training functional models using these tokenizers would be (nearly) impossible.
        - Erroneous: These tokenizers might raise exceptions during use.

        # Development
        `is_invalid` is implemented to always return `True` in some abstract classes where all currently possible subclass instances are valid.
        When adding new subclasses or data members, the developer should check if any such blanket statement of validity still holds and update it as neccesary.

        ## Nesting
        In general, when implementing this method, there is no need to recursively call `is_valid` on nested `_TokenizerElement`s contained in the class.
        In other words, failures of `is_valid` need not bubble up to the top of the nested `_TokenizerElement` tree.
        `MazeTokenizerModular.is_valid` calls `is_valid` on each of its `_TokenizerElement`s individually, so failure at any level will be detected.

        ## Types of Invalidity
        If it's judged to be useful, the types of invalidity could be implemented with an Enum or similar rather than only living in comments.
        This could be used to create more or less stringent filters on the valid `_TokenizerElement` instances.
        """
        raise NotImplementedError


T = TypeVar("T", bound=_TokenizerElement)


def mark_as_unsupported(is_valid: Callable[[T], bool], *args) -> T:
    """mark a _TokenizerElement as unsupported.

    Classes marked with this decorator won't show up in `get_all_tokenizers()` and thus wont be tested.
    The classes marked in release 1.0.0 did work reliably before being marked, but they can't be instantiated since the decorator adds an abstract method.
    The decorator exists to prune the space of tokenizers returned by `all_instances` both for testing and usage.
    Previously, the space was too large, resulting in impractical runtimes.
    These decorators could be removed in future releases to expand the space of possible tokenizers.
    """

    def wrapper(cls):
        cls.is_valid = is_valid
        return cls

    return wrapper


class __TokenizerElementNamespace(abc.ABC):
    """ABC for namespaces

    # Properties
    - key: The binding used in `MazeTokenizerModular` for instances of the classes contained within that `__TokenizerElementNamespace`.
    """

    key: str = NotImplementedError


def _load_tokenizer_element(
    data: dict[str, Any], namespace: type[__TokenizerElementNamespace]
) -> _TokenizerElement:
    """Loads a `TokenizerElement` stored via zanj."""
    key: str = namespace.key
    format: str = data[key]["__format__"]
    cls_name: str = format.split("(")[0]
    cls: type[_TokenizerElement] = getattr(namespace, cls_name)
    kwargs: dict[str, Any] = {
        k: load_item_recursive(data[key][k], tuple()) for k, v in data[key].items()
    }
    if "__format__" in kwargs:
        kwargs.pop("__format__")
    return cls(**kwargs)


class CoordTokenizers(__TokenizerElementNamespace):
    """Namespace for `_CoordTokenizer` subclass hierarchy used by `MazeTokenizerModular`."""

    key = "coord_tokenizer"

    @serializable_dataclass(frozen=True, kw_only=True)
    class _CoordTokenizer(_TokenizerElement, abc.ABC):
        """
        Superclass for classes which tokenize singular coords in a maze.
        """

        @abc.abstractmethod
        def to_tokens(self, coord: Coord | CoordTup) -> list[str]:
            pass

        @classmethod
        def attribute_key(cls) -> str:
            return CoordTokenizers.key

        def is_valid(self) -> bool:
            # No invalid instances possible within data member type hint bounds
            return True

    @serializable_dataclass(frozen=True, kw_only=True)
    class UT(_CoordTokenizer):
        """Unique token coordinate tokenizer."""

        def to_tokens(self, coord: Coord | CoordTup) -> list[str]:
            return ["".join(["(", str(coord[0]), ",", str(coord[1]), ")"])]

    @serializable_dataclass(frozen=True, kw_only=True)
    class CTT(_CoordTokenizer):
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
                *empty_sequence_if_attr_false([VOCAB.COORD_PRE], self, "pre"),
                str(coord[0]),
                *empty_sequence_if_attr_false([VOCAB.COORD_INTRA], self, "intra"),
                str(coord[1]),
                *empty_sequence_if_attr_false([VOCAB.COORD_POST], self, "post"),
            ]


class EdgeGroupings(__TokenizerElementNamespace):
    """Namespace for `_EdgeGrouping` subclass hierarchy used by `_AdjListTokenizer`."""

    key = "edge_grouping"

    class _GroupingTokenParams(TypedDict):
        """A uniform private hyperparameter interface used by `AdjListTokenizer`."""

        connection_token_ordinal: Literal[0, 1, 2]
        intra: bool
        grouped: bool

    @serializable_dataclass(frozen=True, kw_only=True)
    class _EdgeGrouping(_TokenizerElement, abc.ABC):
        """Specifies if/how multiple coord-coord connections are grouped together in a token subsequence called a edge grouping."""

        @classmethod
        def attribute_key(cls) -> str:
            return EdgeGroupings.key

        def is_valid(self) -> bool:
            return True

        @abc.abstractmethod
        def _group_edges(self, edges: ConnectionArray) -> Sequence[ConnectionArray]:
            """Divides a ConnectionArray into groups of edges.
            Shuffles/sequences within each group if applicable.
            """
            pass

        @abc.abstractmethod
        def _token_params(self) -> "EdgeGroupings._GroupingTokenParams":
            """Returns the tok.nization hyperparameters necessary for an `AdjListTokenizer` to tokenize.

            These hyperparameters are not used by `_EdgeGrouping` internally.
            They are located in `_EdgeGrouping` rather than in `AdjListTokenizer`
            since the hyperparameter space is a function of the `_EdgeGrouping` subclass.
            This function resolves the `_EdgeGrouping` hyperparameter space which is non-uniform across subclasses
            into a uniform private interface used by `AdjListTokenizer`.
            """
            pass

    @serializable_dataclass(frozen=True, kw_only=True)
    class Ungrouped(_EdgeGrouping):
        """No grouping occurs, each edge is tokenized individually.

        # Parameters
        - `connection_token_ordinal`: At which index in the edge tokenization the connector (or wall) token appears.
        Edge tokenizations contain 3 parts: a leading coord, a connector (or wall) token, and either a second coord or cardinal direction tokenization.
        """

        connection_token_ordinal: Literal[0, 1, 2] = serializable_field(
            default=1, assert_type=False
        )

        def _token_params(self) -> "EdgeGroupings._GroupingTokenParams":
            return EdgeGroupings._GroupingTokenParams(
                connection_token_ordinal=self.connection_token_ordinal,
                intra=False,
                grouped=False,
            )

        def _group_edges(self, edges: ConnectionList) -> Sequence[ConnectionList]:
            return np.expand_dims(edges, 1)

    @serializable_dataclass(frozen=True, kw_only=True)
    @mark_as_unsupported(lambda self_: False)
    class ByLeadingCoord(_EdgeGrouping):
        """All edges with the same leading coord are grouped together.

        # Parameters
        - `intra`: Whether all edge groupings include a delimiter token between individual edge representations.
        Note that each edge representation will already always include a connector token (`VOCAB.CONNECTOR`, or possibly `)
        - `shuffle_group`: Whether the sequence of edges within the group should be shuffled or appear in a fixed order.
        If false, the fixed order is lexicographical by (row, col).
        In effect, lexicographical sorting sorts edges by their cardinal direction in the sequence NORTH, WEST, EAST, SOUTH, where the directions indicate the position of the trailing coord relative to the leading coord.
        - `connection_token_ordinal`: At which index in token sequence representing a single edge the connector (or wall) token appears.
        Edge tokenizations contain 2 parts: a connector (or wall) token and a coord or cardinal tokenization.
        """

        intra: bool = serializable_field(default=True)
        shuffle_group: bool = serializable_field(default=True)
        connection_token_ordinal: Literal[0, 1] = serializable_field(
            default=0, assert_type=False
        )

        def _token_params(self) -> "EdgeGroupings._GroupingTokenParams":
            return EdgeGroupings._GroupingTokenParams(
                connection_token_ordinal=self.connection_token_ordinal,
                intra=self.intra,
                grouped=True,
            )

        def _group_edges(self, edges: ConnectionArray) -> Sequence[ConnectionArray]:
            # Adapted from: https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function
            index_array: Int[np.ndarray, "sort_indices=edges"] = np.lexsort(
                (edges[:, 1, 1], edges[:, 1, 0], edges[:, 0, 1], edges[:, 0, 0])
            )
            sorted_edges: ConnectionArray = edges[index_array, ...]
            groups: list[ConnectionArray] = np.split(
                sorted_edges,
                np.unique(sorted_edges[:, 0, :], return_index=True, axis=0)[1][1:],
            )
            if self.shuffle_group:
                [numpy_rng.shuffle(g, axis=0) for g in groups]
            return groups


class EdgePermuters(__TokenizerElementNamespace):
    """Namespace for `_EdgePermuter` subclass hierarchy used by `_AdjListTokenizer`."""

    key = "edge_permuter"

    @serializable_dataclass(frozen=True, kw_only=True)
    class _EdgePermuter(_TokenizerElement, abc.ABC):
        """Specifies how to sequence the two coords that encode a lattice edge."""

        @classmethod
        def attribute_key(cls) -> str:
            return EdgePermuters.key

        def is_valid(self) -> bool:
            # No invalid instances possible within data member type hint bounds
            return True

        @staticmethod
        @abc.abstractmethod
        def _permute(lattice_edges: ConnectionArray) -> ConnectionArray:
            """
            Executes a permutation.
            Warning: Caller should be aware that `lattice_edges` may be modified in-place depending on the subclass's implementation.

            # Parameters
            - `lattice_edges`: Array of lattice edges.
            The two coords in shape[1] must be adjacent in the lattice.

            # Returns
            - Array of lattice edges with entries along shape[1] systematically permuted.
            - shape[0] of the returned array is NOT guaranteed to match `lattice_edges.shape[1]`.
            """
            pass

    @serializable_dataclass(frozen=True, kw_only=True)
    class SortedCoords(_EdgePermuter):
        """returns a sorted representation. useful for checking consistency"""

        @staticmethod
        def _permute(lattice_edges: ConnectionArray) -> ConnectionArray:
            return lattice_edges[
                np.lexsort(
                    (
                        lattice_edges[:, 1, 1],
                        lattice_edges[:, 1, 0],
                        lattice_edges[:, 0, 1],
                        lattice_edges[:, 0, 0],
                    )
                ),
                ...,
            ]

    @serializable_dataclass(frozen=True, kw_only=True)
    class RandomCoords(_EdgePermuter):
        """Permutes each edge randomly."""

        @staticmethod
        def _permute(lattice_edges: ConnectionArray) -> ConnectionArray:
            numpy_rng.permuted(lattice_edges, axis=1, out=lattice_edges)
            return lattice_edges

    @serializable_dataclass(frozen=True, kw_only=True)
    class BothCoords(_EdgePermuter):
        """Includes both possible permutations of every edge in the output.
        Since input ConnectionList has only 1 instance of each edge,
        a call to `BothCoords._permute` will modify `lattice_edges` in-place, doubling `shape[0]`.
        """

        @staticmethod
        def _permute(lattice_edges: ConnectionArray) -> ConnectionArray:
            return np.append(lattice_edges, np.flip(lattice_edges, axis=1), axis=0)


class EdgeSubsets(__TokenizerElementNamespace):
    """
    Namespace for `_EdgeSubset` subclass hierarchy used by `_AdjListTokenizer`.
    """

    key = "edge_subset"

    @serializable_dataclass(frozen=True, kw_only=True)
    class _EdgeSubset(_TokenizerElement, abc.ABC):
        """
        Component of an `AdjListTokenizers._AdjListTokenizer` which specifies the subset of lattice edges to be tokenized.
        """

        @classmethod
        def attribute_key(cls) -> str:
            return EdgeSubsets.key

        def is_valid(self) -> bool:
            return True

        @abc.abstractmethod
        def _get_edges(self, maze: LatticeMaze) -> ConnectionArray:
            """
            Returns the set of lattice edges to be tokenized.
            """
            pass

    @serializable_dataclass(frozen=True, kw_only=True)
    class AllLatticeEdges(_EdgeSubset):
        """
        All 2n**2-2n edges of the lattice are tokenized.
        If a wall exists on that edge, the edge is tokenized in the same manner, using `VOCAB.ADJLIST_WALL` in place of `VOCAB.CONNECTOR`.
        """

        def _get_edges(self, maze: LatticeMaze) -> ConnectionArray:
            return lattice_connection_array(maze.grid_n)

    @serializable_dataclass(frozen=True, kw_only=True)
    class ConnectionEdges(_EdgeSubset):
        """
        Only edges which contain a connection are tokenized.
        Alternatively, only edges which contain a wall are tokenized.

        # Parameters
        - `walls`: Whether wall edges or connection edges are tokenized.
        If true, `VOCAB.ADJLIST_WALL` is used in place of `VOCAB.CONNECTOR`.
        """

        walls: bool = serializable_field(default=False)

        def _get_edges(self, maze: LatticeMaze) -> ConnectionArray:
            conn_list: ConnectionList = maze.connection_list
            if self.walls:
                conn_list = np.logical_not(conn_list)
                conn_list[0, -1, :] = False
                conn_list[1, :, -1] = False
            return connection_list_to_adj_list(
                conn_list, shuffle_d0=False, shuffle_d1=False
            )


class AdjListTokenizers(__TokenizerElementNamespace):
    """Namespace for `_AdjListTokenizer` subclass hierarchy used by `MazeTokenizerModular`."""

    key = "adj_list_tokenizer"

    @serializable_dataclass(frozen=True, kw_only=True)
    @mark_as_unsupported(lambda self_: self_.pre is False)
    class _AdjListTokenizer(_TokenizerElement, abc.ABC):
        """
        Specifies how the adjacency list is tokenized.
        Tokenization behavior is decomposed into specification of edge subsets, groupings, and permutations.
        See documentation of `EdgeSubset` and `EdgeGrouping` classes for more details.

        # Parameters
        - `pre`: Whether all edge groupings include a preceding delimiter token
        - `post`: Whether all edge groupings include a following delimiter token
        - `shuffle_d0`: Specifies how to sequence the edge groupings.
        If true, groupings are shuffled randomly. If false, groupings are sorted by the leading coord of each group.
        - `edge_grouping`: Specifies if/how multiple coord-coord connections are grouped together in a token subsequence called an edge grouping.
        - `edge_subset`: Specifies the subset of lattice edges to be tokenized.
        - `edge_permuter`: Specifies, in each edge tokenization, which coord either:
          1. Appears first in the tokenization, for `AdjListCoord`.
          2. Is tokenized directly as a coord, for `AdjListCardinal`.
          - `shuffle`: For each edge, the leading coord is selected randomly.
          - `all`: Each edge appears twice in the tokenization, appearing with both leading coords.
          - `evens`, `odds`: The leading coord is the one belonging to that coord subset. See `EdgeSubsets.ChessboardSublattice` for details.
        """

        pre: bool = serializable_field(default=False, assert_type=False)
        post: bool = serializable_field(default=True)
        shuffle_d0: bool = serializable_field(default=True)
        edge_grouping: EdgeGroupings._EdgeGrouping = serializable_field(
            default=EdgeGroupings.Ungrouped(),
            loading_fn=lambda x: _load_tokenizer_element(x, EdgeGroupings),
        )
        edge_subset: EdgeSubsets._EdgeSubset = serializable_field(
            default=EdgeSubsets.ConnectionEdges(),
            loading_fn=lambda x: _load_tokenizer_element(x, EdgeSubsets),
        )
        edge_permuter: EdgePermuters._EdgePermuter = serializable_field(
            default=EdgePermuters.RandomCoords(),
            loading_fn=lambda x: _load_tokenizer_element(x, EdgePermuters),
        )

        @classmethod
        def attribute_key(cls) -> str:
            return AdjListTokenizers.key

        def is_valid(self) -> bool:
            # No invalid instances possible within data member type hint bounds
            return True

        @abc.abstractmethod
        def _tokenization_callables(
            self,
            edges: ConnectionArray,
            is_conn: Bool[np.ndarray, "edges"],
            coord_tokenizer: CoordTokenizers._CoordTokenizer,
            *args,
            **kwargs,
        ):
            """
            Returns a sequence of callables which take an index in `edges` and return parts of that edge tokenization.

            # Returns
            - `[0]`: leading coord tokens
            - `[1]`: connector tokens
            - `[2]`: trailing coord tokens
            """
            pass

        def _tokenize_edge_grouping(
            self,
            edges: ConnectionArray,
            maze: LatticeMaze,
            coord_tokenizer: CoordTokenizers._CoordTokenizer,
            group_params: EdgeGroupings._GroupingTokenParams,
        ) -> Sequence[str]:
            """
            Tokenizes a single edge grouping.
            """
            cxn_ord: int = group_params["connection_token_ordinal"]
            is_conn: Bool[np.ndarray, "edges"] = is_connection(
                edges, maze.connection_list
            )
            tokenize_callables = self._tokenization_callables(
                edges, is_conn, coord_tokenizer
            )

            if group_params["grouped"]:
                # If grouped
                callable_permutation: list[int] = [1, 2] if cxn_ord == 0 else [2, 1]
                repeated_callables = [
                    tokenize_callables[i] for i in callable_permutation
                ]
                return flatten(
                    [
                        tokenize_callables[0](0),
                        [
                            [
                                *[
                                    tok_callable(i)
                                    for tok_callable in repeated_callables
                                ],
                                *(
                                    (VOCAB.ADJLIST_INTRA,)
                                    if group_params["intra"]
                                    else ()
                                ),
                            ]
                            for i in range(edges.shape[0])
                        ],
                    ]
                )
            else:
                # If ungrouped
                callable_permutation = [0, 2]
                callable_permutation.insert(cxn_ord, 1)
                tokenize_callables = [
                    tokenize_callables[i] for i in callable_permutation
                ]

                return flatten(
                    [
                        [
                            [
                                *[
                                    tok_callable(i)
                                    for tok_callable in tokenize_callables
                                ],
                                *empty_sequence_if_attr_false(
                                    (VOCAB.ADJLIST_INTRA,), group_params, "intra"
                                ),
                            ]
                            for i in range(edges.shape[0])
                        ]
                    ]
                )

        def to_tokens(
            self, maze: LatticeMaze, coord_tokenizer: CoordTokenizers._CoordTokenizer
        ) -> list[str]:
            # Get the set of edges to be tokenized
            edges: ConnectionArray = self.edge_subset._get_edges(maze)
            # Systematically permute the leading coord of each edge
            edges: ConnectionArray = self.edge_permuter._permute(edges)
            group_params: EdgeGroupings._GroupingTokenParams = (
                self.edge_grouping._token_params()
            )
            # then, we need to group the edges
            groups: Sequence[ConnectionArray] = self.edge_grouping._group_edges(edges)
            # shuffle the groups if specified
            if self.shuffle_d0:
                if isinstance(groups, np.ndarray):
                    numpy_rng.shuffle(groups, axis=0)
                elif isinstance(groups, list):
                    random.shuffle(groups)
                else:
                    raise TypeError(
                        f"`groups` is an unexpected type {type(groups)}. Only types `list` and `np.ndarray` are currently supported."
                    )
            # Tokenize each group with optional delimiters
            tokens: list[str] = list(
                flatten(
                    [
                        [
                            *empty_sequence_if_attr_false(
                                (VOCAB.ADJLIST_PRE,), self, "pre"
                            ),
                            *self._tokenize_edge_grouping(
                                group, maze, coord_tokenizer, group_params
                            ),
                            *empty_sequence_if_attr_false(
                                (VOCAB.ADJACENCY_ENDLINE,), self, "post"
                            ),
                        ]
                        for group in groups
                    ]
                )
            )
            return tokens

    @serializable_dataclass(frozen=True, kw_only=True)
    class AdjListCoord(_AdjListTokenizer):
        """Represents an edge group as tokens for the leading coord followed by coord tokens for the other group members."""

        edge_permuter: EdgePermuters._EdgePermuter = serializable_field(
            default=EdgePermuters.RandomCoords(),
            loading_fn=lambda x: _load_tokenizer_element(x, EdgePermuters),
        )

        def _tokenization_callables(
            self,
            edges: ConnectionArray,
            is_conn: Bool[np.ndarray, "edges"],
            coord_tokenizer: CoordTokenizers._CoordTokenizer,
            *args,
            **kwargs,
        ):
            # Map from `is_conn` to the tokens which represent connections and walls
            conn_token_map: dict[bool, str] = {
                True: VOCAB.CONNECTOR,
                False: VOCAB.ADJLIST_WALL,
            }
            return [
                lambda i: coord_tokenizer.to_tokens(edges[i, 0]),
                lambda i: conn_token_map[is_conn[i]],
                lambda i: coord_tokenizer.to_tokens(edges[i, 1]),
            ]

    @serializable_dataclass(frozen=True, kw_only=True)
    class AdjListCardinal(_AdjListTokenizer):
        """Represents an edge group as coord tokens for the leading coord and cardinal tokens relative to the leading coord for the other group members.

        # Parameters
        - `coord_first`: Whether the leading coord token(s) should come before or after the sequence of cardinal tokens.
        """

        edge_permuter: EdgePermuters._EdgePermuter = serializable_field(
            default=EdgePermuters.BothCoords(),
            loading_fn=lambda x: _load_tokenizer_element(x, EdgePermuters),
        )

        def _tokenization_callables(
            self,
            edges: ConnectionArray,
            is_conn: Bool[np.ndarray, "edges"],
            coord_tokenizer: CoordTokenizers._CoordTokenizer,
            *args,
            **kwargs,
        ):
            # Map from `is_conn` to the tokens which represent connections and walls
            conn_token_map: dict[bool, str] = {
                True: VOCAB.CONNECTOR,
                False: VOCAB.ADJLIST_WALL,
            }
            return [
                lambda i: coord_tokenizer.to_tokens(edges[i, 0]),
                lambda i: conn_token_map[is_conn[i]],
                lambda i: get_cardinal_direction(edges[i]),
            ]


class TargetTokenizers(__TokenizerElementNamespace):
    """Namespace for `_TargetTokenizer` subclass hierarchy used by `MazeTokenizerModular`."""

    key = "target_tokenizer"

    @serializable_dataclass(frozen=True, kw_only=True)
    class _TargetTokenizer(_TokenizerElement, abc.ABC):
        """Superclass of tokenizers for maze targets."""

        @abc.abstractmethod
        def to_tokens(
            self,
            targets: Sequence[Coord],
            coord_tokenizer: CoordTokenizers._CoordTokenizer,
        ) -> list[str]:
            """Returns tokens representing the target."""
            pass

        @classmethod
        def attribute_key(cls) -> str:
            return TargetTokenizers.key

    @serializable_dataclass(frozen=True, kw_only=True)
    class Unlabeled(_TargetTokenizer):
        """Targets are simply listed as coord tokens.
        - `post`: Whether all coords include an integral following delimiter token
        """

        post: bool = serializable_field(default=False)

        def to_tokens(
            self,
            targets: Sequence[Coord],
            coord_tokenizer: CoordTokenizers._CoordTokenizer,
        ) -> list[str]:
            return list(
                flatten(
                    [
                        [
                            *coord_tokenizer.to_tokens(target),
                            *empty_sequence_if_attr_false(
                                [VOCAB.TARGET_POST], self, "post"
                            ),
                        ]
                        for target in targets
                    ]
                )
            )

        def is_valid(self) -> bool:
            # No invalid instances possible within data member type hint bounds
            return True


class StepSizes(__TokenizerElementNamespace):
    """Namespace for `_StepSize` subclass hierarchy used by `MazeTokenizerModular`."""

    key = "step_size"

    @serializable_dataclass(frozen=True, kw_only=True)
    class _StepSize(_TokenizerElement, abc.ABC):
        """
        Specifies which coords in `maze.solution` are used to represent the path.
        """

        @classmethod
        def attribute_key(cls) -> str:
            return StepSizes.key

        @abc.abstractmethod  # TODO: make this a static/class method, allowing ForksAndStraightaways to skip object construction at every call
        def _step_single_indices(self, maze: SolvedMaze) -> list[int]:
            """Returns the indices of `maze.solution` corresponding to the steps to be tokenized."""
            raise NotImplementedError(
                "Subclasses must implement `StepSize.step_indices."
            )

        def step_start_end_indices(self, maze) -> list[tuple[int, int]]:
            """Returns steps as tuples of starting and ending positions for each step."""
            indices: list[int] = self._step_single_indices(maze)
            return [(start, end) for start, end in zip(indices[:-1], indices[1:])]

        def is_valid(self) -> bool:
            # No invalid instances possible within data member type hint bounds
            return True

    @serializable_dataclass(frozen=True, kw_only=True)
    class Singles(_StepSize):
        """
        Every coord in `maze.solution` is represented.
        Legacy tokenizers all use this behavior.
        """

        def _step_single_indices(self, maze: SolvedMaze) -> list[int]:
            """Returns the indices of `maze.solution` corresponding to the steps to be tokenized."""
            return list(range(maze.solution.shape[0]))

    @serializable_dataclass(frozen=True, kw_only=True)
    @mark_as_unsupported(lambda self_: False)
    class Straightaways(_StepSize):
        """
        Only coords where the path turns are represented in the path.
        I.e., the path is represented as a sequence of straightaways,
        specified by the coords at the turns.
        """

        def _step_single_indices(self, maze: SolvedMaze) -> list[int]:
            """Returns the indices of `maze.solution` corresponding to the steps to be tokenized."""
            last_turn_coord: Coord = maze.solution[0, ...]
            indices: list[int] = [0]
            for i, coord in enumerate(maze.solution):
                if coord[0] != last_turn_coord[0] and coord[1] != last_turn_coord[1]:
                    indices.append(i - 1)
                    last_turn_coord = maze.solution[i - 1, ...]
            indices.append(i)
            return indices

    @serializable_dataclass(frozen=True, kw_only=True)
    class Forks(_StepSize):
        """
        Only coords at forks, where the path has >=2 options for the next step are included.
        Excludes the option of backtracking.
        The starting and ending coords are always included.
        """

        def _step_single_indices(self, maze: SolvedMaze) -> list[int]:
            """Returns the indices of `maze.solution` corresponding to the steps to be tokenized."""
            return maze.get_solution_forking_points(always_include_endpoints=True)[0]

    @serializable_dataclass(frozen=True, kw_only=True)
    @mark_as_unsupported(lambda self_: False)
    class ForksAndStraightaways(_StepSize):
        """
        Includes the union of the coords included by `Forks` and `Straightaways`.
        See documentation for those classes for details.
        """

        def _step_single_indices(self, maze: SolvedMaze) -> list[int]:
            """Returns the indices of `maze.solution` corresponding to the steps to be tokenized."""
            return list(
                np.unique(
                    np.concatenate(
                        (
                            StepSizes.Straightaways()._step_single_indices(maze),
                            StepSizes.Forks()._step_single_indices(maze),
                        )
                    )
                )
            )


class StepTokenizers(__TokenizerElementNamespace):
    """Namespace for `_StepTokenizer` subclass hierarchy used by `MazeTokenizerModular`."""

    key = "step_tokenizers"

    @serializable_dataclass(frozen=True, kw_only=True)
    class _StepTokenizer(_TokenizerElement, abc.ABC):
        """
        Specifies how a single step (as specified by an instance of `_StepSize`) is tokenized.
        """

        @classmethod
        def attribute_key(cls) -> str:
            return StepTokenizers.key

        @abc.abstractmethod
        def to_tokens(
            self,
            maze: SolvedMaze,
            start_index: int,
            end_index: int,
            **kwargs,
        ) -> list[str]:
            """Tokenizes a single step in the solution.

            # Parameters
            - `maze`: Maze to be tokenized
            - `start_index`: The index of the Coord in `maze.solution` at which the current step starts
            - `end_index`: The index of the Coord in `maze.solution` at which the current step ends
            """
            raise NotImplementedError(
                "Subclasses must implement `StepTokenizer.to_tokens."
            )

        def is_valid(self) -> bool:
            # No invalid instances possible within data member type hint bounds
            return True

    @serializable_dataclass(frozen=True, kw_only=True)
    class Coord(_StepTokenizer):
        """
        A direct tokenization of the end position coord represents the step.
        """

        def to_tokens(
            self,
            maze: SolvedMaze,
            start_index: int,
            end_index: int,
            coord_tokenizer: CoordTokenizers._CoordTokenizer,
        ) -> list[str]:
            return coord_tokenizer.to_tokens(maze.solution[end_index, ...])

    @serializable_dataclass(frozen=True, kw_only=True)
    class Cardinal(_StepTokenizer):
        """
        A step is tokenized with a cardinal direction token.
        It is the direction of the step from the starting position along the solution.
        """

        def to_tokens(
            self, maze: SolvedMaze, start_index: int, end_index: int, **kwargs
        ) -> list[str]:
            return [
                get_cardinal_direction(maze.solution[start_index : start_index + 2])
            ]

    @serializable_dataclass(frozen=True, kw_only=True)
    class Relative(_StepTokenizer):
        """Tokenizes a solution step using relative first-person directions (right, left, forward, etc.).
        To simplify the indeterminacy, at the start of a solution the "agent" solving the maze is assumed to be facing NORTH.
        Similarly to `Cardinal`, the direction is that of the step from the starting position.
        """

        def to_tokens(
            self, maze: SolvedMaze, start_index: int, end_index: int, **kwargs
        ) -> list[str]:
            if start_index == 0:
                start = maze.solution[0]
                previous = start + np.array([1, 0])
                return [
                    get_relative_direction(
                        np.concatenate(
                            (
                                np.expand_dims(previous, 0),
                                maze.solution[start_index : start_index + 2],
                            ),
                            axis=0,
                        )
                    )
                ]
            return [
                get_relative_direction(maze.solution[start_index - 1 : start_index + 2])
            ]

    @serializable_dataclass(frozen=True, kw_only=True)
    class Distance(_StepTokenizer):
        """
        A count of the number of individual steps from the starting point to the end point.
        Contains no information about directionality, only the distance traveled in the step.
        `Distance` must be combined with at least one other `_StepTokenizer` in a `StepTokenizerPermutation`.
        This constraint is enforced in `_PathTokenizer.is_valid`.
        """

        def to_tokens(
            self, maze: SolvedMaze, start_index: int, end_index: int, **kwargs
        ) -> list[str]:
            d: int = end_index - start_index
            return [getattr(VOCAB, f"I_{d:03}")]

    """
    `StepTokenizerPermutation`
    A sequence of unique `_StepTokenizer`s.
    This type exists mostly just for the clarity and convenience of `_PathTokenizer` code.
    """
    StepTokenizerPermutation: type = (
        tuple[_StepTokenizer]
        | tuple[_StepTokenizer, _StepTokenizer]
        | tuple[_StepTokenizer, _StepTokenizer, _StepTokenizer]
        | tuple[_StepTokenizer, _StepTokenizer, _StepTokenizer, _StepTokenizer]
    )


class PathTokenizers(__TokenizerElementNamespace):
    """Namespace for `_PathTokenizer` subclass hierarchy used by `MazeTokenizerModular`."""

    key = "path_tokenizer"

    @serializable_dataclass(frozen=True, kw_only=True)
    class _PathTokenizer(_TokenizerElement, abc.ABC):
        """Superclass of tokenizers for maze solution paths."""

        @abc.abstractmethod
        def to_tokens(
            self, maze: SolvedMaze, coord_tokenizer: CoordTokenizers._CoordTokenizer
        ) -> list[str]:
            """Returns tokens representing the solution path."""
            pass

        @classmethod
        def attribute_key(cls) -> str:
            return PathTokenizers.key

    @serializable_dataclass(frozen=True, kw_only=True)
    class StepSequence(_PathTokenizer, abc.ABC):
        """Any `PathTokenizer` where the tokenization may be assembled from token subsequences, each of which represents a step along the path.
        Allows for a sequence of leading and trailing tokens which don't fit the step pattern.

        # Parameters
        - `step_size`: Selects the size of a single step in the sequence
        - `step_tokenizers`: Selects the combination and permutation of tokens
        - `pre`: Whether all steps include an integral preceding delimiter token
        - `intra`: Whether all steps include a delimiter token after each individual `_StepTokenizer` tokenization.
        - `post`: Whether all steps include an integral following delimiter token
        """

        step_size: StepSizes._StepSize = serializable_field(
            default=StepSizes.Singles(),
            loading_fn=lambda x: _load_tokenizer_element(x, StepSizes),
        )
        step_tokenizers: StepTokenizers.StepTokenizerPermutation = serializable_field(
            default=(StepTokenizers.Coord(),),
            serialization_fn=lambda x: [y.serialize() for y in x],
            loading_fn=lambda x: tuple(x[StepTokenizers.key]),
        )
        pre: bool = serializable_field(default=False)
        intra: bool = serializable_field(default=False)
        post: bool = serializable_field(default=False)

        def to_tokens(
            self, maze: SolvedMaze, coord_tokenizer: CoordTokenizers._CoordTokenizer
        ) -> list[str]:
            return [
                *self._leading_tokens(maze, coord_tokenizer),
                *flatten(
                    [
                        self._single_step_tokens(maze, start, end, coord_tokenizer)
                        for start, end in self.step_size.step_start_end_indices(maze)
                    ]
                ),
                *self._trailing_tokens(maze, coord_tokenizer),
            ]

        def _single_step_tokens(
            self,
            maze: SolvedMaze,
            i: int,
            j: int,
            coord_tokenizer: CoordTokenizers._CoordTokenizer,
        ) -> list[str]:
            """Returns the token sequence representing a single step along the path."""
            step_rep_tokens: list[list[str]] = [
                step_tokenizer.to_tokens(maze, i, j, coord_tokenizer=coord_tokenizer)
                for step_tokenizer in self.step_tokenizers
            ]
            if self.intra:
                step_rep_tokens_and_intra: list[str] = [None] * (
                    len(step_rep_tokens) * 2
                )
                step_rep_tokens_and_intra[::2] = step_rep_tokens
                step_rep_tokens_and_intra[1::2] = [VOCAB.PATH_INTRA] * len(
                    step_rep_tokens
                )
                step_rep_tokens = list(flatten(step_rep_tokens_and_intra))
            all_tokens: list[str] = [
                *empty_sequence_if_attr_false((VOCAB.PATH_PRE,), self, "pre"),
                *flatten(step_rep_tokens),
                *empty_sequence_if_attr_false((VOCAB.PATH_POST,), self, "post"),
            ]
            return all_tokens

        def _leading_tokens(
            self, maze: SolvedMaze, coord_tokenizer: CoordTokenizers._CoordTokenizer
        ) -> list[str]:
            """Returns tokens preceding those from the sequence from `_single_step_tokens`.
            Since the for loop in `to_tokens` iterates `len(path)-1` times, a fencepost problem exists with `StepTokenizers.Coord`.
            <PATH_START> should NOT be included.
            """
            if StepTokenizers.Coord() in self.step_tokenizers:
                return [
                    *empty_sequence_if_attr_false((VOCAB.PATH_PRE,), self, "pre"),
                    *coord_tokenizer.to_tokens(maze.solution[0, ...]),
                    *empty_sequence_if_attr_false((VOCAB.PATH_INTRA,), self, "intra"),
                ]
            return []

        def _trailing_tokens(
            self, c: Coord, coord_tokenizer: CoordTokenizers._CoordTokenizer
        ) -> list[str]:
            """Returns tokens following those from the sequence from `_single_step_tokens`.
            <PATH_END> should NOT be included.
            """
            return []

        def is_valid(self) -> bool:
            if len(set(self.step_tokenizers)) != len(self.step_tokenizers):
                # Uninteresting: repeated elements are not useful
                return False

            if len(self.step_tokenizers) == 1 and isinstance(
                self.step_tokenizers[0], StepTokenizers.Distance
            ):
                # Untrainable: `Distance` alone cannot encode a path. >=1 `StepTokenizer` which indicates direction/location is required.
                return False
            else:
                return True


class PromptSequencers(__TokenizerElementNamespace):
    """Namespace for `_PromptSequencer` subclass hierarchy used by `MazeTokenizerModular`."""

    key = "prompt_sequencer"

    @serializable_dataclass(frozen=True, kw_only=True)
    class _PromptSequencer(_TokenizerElement, abc.ABC):
        """
        Sequences token regions into a complete maze tokenization.

        # Parameters
        - `coord_tokenizer`: Tokenizer element which tokenizes a single `Coord` aka maze position.
        - `adj_list_tokenizer`: Tokenizer element which tokenizes the adjacency list of a `LatticeMaze`.
        Uses `coord_tokenizer` to tokenize coords if needed in other `TokenizerElement`s.
        """

        coord_tokenizer: CoordTokenizers._CoordTokenizer = serializable_field(
            default=CoordTokenizers.UT(),
            loading_fn=lambda x: _load_tokenizer_element(x, CoordTokenizers),
        )
        adj_list_tokenizer: AdjListTokenizers._AdjListTokenizer = serializable_field(
            default=AdjListTokenizers.AdjListCoord(),
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
            maze: LatticeMaze,
            *args,
            **kwargs,
        ) -> list[str]:
            """Returns a complete list of tokens for a given set of maze elements."""
            untrimmed: list[str] = self._sequence_tokens(
                *self._get_prompt_regions(maze)
            )
            return self._trim_if_unsolved_maze(
                untrimmed, not hasattr(maze, "start_pos"), not hasattr(maze, "solution")
            )

        def _get_prompt_regions(
            self,
            maze: LatticeMaze,
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
            origin: Coord | None = getattr(maze, "start_pos", None)
            target: list[Coord] | None = [
                getattr(maze, "end_pos", None)
            ]  # TargetTokenizer requires target: Sequence[Coord]

            return [
                (
                    self.adj_list_tokenizer.to_tokens(
                        maze, coord_tokenizer=self.coord_tokenizer
                    )
                    if hasattr(self, "adj_list_tokenizer")
                    else []
                ),
                self.coord_tokenizer.to_tokens(origin) if origin is not None else [],
                (
                    self.target_tokenizer.to_tokens(
                        target, coord_tokenizer=self.coord_tokenizer
                    )
                    if target[0] is not None and hasattr(self, "target_tokenizer")
                    else []
                ),
                (
                    self.path_tokenizer.to_tokens(
                        maze, coord_tokenizer=self.coord_tokenizer
                    )
                    if hasattr(maze, "solution") and hasattr(self, "path_tokenizer")
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

        def is_valid(self) -> bool:
            # No invalid instances possible within data member type hint bounds
            return True

    @serializable_dataclass(frozen=True, kw_only=True)
    class AOTP(_PromptSequencer):
        """
        Sequences a prompt as [adjacency list, origin, target, path].

        # Parameters
        - `target_tokenizer`: Tokenizer element which tokenizes the target(s) of a `TargetedLatticeMaze`.
        Uses `coord_tokenizer` to tokenize coords if that is part of the design of that `TargetTokenizer`.
        - `path_tokenizer`: Tokenizer element which tokenizes the solution path of a `SolvedMaze`.
        Uses `coord_tokenizer` to tokenize coords if that is part of the design of that `PathTokenizer`.

        """

        target_tokenizer: TargetTokenizers._TargetTokenizer = serializable_field(
            default=TargetTokenizers.Unlabeled(),
            loading_fn=lambda x: _load_tokenizer_element(x, TargetTokenizers),
        )
        path_tokenizer: PathTokenizers._PathTokenizer = serializable_field(
            default=PathTokenizers.StepSequence(),
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
    class AOP(_PromptSequencer):
        """Sequences a prompt as [adjacency list, origin, path].
        Still includes "<TARGET_START>" and "<TARGET_END>" tokens, but no representation of the target itself.

        # Parameters
        - `path_tokenizer`: Tokenizer element which tokenizes the solution path of a `SolvedMaze`.
        Uses `coord_tokenizer` to tokenize coords if that is part of the design of that `PathTokenizer`.
        """

        path_tokenizer: PathTokenizers._PathTokenizer = serializable_field(
            default=PathTokenizers.StepSequence(),
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


@serializable_dataclass(
    frozen=True,
    kw_only=True,
    properties_to_serialize=["tokenizer_element_tree_concrete", "name"],
)
class MazeTokenizerModular(SerializableDataclass):
    """Tokenizer for mazes

    # Parameters
    - `prompt_sequencer`: Tokenizer element which assembles token regions (adjacency list, origin, target, path) into a complete prompt.

    # Development
    - To ensure backwards compatibility, the default constructor must always return a tokenizer equivalent to the legacy `TokenizationMode.AOTP_UT_Uniform`.
    - Furthermore, the mapping reflected in `from_legacy` must also be maintained.
    - Updates to `MazeTokenizerModular` or the `_TokenizerElement` hierarchy must maintain that behavior.
    """

    prompt_sequencer: PromptSequencers._PromptSequencer = serializable_field(
        default=PromptSequencers.AOTP(),
        loading_fn=lambda x: _load_tokenizer_element(x, PromptSequencers),
    )

    def hash_int(self) -> int:
        return int.from_bytes(
            hashlib.blake2b(self.name.encode("utf-8")).digest(),
            byteorder="big",
        )

    def __hash__(self):
        "Stable hash to identify unique `MazeTokenizerModular` instances. uses name"
        return self.hash_int()

    def hash_b64(self, n_bytes: int = 8) -> str:
        """filename-safe base64 encoding of the hash"""
        # Use modulus to ensure the integer fits within n_bytes * 8 bits
        hash_mod: int = self.hash_int() % (1 << (n_bytes * 8))

        encoded = base64.b64encode(
            hash_mod.to_bytes(n_bytes, byteorder="big"), altchars=b"-_"
        ).decode()

        # Remove any padding equals signs
        return encoded.rstrip("=")

    # Information Querying Methods

    @cached_property
    def tokenizer_elements(self) -> list[_TokenizerElement]:
        return [self.prompt_sequencer, *self.prompt_sequencer.tokenizer_elements()]

    def tokenizer_element_tree(self, abstract: bool = False) -> str:
        """
        Returns a string representation of the tree of tokenizer elements contained in `self`.

        # Parameters
        - `abstract: bool`: Whether to print the name of the abstract base class or the concrete class for each `_TokenizerElement` instance.
        """

        return "\n".join(
            [
                type(self).__name__,
                self.prompt_sequencer.tokenizer_element_tree(
                    abstract=abstract, depth=1
                ),
            ]
        )

    @property
    def tokenizer_element_tree_concrete(self):
        """
        Property wrapper for `tokenizer_element_tree` so that it can be used in `properties_to_serialize`.
        """
        return self.tokenizer_element_tree()

    def tokenizer_element_dict(self) -> dict:
        """
        Nested dictionary of the internal `TokenizerElement`s.
        """
        return {type(self).__name__: self.prompt_sequencer.tokenizer_element_dict()}

    @property
    def name(self) -> str:
        """Serializes MazeTokenizer into a key for encoding in zanj"""
        return "-".join([type(self).__name__, self.prompt_sequencer.name])

    def summary(self) -> dict[str, str]:
        """
        Single-level dictionary of the internal `TokenizerElement`s.
        """
        return {
            # "prompt_sequencer": self.prompt_sequencer.name,
            **{elem.attribute_key(): elem.name for elem in self.tokenizer_elements}
        }

    @staticmethod
    def _type_check(obj: any) -> None:
        """Helper method for `has_element`"""
        if not (
            isinstance(obj, _TokenizerElement)
            or (isinstance(obj, type) and issubclass(obj, _TokenizerElement))
        ):
            raise TypeError(f"{obj} is not a `_TokenizerElement` instance or subclass.")

    def _has_element_singular(self, el: type[_TokenizerElement] | _TokenizerElement):
        """Helper method for `has_element`"""
        self._type_check(el)
        if isinstance(el, type):
            return any([isinstance(e, el) for e in self.tokenizer_elements])
        else:
            return el in self.tokenizer_elements

    def has_element(
        self,
        *elements: Sequence[type[_TokenizerElement] | _TokenizerElement],
    ) -> bool:
        """Returns True if the `MazeTokenizerModular` instance contains ALL of the items specified in `elements`.

        Querying with a partial subset of `_TokenizerElement` fields is not currently supported.
        To do such a query, assemble multiple calls to `has_elements`.

        # Parameters
        - `elements`: Singleton or iterable of `_TokenizerElement` instances or classes.
        If an instance is provided, then comparison is done via instance equality.
        If a class is provided, then comparison isdone via `isinstance`. I.e., any instance of that class is accepted.
        """
        if len(elements) == 1 and isinstance(elements[0], Iterable):
            elements = elements[0]
        return all([self._has_element_singular(e) for e in elements])

    def is_valid(self):
        """
        Returns `True` if `self` is a valid tokenizer.
        Evaluates the validity of all of `self.tokenizer_elements` according to each one's method.
        """
        return all([el.is_valid() for el in self.tokenizer_elements])

    def is_legacy_equivalent(self) -> bool:
        """Returns if `self` has identical stringification behavior as any legacy `MazeTokenizer`."""
        return any(
            [
                self == MazeTokenizerModular.from_legacy(tok_mode)
                for tok_mode in TokenizationMode
            ]
        )

    def is_tested_tokenizer(self, do_assert: bool = False) -> bool:
        """Returns if the tokenizer is returned by `all_tokenizers.get_all_tokenizers`, the set of tested and reliable tokenizers.

        Since evaluating `all_tokenizers.get_all_tokenizers` is expensive,
        instead checks for membership of `self`'s hash in `get_all_tokenizer_hashes()`.

        if `do_assert` is `True`, raises an `AssertionError` if the tokenizer is not tested.
        """
        all_tokenizer_hashes: Int64[np.ndarray, "n_tokenizers"] = (
            get_all_tokenizer_hashes()
        )
        hash_index: int = np.searchsorted(all_tokenizer_hashes, hash(self))

        in_range: bool = hash_index < len(all_tokenizer_hashes)
        hashes_match: bool = all_tokenizer_hashes[hash_index] == hash(self)
        is_valid: bool = self.is_valid()

        if do_assert:
            assert (
                in_range
            ), f"{hash_index = } is invalid, must be at most {len(all_tokenizer_hashes) - 1}"
            assert (
                hashes_match
            ), f"{all_tokenizer_hashes[hash_index] = } != {hash(self) = }"
            assert is_valid, f"self.is_valid returns False"
            return True
        else:
            return in_range and hashes_match and is_valid

    def is_AOTP(self) -> bool:
        return self.has_element(PromptSequencers.AOTP)

    def is_UT(self) -> bool:
        return self.has_element(CoordTokenizers.UT)

    # Alternate Constructors
    # ======================

    @classmethod
    def from_legacy(
        cls, legacy_maze_tokenizer: MazeTokenizer | TokenizationMode
    ) -> "MazeTokenizerModular":
        """Maps a legacy `MazeTokenizer` or `TokenizationMode` to its equivalent `MazeTokenizerModular` instance."""
        if isinstance(legacy_maze_tokenizer, MazeTokenizer):
            legacy_maze_tokenizer = legacy_maze_tokenizer.tokenization_mode
        return {
            TokenizationMode.AOTP_UT_uniform: MazeTokenizerModular(),
            TokenizationMode.AOTP_UT_rasterized: MazeTokenizerModular(),
            TokenizationMode.AOTP_CTT_indexed: MazeTokenizerModular(
                prompt_sequencer=PromptSequencers.AOTP(
                    coord_tokenizer=CoordTokenizers.CTT()
                )
            ),
        }[legacy_maze_tokenizer]

    # Simple properties
    # =================
    @classmethod
    def from_tokens(
        cls,
        tokens: str | list[str],
    ) -> "MazeTokenizerModular":
        """
        Infers most `MazeTokenizerModular` parameters from a full sequence of tokens.
        """
        raise NotImplementedError(
            "Recovering tokenizer objects from MazeTokenizerModular-produced strings is not supported"
        )

    @property
    def token_arr(self) -> list[str] | None:
        """map from index to token"""
        return VOCAB_LIST

    @property
    def tokenizer_map(self) -> dict[str, int]:
        """map from token to index"""
        return VOCAB_TOKEN_TO_INDEX

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the static vocab"""
        return len(VOCAB_LIST)

    @property
    def n_tokens(self) -> int:
        raise NameError(
            "`MazeTokenizerModular.n_tokens` has been removed. Use `len(maze_dataset.VOCAB_LIST)` instead."
        )

    @property
    def padding_token_index(self) -> int:
        return VOCAB_TOKEN_TO_INDEX[VOCAB.PADDING]

    # conversion functions
    # ============================================================

    def to_tokens(
        self,
        maze: LatticeMaze,
    ) -> list[str]:
        """Converts maze into a list of tokens."""
        return self.prompt_sequencer.to_tokens(maze)

    def coords_to_strings(self, coords: list[CoordTup | Coord]) -> list[str]:
        return list(
            flatten(
                [self.prompt_sequencer.coord_tokenizer.to_tokens(c) for c in coords]
            )
        )

    @staticmethod
    def strings_to_coords(
        text: str,
        when_noncoord: WhenMissing = "skip",
    ) -> list[str | CoordTup]:
        warnings.warn(
            "`MazeTokenizerModular.strings_to_coords` only supports legacy UT strings.",
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


_ALL_TOKENIZER_HASHES: Int64[np.ndarray, "n_tokenizers"]
"private array of all tokenizer hashes"
_TOKENIZER_HASHES_PATH: Path = Path(__file__).parent / "MazeTokenizerModular_hashes.npz"
"path to where we expect the hashes file -- in the same dir as this file, by default. change with `set_tokenizer_hashes_path`"


def set_tokenizer_hashes_path(path: Path):
    """set path to tokenizer hashes, and reload the hashes if needed

    the hashes are expected to be stored in and read from `_TOKENIZER_HASHES_PATH`,
    which by default is `Path(__file__).parent / "MazeTokenizerModular_hashes.npz"` or in this file's directory.

    However, this might not always work, so we provide a way to change this.
    """
    global _TOKENIZER_HASHES_PATH
    global _ALL_TOKENIZER_HASHES

    path = Path(path)
    if path.is_dir():
        path = path / "MazeTokenizerModular_hashes.npz"

    if not path.is_file():
        raise FileNotFoundError(f"could not find maze tokenizer hashes file at: {path}")

    if _TOKENIZER_HASHES_PATH.absolute() != path.absolute():
        # reload if they aren't equal
        _TOKENIZER_HASHES_PATH = path
        _ALL_TOKENIZER_HASHES = _load_tokenizer_hashes()
    else:
        # always set to new path
        _TOKENIZER_HASHES_PATH = path


def _load_tokenizer_hashes() -> Int64[np.ndarray, "n_tokenizers"]:
    """Loads the sorted list of `all_tokenizers.get_all_tokenizers()` hashes from disk."""
    global _TOKENIZER_HASHES_PATH
    try:
        path: Path = _TOKENIZER_HASHES_PATH
        return np.load(path)["hashes"]
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Tokenizers hashes cannot be loaded. To fix this, run",
            "\n`python -m maze-dataset.tokenization.save_hashes` which will save the hashes to",
            "\n`data/MazeTokenizerModular_hashes.npz`",
            "relative to the current working directory -- this is where the code looks for them.",
        ) from e


def get_all_tokenizer_hashes() -> Int64[np.ndarray, "n_tokenizers"]:
    global _ALL_TOKENIZER_HASHES
    try:
        got_tokenizers: bool = len(_ALL_TOKENIZER_HASHES) > 0
        if got_tokenizers:
            return _ALL_TOKENIZER_HASHES
        else:
            _ALL_TOKENIZER_HASHES = _load_tokenizer_hashes()
    except NameError:
        _ALL_TOKENIZER_HASHES = _load_tokenizer_hashes()

    return _ALL_TOKENIZER_HASHES
