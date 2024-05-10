import cProfile
import math
import pstats
import timeit
import typing
from typing import (
    Any, 
    Callable, 
    Iterable, 
    Literal, 
    Mapping, 
    NamedTuple, 
    TypeVar,
    Generator,
    Protocol,
    runtime_checkable,
    get_args,
    get_origin,
    ClassVar,
    )
from types import UnionType
from dataclasses import field
import itertools
import enum

import numpy as np
from jaxtyping import Bool
from muutils.statcounter import StatCounter

WhenMissing = Literal["except", "skip", "include"]


@runtime_checkable
class IsDataclass(Protocol):
    # Type hint for any dataclass instance
    # https://stackoverflow.com/questions/54668000/type-hint-for-an-instance-of-a-non-specific-dataclass
    __dataclass_fields__: ClassVar[dict[str, Any]]

    
FiniteValued = TypeVar("FiniteValued", bool, IsDataclass, enum.Enum)


def bool_array_from_string(
    string: str, shape: list[int], true_symbol: str = "T"
) -> Bool[np.ndarray, "*shape"]:
    """Transform a string into an ndarray of bools.

    Parameters
    ----------
    string: str
        The string representation of the array
    shape: list[int]
        The shape of the resulting array
    true_symbol:
        The character to parse as True. Whitespace will be removed. All other characters will be parsed as False.

    Returns
    -------
    np.ndarray
        A ndarray with dtype bool of shape `shape`

    Examples
    --------
    >>> bool_array_from_string(
    ...     "TT TF", shape=[2,2]
    ... )
    array([[ True,  True],
        [ True, False]])
    """
    stripped = "".join(string.split())

    expected_symbol_count = math.prod(shape)
    symbol_count = len(stripped)
    if len(stripped) != expected_symbol_count:
        raise ValueError(
            f"Connection List contains the wrong number of symbols. Expected {expected_symbol_count}. Found {symbol_count} in {stripped}."
        )

    bools = [True if symbol == true_symbol else False for symbol in stripped]
    return np.array(bools).reshape(*shape)


def corner_first_ndindex(n: int, ndim: int = 2) -> list[tuple]:
    """returns an array of indices, sorted by distance from the corner

    this gives the property that `np.ndindex((n,n))` is equal to
    the first n^2 elements of `np.ndindex((n+1, n+1))`

    ```
    >>> corner_first_ndindex(1)
    [(0, 0)]
    >>> corner_first_ndindex(2)
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    >>> corner_first_ndindex(3)
    [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2)]
    ```
    """

    unsorted: list = list(np.ndindex(tuple([n for _ in range(ndim)])))
    return sorted(unsorted, key=lambda x: (max(x), x if x[0] % 2 == 0 else x[::-1]))

    # alternate numpy version from GPT-4:
    """
    # Create all index combinations
    indices = np.indices([n]*ndim).reshape(ndim, -1).T
    # Find the max value for each index
    max_indices = np.max(indices, axis=1)
    # Identify the odd max values
    odd_mask = max_indices % 2 != 0
    # Make a copy of indices to avoid changing the original one
    indices_copy = indices.copy()
    # Reverse the order of the coordinates for indices with odd max value
    indices_copy[odd_mask] = indices_copy[odd_mask, ::-1]
    # Sort by max index value, then by coordinates
    sorted_order = np.lexsort((*indices_copy.T, max_indices))
    return indices[sorted_order]
    """


def adj_list_to_nested_set(adj_list: list) -> set:
    """Used for comparison of adj_lists

    Adj_list looks like [[[0, 1], [1, 1]], [[0, 0], [0, 1]], ...]
    We don't care about order of coordinate pairs within
    the adj_list or coordinates within each coordinate pair."""
    return {
        frozenset([tuple(start_coord), tuple(end_coord)])
        for start_coord, end_coord in adj_list
    }


_AM_K = typing.TypeVar("_AM_K")
_AM_V = typing.TypeVar("_AM_V")


def apply_mapping(
    mapping: Mapping[_AM_K, _AM_V],
    iter: Iterable[_AM_K],
    when_missing: WhenMissing = "skip",
) -> list[_AM_V]:
    """Given an and a mapping, apply the mapping to the iterable with certain options"""
    output: list[_AM_V] = list()
    item: _AM_K
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


def apply_mapping_chain(
    mapping: Mapping[_AM_K, Iterable[_AM_V]],
    iter: Iterable[_AM_K],
    when_missing: WhenMissing = "skip",
) -> list[_AM_V]:
    """Given a list and a mapping, apply the mapping to the list"""
    output: list[_AM_V] = list()
    item: _AM_K
    for item in iter:
        if item in mapping:
            output.extend(mapping[item])
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


T = TypeVar("T")

FancyTimeitResult = NamedTuple(
    "FancyTimeitResult",
    [
        ("timings", StatCounter),
        ("return_value", T | None),
        ("profile", pstats.Stats | None),
    ],
)


def timeit_fancy(
    cmd: Callable[[], T] | str,
    setup: str = lambda: None,
    repeats: int = 5,
    namespace: dict[str, Any] | None = None,
    get_return: bool = True,
    do_profiling: bool = False,
) -> FancyTimeitResult:
    """
    Wrapper for `timeit` to get the fastest run of a callable with more customization options.

    Approximates the functionality of the %timeit magic or command line interface in a Python callable.

    # Parameters
    - `cmd: Callable[[], T] | str`
        The callable to time. If a string, it will be passed to `timeit.Timer` as the `stmt` argument.
    - `setup: str`
        The setup code to run before `cmd`. If a string, it will be passed to `timeit.Timer` as the `setup` argument.
    - `repeats: int`
        The number of times to run `cmd` to get a reliable measurement.
    - `namespace: dict[str, Any]`
        Passed to `timeit.Timer` constructor.
        If `cmd` or `setup` use local or global variables, they must be passed here. See `timeit` documentation for details.
    - `get_return: bool`
        Whether to pass the value returned from `cmd`. If True, the return value will be appended in a tuple with execution time.
        This is for speed and convenience so that `cmd` doesn't need to be run again in the calling scope if the return values are needed.
        (default: `False`)
    - `do_profiling: bool`
        Whether to return a `pstats.Stats` object in addition to the time and return value.
        (default: `False`)

    # Returns
    `FancyTimeitResult`, which is a NamedTuple with the following fields:
    - `time: float`
        The time in seconds it took to run `cmd` the minimum number of times to get a reliable measurement.
    - `return_value: T|None`
        The return value of `cmd` if `get_return` is `True`, otherwise `None`.
    - `profile: pstats.Stats|None`
        A `pstats.Stats` object if `do_profiling` is `True`, otherwise `None`.
    """
    timer: timeit.Timer = timeit.Timer(cmd, setup, globals=namespace)

    # Perform the timing
    times: list[float] = timer.repeat(repeats, 1)

    # Optionally capture the return value
    return_value: T | None = None
    profile: pstats.Stats | None = None

    if get_return or do_profiling:
        # Optionally perform profiling
        if do_profiling:
            profiler = cProfile.Profile()
            profiler.enable()

        return_value: T = cmd()

        if do_profiling:
            profiler.disable()
            profile = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")

    # reset the return value if it wasn't requested
    if not get_return:
        return_value = None

    return FancyTimeitResult(
        timings=StatCounter(times),
        return_value=return_value,
        profile=profile,
    )


def unpackable_if_true_attribute(
    iterable: Iterable[any], attr_owner: Any, attr_name: str
) -> Iterable[any]:
    """Returns `iterable` if `attr_owner` has the attribute `attr_name` and it boolean casts to `True`.
    Particularly useful for optionally inserting delimiters into a sequence depending on an `TokenizerElement` attribute.
    """
    return iterable if bool(getattr(attr_owner, attr_name, False)) else ()


def flatten(it: Iterable[any], levels_to_flatten: int | None = None) -> Generator:
    """
    Flattens an arbitrarily nested iterable.
    Flattens all iterable data types except for `str` and `bytes`.

    # Returns
    Generator over the flattened sequence.

    # Parameters
    - `it`: Any arbitrarily nested iterable.
    - `levels_to_flatten`: Number of levels to flatten by. If `None`, performs full flattening.
    """
    for x in it:
        # TODO: swap type check with more general check for __iter__() or __next__() or whatever
        if (
            hasattr(x, "__iter__")
            and not isinstance(x, (str, bytes))
            and (levels_to_flatten is None or levels_to_flatten > 0)
        ):
            yield from flatten(
                x, None if levels_to_flatten == None else levels_to_flatten - 1
            )
        else:
            yield x


def is_abstract(cls):
    if not hasattr(cls, "__abstractmethods__"):
        return False # an ordinary class
    elif len(cls.__abstractmethods__) == 0:
        return False # a concrete implementation of an abstract class
    else:
        return True # an abstract class


def get_all_subclasses(class_: type, include_self=False) -> set[type]:
    """
    Returns a set containing all child classes in the subclass graph of `class_`.
    I.e., includes subclasses of subclasses, etc.

    # Parameters
    - `include_self`: Whether to include `class_` itself in the returned list
    - `class_`: Superclass
    
    # Development
    Since most class hierarchies are small, the inefficiencies of the existing recursive implementation aren't problematic.
    It might be valuable to refactor with memoization if the need arises to use this function on a very large class hierarchy.
    """
    subs: list[set] = [
        get_all_subclasses(sub, include_self=True)
        for sub in class_.__subclasses__()
        if sub is not None
    ]
    subs: set = set(flatten(subs))
    if include_self:
        subs.add((class_))
    return subs
    

def all_instances(type_: FiniteValued) -> list[FiniteValued]:
    """Returns all possible values of an instance of `type_` if finite instances exist.
    Do not use with types whose members contain circular references.
    Function is susceptible to infinite recursion if `type_` is a dataclass whose member tree includes another instance of `type_`.
    """
    if type_ == bool:
        return [True, False]
    elif hasattr(type_, "__dataclass_fields__") and not is_abstract(type_):
        fields: list[field] = type_.__dataclass_fields__
        fields_to_types: dict[str, type] = {f: fields[f].type for f in fields}
        all_arg_sequences: Iterable = itertools.product(*[all_instances(arg_type) for arg_type in fields_to_types.values()])
        return [type_(**{fld: arg for fld, arg in zip(fields_to_types.keys(), args)}) 
                for args in all_arg_sequences]
    elif hasattr(type_, "__dataclass_fields__") and is_abstract(type_):
        return list(flatten([all_instances(sub) for sub in type_.__subclasses__()], levels_to_flatten=1))
    elif get_origin(type_) == tuple: # Only matches Generic type tuple since regular tuple is not Finite-valued
        ...
        # TODO: figure this out for weird possible tuple variants
    elif get_origin(type_) == UnionType: # Union: get all possible values for each Union arg
        list(flatten([all_instances(sub) for sub in get_args(type_)], levels_to_flatten=1))
    elif type(type_) == enum.EnumMeta: # `issubclass(type_, enum.Enum)` doesn't work
        raise NotImplementedError(f"Support for Enums not yet implemented.")
    else:
        raise TypeError(f"Type {type_} either has unbounded possible values or is not supported.")
    
    
def dataclass_set_equals(coll1: Iterable[IsDataclass], coll2: Iterable[IsDataclass]) -> bool:
    """Compares 2 collections of dataclass instances as if they were sets.
    Duplicates are ignored in the same manner as a set.
    Unfrozen dataclasses can't be placed in sets since they're not hashable.
    Collections of them may be compared using this function.
    """
    def get_hashable_eq_attrs(dc: IsDataclass) -> tuple[Any]:
        """Returns a tuple of all fields used for equality comparison.
        Essentially used to generate a hashable dataclass representation of a dataclass for equality comparison even if it's not frozen.
        """
        return *(getattr(dc, fld.name) for fld in filter(lambda x: x.compare, dc.__dataclass_fields__.values())), type(dc)
    
    return {get_hashable_eq_attrs(x) for x in coll1} == {get_hashable_eq_attrs(y) for y in coll2}