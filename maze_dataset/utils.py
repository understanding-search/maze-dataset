import cProfile
import enum
import itertools
import math
import pstats
import timeit
import typing
from dataclasses import field
from functools import wraps
from types import UnionType
from typing import (
    Any,
    Callable,
    ClassVar,
    Generator,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    TypeVar,
    get_args,
    get_origin,
    runtime_checkable,
)

import frozendict
import numpy as np
from jaxtyping import Bool, Int, Int8
from muutils.statcounter import StatCounter

WhenMissing = Literal["except", "skip", "include"]


@runtime_checkable
class IsDataclass(Protocol):
    # Type hint for any dataclass instance
    # https://stackoverflow.com/questions/54668000/type-hint-for-an-instance-of-a-non-specific-dataclass
    __dataclass_fields__: ClassVar[dict[str, Any]]


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


def manhattan_distance(
    edges: (
        Int[np.ndarray, "edges coord=2 row_col=2"]
        | Int[np.ndarray, "coord=2 row_col=2"]
    ),
) -> Int[np.ndarray, "edges"] | Int[np.ndarray, ""]:
    """Returns the Manhattan distance between two coords."""
    if len(edges.shape) == 3:
        return np.linalg.norm(edges[:, 0, :] - edges[:, 1, :], axis=1, ord=1).astype(
            np.int8
        )
    elif len(edges.shape) == 2:
        return np.linalg.norm(edges[0, :] - edges[1, :], ord=1).astype(np.int8)
    else:
        raise ValueError(
            f"{edges} has shape {edges.shape}, but must be match the shape in the type hints."
        )


def lattice_max_degrees(n: int) -> Int8[np.ndarray, "row col"]:
    """
    Returns an array with the maximum possible degree for each coord.
    """
    out = np.full((n, n), 2)
    out[1:-1, :] += 1
    out[:, 1:-1] += 1
    return out


def lattice_connection_array(
    n: int,
) -> Int8[np.ndarray, "edges=2*n*(n-1) leading_trailing_coord=2 row_col=2"]:
    """
    Returns a 3D NumPy array containing all the edges in a 2D square lattice of size n x n.
    Thanks Claude.

    # Parameters
    - `n`: The size of the square lattice.

    # Returns
    np.ndarray: A 3D NumPy array of shape containing the coordinates of the edges in the 2D square lattice.
    In each pair, the coord with the smaller sum always comes first.
    """
    row_coords, col_coords = np.meshgrid(
        np.arange(n, dtype=np.int8),
        np.arange(n, dtype=np.int8),
        indexing="ij",
    )

    # Horizontal edges
    horiz_edges = np.column_stack(
        (
            row_coords[:, :-1].ravel(),
            col_coords[:, :-1].ravel(),
            row_coords[:, 1:].ravel(),
            col_coords[:, 1:].ravel(),
        )
    )

    # Vertical edges
    vert_edges = np.column_stack(
        (
            row_coords[:-1, :].ravel(),
            col_coords[:-1, :].ravel(),
            row_coords[1:, :].ravel(),
            col_coords[1:, :].ravel(),
        )
    )

    return np.concatenate(
        (horiz_edges.reshape(n**2 - n, 2, 2), vert_edges.reshape(n**2 - n, 2, 2)),
        axis=0,
    )


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


def empty_sequence_if_attr_false(
    itr: Iterable[Any],
    attr_owner: Any,
    attr_name: str,
) -> Iterable[any]:
    """Returns `itr` if `attr_owner` has the attribute `attr_name` and it boolean casts to `True`. Returns an empty sequence otherwise.

    Particularly useful for optionally inserting delimiters into a sequence depending on an `TokenizerElement` attribute.

    # Parameters:
    - `itr: Iterable[Any]`
        The iterable to return if the attribute is `True`.
    - `attr_owner: Any`
        The object to check for the attribute.
    - `attr_name: str`
        The name of the attribute to check.

    # Returns:
    - `itr: Iterable` if `attr_owner` has the attribute `attr_name` and it boolean casts to `True`, otherwise an empty sequence.
    - `()` an empty sequence if the attribute is `False` or not present.
    """
    return itr if bool(getattr(attr_owner, attr_name, False)) else ()


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
        return False  # an ordinary class
    elif len(cls.__abstractmethods__) == 0:
        return False  # a concrete implementation of an abstract class
    else:
        return True  # an abstract class


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


"""
# `FiniteValued`
The details of this type are not possible to fully define via the Python 3.10 typing library.
This custom generic type is a generic domain of many types which have a finite, discrete, and well-defined range space.
`FiniteValued` defines the domain of supported types for the `all_instances` function, since that function relies heavily on static typing. 
These types may be nested in an arbitrarily deep tree via Container Types and Superclass Types (see below).
The leaves of the tree must always be Primitive Types. 

# `FiniteValued` Subtypes
*: Indicates that this subtype is not yet supported by `all_instances`

## Non-`FiniteValued` (Unbounded) Types
These are NOT valid subtypes, and are listed for illustrative purposes only.
This list is not comprehensive.
While the finite and discrete nature of digital computers means that the cardinality of these types is technically finite, 
they are considered unbounded types in this context.
- No Container subtype may contain any of these unbounded subtypes.
- `int`
- `float`
- `str`
- `list`
- `set`: Set types without a `FiniteValued` argument are unbounded
- `tuple`: Tuple types without a fixed length are unbounded

## Primitive Types
Primitive types are non-nested types which resolve directly to a concrete range of values 
- `bool`: has 2 possible values
- `enum.Enum`: The range of a concrete `Enum` subclass is its set of enum members
- `typing.Literal`: Every type constructed using `Literal` has a finite set of possible literal values in its definition.
This is the preferred way to include limited ranges of non-`FiniteValued` types such as `int` or `str` in a `FiniteValued` hierarchy.

## Container Types
Container types are types which contain zero or more fields of `FiniteValued` type.
The range of a container type is the cartesian product of their field types, except for `set[FiniteValued]`.
- `tuple[FiniteValued]`: Tuples of fixed length whose elements are each `FiniteValued`.
- `IsDataclass`: Concrete dataclasses whose fields are `FiniteValued`.
- *Standard concrete class: Regular classes could be supported just like dataclasses if all their data members are `FiniteValued`-typed.
- *`set[FiniteValued]`: Sets of fixed length of a `FiniteValued` type.

## Superclass Types
Superclass types don't directly contain data members like container types.
Their range is the union of the ranges of their subtypes.
- Abstract dataclasses: Abstract dataclasses whose subclasses are all `FiniteValued` superclass or container types
- *`IsDataclass`: Concrete dataclasses which also have their own subclasses.
- *Standard abstract classes: Abstract dataclasses whose subclasses are all `FiniteValued` superclass or container types
- `UnionType`: Any union of `FiniteValued` types, e.g., bool | Literal[2, 3]
"""
FiniteValued = TypeVar("FiniteValued", bound=bool | IsDataclass | enum.Enum)


def _apply_validation_func(
    type_: FiniteValued,
    vals: Generator[FiniteValued, None, None],
    validation_funcs: (
        frozendict.frozendict[FiniteValued, Callable[[FiniteValued], bool]] | None
    ) = None,
) -> Generator[FiniteValued, None, None]:
    """Helper function for `all_instances`"""
    if validation_funcs is None:
        return vals
    if type_ in validation_funcs:
        return filter(validation_funcs[type_], vals)
    elif hasattr(type_, "__mro__"):  # UnionType doesn't have `__mro__`
        for superclass in type_.__mro__:
            if superclass not in validation_funcs:
                continue
            vals = filter(validation_funcs[superclass], vals)
            break  # Only the first validation function hit in the mro is applied
    elif get_origin(type_) == Literal:
        return flatten(
            (
                _apply_validation_func(type(v), [v], validation_funcs)
                for v in get_args(type_)
            ),
            levels_to_flatten=1,
        )
    return vals


def _all_instances_wrapper(f):
    """
    Converts dicts to frozendicts to allow caching and applies `_apply_validation_func`.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if len(args) >= 2 and args[1] is not None:
            validation_funcs = frozendict.frozendict(args[1])
        elif "validation_funcs" in kwargs and kwargs["validation_funcs"] is not None:
            validation_funcs = frozendict.frozendict(kwargs["validation_funcs"])
        else:
            validation_funcs = None
        return _apply_validation_func(
            args[0], f(args[0], validation_funcs), validation_funcs
        )

    return wrapper


@_all_instances_wrapper
# @cache
def all_instances(
    type_: FiniteValued,
    validation_funcs: dict[FiniteValued, Callable[[FiniteValued], bool]] | None = None,
) -> Generator[FiniteValued, None, None]:
    """
    Returns all possible values of an instance of `type_` if finite instances exist.
    Uses type hinting to construct the possible values.
    All nested elements of `type_` must themselves be typed.
    Do not use with types whose members contain circular references.
    Function is susceptible to infinite recursion if `type_` is a dataclass whose member tree includes another instance of `type_`.

    # Parameters
    - `type_`: A finite-valued type. See docstring on `FiniteValued` for full details.
    - `validation_funcs`: A mapping of types to auxiliary functions to validate instances of that type.
    This optional argument can provide an additional, more precise layer of validation for the instances generated beyond what type hinting alone can provide.
    See `validation_funcs` Details section below.

    ## Supported `type_` Values
    See docstring on `FiniteValued` for full details.
    `type_` may be:
    - `FiniteValued`
    - A finite-valued, fixed-length Generic tuple type.
    E.g., `tuple[bool]`, `tuple[bool, MyEnum]` are OK.
    `tuple[bool, ...]` is NOT supported, since the length of the tuple is not fixed.
    - Nested versions of any of the types in this list
    - A `UnionType` of any of the types in this list

    ## `validation_funcs` Details
    `validation_funcs` is applied after all instances have been generated according to type hints.
    If `type_` is in `validation_funcs`, then the list of instances is filtered by `validation_funcs[type_](instance)`.
    `validation_funcs` is passed down for all recursive calls of `all_instances`.
    This allows for improved performance through maximal pruning of the exponential tree.
    `validation_funcs` supports subclass checking.
    If `type_` is not found in `validation_funcs`, then the search is performed iteratively in mro order.
    If a superclass of `type_` is found while searching in mro order, that validation function is applied and the list is returned.
    If no superclass of `type_` is found, then no filter is applied.
    """
    if type_ == bool:
        yield from [True, False]
    elif hasattr(type_, "__dataclass_fields__") and not is_abstract(type_):
        # Concrete dataclass: construct dataclass instances with all possible combinations of fields
        fields: list[field] = type_.__dataclass_fields__
        fields_to_types: dict[str, type] = {f: fields[f].type for f in fields}
        all_arg_sequences: Iterable = itertools.product(
            *[
                all_instances(arg_type, validation_funcs)
                for arg_type in fields_to_types.values()
            ]
        )
        yield from (
            type_(**{fld: arg for fld, arg in zip(fields_to_types.keys(), args)})
            for args in all_arg_sequences
        )
    elif hasattr(type_, "__dataclass_fields__") and is_abstract(type_):
        # Abstract dataclass: call `all_instances` on each subclass
        yield from flatten(
            (all_instances(sub, validation_funcs) for sub in type_.__subclasses__()),
            levels_to_flatten=1,
        )
    elif (
        get_origin(type_) == tuple
    ):  # Only matches Generic type tuple since regular tuple is not finite-valued
        # Generic tuple: Similar to concrete dataclass. Construct all possible combinations of tuple fields.
        yield from (
            tuple(combo)
            for combo in itertools.product(
                *(
                    all_instances(tup_item, validation_funcs)
                    for tup_item in get_args(type_)
                )
            )
        )
    elif get_origin(type_) in (UnionType, typing.Union):
        # Union: call `all_instances` for each type in the Union
        yield from flatten(
            [all_instances(sub, validation_funcs) for sub in get_args(type_)],
            levels_to_flatten=1,
        )
    elif get_origin(type_) is Literal:
        # Literal: return all Literal arguments
        yield from get_args(type_)
    elif type(type_) == enum.EnumMeta:  # `issubclass(type_, enum.Enum)` doesn't work
        # Enum: return all Enum members
        raise NotImplementedError(f"Support for Enums not yet implemented.")
    else:
        raise TypeError(
            f"Type {type_} either has unbounded possible values or is not supported."
        )


def get_hashable_eq_attrs(dc: IsDataclass) -> tuple[Any]:
    """Returns a tuple of all fields used for equality comparison.
    Essentially used to generate a hashable dataclass representation of a dataclass for equality comparison even if it's not frozen.
    """
    return *(
        getattr(dc, fld.name)
        for fld in filter(lambda x: x.compare, dc.__dataclass_fields__.values())
    ), type(dc)


def dataclass_set_equals(
    coll1: Iterable[IsDataclass], coll2: Iterable[IsDataclass]
) -> bool:
    """Compares 2 collections of dataclass instances as if they were sets.
    Duplicates are ignored in the same manner as a set.
    Unfrozen dataclasses can't be placed in sets since they're not hashable.
    Collections of them may be compared using this function.
    """

    return {get_hashable_eq_attrs(x) for x in coll1} == {
        get_hashable_eq_attrs(y) for y in coll2
    }


def isinstance_by_type_name(o: object, type_name: str):
    """Behaves like stdlib `isinstance` except it accepts a string representation of the type rather than the type itself.
    This is a hacky function intended to circumvent the need to import a type into a module.
    It is susceptible to type name collisions.

    # Parameters
    `o`: Object (not the type itself) whose type to interrogate
    `type_name`: The string returned by `type_.__name__`.
    Generic types are not supported, only types that would appear in `type_.__mro__`.
    """
    return type_name in {s.__name__ for s in type(o).__mro__}
