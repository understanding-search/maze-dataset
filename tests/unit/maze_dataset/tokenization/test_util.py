import pytest

from maze_dataset.tokenization.util import flatten, get_all_subclasses

# Testing the flatten function
def test_flatten_full_flattening():
    assert list(flatten([1, [2, [3, 4]], 5])) == [1, 2, 3, 4, 5]
    assert list(flatten([1, [2, [3, [4, [5]]]]])) == [1, 2, 3, 4, 5]
    assert list(flatten([])) == []

def test_flatten_partial_flattening():
    assert list(flatten([1, [2, [3, 4]], 5], levels_to_flatten=1)) == [1, 2, [3, 4], 5]
    assert list(flatten([1, [2, [3, [4, [5]]]]], levels_to_flatten=2)) == [1, 2, 3, [4, [5]]]

def test_flatten_with_non_iterables():
    assert list(flatten([1, 2, 3])) == [1, 2, 3]
    assert list(flatten([1, "abc", 2, [3, 4], 5])) == [1, "abc", 2, 3, 4, 5]

# Testing the get_all_subclasses function
class A:
    pass

class B(A):
    pass

class C(B):
    pass

def test_get_all_subclasses():
    assert get_all_subclasses(A) == {B, C}
    assert get_all_subclasses(B) == {C}
    assert get_all_subclasses(C) == set()

def test_get_all_subclasses_include_self():
    assert get_all_subclasses(A, include_self=True) == {A, B, C}
    assert get_all_subclasses(B, include_self=True) == {B, C}
    assert get_all_subclasses(C, include_self=True) == {C}
