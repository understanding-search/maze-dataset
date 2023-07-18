import pytest
import numpy as np

from maze_dataset.constants import SPECIAL_TOKENS
from maze_dataset.tokenization.token_utils import (
    _coord_to_strings_UT,
    _coord_to_strings_indexed,
    str_is_coord,
    coord_str_to_tuple,
    coord_str_to_tuple_noneable,
    strings_to_coords,
    coords_to_strings,
    coord_str_to_coord_np,
    coords_string_split,
)

def test_coord_to_strings():
    assert _coord_to_strings_UT((1,2)) == ["(1,2)"]
    assert _coord_to_strings_UT((-1,0)) == ["(-1,0)"]

    assert _coord_to_strings_indexed((1,2)) == ["(", "1", ",", "2", ")"]
    assert _coord_to_strings_indexed((-1,0)) == ["(", "-1", ",", "0", ")"]

def test_str_is_coord():
    assert str_is_coord("(1,2)") == True
    assert str_is_coord("(-1,0)") == True
    assert str_is_coord("(1,2,3)") == True
    assert str_is_coord("1,2") == False
    assert str_is_coord("(1, 2)") == True
    assert str_is_coord("(1, 2)", allow_whitespace=False) == False

def test_coord_str_to_tuple():
    assert coord_str_to_tuple("(1,2)") == (1,2)
    assert coord_str_to_tuple("(-1,0)") == (-1,0)
    assert coord_str_to_tuple("(1,2,3)") == (1,2,3)
    assert coord_str_to_tuple("(1, 2)") == (1,2)
    assert coord_str_to_tuple("(1, 2)", allow_whitespace=False) == (1,2)

def test_coord_str_to_coord_np():
    assert (coord_str_to_coord_np("(1,2)") == np.array([1,2])).all()
    assert (coord_str_to_coord_np("(-1,0)") == np.array([-1,0])).all()
    assert (coord_str_to_coord_np("(1,2,3)") == np.array([1,2,3])).all()
    assert (coord_str_to_coord_np("(1, 2)") == np.array([1,2])).all()
    assert (coord_str_to_coord_np("(1, 2)", allow_whitespace=False) == np.array([1,2])).all()

def test_coord_str_to_tuple_noneable():
    assert coord_str_to_tuple_noneable("(1,2)") == (1,2)
    assert coord_str_to_tuple_noneable("(-1,0)") == (-1,0)
    assert coord_str_to_tuple_noneable("(1,2,3)") == (1,2,3)
    assert coord_str_to_tuple_noneable("(1, 2)") == (1,2)
    assert coord_str_to_tuple_noneable("1,2") == None

def test_coords_string_split():
    assert coords_string_split("(1,2) <ADJLIST_START> (5,6)") == ["(1,2)", "<ADJLIST_START>", "(5,6)"]
    assert coords_string_split("(1,2) (5,6)") == ["(1,2)", "(5,6)"]

def test_strings_to_coords():
    assert strings_to_coords("(1,2) <ADJLIST_START> (5,6)") == [(1,2), (5,6)]
    assert strings_to_coords("(1,2) <ADJLIST_START> (5,6)", when_noncoord="skip") == [(1,2), (5,6)]
    assert strings_to_coords("(1,2) <ADJLIST_START> (5,6)", when_noncoord="include") == [(1,2), "<ADJLIST_START>", (5,6)]
    with pytest.raises(ValueError):
        strings_to_coords("(1,2) <ADJLIST_START> (5,6)", when_noncoord="error")

def test_coords_to_strings():
    assert coords_to_strings([(1,2), "<ADJLIST_START>", (5,6)], _coord_to_strings_UT) == ["(1,2)", "(5,6)"]
    assert coords_to_strings([(1,2), "<ADJLIST_START>", (5,6)], _coord_to_strings_UT, when_noncoord="skip") == ["(1,2)", "(5,6)"]
    assert coords_to_strings([(1,2), "<ADJLIST_START>", (5,6)], _coord_to_strings_UT, when_noncoord="include") == ["(1,2)", "<ADJLIST_START>", "(5,6)"]
    with pytest.raises(ValueError):
        coords_to_strings([(1,2), "<ADJLIST_START>", (5,6)], _coord_to_strings_UT, when_noncoord="error")

    assert coords_to_strings([(1,2), "<ADJLIST_START>", (5,6)], _coord_to_strings_indexed) == ["(", "1", ",", "2", ")", "(", "5", ",", "6", ")"]
    assert coords_to_strings([(1,2), "<ADJLIST_START>", (5,6)], _coord_to_strings_indexed, when_noncoord="skip") == ["(", "1", ",", "2", ")", "(", "5", ",", "6", ")"]
    assert coords_to_strings([(1,2), "<ADJLIST_START>", (5,6)], _coord_to_strings_indexed, when_noncoord="include") == ["(", "1", ",", "2", ")", "<ADJLIST_START>", "(", "5", ",", "6", ")"]




def test_str_is_coord():
    assert str_is_coord("(1,2)")
    assert not str_is_coord("1,2")
    assert not str_is_coord("(1,2")
    assert not str_is_coord("1,2)")
    assert not str_is_coord("(1, a)")
    assert not str_is_coord("()")


def test_coord_str_to_tuple():
    assert coord_str_to_tuple("(1,2)") == (1, 2)
    with pytest.raises(ValueError):
        coord_str_to_tuple("(1, a)")
    with pytest.raises(ValueError):
        coord_str_to_tuple("()")


def test_coord_str_to_tuple_noneable():
    assert coord_str_to_tuple_noneable("(1,2)") == (1, 2)
    assert coord_str_to_tuple_noneable("1,2") is None
    assert coord_str_to_tuple_noneable("(1,2") is None
    assert coord_str_to_tuple_noneable("1,2)") is None
    assert coord_str_to_tuple_noneable("(1, a)") is None
    assert coord_str_to_tuple_noneable("()") is None


def test_coord_to_str():
    assert _coord_to_strings_UT((1, 2)) == ["(1,2)"]
    assert _coord_to_strings_UT((10, 20)) == ["(10,20)"]
    assert _coord_to_strings_UT((0, 0)) == ["(0,0)"]
    with pytest.raises(TypeError):
        _coord_to_strings_UT(1)

    assert _coord_to_strings_indexed((1,2)) == ["(", "1", ",", "2", ")"]
    assert _coord_to_strings_indexed((10,20)) == ["(", "10", ",", "20", ")"]
    assert _coord_to_strings_indexed((0,0)) == ["(", "0", ",", "0", ")"]
    with pytest.raises(TypeError):
        _coord_to_strings_indexed(1)
