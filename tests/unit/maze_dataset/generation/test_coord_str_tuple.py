import pytest

from maze_dataset.maze.lattice_maze import (
    coord_str_to_tuple,
    coord_str_to_tuple_noneable,
    coord_to_str,
    str_is_coord,
)


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
    assert coord_to_str((1, 2)) == "(1,2)"
    assert coord_to_str((10, 20)) == "(10,20)"
    assert coord_to_str((0, 0)) == "(0,0)"
    with pytest.raises(TypeError):
        coord_to_str(1)
