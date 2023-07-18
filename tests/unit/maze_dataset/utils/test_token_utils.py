import pytest

from maze_dataset.dataset.maze_dataset import MazeDatasetConfig
from maze_dataset.tokenization.token_utils import (
    _coord_to_strings_UT,
    coords_to_strings,
    get_adj_list_tokens,
    get_origin_tokens,
    get_path_tokens,
    get_target_tokens,
    get_tokens_up_to_path_start,
    strings_to_coords,
    tokens_between,
)

MAZE_TOKENS = "<ADJLIST_START> (0,1) <--> (1,1) ; (1,0) <--> (1,1) ; (0,1) <--> (0,0) ; <ADJLIST_END> <ORIGIN_START> (1,0) <ORIGIN_END> <TARGET_START> (1,1) <TARGET_END> <PATH_START> (1,0) (1,1) <PATH_END>".split()


def test_tokens_between():
    result = tokens_between(MAZE_TOKENS, "<PATH_START>", "<PATH_END>")
    assert result == ["(1,0)", "(1,1)"]

    # Normal case
    tokens = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    start_value = "quick"
    end_value = "over"
    assert tokens_between(tokens, start_value, end_value) == ["brown", "fox", "jumps"]

    # Including start and end values
    assert tokens_between(tokens, start_value, end_value, True, True) == [
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
    ]

    # When start_value or end_value is not unique and except_when_tokens_not_unique is True
    with pytest.raises(ValueError):
        tokens_between(tokens, "the", "dog", False, False, True)

    # When start_value or end_value is not unique and except_when_tokens_not_unique is False
    assert tokens_between(tokens, "the", "dog", False, False, False) == [
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "the",
        "lazy",
    ]

    # Empty tokens list
    with pytest.raises(ValueError):
        tokens_between([], "start", "end")

    # start_value and end_value are the same
    with pytest.raises(ValueError):
        tokens_between(tokens, "fox", "fox")

    # start_value or end_value not in the tokens list
    with pytest.raises(ValueError):
        tokens_between(tokens, "start", "end")

    # start_value comes after end_value in the tokens list
    with pytest.raises(AssertionError):
        tokens_between(tokens, "over", "quick")

    # start_value and end_value are at the beginning and end of the tokens list, respectively
    assert tokens_between(tokens, "the", "dog", True, True) == tokens

    # Single element in the tokens list, which is the same as start_value and end_value
    with pytest.raises(ValueError):
        tokens_between(["fox"], "fox", "fox", True, True)


def test_tokens_between_out_of_order():
    with pytest.raises(AssertionError):
        tokens_between(MAZE_TOKENS, "<PATH_END>", "<PATH_START>")


def test_get_adj_list_tokens():
    result = get_adj_list_tokens(MAZE_TOKENS)
    expected = "(0,1) <--> (1,1) ; (1,0) <--> (1,1) ; (0,1) <--> (0,0) ;".split()
    assert result == expected


def test_get_path_tokens():
    result_notrim = get_path_tokens(MAZE_TOKENS)
    assert result_notrim == ["<PATH_START>", "(1,0)", "(1,1)", "<PATH_END>"]
    result_trim = get_path_tokens(MAZE_TOKENS, trim_end=True)
    assert result_trim == ["(1,0)", "(1,1)"]


def test_get_origin_tokens():
    result = get_origin_tokens(MAZE_TOKENS)
    assert result == ["(1,0)"]


def test_get_target_token():
    result = get_target_tokens(MAZE_TOKENS)
    assert result == ["(1,1)"]


def test_get_tokens_up_to_path_start_including_start():
    result = get_tokens_up_to_path_start(MAZE_TOKENS)
    expected = "<ADJLIST_START> (0,1) <--> (1,1) ; (1,0) <--> (1,1) ; (0,1) <--> (0,0) ; <ADJLIST_END> <ORIGIN_START> (1,0) <ORIGIN_END> <TARGET_START> (1,1) <TARGET_END> <PATH_START> (1,0)".split()
    assert result == expected


def test_get_tokens_up_to_path_start_excluding_start():
    result = get_tokens_up_to_path_start(MAZE_TOKENS, include_start_coord=False)
    expected = "<ADJLIST_START> (0,1) <--> (1,1) ; (1,0) <--> (1,1) ; (0,1) <--> (0,0) ; <ADJLIST_END> <ORIGIN_START> (1,0) <ORIGIN_END> <TARGET_START> (1,1) <TARGET_END> <PATH_START>".split()
    assert result == expected


def test_strings_to_coords():
    adj_list = get_adj_list_tokens(MAZE_TOKENS)
    skipped = strings_to_coords(adj_list, when_noncoord="skip")
    included = strings_to_coords(adj_list, when_noncoord="include")

    assert skipped == [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, 1),
        (0, 1),
        (0, 0),
    ]

    assert included == [
        (0, 1),
        "<-->",
        (1, 1),
        ";",
        (1, 0),
        "<-->",
        (1, 1),
        ";",
        (0, 1),
        "<-->",
        (0, 0),
        ";",
    ]

    with pytest.raises(ValueError):
        strings_to_coords(adj_list, when_noncoord="except")


def test_coords_to_strings():
    adj_list = get_adj_list_tokens(MAZE_TOKENS)
    config = MazeDatasetConfig(name="test", grid_n=2, n_mazes=1)
    coords = strings_to_coords(adj_list, when_noncoord="include")

    skipped = coords_to_strings(
        coords, coord_to_strings_func=_coord_to_strings_UT, when_noncoord="skip"
    )
    included = coords_to_strings(
        coords, coord_to_strings_func=_coord_to_strings_UT, when_noncoord="include"
    )

    assert skipped == [
        "(0,1)",
        "(1,1)",
        "(1,0)",
        "(1,1)",
        "(0,1)",
        "(0,0)",
    ]

    assert included == [
        "(0,1)",
        "<-->",
        "(1,1)",
        ";",
        "(1,0)",
        "<-->",
        "(1,1)",
        ";",
        "(0,1)",
        "<-->",
        "(0,0)",
        ";",
    ]

    with pytest.raises(ValueError):
        coords_to_strings(
            coords, coord_to_strings_func=_coord_to_strings_UT, when_noncoord="except"
        )
