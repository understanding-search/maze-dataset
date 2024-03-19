import pytest
from pytest import mark, param

from maze_dataset.dataset.maze_dataset import MazeDatasetConfig
from maze_dataset.tokenization.token_utils import (
    get_adj_list_tokens,
    get_origin_tokens,
    get_path_tokens,
    get_target_tokens,
    get_tokens_up_to_path_start,
    tokens_between,
)
from maze_dataset.tokenization.util import (
    _coord_to_strings_UT,
    coords_to_strings,
    strings_to_coords,
)

MAZE_TOKENS = (
    "<ADJLIST_START> (0,1) <--> (1,1) ; (1,0) <--> (1,1) ; (0,1) <--> (0,0) ; <ADJLIST_END> <ORIGIN_START> (1,0) <ORIGIN_END> <TARGET_START> (1,1) <TARGET_END> <PATH_START> (1,0) (1,1) <PATH_END>".split(),
    "AOTP_UT",
)
# setattr(MAZE_TOKENS, "name", 'AOTP_UT')
MAZE_TOKENS_AOTP_CTT_indexed = (
    "<ADJLIST_START> ( 0 , 1 ) <--> ( 1 , 1 ) ; ( 1 , 0 ) <--> ( 1 , 1 ) ; ( 0 , 1 ) <--> ( 0 , 0 ) ; <ADJLIST_END> <ORIGIN_START> ( 1 , 0 ) <ORIGIN_END> <TARGET_START> ( 1 , 1 ) <TARGET_END> <PATH_START> ( 1 , 0 ) ( 1 , 1 ) <PATH_END>".split(),
    "AOTP_CTT_indexed",
)
TEST_TOKEN_LISTS = [MAZE_TOKENS, MAZE_TOKENS_AOTP_CTT_indexed]


@mark.parametrize(
    "toks, tokenizer_name",
    [
        param(
            token_list[0],
            token_list[1],
            id=f"{token_list[1]}",
        )
        for token_list in TEST_TOKEN_LISTS
    ],
)
def test_tokens_between(toks: list[str], tokenizer_name: str):
    result = tokens_between(toks, "<PATH_START>", "<PATH_END>")
    match tokenizer_name:
        case "AOTP_UT":
            assert result == ["(1,0)", "(1,1)"]
        case "AOTP_CTT_indexed":
            assert result == ["(", "1", ",", "0", ")", "(", "1", ",", "1", ")"]

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


@mark.parametrize(
    "toks, tokenizer_name",
    [
        param(
            token_list[0],
            token_list[1],
            id=f"{token_list[1]}",
        )
        for token_list in TEST_TOKEN_LISTS
    ],
)
def test_tokens_between_out_of_order(toks: list[str], tokenizer_name: str):
    with pytest.raises(AssertionError):
        tokens_between(toks, "<PATH_END>", "<PATH_START>")


@mark.parametrize(
    "toks, tokenizer_name",
    [
        param(
            token_list[0],
            token_list[1],
            id=f"{token_list[1]}",
        )
        for token_list in TEST_TOKEN_LISTS
    ],
)
def test_get_adj_list_tokens(toks: list[str], tokenizer_name: str):
    result = get_adj_list_tokens(toks)
    match tokenizer_name:
        case "AOTP_UT":
            expected = (
                "(0,1) <--> (1,1) ; (1,0) <--> (1,1) ; (0,1) <--> (0,0) ;".split()
            )
        case "AOTP_CTT_indexed":
            expected = "( 0 , 1 ) <--> ( 1 , 1 ) ; ( 1 , 0 ) <--> ( 1 , 1 ) ; ( 0 , 1 ) <--> ( 0 , 0 ) ;".split()
    assert result == expected


@mark.parametrize(
    "toks, tokenizer_name",
    [
        param(
            token_list[0],
            token_list[1],
            id=f"{token_list[1]}",
        )
        for token_list in TEST_TOKEN_LISTS
    ],
)
def test_get_path_tokens(toks: list[str], tokenizer_name: str):
    result_notrim = get_path_tokens(toks)
    result_trim = get_path_tokens(toks, trim_end=True)
    match tokenizer_name:
        case "AOTP_UT":
            assert result_notrim == ["<PATH_START>", "(1,0)", "(1,1)", "<PATH_END>"]
            assert result_trim == ["(1,0)", "(1,1)"]
        case "AOTP_CTT_indexed":
            assert (
                result_notrim == "<PATH_START> ( 1 , 0 ) ( 1 , 1 ) <PATH_END>".split()
            )
            assert result_trim == "( 1 , 0 ) ( 1 , 1 )".split()


@mark.parametrize(
    "toks, tokenizer_name",
    [
        param(
            token_list[0],
            token_list[1],
            id=f"{token_list[1]}",
        )
        for token_list in TEST_TOKEN_LISTS
    ],
)
def test_get_origin_tokens(toks: list[str], tokenizer_name: str):
    result = get_origin_tokens(toks)
    match tokenizer_name:
        case "AOTP_UT":
            assert result == ["(1,0)"]
        case "AOTP_CTT_indexed":
            assert result == "( 1 , 0 )".split()


@mark.parametrize(
    "toks, tokenizer_name",
    [
        param(
            token_list[0],
            token_list[1],
            id=f"{token_list[1]}",
        )
        for token_list in TEST_TOKEN_LISTS
    ],
)
def test_get_target_tokens(toks: list[str], tokenizer_name: str):
    result = get_target_tokens(toks)
    match tokenizer_name:
        case "AOTP_UT":
            assert result == ["(1,1)"]
        case "AOTP_CTT_indexed":
            assert result == "( 1 , 1 )".split()


@mark.parametrize(
    "toks, tokenizer_name",
    [
        param(
            token_list[0],
            token_list[1],
            id=f"{token_list[1]}",
        )
        for token_list in [MAZE_TOKENS]
    ],
)
def test_get_tokens_up_to_path_start_including_start(
    toks: list[str], tokenizer_name: str
):
    # Dont test on `MAZE_TOKENS_AOTP_CTT_indexed` because this function doesn't support `AOTP_CTT_indexed` when `include_start_coord=True`.
    result = get_tokens_up_to_path_start(toks, include_start_coord=True)
    match tokenizer_name:
        case "AOTP_UT":
            expected = "<ADJLIST_START> (0,1) <--> (1,1) ; (1,0) <--> (1,1) ; (0,1) <--> (0,0) ; <ADJLIST_END> <ORIGIN_START> (1,0) <ORIGIN_END> <TARGET_START> (1,1) <TARGET_END> <PATH_START> (1,0)".split()
        case "AOTP_CTT_indexed":
            expected = "<ADJLIST_START> ( 0 , 1 ) <--> ( 1 , 1 ) ; ( 1 , 0 ) <--> ( 1 , 1 ) ; ( 0 , 1 ) <--> ( 0 , 0 ) ; <ADJLIST_END> <ORIGIN_START> ( 1 , 0 ) <ORIGIN_END> <TARGET_START> ( 1 , 1 ) <TARGET_END> <PATH_START> ( 1 , 0 )".split()
    assert result == expected


@mark.parametrize(
    "toks, tokenizer_name",
    [
        param(
            token_list[0],
            token_list[1],
            id=f"{token_list[1]}",
        )
        for token_list in TEST_TOKEN_LISTS
    ],
)
def test_get_tokens_up_to_path_start_excluding_start(
    toks: list[str], tokenizer_name: str
):
    result = get_tokens_up_to_path_start(toks, include_start_coord=False)
    match tokenizer_name:
        case "AOTP_UT":
            expected = "<ADJLIST_START> (0,1) <--> (1,1) ; (1,0) <--> (1,1) ; (0,1) <--> (0,0) ; <ADJLIST_END> <ORIGIN_START> (1,0) <ORIGIN_END> <TARGET_START> (1,1) <TARGET_END> <PATH_START>".split()
        case "AOTP_CTT_indexed":
            expected = "<ADJLIST_START> ( 0 , 1 ) <--> ( 1 , 1 ) ; ( 1 , 0 ) <--> ( 1 , 1 ) ; ( 0 , 1 ) <--> ( 0 , 0 ) ; <ADJLIST_END> <ORIGIN_START> ( 1 , 0 ) <ORIGIN_END> <TARGET_START> ( 1 , 1 ) <TARGET_END> <PATH_START>".split()
    assert result == expected


@mark.parametrize(
    "toks, tokenizer_name",
    [
        param(
            token_list[0],
            token_list[1],
            id=f"{token_list[1]}",
        )
        for token_list in TEST_TOKEN_LISTS
    ],
)
def test_strings_to_coords(toks: list[str], tokenizer_name: str):
    adj_list = get_adj_list_tokens(toks)
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
        strings_to_coords(adj_list, when_noncoord="error")

    assert strings_to_coords("(1,2) <ADJLIST_START> (5,6)") == [(1, 2), (5, 6)]
    assert strings_to_coords("(1,2) <ADJLIST_START> (5,6)", when_noncoord="skip") == [
        (1, 2),
        (5, 6),
    ]
    assert strings_to_coords(
        "(1,2) <ADJLIST_START> (5,6)", when_noncoord="include"
    ) == [(1, 2), "<ADJLIST_START>", (5, 6)]
    with pytest.raises(ValueError):
        strings_to_coords("(1,2) <ADJLIST_START> (5,6)", when_noncoord="error")


@mark.parametrize(
    "toks, tokenizer_name",
    [
        param(
            token_list[0],
            token_list[1],
            id=f"{token_list[1]}",
        )
        for token_list in TEST_TOKEN_LISTS
    ],
)
def test_coords_to_strings(toks: list[str], tokenizer_name: str):
    adj_list = get_adj_list_tokens(toks)
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
            coords, coord_to_strings_func=_coord_to_strings_UT, when_noncoord="error"
        )
