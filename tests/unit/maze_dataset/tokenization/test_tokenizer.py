import itertools
import re
from collections import namedtuple
from itertools import product
from typing import Iterable

import numpy as np
from pytest import mark, param

from maze_dataset import (
    Coord,
    CoordTup,
    LatticeMaze,
    LatticeMazeGenerators,
    MazeDataset,
    MazeDatasetConfig,
    SolvedMaze,
    TargetedLatticeMaze,
)
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.plotting.print_tokens import color_maze_tokens_AOTP
from maze_dataset.tokenization import (
    AdjListTokenizers,
    CoordTokenizers,
    MazeTokenizer,
    MazeTokenizer2,
    PathTokenizers,
    PromptSequencers,
    StepTokenizers,
    TargetTokenizers,
    TokenizationMode,
    TokenizerElement,
)
from maze_dataset.util import equal_except_adj_list_sequence
from maze_dataset.testing_utils import (
    MIXED_MAZES,
    MAZE_DATASET,
)


def test_tokenizer():
    cfg: MazeDatasetConfig = MazeDatasetConfig(
        name="test",
        grid_n=5,
        n_mazes=3,
        maze_ctor=LatticeMazeGenerators.gen_dfs,
    )
    # to create a dataset, just call MazeDataset.from_config
    dataset: MazeDataset = MazeDataset.from_config(
        cfg,
        do_download=False,
        load_local=False,
        do_generate=True,
        save_local=False,
        verbose=True,
        gen_parallel=False,
    )

    for mode in (
        TokenizationMode.AOTP_UT_rasterized,
        TokenizationMode.AOTP_UT_uniform,
        TokenizationMode.AOTP_CTT_indexed,
    ):
        tokenizer: MazeTokenizer = MazeTokenizer(
            tokenization_mode=mode, max_grid_size=100
        )

        assert tokenizer.name == f"maze_tokenizer-{mode.name}-g{100}"

        if mode == TokenizationMode.AOTP_CTT_indexed:
            # TODO: fix these asserts
            assert tokenizer.node_strings_map is not None
            # assert len(tokenizer.node_strings_map) == 100  # `tokenizer.node_strings_map` is a `Kappa` which has no length
            assert 100 < tokenizer.vocab_size < 200
        elif mode in (
            TokenizationMode.AOTP_UT_rasterized,
            TokenizationMode.AOTP_UT_uniform,
        ):
            assert tokenizer.node_strings_map is None
            assert tokenizer.vocab_size > 10000

        assert isinstance(tokenizer.token_arr, Iterable)
        assert all(isinstance(token, str) for token in tokenizer.token_arr)
        assert len(tokenizer.token_arr) == tokenizer.vocab_size

        print(tokenizer.summary())

        for maze in dataset:
            # clear the cache here so we test if it works fine on the next loop
            tokenizer.clear_cache()

            maze_tok = maze.as_tokens(maze_tokenizer=tokenizer)

            maze_encoded = tokenizer.encode(maze_tok)
            maze_decoded = tokenizer.decode(maze_encoded)

            assert maze_tok == maze_decoded

            # you can view the tokens directly
            print("\nRaw tokens:\n")
            print(" ".join(maze_tok))

            maze_recovered = SolvedMaze.from_tokens(maze_tok, maze_tokenizer=tokenizer)

            assert (maze.connection_list == maze_recovered.connection_list).all()

            # or color and print them in various formats
            print("\nColored tokens, raw html:\n")
            print(color_maze_tokens_AOTP(maze_tok, fmt="html"))
            print("\nColored tokens, raw latex:\n")
            print(color_maze_tokens_AOTP(maze_tok, fmt="latex"))
            print("\nColored tokens, terminal:\n")
            print(color_maze_tokens_AOTP(maze_tok, fmt="terminal"))


_MANUAL_MAZE = namedtuple(
    "_MANUAL_MAZE", ["tokens", "ascii", "straightaway_footprints"]
)
_ASCII_MAZES: dict[str, tuple[str, list[str]]] = dict(
    small_3x3=_MANUAL_MAZE(
        tokens="<ADJLIST_START> (2,0) <--> (2,1) ; (0,0) <--> (0,1) ; (0,0) <--> (1,0) ; (0,2) <--> (1,2) ; (1,0) <--> (2,0) ; (0,2) <--> (0,1) ; (2,2) <--> (2,1) ; (1,1) <--> (2,1) ; <ADJLIST_END> <ORIGIN_START> (0,0) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (0,0) (1,0) (2,0) (2,1) <PATH_END>",
        ascii=[
            "#######",
            "#S    #",
            "#X### #",
            "#X# # #",
            "#X# ###",
            "#XXE  #",
            "#######",
        ],
        straightaway_footprints=np.array(
            [
                [0, 0],
                [2, 0],
                [2, 1],
            ]
        ),
    ),
    big_10x10=_MANUAL_MAZE(
        tokens="<ADJLIST_START> (8,2) <--> (8,3) ; (3,7) <--> (3,6) ; (6,7) <--> (6,8) ; (4,6) <--> (5,6) ; (9,5) <--> (9,4) ; (3,3) <--> (3,4) ; (5,1) <--> (4,1) ; (2,6) <--> (2,7) ; (8,5) <--> (8,4) ; (1,9) <--> (2,9) ; (4,1) <--> (4,2) ; (0,8) <--> (0,7) ; (5,4) <--> (5,3) ; (6,3) <--> (6,4) ; (5,0) <--> (4,0) ; (5,3) <--> (5,2) ; (3,1) <--> (2,1) ; (9,1) <--> (9,0) ; (3,5) <--> (3,6) ; (5,5) <--> (6,5) ; (7,1) <--> (7,2) ; (0,1) <--> (1,1) ; (7,8) <--> (8,8) ; (3,9) <--> (4,9) ; (4,6) <--> (4,7) ; (0,6) <--> (0,7) ; (3,4) <--> (3,5) ; (6,0) <--> (5,0) ; (7,7) <--> (7,6) ; (1,6) <--> (0,6) ; (6,1) <--> (6,0) ; (8,6) <--> (8,7) ; (9,9) <--> (9,8) ; (1,8) <--> (1,9) ; (2,1) <--> (2,2) ; (9,2) <--> (9,3) ; (5,9) <--> (6,9) ; (3,2) <--> (2,2) ; (0,8) <--> (0,9) ; (5,6) <--> (5,7) ; (2,3) <--> (2,4) ; (4,5) <--> (4,4) ; (8,9) <--> (8,8) ; (9,6) <--> (8,6) ; (3,7) <--> (3,8) ; (8,0) <--> (7,0) ; (6,1) <--> (6,2) ; (0,1) <--> (0,0) ; (7,3) <--> (7,4) ; (9,4) <--> (9,3) ; (9,6) <--> (9,5) ; (8,7) <--> (7,7) ; (5,2) <--> (5,1) ; (0,0) <--> (1,0) ; (7,2) <--> (7,3) ; (2,5) <--> (2,6) ; (4,9) <--> (5,9) ; (5,5) <--> (5,4) ; (5,6) <--> (6,6) ; (7,8) <--> (7,9) ; (1,7) <--> (2,7) ; (4,6) <--> (4,5) ; (1,1) <--> (1,2) ; (3,1) <--> (3,0) ; (1,5) <--> (1,6) ; (8,3) <--> (8,4) ; (9,9) <--> (8,9) ; (8,5) <--> (7,5) ; (1,4) <--> (2,4) ; (3,0) <--> (4,0) ; (3,3) <--> (4,3) ; (6,9) <--> (6,8) ; (1,0) <--> (2,0) ; (6,0) <--> (7,0) ; (8,0) <--> (9,0) ; (2,3) <--> (2,2) ; (2,8) <--> (3,8) ; (5,7) <--> (6,7) ; (1,3) <--> (0,3) ; (9,7) <--> (9,8) ; (7,5) <--> (7,4) ; (1,8) <--> (2,8) ; (6,5) <--> (6,4) ; (0,2) <--> (1,2) ; (0,7) <--> (1,7) ; (0,3) <--> (0,2) ; (4,3) <--> (4,2) ; (5,8) <--> (4,8) ; (9,1) <--> (8,1) ; (9,2) <--> (8,2) ; (1,3) <--> (1,4) ; (2,9) <--> (3,9) ; (4,8) <--> (4,7) ; (0,5) <--> (0,4) ; (8,1) <--> (7,1) ; (0,3) <--> (0,4) ; (9,7) <--> (9,6) ; (7,6) <--> (6,6) ; (1,5) <--> (0,5) ; <ADJLIST_END> <ORIGIN_START> (6,2) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (6,2) (6,1) (6,0) (5,0) (4,0) (3,0) (3,1) (2,1) <PATH_END>",
        ascii=[
            "#####################",
            "#   #       #       #",
            "# # # # ### # # #####",
            "# #   #   #   # #   #",
            "# ####### ##### # # #",
            "# #E      #     # # #",
            "###X# ########### # #",
            "#XXX# #           # #",
            "#X##### ########### #",
            "#X#     #         # #",
            "#X# ######### ### # #",
            "#X#         #   # # #",
            "#X######### # # ### #",
            "#XXXXS#     # #     #",
            "# ########### #######",
            "# #         #   #   #",
            "# # ####### ### # ###",
            "# # #       #   #   #",
            "# # # ####### ##### #",
            "#   #               #",
            "#####################",
        ],
        straightaway_footprints=np.array(
            [
                [6, 2],
                [6, 0],
                [3, 0],
                [3, 1],
                [2, 1],
            ]
        ),
    ),
    longer_10x10=_MANUAL_MAZE(
        tokens="<ADJLIST_START> (8,2) <--> (8,3) ; (3,7) <--> (3,6) ; (6,7) <--> (6,8) ; (4,6) <--> (5,6) ; (9,5) <--> (9,4) ; (3,3) <--> (3,4) ; (5,1) <--> (4,1) ; (2,6) <--> (2,7) ; (8,5) <--> (8,4) ; (1,9) <--> (2,9) ; (4,1) <--> (4,2) ; (0,8) <--> (0,7) ; (5,4) <--> (5,3) ; (6,3) <--> (6,4) ; (5,0) <--> (4,0) ; (5,3) <--> (5,2) ; (3,1) <--> (2,1) ; (9,1) <--> (9,0) ; (3,5) <--> (3,6) ; (5,5) <--> (6,5) ; (7,1) <--> (7,2) ; (0,1) <--> (1,1) ; (7,8) <--> (8,8) ; (3,9) <--> (4,9) ; (4,6) <--> (4,7) ; (0,6) <--> (0,7) ; (3,4) <--> (3,5) ; (6,0) <--> (5,0) ; (7,7) <--> (7,6) ; (1,6) <--> (0,6) ; (6,1) <--> (6,0) ; (8,6) <--> (8,7) ; (9,9) <--> (9,8) ; (1,8) <--> (1,9) ; (2,1) <--> (2,2) ; (9,2) <--> (9,3) ; (5,9) <--> (6,9) ; (3,2) <--> (2,2) ; (0,8) <--> (0,9) ; (5,6) <--> (5,7) ; (2,3) <--> (2,4) ; (4,5) <--> (4,4) ; (8,9) <--> (8,8) ; (9,6) <--> (8,6) ; (3,7) <--> (3,8) ; (8,0) <--> (7,0) ; (6,1) <--> (6,2) ; (0,1) <--> (0,0) ; (7,3) <--> (7,4) ; (9,4) <--> (9,3) ; (9,6) <--> (9,5) ; (8,7) <--> (7,7) ; (5,2) <--> (5,1) ; (0,0) <--> (1,0) ; (7,2) <--> (7,3) ; (2,5) <--> (2,6) ; (4,9) <--> (5,9) ; (5,5) <--> (5,4) ; (5,6) <--> (6,6) ; (7,8) <--> (7,9) ; (1,7) <--> (2,7) ; (4,6) <--> (4,5) ; (1,1) <--> (1,2) ; (3,1) <--> (3,0) ; (1,5) <--> (1,6) ; (8,3) <--> (8,4) ; (9,9) <--> (8,9) ; (8,5) <--> (7,5) ; (1,4) <--> (2,4) ; (3,0) <--> (4,0) ; (3,3) <--> (4,3) ; (6,9) <--> (6,8) ; (1,0) <--> (2,0) ; (6,0) <--> (7,0) ; (8,0) <--> (9,0) ; (2,3) <--> (2,2) ; (2,8) <--> (3,8) ; (5,7) <--> (6,7) ; (1,3) <--> (0,3) ; (9,7) <--> (9,8) ; (7,5) <--> (7,4) ; (1,8) <--> (2,8) ; (6,5) <--> (6,4) ; (0,2) <--> (1,2) ; (0,7) <--> (1,7) ; (0,3) <--> (0,2) ; (4,3) <--> (4,2) ; (5,8) <--> (4,8) ; (9,1) <--> (8,1) ; (9,2) <--> (8,2) ; (1,3) <--> (1,4) ; (2,9) <--> (3,9) ; (4,8) <--> (4,7) ; (0,5) <--> (0,4) ; (8,1) <--> (7,1) ; (0,3) <--> (0,4) ; (9,7) <--> (9,6) ; (7,6) <--> (6,6) ; (1,5) <--> (0,5) ; <ADJLIST_END> <ORIGIN_START> (6,2) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (6,2) (6,1) (6,0) (5,0) (4,0) (3,0) (3,1) (2,1) (2,2) (2,3) (2,4) (1,4) (1,3) (0,3) (0,4) (0,5) (1,5) (1,6) (0,6) (0,7) (0,8) <PATH_END>",
        ascii=[
            "#####################",
            "#   #  XXXXX#XXXXE  #",
            "# # # #X###X#X# #####",
            "# #   #XXX#XXX# #   #",
            "# #######X##### # # #",
            "# #XXXXXXX#     # # #",
            "###X# ########### # #",
            "#XXX# #           # #",
            "#X##### ########### #",
            "#X#     #         # #",
            "#X# ######### ### # #",
            "#X#         #   # # #",
            "#X######### # # ### #",
            "#XXXXS#     # #     #",
            "# ########### #######",
            "# #         #   #   #",
            "# # ####### ### # ###",
            "# # #       #   #   #",
            "# # # ####### ##### #",
            "#   #               #",
            "#####################",
        ],
        straightaway_footprints=np.array(
            [
                [6, 2],
                [6, 0],
                [3, 0],
                [3, 1],
                [2, 1],
                [2, 4],
                [1, 4],
                [1, 3],
                [0, 3],
                [0, 5],
                [1, 5],
                [1, 6],
                [0, 6],
                [0, 8],
            ]
        ),
    ),
)


@mark.parametrize(
    "maze_ascii, tok_mode, tokens",
    [
        param(
            _ASCII_MAZES[maze_ascii_key][1],  # maze_ascii
            tok_mode,  # tok_mode
            _ASCII_MAZES[maze_ascii_key][0],  # tokens
            id=f"{tok_mode.name}_{maze_ascii_key}",
        )
        for maze_ascii_key, tok_mode in product(
            ["small_3x3", "big_10x10"],
            [
                TokenizationMode.AOTP_UT_uniform,
                TokenizationMode.AOTP_UT_rasterized,
                TokenizationMode.AOTP_CTT_indexed,
            ],
        )
    ],
)
def test_maze_to_tokens_roundtrip(
    maze_ascii: list[str],
    tok_mode: TokenizationMode,
    tokens: str,
):
    if tok_mode == TokenizationMode.AOTP_CTT_indexed:
        # The hardcoded `tokens` assumes a UT tokenizer.
        # Here we modify `tokens` to match what a `AOTP_CTT_indexed` tokenizer would produce.
        tokens = re.sub(r"\(([0-9]),([0-9])\)", r"(\1 , \2)", tokens)
        tokens = re.sub(r"\(([0-9]+ ,)", r"( \1", tokens)
        tokens = re.sub(r"(, [0-9]+)\)", r"\1 )", tokens)
    tokens_original_split: list[str] = tokens.split()

    # join into a single string, and get a maze out
    ascii_str: str = "\n".join(maze_ascii)
    maze: SolvedMaze = SolvedMaze.from_ascii(ascii_str)
    # init tokenizer
    tokenizer: MazeTokenizer = MazeTokenizer(tokenization_mode=tok_mode)

    # maze as tokens
    tokens_from_maze: list[str] = maze.as_tokens(tokenizer)

    # maze round trip
    maze_roundtrip: SolvedMaze = SolvedMaze.from_tokens(tokens_from_maze, tokenizer)
    tokens_roundtrip: list[str] = maze_roundtrip.as_tokens(tokenizer)

    # check that the mazes and tokens are all equivalent
    assert maze == maze_roundtrip
    assert equal_except_adj_list_sequence(tokens_original_split, tokens_from_maze)
    assert equal_except_adj_list_sequence(tokens_original_split, tokens_roundtrip)


@mark.parametrize(
    "tok_mode, max_grid_size, result",
    [
        param(
            tok_mode,
            max_grid_size,
            MazeTokenizer(tokenization_mode=tok_mode, max_grid_size=max_grid_size),
            id=f"{tok_mode}-{max_grid_size}",
        )
        for tok_mode, max_grid_size in [
            (TokenizationMode.AOTP_CTT_indexed, None),
            (TokenizationMode.AOTP_UT_rasterized, None),
            (TokenizationMode.AOTP_UT_uniform, None),
            (TokenizationMode.AOTP_CTT_indexed, 5),
        ]
    ],
)
def test_to_legacy_tokenizer(
    tok_mode: TokenizationMode, max_grid_size: int | None, result: MazeTokenizer
):
    assert tok_mode.to_legacy_tokenizer(max_grid_size) == result


# MazeTokenizer2 tests
# =====================

# Backwards compatibility tests
# =============================


@mark.parametrize(
    "maze,legacy_tokenizer",
    [
        param(maze[0], tok_spec, id=f"{tok_spec.value}-maze{maze[1]}")
        for maze, tok_spec in itertools.product(
            [(maze, i) for i, maze in enumerate(MIXED_MAZES)],
            [tok_mode for tok_mode in TokenizationMode],
        )
    ],
)
def test_to_tokens_backwards_compatible(
    maze: SolvedMaze, legacy_tokenizer: TokenizationMode
):
    tokenizer: MazeTokenizer2 = MazeTokenizer2.from_legacy(legacy_tokenizer)
    toks: list[str] = maze.as_tokens(tokenizer)
    toks2: list[str] = tokenizer.to_tokens(maze)
    toks_legacy: list[str] = maze.as_tokens(legacy_tokenizer)

    try:
        assert equal_except_adj_list_sequence(toks, toks_legacy)
        assert equal_except_adj_list_sequence(toks2, toks_legacy)
    except AssertionError as e:
        raise AssertionError(
            "Tokens from `as_tokens` and `to_tokens` should be equal to tokens from `as_tokens` with the legacy tokenizer.\n"
            f"{len(toks) = }, {len(toks2) = }, {len(toks_legacy) = }\n"
            f"{toks = }\n{toks2 = }\n{toks_legacy = }",
        ) from e


@mark.parametrize(
    "coords, legacy_tok_mode",
    [
        param(
            coords,
            tok_mode,
            id=f"{tok_mode.value}-coords(type={type(coords[0])},len={len(coords)})",
        )
        for tok_mode, coords in itertools.product(
            [tok_mode for tok_mode in TokenizationMode],
            [
                *[[maze.start_pos] for maze in MAZE_DATASET.mazes[:2]],
                [maze.start_pos for maze in MAZE_DATASET.mazes],
                *[[tuple(maze.start_pos)] for maze in MAZE_DATASET.mazes[:2]],
                [tuple(maze.start_pos) for maze in MAZE_DATASET.mazes],
            ],
        )
    ],
)
def test_coords_to_strings_backwards_compatible(
    coords: list[Coord, CoordTup], legacy_tok_mode: TokenizationMode
):
    tokenizer: MazeTokenizer2 = MazeTokenizer2.from_legacy(legacy_tok_mode)
    legacy_tokenizer = MazeTokenizer(tokenization_mode=legacy_tok_mode)
    strings: list[str] = tokenizer.coords_to_strings(coords)
    strings_legacy: list[str] = legacy_tokenizer.coords_to_strings(coords)
    assert strings == strings_legacy


@mark.parametrize(
    "maze,tok_mode",
    [
        param(maze[0], tok_spec, id=f"{tok_spec.value}-maze{maze[1]}")
        for maze, tok_spec in itertools.product(
            [(maze, i) for i, maze in enumerate(MIXED_MAZES)],
            [tok_mode for tok_mode in TokenizationMode],
        )
    ],
)
def test_from_tokens_backwards_compatible(
    maze: LatticeMaze, tok_mode: TokenizationMode
):
    tokenizer = MazeTokenizer2.from_legacy(tok_mode)
    toks = maze.as_tokens(tok_mode)
    # Equality test of `as_tokens` output done in a separate unit test
    maze_legacy: LatticeMaze = LatticeMaze.from_tokens(toks, tok_mode)
    maze: LatticeMaze = LatticeMaze.from_tokens(toks, tokenizer)
    assert maze == maze_legacy


# General functionality tests
# ===========================


@mark.parametrize(
    "el, result",
    [
        param(elem, result, id=elem.name)
        for elem, result in [
            (CoordTokenizers.CTT(), True),
            (CoordTokenizers.CTT(intra=True), True),
            (CoordTokenizers.UT(), True),
            (AdjListTokenizers.AdjListCoord(), True),
            # (AdjListTokenizers.AdjListCoord(post=True), True), # TODO: this breaks collecting tests
            (TargetTokenizers.Unlabeled(post=True), True),
            (PathTokenizers.StepSequence(), True),
            (
                PathTokenizers.StepSequence(step_tokenizers=(StepTokenizers.Coord(),)),
                True,
            ),
            (
                PathTokenizers.StepSequence(
                    step_tokenizers=(
                        StepTokenizers.Coord(),
                        StepTokenizers.Coord(),
                    )
                ),
                False,
            ),
            (PromptSequencers.AOP(), True),
            (PromptSequencers.AOP(path_tokenizer=PathTokenizers.StepSequence()), True),
            (
                PromptSequencers.AOP(
                    path_tokenizer=PathTokenizers.StepSequence(
                        step_tokenizers=(StepTokenizers.Coord(),)
                    )
                ),
                True,
            ),
            (
                PromptSequencers.AOP(
                    path_tokenizer=PathTokenizers.StepSequence(
                        step_tokenizers=(
                            StepTokenizers.Coord(),
                            StepTokenizers.Coord(),
                        )
                    )
                ),
                True,
            ),
        ]
    ],
)
def test_tokenizer_element_is_valid(el: TokenizerElement, result: bool):
    assert el.is_valid() == result


@mark.parametrize(
    "tokenizer, result",
    [
        param(tokenizer, result, id=str(tokenizer))
        for tokenizer, result in [
            (MazeTokenizer2(), True),
            (MazeTokenizer2.from_legacy(TokenizationMode.AOTP_CTT_indexed), True),
            (MazeTokenizer2(prompt_sequencer=PromptSequencers.AOP()), False),
        ]
    ],
)
def test_is_legacy_equivalent(tokenizer: MazeTokenizer2, result: bool):
    assert tokenizer.is_legacy_equivalent() == result
