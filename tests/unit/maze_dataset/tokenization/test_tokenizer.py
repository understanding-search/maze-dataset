import itertools
import re
from collections import namedtuple
from itertools import product
from typing import Iterable

import numpy as np
from pytest import mark, param

from maze_dataset import (
    ConnectionArray,
    Coord,
    CoordArray,
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
    EdgeGroupings,
    EdgePermuters,
    EdgeSubsets,
    MazeTokenizer,
    MazeTokenizer2,
    PathTokenizers,
    PromptSequencers,
    StepSizes,
    StepTokenizers,
    TargetTokenizers,
    TokenizationMode,
    TokenizerElement,
)
from maze_dataset.util import (
    connection_list_to_adj_list,
    equal_except_adj_list_sequence,
)
from maze_dataset.utils import all_instances, flatten, manhattan_distance

NUM_TOKENIZERS_TO_TEST = 100
GRID_N = 5
N_MAZES = 5
CFG: MazeDatasetConfig = MazeDatasetConfig(
    name="test",
    grid_n=GRID_N,
    n_mazes=N_MAZES,
    maze_ctor=LatticeMazeGenerators.gen_dfs,
)
MAZE_DATASET: MazeDataset = MazeDataset.from_config(
    CFG,
    do_download=False,
    load_local=False,
    do_generate=True,
    save_local=False,
    verbose=True,
    gen_parallel=False,
)
LATTICE_MAZES: list[LatticeMaze] = [
    LatticeMazeGenerators.gen_dfs(np.array([GRID_N, GRID_N])) for _ in range(N_MAZES)
]
_PATHS = [maze.generate_random_path() for maze in LATTICE_MAZES]
TARGETED_MAZES: list[TargetedLatticeMaze] = [
    TargetedLatticeMaze.from_lattice_maze(maze, path[0], path[-1])
    for maze, path in zip(LATTICE_MAZES, _PATHS)
]
# MIXED_MAZES alternates the maze types, so you can slice a contiguous subset and still get all types
MIXED_MAZES: list[LatticeMaze | TargetedLatticeMaze | SolvedMaze] = [
    x
    for x in itertools.chain.from_iterable(
        itertools.zip_longest(MAZE_DATASET.mazes, TARGETED_MAZES, LATTICE_MAZES)
    )
]


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
    assert equal_except_adj_list_sequence(toks, toks_legacy)
    assert equal_except_adj_list_sequence(toks2, toks_legacy)


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
            (AdjListTokenizers.AdjListCoord(post=True), True),
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


def _helper_test_path_tokenizers(
    pt: PathTokenizers.PathTokenizer,
    maze: SolvedMaze,
    footprint_inds: Sequence[int],
):
    ct: CoordTokenizers.CoordTokenizer = CoordTokenizers.UT()
    path_toks: list[str] = pt.to_tokens(maze, ct)
    path_toks_set: set[str] = set(path_toks)
    footprint_inds: Int[np.ndarray, "footprint_index"] = np.array(footprint_inds)
    footprints: Int[np.ndarray, "footprint_index row_col=2"] = maze.solution[
        footprint_inds
    ]
    if StepTokenizers.Coord() in pt.step_tokenizers:
        non_steps: set[CoordTup] = set(tuple(c) for c in maze.solution) - set(
            tuple(c) for c in footprints
        )
        assert all([ct.to_tokens(coord)[0] in path_toks_set for coord in footprints])
        assert all([ct.to_tokens(coord)[0] not in path_toks_set for coord in non_steps])
    if StepTokenizers.Distance() in pt.step_tokenizers:
        distances: list[int] = footprint_inds[1:] - footprint_inds[:-1]
        assert (
            len(
                Counter(getattr(VOCAB, f"I_{d:03}") for d in distances)
                - Counter(path_toks)
            )
            == 0
        )
    # TODO: Uncomment tests when restoring full breadth of TokenizerElements
    # if StepTokenizers.Cardinal() in pt.step_tokenizers:
    #     c = Counter(path_toks)
    #     assert c[VOCAB.PATH_NORTH] + c[VOCAB.PATH_SOUTH] + c[VOCAB.PATH_EAST] + c[VOCAB.PATH_WEST] == len(footprint_inds)-1
    # if StepTokenizers.Relative() in pt.step_tokenizers:
    #     c = Counter(path_toks)
    #     assert c[VOCAB.PATH_LEFT] + c[VOCAB.PATH_RIGHT] + c[VOCAB.PATH_FORWARD] + c[VOCAB.PATH_BACKWARD] == len(footprint_inds)-1


@mark.parametrize(
    "pt,manual_maze",
    [
        param(tokenizer, maze_kv[1], id=f"{tokenizer.name}-{maze_kv[0]}")
        for maze_kv, tokenizer in itertools.product(
            _ASCII_MAZES.items(),
            random.sample(
                all_instances(
                    PathTokenizers.PathTokenizer,
                    frozendict.frozendict({TokenizerElement: lambda x: x.is_valid()}),
                ),
                min(
                    3, NUM_TOKENIZERS_TO_TEST
                ),  # TODO: Get rid of "3" when reinstantiating all `StepTokenizer` leaf classes
            ),
        )
    ],
)
def test_path_tokenizers(pt: PathTokenizers.PathTokenizer, manual_maze: _MANUAL_MAZE):
    solved_maze: SolvedMaze = SolvedMaze.from_ascii("\n".join(manual_maze.ascii))
    match type(pt.step_size):
        case StepSizes.Singles:
            footprint_inds = range(solved_maze.solution.shape[0])
        case StepSizes.Straightaways:
            swy_coordtup_set: set[CoordTup] = set(
                tuple(c) for c in manual_maze.straightaway_footprints
            )
            footprint_inds: list[int] = [
                i
                for i, c in enumerate(solved_maze.solution)
                if tuple(c) in swy_coordtup_set
            ]
        case StepSizes.Forks:
            footprint_inds = solved_maze.get_solution_forking_points(
                always_include_endpoints=True
            )[0]
        case StepSizes.ForksAndStraightaways:
            swy_step_inds: list[int] = StepSizes.Straightaways()._step_single_indices(
                solved_maze
            )
            footprint_inds: Int[np.ndarray, "footprint_index"] = np.concatenate(
                (
                    solved_maze.get_solution_forking_points(
                        always_include_endpoints=True
                    )[0],
                    swy_step_inds,
                )
            )
            footprint_inds, _ = np.unique(footprint_inds, axis=0, return_index=True)
    _helper_test_path_tokenizers(
        pt,
        solved_maze,
        footprint_inds,
    )


@mark.parametrize(
    "ep,maze",
    [
        param(tokenizer, maze, id=f"{tokenizer.name}-maze[{i}]")
        for (i, maze), tokenizer in itertools.product(
            enumerate(MIXED_MAZES[:6]),
            all_instances(
                EdgePermuters.EdgePermuter,
                frozendict.frozendict({TokenizerElement: lambda x: x.is_valid()}),
            ),
        )
    ],
)
def test_edge_permuters(ep: EdgePermuters.EdgePermuter, maze: LatticeMaze):
    edges: ConnectionArray = connection_list_to_adj_list(maze.connection_list)
    edges_copy = np.copy(edges)
    old_shape = edges.shape
    permuted: ConnectionArray = ep._permute(edges)
    match ep:
        case EdgePermuters.RandomCoord():
            assert permuted.shape == old_shape
            assert edges is permuted
            i = 0
            while np.array_equal(permuted, edges_copy) and i < 5:
                # Permute again in case for small mazes the random selection happened to not change anything
                permuted: ConnectionArray = ep._permute(permuted)
                i += 1
            assert not np.array_equal(permuted, edges_copy)
        case EdgePermuters.BothCoords():
            new_shape = old_shape[0] * 2, *old_shape[1:]
            n = old_shape[0]
            assert permuted.shape == new_shape
            assert np.array_equal(permuted[:n, ...], edges_copy)
            assert np.array_equal(permuted[:n, 0, :], permuted[n:, 1, :])
            assert np.array_equal(permuted[:n, 1, :], permuted[n:, 0, :])
            assert edges is not permuted


@mark.parametrize(
    "es,maze",
    [
        param(tokenizer, maze, id=f"{tokenizer.name}-maze[{i}]")
        for (i, maze), tokenizer in itertools.product(
            enumerate(MIXED_MAZES[:6]),
            all_instances(
                EdgeSubsets.EdgeSubset,
                frozendict.frozendict({TokenizerElement: lambda x: x.is_valid()}),
            ),
        )
    ],
)
def test_edge_subsets(es: EdgeSubsets.EdgeSubset, maze: LatticeMaze):
    edges: ConnectionArray = es._get_edges(maze)
    n: int = maze.grid_n
    match type(es):
        case EdgeSubsets.AllLatticeEdges:
            assert_shape: tuple = (4 * n * (n - 1), 2, 2)
        case EdgeSubsets.ConnectionEdges:
            if not es.walls:
                assert_shape: tuple = (np.count_nonzero(maze.connection_list), 2, 2)
            else:
                assert_shape: tuple = (
                    2 * n * (n - 1) - np.count_nonzero(maze.connection_list),
                    2,
                    2,
                )
    assert edges.dtype == np.int8
    assert assert_shape == tuple(edges.shape)
    assert assert_shape == tuple(
        np.unique(edges, axis=0).shape
    )  # All edges are unique (swapping leading/trailing coords is considered different)
    assert np.array_equal(
        manhattan_distance(edges), np.array([1] * assert_shape[0], dtype=np.int8)
    )


@mark.parametrize(
    "tok_elem,es,maze",
    [
        param(tok_elem, es, maze, id=f"{tok_elem.name}-{es.name}-maze[{i}]")
        for (i, maze), tok_elem, es in itertools.product(
            enumerate(MIXED_MAZES[:6]),
            all_instances(
                EdgeGroupings.EdgeGrouping,
                frozendict.frozendict(
                    {
                        TokenizerElement: lambda x: x.is_valid(),
                        # Add a condition to prune the range space that doesn't affect functionality being tested
                        EdgeGroupings.ByLeadingCoord: lambda x: x.intra
                        and x.connection_token_ordinal == 1,
                    }
                ),
            ),
            all_instances(
                EdgeSubsets.EdgeSubset,
                frozendict.frozendict({TokenizerElement: lambda x: x.is_valid()}),
            ),
        )
    ],
)
def test_edge_subsets(
    tok_elem: EdgeGroupings.EdgeGrouping, es: EdgeSubsets.EdgeSubset, maze: LatticeMaze
):
    edges: ConnectionArray = es._get_edges(maze)
    n: int = maze.grid_n
    groups: Sequence[ConnectionArray] = tok_elem._group_edges(edges)
    match type(tok_elem):
        case EdgeGroupings.Ungrouped:
            assert_shape = edges.shape[0], 1, 2, 2
            assert tuple(groups.shape) == assert_shape
        case EdgeGroupings.ByLeadingCoord:
            assert len(groups) == np.unique(edges[:, 0, :], axis=0).shape[0]
            assert sum(g.shape[0] for g in groups) == edges.shape[0]
            trailing_coords: list[CoordArray] = [g[:, 1, :] for g in groups]
            # vector_diffs is the position vector difference between the trailing coords of each group
            # These are stacked into a single array since we don't care about maintaining group separation
            vector_diffs: CoordArray = np.stack(
                list(flatten([np.diff(g[:, 1, :], axis=0) for g in groups], 1))
            )
            if tok_elem.shuffle_group:
                allowed_diffs = {(1, -1), (1, 1), (0, 2), (2, 0)}
                # The set of all 2D vectors between any 2 coords adjacent to a central coord
                allowed_diffs = allowed_diffs.union(
                    {(-d[0], -d[1]) for d in allowed_diffs}
                )
            else:
                # If vector_diffs are lexicographically sorted, these are the only possible values. Any other value indicates an error in sorting
                allowed_diffs = {(1, -1), (1, 1), (0, 2), (2, 0)}
            assert all(
                tuple(diff) in allowed_diffs for diff in np.unique(vector_diffs, axis=0)
            )
