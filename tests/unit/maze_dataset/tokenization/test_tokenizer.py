import itertools
import os
import re
from collections import Counter, namedtuple
from itertools import product
from typing import Iterable, Callable, Hashable
import frozendict
import random
from jaxtyping import Int

import numpy as np
import pytest
from pytest import mark, param
from zanj import ZANJ

from maze_dataset import (
    VOCAB,
    VOCAB_LIST,
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
    ALL_TOKENIZER_HASHES,
    MazeTokenizer,
    MazeTokenizer2,
    TokenizerElement,
    CoordTokenizers,
    PromptSequencers,
    AdjListTokenizers,
    StepSizes,
    StepTokenizers,
    PathTokenizers,
    TargetTokenizers,
    TokenizationMode,
)
from maze_dataset.utils import all_instances
from maze_dataset.tokenization.maze_tokenizer import _load_tokenizer_hashes
from maze_dataset.util import equal_except_adj_list_sequence
from maze_dataset.token_utils import get_path_tokens
from maze_dataset.tokenization.all_tokenizers import (
    sample_tokenizers_for_test, 
    EVERY_TEST_TOKENIZERS,
    ALL_TOKENIZERS,
    _get_all_tokenizers,
    save_hashes,
)

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


_MANUAL_MAZE = namedtuple("_MANUAL_MAZE", ["tokens", "ascii", "straightaway_steps"])
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
        straightaway_steps=np.array([[0,0],[2,0],[2,1],]),
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
        straightaway_steps=np.array([[6,2],[6,0],[3,0],[3,1],[2,1],]),
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
        straightaway_steps=np.array([[6,2],[6,0],[3,0],[3,1],[2,1],[2,4],[1,4],[1,3],[0,3],[0,5],[1,5],[1,6],[0,6],[0,8],]),
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
            id=f"{tok_mode}-{max_grid_size}")
        for tok_mode, max_grid_size in 
        [
            (TokenizationMode.AOTP_CTT_indexed, None),
            (TokenizationMode.AOTP_UT_rasterized, None),
            (TokenizationMode.AOTP_UT_uniform, None),
            (TokenizationMode.AOTP_CTT_indexed, 5),            
        ]
    ],
)
def test_to_legacy_tokenizer(tok_mode: TokenizationMode, max_grid_size: int | None, result: MazeTokenizer):
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
            [(maze, i) for i, maze in enumerate(MIXED_MAZES[:6])],
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
            [(maze, i) for i, maze in enumerate(MIXED_MAZES[:6])],
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

def test_all_tokenizers():
    assert len(ALL_TOKENIZERS) > 400
    assert len(_get_all_tokenizers()) == len(ALL_TOKENIZERS)
    assert len({hash(mt) for mt in ALL_TOKENIZERS}) == len(ALL_TOKENIZERS)


@mark.parametrize(
    "class_",
    [
        param(c, id=c.__name__)
        for c in TokenizerElement.__subclasses__()
    ],
)
def test_all_instances_tokenizerelement(class_: type):
    all_vals = all_instances(
        class_, 
        validation_funcs=frozendict.frozendict({
            TokenizerElement: lambda x: x.is_valid(),
        })
    )
    assert len({hash(elem) for elem in all_vals}) == len(all_vals)
    


sample_min: int = len(EVERY_TEST_TOKENIZERS)

@mark.parametrize(
    "n, result",
    [
        param(i, result)
        for i, result in [
            (sample_min-1, ValueError),
            (sample_min, None),
            (sample_min+5, None),
            (sample_min+200, None),
            ]
    ],
)
def test_sample_tokenizers_for_test(n: int, result: type[Exception] | None):
    if isinstance(result, type) and issubclass(result, Exception):
        with pytest.raises(result):
            sample_tokenizers_for_test(n)
        return
    mts: list[MazeTokenizer2] = sample_tokenizers_for_test(n)
    mts_set: set[MazeTokenizer2] = set(mts)
    assert len(mts) == len(mts_set)
    assert set(EVERY_TEST_TOKENIZERS).issubset(mts_set)
    if n > sample_min + 1:
        mts2: list[MazeTokenizer2] = sample_tokenizers_for_test(n)
        assert set(mts2) != mts_set  # Check that succesive samples are different
        

@mark.parametrize(
    "maze,tokenizer",
    [
        param(maze[0], tokenizer, id=f"{type(maze[0])}{maze[1]}-{tokenizer.name}")
        for maze, tokenizer in itertools.product(
            [(maze, i) for i, maze in enumerate(MIXED_MAZES[:6])], sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST)
        )
    ],
)
def test_token_region_delimiters(maze: LatticeMaze, tokenizer: MazeTokenizer2):
    """<PATH_START> and similar token region delimiters should appear at most 1 time, regardless of tokenizer."""
    counts: Counter = Counter(maze.as_tokens(tokenizer))
    assert all([counts[tok] < 2 for tok in VOCAB_LIST[:8]])


@mark.parametrize(
    "tokenizer", [param(tokenizer, id=tokenizer.name) for tokenizer in ALL_TOKENIZERS]
)
def test_tokenizer_properties(tokenizer: MazeTokenizer2):
    # Just make sure the call doesn't raise exception
    assert len(tokenizer.name) > 5

    assert tokenizer.vocab_size == 4096
    assert isinstance(tokenizer.token_arr, Iterable)
    assert all(isinstance(token, str) for token in tokenizer.token_arr)
    assert tokenizer.token_arr[tokenizer.padding_token_index] == VOCAB.PADDING

    # Just make sure the call doesn't raise exception
    print(tokenizer.summary())


@mark.parametrize(
    "maze,tokenizer",
    [
        param(maze[0], tokenizer, id=f"{tokenizer.name}-maze{maze[1]}")
        for maze, tokenizer in itertools.product(
            [(maze, i) for i, maze in enumerate(MIXED_MAZES[:6])], sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST)
        )
    ],
)
def test_encode_decode(maze: LatticeMaze, tokenizer: MazeTokenizer2):
    maze_tok: list[str] = maze.as_tokens(maze_tokenizer=tokenizer)
    maze_encoded: list[int] = tokenizer.encode(maze_tok)
    maze_decoded: LatticeMaze = tokenizer.decode(maze_encoded)
    assert maze_tok == maze_decoded


@mark.parametrize(
    "tokenizer", [param(tokenizer, id=tokenizer.name) for tokenizer in sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST)]
)
def test_zanj_save_read(tokenizer: MazeTokenizer2):
    path = os.path.abspath(
        os.path.join(
            os.path.curdir, "data", "MazeTokenizer2_" + hex(hash(tokenizer)) + ".zanj"
        )
    )
    zanj = ZANJ()
    zanj.save(tokenizer, path)
    assert zanj.read(path) == tokenizer


@mark.parametrize(
    "tokenizer", [param(tokenizer, id=tokenizer.name) for tokenizer in sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST)]
)
def test_is_AOTP(tokenizer: MazeTokenizer2):
    if isinstance(tokenizer.prompt_sequencer, PromptSequencers.AOTP):
        assert tokenizer.is_AOTP()
    else:
        assert not tokenizer.is_AOTP()


@mark.parametrize(
    "tokenizer", [param(tokenizer, id=tokenizer.name) for tokenizer in sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST)]
)
def test_is_UT(tokenizer: MazeTokenizer2):
    if isinstance(tokenizer.prompt_sequencer.coord_tokenizer, CoordTokenizers.UT):
        assert tokenizer.is_UT()
    else:
        assert not tokenizer.is_UT()


_has_elems_type = type[TokenizerElement] | TokenizerElement | Iterable[type[TokenizerElement] | TokenizerElement]


@mark.parametrize(
    "tokenizer, elems, result_func",
    [
        param(tokenizer, 
              elems_tuple[0],
              elems_tuple[1],
              id=f"{tokenizer.name}-{elems_tuple[0]}")
        for tokenizer, elems_tuple in itertools.product(
            sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST),
            [
                ([PromptSequencers.AOTP()], 
                 lambda mt, els: mt.prompt_sequencer == els[0]
                 ),
                (PromptSequencers.AOTP(), 
                 lambda mt, els: mt.prompt_sequencer == els
                 ),
                ([CoordTokenizers.CTT()], 
                 lambda mt, els: mt.prompt_sequencer.coord_tokenizer == els[0]
                 ),
                (CoordTokenizers.CTT(intra=False), 
                 lambda mt, els: mt.prompt_sequencer.coord_tokenizer == els
                 ),
                ([CoordTokenizers.CTT], 
                 lambda mt, els: isinstance(mt.prompt_sequencer.coord_tokenizer, els[0])
                 ),
                (CoordTokenizers.CoordTokenizer, 
                 lambda mt, els: isinstance(mt.prompt_sequencer.coord_tokenizer, els)
                 ),
                ([CoordTokenizers.CTT, PathTokenizers.StepSequence], 
                 lambda mt, els: isinstance(mt.prompt_sequencer.coord_tokenizer, els[0]) and isinstance(mt.prompt_sequencer.path_tokenizer, els[1])
                 ),
                # ((a for a in [CoordTokenizers.CTT, PathTokenizers.Coords]), 
                #  lambda mt, els: isinstance(mt.coord_tokenizer, list(els)[0]) and isinstance(mt.path_tokenizer, list(els)[1])
                #  ),
                ([CoordTokenizers.CTT, PathTokenizers.StepSequence(post=False)], 
                 lambda mt, els: isinstance(mt.prompt_sequencer.coord_tokenizer, els[0]) and mt.prompt_sequencer.path_tokenizer == els[1]
                 ),
                ([CoordTokenizers.CTT, PathTokenizers.StepSequence, PromptSequencers.AOP()], 
                 lambda mt, els: isinstance(mt.prompt_sequencer.coord_tokenizer, els[0]) and isinstance(mt.prompt_sequencer.path_tokenizer, els[1]) and mt.prompt_sequencer == els[2]
                 ),
            ]
        )
    ],
)
def test_has_element(
    tokenizer: MazeTokenizer2, 
    elems: _has_elems_type,
    result_func: Callable[[MazeTokenizer2, _has_elems_type], bool]):
    assert tokenizer.has_element(elems) == result_func(tokenizer, elems)


@mark.parametrize(
    "el, result",
    [
        param(elem, result, id=elem.name)
        for elem, result in [
            (CoordTokenizers.CTT(), True),
            (CoordTokenizers.CTT(intra=True), True),
            (CoordTokenizers.UT(), True),
            (AdjListTokenizers.Coords(), True),
            (AdjListTokenizers.Coords(walls=True), True),
            (TargetTokenizers.Unlabeled(post=True), True),
            (PathTokenizers.StepSequence(), True),
            (PathTokenizers.StepSequence(
                step_tokenizers=(StepTokenizers.Coord(), )
                ),
             True),
            (PathTokenizers.StepSequence(
                step_tokenizers=(StepTokenizers.Coord(), StepTokenizers.Coord(),)
                ),
             False),
            (PromptSequencers.AOP(), True),
            (PromptSequencers.AOP(
                path_tokenizer=PathTokenizers.StepSequence()
                ), 
             True),
            (PromptSequencers.AOP(
                path_tokenizer=PathTokenizers.StepSequence(
                    step_tokenizers=(StepTokenizers.Coord(),)
                    )
                ), 
             True),
            (PromptSequencers.AOP(
                path_tokenizer=PathTokenizers.StepSequence(
                    step_tokenizers=(StepTokenizers.Coord(), StepTokenizers.Coord(),)
                    )
                ), 
             True),
            ]
    ],
)
def test_tokenizer_element_is_valid(el: TokenizerElement, result: bool):
    assert el.is_valid() == result
    
    
def test_all_tokenizer_hashes():
    loaded_hashes = save_hashes()
    assert np.array_equal(_load_tokenizer_hashes(), loaded_hashes)
    

@mark.parametrize(
    "tokenizer", [param(tokenizer, id=tokenizer.name) for tokenizer in sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST)]
)
def test_is_tested_tokenizer(tokenizer: MazeTokenizer2):
    assert tokenizer.is_tested_tokenizer()


@mark.parametrize(
    "tokenizer, result", 
    [param(tokenizer, result, id=str(tokenizer)) 
    for tokenizer, result in 
    [
        (MazeTokenizer2(), True),
        (MazeTokenizer2.from_legacy(TokenizationMode.AOTP_CTT_indexed), True),
        (MazeTokenizer2(prompt_sequencer=PromptSequencers.AOP()), False),
    ]
    ]
)
def test_is_legacy_equivalent(tokenizer: MazeTokenizer2, result: bool):
    assert tokenizer.is_legacy_equivalent() == result
    

@mark.parametrize(
    "pt,maze",
    [
        param(tokenizer, maze_kv[1], id=f"{tokenizer.name}-{maze_kv[0]}")
        for maze_kv, tokenizer in itertools.product(
            _ASCII_MAZES.items(), 
            random.sample(
                all_instances(
                    PathTokenizers.PathTokenizer,
                    frozendict.frozendict({PathTokenizers.PathTokenizer: lambda x: x.is_valid()})
                ),
                NUM_TOKENIZERS_TO_TEST
            )
        )
    ],
)    
def test_path_tokenizers(pt: PathTokenizers.PathTokenizer, maze: _MANUAL_MAZE):
    solved_maze = SolvedMaze.from_ascii("\n".join(maze.ascii))
    ct: CoordTokenizers.CoordTokenizer = CoordTokenizers.UT()
    path_toks: list[str] = pt.to_tokens(solved_maze, ct)
    path_toks_set: set[str] = set(path_toks)
    match type(pt.step_size):
        case StepSizes.Singles:
            if StepTokenizers.Coord() in pt.step_tokenizers:
                assert all([tok in path_toks for tok in [ct.to_tokens(c)[0] for c in solved_maze.solution]])
            if StepTokenizers.Distance() in pt.step_tokenizers:
                assert Counter(path_toks)[VOCAB.I_001] == len(solved_maze.solution)-1
            if StepTokenizers.Cardinal() in pt.step_tokenizers:
                c = Counter(path_toks)
                assert c[VOCAB.PATH_NORTH] + c[VOCAB.PATH_SOUTH] + c[VOCAB.PATH_EAST] + c[VOCAB.PATH_WEST] == len(solved_maze.solution)-1
        case StepSizes.Straightaways:
            if StepTokenizers.Coord() in pt.step_tokenizers:
                non_steps = set(tuple(c) for c in solved_maze.solution) - set(tuple(c) for c in maze.straightaway_steps)
                assert all([ct.to_tokens(tok)[0] in path_toks_set for tok in maze.straightaway_steps])
                assert all([ct.to_tokens(tok)[0] not in path_toks_set for tok in non_steps])
            if StepTokenizers.Distance() in pt.step_tokenizers:
                distances: list[int] = [max(abs(c1[0]-c0[0]), abs(c1[1]-c0[1])) for c0, c1 in zip(maze.straightaway_steps[:-1], maze.straightaway_steps[1:])]
                assert len(Counter(getattr(VOCAB, f"I_{d:03}") for d in distances) - Counter(path_toks)) == 0
            if StepTokenizers.Cardinal() in pt.step_tokenizers:
                c = Counter(path_toks)
                assert c[VOCAB.PATH_NORTH] + c[VOCAB.PATH_SOUTH] + c[VOCAB.PATH_EAST] + c[VOCAB.PATH_WEST] == len(maze.straightaway_steps)-1
        case StepSizes.Forks:
            if StepTokenizers.Coord() in pt.step_tokenizers:
                non_steps: set[CoordTup] = set(tuple(c) for c in solved_maze.solution) - set(tuple(c) for c in solved_maze.get_solution_forking_points(always_include_endpoints=True)[1])
                assert all([ct.to_tokens(tok)[0] in path_toks_set for tok in solved_maze.get_solution_forking_points(always_include_endpoints=True)[1]])
                assert all([ct.to_tokens(tok)[0] not in path_toks_set for tok in non_steps])
            if StepTokenizers.Distance() in pt.step_tokenizers:
                footprint_inds: np.ndarray = np.array(solved_maze.get_solution_forking_points(always_include_endpoints=True)[0])
                distances: list[int] = footprint_inds[1:] - footprint_inds[:-1]
                assert len(Counter(getattr(VOCAB, f"I_{d:03}") for d in distances) - Counter(path_toks)) == 0
            if StepTokenizers.Cardinal() in pt.step_tokenizers:
                c = Counter(path_toks)
                assert c[VOCAB.PATH_NORTH] + c[VOCAB.PATH_SOUTH] + c[VOCAB.PATH_EAST] + c[VOCAB.PATH_WEST] == len(solved_maze.get_solution_forking_points(always_include_endpoints=True)[1])-1
        case StepSizes.ForksAndStraightaways:
            swy_step_inds: list[int] = StepSizes.Straightaways()._step_single_indices(solved_maze)
            footprint_inds: Int[np.ndarray, "footprint_index"] = np.concatenate((solved_maze.get_solution_forking_points(always_include_endpoints=True)[0], swy_step_inds))
            footprint_inds, _ = np.unique(footprint_inds, axis=0, return_index=True)
            footprints: Int[np.ndarray, "footprint_index, row_col=2"] = solved_maze.solution[footprint_inds]
            if StepTokenizers.Coord() in pt.step_tokenizers:
                non_steps: set[CoordTup] = set(tuple(c) for c in solved_maze.solution) - set(tuple(c) for c in footprints)
                assert all([ct.to_tokens(coord)[0] in path_toks_set for coord in footprints])
                assert all([ct.to_tokens(coord)[0] not in path_toks_set for coord in non_steps])
            if StepTokenizers.Distance() in pt.step_tokenizers:
                distances: list[int] = footprint_inds[1:] - footprint_inds[:-1]
                assert len(Counter(getattr(VOCAB, f"I_{d:03}") for d in distances) - Counter(path_toks)) == 0
            if StepTokenizers.Cardinal() in pt.step_tokenizers:
                c = Counter(path_toks)
                assert c[VOCAB.PATH_NORTH] + c[VOCAB.PATH_SOUTH] + c[VOCAB.PATH_EAST] + c[VOCAB.PATH_WEST] == len(footprint_inds)-1