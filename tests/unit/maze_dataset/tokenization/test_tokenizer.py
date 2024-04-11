import re
import numpy as np
import itertools
from itertools import product
from typing import Iterable

from pytest import mark, param

from maze_dataset import (
    LatticeMazeGenerators,
    MazeDataset,
    MazeDatasetConfig,
    SolvedMaze,
    TargetedLatticeMaze,
    LatticeMaze,
    Coord,
    CoordTup,
    VOCAB
)
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.generation.default_generators import DEFAULT_GENERATORS
from maze_dataset.generation.generators import GENERATORS_MAP

from maze_dataset.plotting.print_tokens import color_maze_tokens_AOTP
from maze_dataset.tokenization.util import equal_except_adj_list_sequence
from maze_dataset.tokenization import (
    MazeTokenizer2,
    MazeTokenizer,
    TokenizationMode
)


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
LATTICE_MAZES: list[LatticeMaze] = [LatticeMazeGenerators.gen_dfs(np.array([GRID_N, GRID_N])) for _ in range(N_MAZES)]
_PATHS = [maze.generate_random_path() for maze in LATTICE_MAZES]
TARGETED_MAZES: list[TargetedLatticeMaze] = [TargetedLatticeMaze.from_lattice_maze(maze, path[0], path[-1]) for maze, path in zip(LATTICE_MAZES, _PATHS)]
# MIXED_MAZES alternates the maze types, so you can slice a contiguous subset and still get all types
MIXED_MAZES: list[LatticeMaze | TargetedLatticeMaze | SolvedMaze] = [x for x in itertools.chain.from_iterable(itertools.zip_longest(MAZE_DATASET.mazes, TARGETED_MAZES, LATTICE_MAZES))]


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


_ASCII_MAZES: dict[str, tuple[str, list[str]]] = dict(
    small_3x3=(
        "<ADJLIST_START> (2,0) <--> (2,1) ; (0,0) <--> (0,1) ; (0,0) <--> (1,0) ; (0,2) <--> (1,2) ; (1,0) <--> (2,0) ; (0,2) <--> (0,1) ; (2,2) <--> (2,1) ; (1,1) <--> (2,1) ; <ADJLIST_END> <ORIGIN_START> (0,0) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (0,0) (1,0) (2,0) (2,1) <PATH_END>",
        [
            "#######",
            "#S    #",
            "#X### #",
            "#X# # #",
            "#X# ###",
            "#XXE  #",
            "#######",
        ],
    ),
    big_10x10=(
        "<ADJLIST_START> (8,2) <--> (8,3) ; (3,7) <--> (3,6) ; (6,7) <--> (6,8) ; (4,6) <--> (5,6) ; (9,5) <--> (9,4) ; (3,3) <--> (3,4) ; (5,1) <--> (4,1) ; (2,6) <--> (2,7) ; (8,5) <--> (8,4) ; (1,9) <--> (2,9) ; (4,1) <--> (4,2) ; (0,8) <--> (0,7) ; (5,4) <--> (5,3) ; (6,3) <--> (6,4) ; (5,0) <--> (4,0) ; (5,3) <--> (5,2) ; (3,1) <--> (2,1) ; (9,1) <--> (9,0) ; (3,5) <--> (3,6) ; (5,5) <--> (6,5) ; (7,1) <--> (7,2) ; (0,1) <--> (1,1) ; (7,8) <--> (8,8) ; (3,9) <--> (4,9) ; (4,6) <--> (4,7) ; (0,6) <--> (0,7) ; (3,4) <--> (3,5) ; (6,0) <--> (5,0) ; (7,7) <--> (7,6) ; (1,6) <--> (0,6) ; (6,1) <--> (6,0) ; (8,6) <--> (8,7) ; (9,9) <--> (9,8) ; (1,8) <--> (1,9) ; (2,1) <--> (2,2) ; (9,2) <--> (9,3) ; (5,9) <--> (6,9) ; (3,2) <--> (2,2) ; (0,8) <--> (0,9) ; (5,6) <--> (5,7) ; (2,3) <--> (2,4) ; (4,5) <--> (4,4) ; (8,9) <--> (8,8) ; (9,6) <--> (8,6) ; (3,7) <--> (3,8) ; (8,0) <--> (7,0) ; (6,1) <--> (6,2) ; (0,1) <--> (0,0) ; (7,3) <--> (7,4) ; (9,4) <--> (9,3) ; (9,6) <--> (9,5) ; (8,7) <--> (7,7) ; (5,2) <--> (5,1) ; (0,0) <--> (1,0) ; (7,2) <--> (7,3) ; (2,5) <--> (2,6) ; (4,9) <--> (5,9) ; (5,5) <--> (5,4) ; (5,6) <--> (6,6) ; (7,8) <--> (7,9) ; (1,7) <--> (2,7) ; (4,6) <--> (4,5) ; (1,1) <--> (1,2) ; (3,1) <--> (3,0) ; (1,5) <--> (1,6) ; (8,3) <--> (8,4) ; (9,9) <--> (8,9) ; (8,5) <--> (7,5) ; (1,4) <--> (2,4) ; (3,0) <--> (4,0) ; (3,3) <--> (4,3) ; (6,9) <--> (6,8) ; (1,0) <--> (2,0) ; (6,0) <--> (7,0) ; (8,0) <--> (9,0) ; (2,3) <--> (2,2) ; (2,8) <--> (3,8) ; (5,7) <--> (6,7) ; (1,3) <--> (0,3) ; (9,7) <--> (9,8) ; (7,5) <--> (7,4) ; (1,8) <--> (2,8) ; (6,5) <--> (6,4) ; (0,2) <--> (1,2) ; (0,7) <--> (1,7) ; (0,3) <--> (0,2) ; (4,3) <--> (4,2) ; (5,8) <--> (4,8) ; (9,1) <--> (8,1) ; (9,2) <--> (8,2) ; (1,3) <--> (1,4) ; (2,9) <--> (3,9) ; (4,8) <--> (4,7) ; (0,5) <--> (0,4) ; (8,1) <--> (7,1) ; (0,3) <--> (0,4) ; (9,7) <--> (9,6) ; (7,6) <--> (6,6) ; (1,5) <--> (0,5) ; <ADJLIST_END> <ORIGIN_START> (6,2) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (6,2) (6,1) (6,0) (5,0) (4,0) (3,0) (3,1) (2,1) <PATH_END>",
        [
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


# MazeTokenizer2 tests
# =====================

# Backwards compatibility tests
# =============================

@mark.parametrize(
    "maze,legacy_tokenizer",
    [
        param(
            maze[0],
            tok_spec,
            id=f"{tok_spec.value}-maze{maze[1]}"
        )
        for maze, tok_spec in itertools.product(
            [(maze, i) for i, maze in enumerate(MIXED_MAZES[:6])],
            [tok_mode for tok_mode in TokenizationMode]
        )
    ]
)
def test_to_tokens_backwards_compatible(maze: SolvedMaze, legacy_tokenizer: TokenizationMode):
    tokenizer: MazeTokenizer2 = MazeTokenizer2.from_legacy(legacy_tokenizer)
    toks: list[str] = maze.as_tokens(tokenizer)
    toks_legacy: list[str] = maze.as_tokens(legacy_tokenizer)
    assert equal_except_adj_list_sequence(toks, toks_legacy)
    

@mark.parametrize(
    "coords, legacy_tok_mode",
    [
        param(
            coords,
            tok_mode,
            id=f"{tok_mode.value}-coords(type={type(coords[0])},len={len(coords)})"
        )
        for tok_mode, coords in itertools.product(
            [tok_mode for tok_mode in TokenizationMode],
            [
                *[[maze.start_pos] for maze in MAZE_DATASET.mazes[:2]],
                [maze.start_pos for maze in MAZE_DATASET.mazes],
                *[[tuple(maze.start_pos)] for maze in MAZE_DATASET.mazes[:2]],
                [tuple(maze.start_pos) for maze in MAZE_DATASET.mazes],
                ]
        )
    ]
)
def test_coords_to_strings_backwards_compatible(coords: list[Coord, CoordTup], legacy_tok_mode: TokenizationMode):
    tokenizer: MazeTokenizer2 = MazeTokenizer2.from_legacy(legacy_tok_mode)
    legacy_tokenizer = MazeTokenizer(tokenization_mode=legacy_tok_mode)
    strings: list[str] = tokenizer.coords_to_strings(coords)
    strings_legacy: list[str] = legacy_tokenizer.coords_to_strings(coords)
    assert strings == strings_legacy


@mark.parametrize(
    "maze,tok_mode",
    [
        param(
            maze[0],
            tok_spec,
            id=f"{tok_spec.value}-maze{maze[1]}"
        )
        for maze, tok_spec in itertools.product(
            [(maze, i) for i, maze in enumerate(MIXED_MAZES[:6])],
            [tok_mode for tok_mode in TokenizationMode]
        )
    ]
)
def test_from_tokens_backwards_compatible(maze: LatticeMaze, tok_mode: TokenizationMode):
    tokenizer = MazeTokenizer2.from_legacy(tok_mode)
    toks = maze.as_tokens(tok_mode)
    # Equality test of `as_tokens` output done in a separate unit test
    maze_legacy: LatticeMaze = LatticeMaze.from_tokens(toks, tok_mode)
    maze: LatticeMaze = LatticeMaze.from_tokens(toks, tokenizer)        
    assert maze == maze_legacy
    
# General functionality tests
# ===========================

@mark.parametrize(
    "tokenizer",
    [
        param(
            MazeTokenizer2.from_legacy(tok_mode),
        )
        for tok_mode in TokenizationMode
    ]
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

    # Just make sure the call doesn't raise exception
    tokenizer.clear_cache()


@mark.parametrize(
    "maze,tokenizer",
    [
        param(
            maze[0],
            tok_spec,
            id=f"{tok_spec.name}-maze{maze[1]}"
        )
        for maze, tok_spec in itertools.product(
            [(maze, i) for i, maze in enumerate(MIXED_MAZES[:6])],
            [MazeTokenizer2.from_legacy(tok_mode) for tok_mode in TokenizationMode]
        )
    ]
)
def test_encode_decode(maze: LatticeMaze, tokenizer: MazeTokenizer2):
    # Just make sure the call doesn't raise exception
 
    maze_tok = maze.as_tokens(maze_tokenizer=tokenizer)

    maze_encoded = tokenizer.encode(maze_tok)
    maze_decoded = tokenizer.decode(maze_encoded)

    assert maze_tok == maze_decoded

    maze_recovered = SolvedMaze.from_tokens(maze_tok, maze_tokenizer=tokenizer)

    assert (maze.connection_list == maze_recovered.connection_list).all()
