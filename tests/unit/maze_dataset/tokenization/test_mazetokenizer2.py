from pytest import mark, param
import itertools
import numpy as np

from maze_dataset import (
    LatticeMazeGenerators,
    MazeDataset,
    MazeDatasetConfig,
    SolvedMaze,
    TargetedLatticeMaze,
    LatticeMaze,
    Coord,
    CoordTup
)
from maze_dataset.generation.default_generators import DEFAULT_GENERATORS
from maze_dataset.generation.generators import GENERATORS_MAP
from maze_dataset.tokenization.util import equal_except_adj_list_sequence

from maze_dataset.tokenization import (
    TokenizerElement,
    MazeTokenizer2,
    PromptSequencers,
    CoordTokenizers,
    AdjListTokenizers,
    PathTokenizers,
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

# General functionality tests
# ===========================

def test_maze_to_tokens_roundtrip():
    # TODO: implement when `from_tokens` ready
    pass
