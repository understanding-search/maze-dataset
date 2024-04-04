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
    

# def test_from_legacy(maze: SolvedMaze, )