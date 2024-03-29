from pytest import mark, param
from itertools import product

from maze_dataset import (
    LatticeMazeGenerators,
    MazeDataset,
    MazeDatasetConfig,
    SolvedMaze,
)

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


CFG: MazeDatasetConfig = MazeDatasetConfig(
        name="test",
        grid_n=5,
        n_mazes=5,
        maze_ctor=LatticeMazeGenerators.gen_dfs,
    )
DATASET = MazeDataset.from_config(
    CFG,
    do_download=False,
    load_local=False,
    do_generate=True,
    save_local=False,
    verbose=True,
    gen_parallel=False,
)


@mark.parametrize(
    "maze, tokenizer, legacy_tokenizer",
    [
        param(
            maze[0],
            tok_spec[0],
            tok_spec[1],
            id=f"{tok_spec[1].value}-maze{maze[1]}"
        )
        for maze, tok_spec in product(
            [(maze, i) for i, maze in enumerate(DATASET.mazes[:2])],
            [
                (MazeTokenizer2.from_legacy(tok_mode), tok_mode)
                for tok_mode in TokenizationMode
            ]
        )
    ]
)
def test_to_tokens_backwards_compatible(maze: SolvedMaze, tokenizer: MazeTokenizer2, legacy_tokenizer: TokenizationMode):
    # tokenizer = MazeTokenizer2()
    toks: list[str] = tokenizer.to_tokens(maze.connection_list, maze.start_pos, maze.end_pos, maze.solution)
    toks_legacy: list[str] = maze.as_tokens(legacy_tokenizer)
    assert equal_except_adj_list_sequence(toks, toks_legacy)
    

# def test_from_legacy(maze: SolvedMaze, )