from pytest import mark, param
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
        n_mazes=3,
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
    "maze",
    [
        param(maze)
        for maze in DATASET.mazes
    ]
)
# def test_to_tokens(tokenizer: MazeTokenizer2, maze: SolvedMaze):
def test_to_tokens(maze: SolvedMaze):
    tokenizer = MazeTokenizer2()
    toks: list[str] = tokenizer.to_tokens(maze.connection_list, maze.start_pos, maze.end_pos, maze.solution)
    toks_legacy: list[str] = maze.as_tokens(TokenizationMode.AOTP_UT_uniform)
    print(toks)
    assert equal_except_adj_list_sequence(toks, toks_legacy)