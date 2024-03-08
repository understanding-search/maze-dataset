from pytest import mark, param

from maze_dataset import (
    LatticeMazeGenerators,
    MazeDataset,
    MazeDatasetConfig,
    SolvedMaze,
)
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode


@mark.parametrize(
    "tok_mode",
    [
        param(
            tok_mode,
            id=tok_mode.name
        )
        for tok_mode in TokenizationMode
    ],
)
def test_tokenization_roundtrip(tok_mode: TokenizationMode):
    dataset: MazeDataset = MazeDataset.from_config(
        MazeDatasetConfig(
            name="test",
            grid_n=5,
            n_mazes=5,
            maze_ctor=LatticeMazeGenerators.gen_dfs,
        )
    )
    tokenizer: MazeTokenizer = MazeTokenizer(
        tokenization_mode=tok_mode,
        max_grid_size=20,
    )

    dataset_tokenized: list[list[str]] = dataset.as_tokens(tokenizer)
    dataset_tokenized_individual: list[list[str]] = [
        maze.as_tokens(tokenizer) for maze in dataset.mazes
    ]

    mazes_roundtrip: list[SolvedMaze] = [
        SolvedMaze.from_tokens(
            tokens=maze_tokens,
            maze_tokenizer=tokenizer,
        )
        for maze_tokens in dataset_tokenized
    ]

    mazes_roundtrip_individual: list[SolvedMaze] = [
        SolvedMaze.from_tokens(
            tokens=maze_tokens,
            maze_tokenizer=tokenizer,
        )
        for maze_tokens in dataset_tokenized_individual
    ]

    # NOTE: can't test the tokenization explicitly because order in adjacency list is random
    # test both tokenized as a whole and tokenized individually
    # for maze_tok, maze_tok_indiv in zip(dataset_tokenized, dataset_tokenized_individual):
    #     assert all(
    #         x == y
    #         for x, y in zip(maze_tok, maze_tok_indiv)
    #     ), f"maze_tok: {' '.join(maze_tok)}, maze_tok_indiv: {' '.join(maze_tok_indiv)}"

    # test roundtrip
    for maze, maze_rt, maze_rt_indiv in zip(
        dataset.mazes, mazes_roundtrip, mazes_roundtrip_individual
    ):
        assert maze == maze_rt, f"maze: {maze}, maze_rt: {maze_rt}"
        assert maze == maze_rt_indiv, f"maze: {maze}, maze_rt_indiv: {maze_rt_indiv}"
