from pytest import mark, param

from maze_dataset import SolvedMaze
from maze_dataset.generation.generators import get_maze_with_solution
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
def test_from_tokens(tok_mode: TokenizationMode):
    tokenizer: MazeTokenizer = MazeTokenizer(
        tokenization_mode=tok_mode,
        max_grid_size=20,
    )
    maze_size: int = 2
    solved_maze: SolvedMaze = get_maze_with_solution("gen_dfs", (maze_size, maze_size))

    tokenized_maze: list[str] = solved_maze.as_tokens(tokenizer)

    solved_maze_rt: SolvedMaze = SolvedMaze.from_tokens(tokenized_maze, tokenizer)
    assert (
        solved_maze == solved_maze_rt
    ), f"solved_maze: {solved_maze}, solved_maze_rt: {solved_maze_rt}"
    assert (
        solved_maze.connection_list == solved_maze_rt.connection_list
    ).all(), f"solved_maze.connection_list: {solved_maze.connection_list}, solved_maze_rt.connection_list: {solved_maze_rt.connection_list}"
    assert (
        solved_maze.solution == solved_maze_rt.solution
    ).all(), f"solved_maze.solution: {solved_maze.solution}, solved_maze_rt.solution: {solved_maze_rt.solution}"
