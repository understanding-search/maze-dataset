import pytest

from maze_dataset.dataset.maze_dataset import MazeDatasetConfig
from maze_dataset.generation.generators import get_maze_with_solution
from maze_dataset.generation.lattice_maze import SolvedMaze


@pytest.mark.skip(
    "Currently impossible to compare LatticeMazes - https://github.com/AISC-understanding-search/maze-transformer/issues/135"
)
def test_from_tokens():
    maze_size = 2
    maze, solution = get_maze_with_solution("gen_dfs", (maze_size, maze_size))

    # See https://github.com/AISC-understanding-search/maze-transformer/issues/77
    config = MazeDatasetConfig(grid_n=maze_size, name="test", n_mazes=1)
    tokenized_maze = maze.as_tokens(config.node_token_map)

    new_maze, new_solution = SolvedMaze.from_tokens(tokenized_maze, config)
    assert new_maze == maze
    assert new_solution == solution
