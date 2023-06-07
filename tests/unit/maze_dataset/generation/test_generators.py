import numpy as np

from maze_dataset.generation.generators import (
    GENERATORS_MAP,
    LatticeMazeGenerators,
    get_maze_with_solution,
)
from maze_dataset.generation.lattice_maze import Coord, SolvedMaze


def test_gen_dfs_square():
    three_by_three: Coord = np.array([3, 3])
    maze = LatticeMazeGenerators.gen_dfs(three_by_three)

    assert maze.connection_list.shape == (2, 3, 3)


def test_gen_dfs_oblong():
    three_by_four: Coord = np.array([3, 4])
    maze = LatticeMazeGenerators.gen_dfs(three_by_four)

    assert maze.connection_list.shape == (2, 3, 4)


def test_get_maze_with_solution():
    three_by_three: Coord = np.array([3, 3])
    for key in GENERATORS_MAP:
        maze: SolvedMaze = get_maze_with_solution(key, three_by_three)

        assert maze.connection_list.shape == (2, 3, 3)
        assert len(maze.solution[0]) == 2
