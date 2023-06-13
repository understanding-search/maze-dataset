import typing
import warnings
from dataclasses import dataclass
from itertools import chain
from typing import cast

import numpy as np
from jaxtyping import Bool, Int, Shaped
from muutils.json_serialize.serializable_dataclass import (
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)
from muutils.misc import list_split
from muutils.tensor_utils import NDArray

from maze_dataset.constants import (
    NEIGHBORS_MASK,
    SPECIAL_TOKENS,
    Coord,
    CoordArray,
    CoordTup,
)
from maze_dataset.tokenization.token_utils import (
    get_adj_list_tokens,
    get_path_tokens,
    tokens_to_coords,
)

from maze_dataset.maze.lattice_maze import (
    ConnectionList,
    PixelGrid,
    BinaryPixelGrid,
    ConnectionList,
    LatticeMaze,
    PixelColors,
    AsciiChars,
    ASCII_PIXEL_PAIRINGS,
    TargetedLatticeMaze,
)

@serializable_dataclass(frozen=True, kw_only=True)
class SolvedMaze(TargetedLatticeMaze):
    """Stores a maze and a solution"""

    solution: CoordArray

    def __init__(
        self,
        connection_list: ConnectionList,
        solution: CoordArray,
        generation_meta: dict | None = None,
        start_pos: Coord | None = None,
        end_pos: Coord | None = None,
    ) -> None:
        super().__init__(
            connection_list=connection_list,
            generation_meta=generation_meta,
            start_pos=np.array(solution[0]),
            end_pos=np.array(solution[-1]),
        )
        self.__dict__["solution"] = solution

        if start_pos is not None:
            assert np.array_equal(
                np.array(start_pos), self.start_pos
            ), f"when trying to create a SolvedMaze, the given start_pos does not match the one in the solution: given={start_pos}, solution={self.start_pos}"
        if end_pos is not None:
            assert np.array_equal(
                np.array(end_pos), self.end_pos
            ), f"when trying to create a SolvedMaze, the given end_pos does not match the one in the solution: given={end_pos}, solution={self.end_pos}"

    def __hash__(self) -> int:
        return hash((self.connection_list.tobytes(), self.solution.tobytes()))

    def get_solution_tokens(self, node_token_map: dict[CoordTup, str]) -> list[str]:
        return [
            SPECIAL_TOKENS["path_start"],
            *[node_token_map[tuple(c.tolist())] for c in self.solution],
            SPECIAL_TOKENS["path_end"],
        ]

    # for backwards compatibility
    @property
    def maze(self) -> LatticeMaze:
        warnings.warn(
            "maze is deprecated, SolvedMaze now inherits from LatticeMaze.",
            DeprecationWarning,
        )
        return LatticeMaze(connection_list=self.connection_list)

    @classmethod
    def from_lattice_maze(
        cls, lattice_maze: LatticeMaze, solution: list[CoordTup]
    ) -> "SolvedMaze":
        return cls(
            connection_list=lattice_maze.connection_list,
            solution=solution,
            generation_meta=lattice_maze.generation_meta,
        )

    @classmethod
    def from_targeted_lattice_maze(
        cls, targeted_lattice_maze: TargetedLatticeMaze
    ) -> "SolvedMaze":
        """solves the given targeted lattice maze and returns a SolvedMaze"""
        solution: list[CoordTup] = targeted_lattice_maze.find_shortest_path(
            targeted_lattice_maze.start_pos,
            targeted_lattice_maze.end_pos,
        )
        return cls(
            connection_list=targeted_lattice_maze.connection_list,
            solution=np.array(solution),
            generation_meta=targeted_lattice_maze.generation_meta,
        )

    @classmethod
    def from_tokens(cls, tokens: list[str] | str, data_cfg) -> "SolvedMaze":
        if type(tokens) == str:
            tokens = tokens.split(" ")

        maze: LatticeMaze = LatticeMaze.from_tokens(tokens)
        path_tokens: list[str] = get_path_tokens(tokens)
        solution: list[str | tuple[int, int]] = tokens_to_coords(path_tokens, data_cfg)

        assert len(solution) > 0, f"No solution found: {solution = }"

        try:
            solution_cast: list[CoordTup] = cast(list[CoordTup], solution)
            solution_np: CoordArray = np.array(solution_cast)
        except ValueError as e:
            raise ValueError(f"Invalid solution: {solution = }") from e

        return cls.from_lattice_maze(lattice_maze=maze, solution=solution_np)