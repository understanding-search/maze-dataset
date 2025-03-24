"""generation functions have signature `(grid_shape: Coord, **kwargs) -> LatticeMaze` and are methods in `LatticeMazeGenerators`

`DEFAULT_GENERATORS` is a list of generator name, generator kwargs pairs used in tests and demos

"""

from maze_dataset.generation.generators import (
	GENERATORS_MAP,
	LatticeMazeGenerators,
	get_maze_with_solution,
	numpy_rng,
)

__all__ = [
	# submodules
	"default_generators",
	"generators",
	"seed",
	# imports
	"LatticeMazeGenerators",
	"GENERATORS_MAP",
	"get_maze_with_solution",
	"numpy_rng",
]
