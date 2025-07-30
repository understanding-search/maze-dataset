"""generation functions have signature `(grid_shape: Coord, **kwargs) -> LatticeMaze` and are methods in `LatticeMazeGenerators`

`DEFAULT_GENERATORS` is a list of generator name, generator kwargs pairs used in tests and demos

you can add your own maze generators by:
- adding a static method implementing your generation function to `LatticeMazeGenerators`, with the signature `(grid_shape: Coord, **kwargs) -> LatticeMaze`
- adding the pair to `GENERATORS_MAP`

If you implement a new generation function, please make a pull request!
https://github.com/understanding-search/maze-dataset/pulls
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
