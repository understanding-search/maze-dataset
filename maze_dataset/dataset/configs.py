"`MAZE_DATASET_CONFIGS` contains some default configs for tests and demos"

import copy
from typing import Iterator, Mapping

from maze_dataset.dataset.maze_dataset import MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators

_MAZE_DATASET_CONFIGS_SRC: dict[str, MazeDatasetConfig] = {
	cfg.to_fname(): cfg
	for cfg in [
		MazeDatasetConfig(
			name="test",
			grid_n=3,
			n_mazes=5,
			maze_ctor=LatticeMazeGenerators.gen_dfs,
		),
		MazeDatasetConfig(
			name="test-perc",
			grid_n=3,
			n_mazes=5,
			maze_ctor=LatticeMazeGenerators.gen_dfs_percolation,
			maze_ctor_kwargs={"p": 0.7},
		),
		MazeDatasetConfig(
			name="demo_small",
			grid_n=3,
			n_mazes=100,
			maze_ctor=LatticeMazeGenerators.gen_dfs,
		),
		MazeDatasetConfig(
			name="demo",
			grid_n=6,
			n_mazes=10000,
			maze_ctor=LatticeMazeGenerators.gen_dfs,
		),
	]
}


class _MazeDatsetConfigsWrapper(Mapping[str, MazeDatasetConfig]):
	"wrap the default configs in a read-only dict-like object"

	def __init__(self, configs: dict[str, MazeDatasetConfig]) -> None:
		"initialize with a dict of configs"
		self._configs = configs

	def __getitem__(self, item: str) -> MazeDatasetConfig:
		return self._configs[item]

	def __len__(self) -> int:
		return len(self._configs)

	def __iter__(self) -> Iterator:
		"iterate over the keys"
		return iter(self._configs)

	# TYPING: error: Return type "list[str]" of "keys" incompatible with return type "KeysView[str]" in supertype "Mapping"  [override]
	def keys(self) -> list[str]:  # type: ignore[override]
		"return the keys"
		return list(self._configs.keys())

	# TYPING: error: Return type "list[tuple[str, MazeDatasetConfig]]" of "items" incompatible with return type "ItemsView[str, MazeDatasetConfig]" in supertype "Mapping"  [override]
	def items(self) -> list[tuple[str, MazeDatasetConfig]]:  # type: ignore[override]
		"return the items"
		return [(k, copy.deepcopy(v)) for k, v in self._configs.items()]

	# TYPING: error: Return type "list[MazeDatasetConfig]" of "values" incompatible with return type "ValuesView[MazeDatasetConfig]" in supertype "Mapping"  [override]
	def values(self) -> list[MazeDatasetConfig]:  # type: ignore[override]
		return [copy.deepcopy(v) for v in self._configs.values()]


MAZE_DATASET_CONFIGS: _MazeDatsetConfigsWrapper = _MazeDatsetConfigsWrapper(
	_MAZE_DATASET_CONFIGS_SRC,
)


def _get_configs_for_examples() -> list[dict]:
	"""Generate a list of diverse maze configurations"""
	configs: list[dict] = []

	# Basic maze configurations with different algorithms
	for grid_n in [5, 8, 12]:
		# DFS with different options
		configs.append(
			{
				"name": "basic",
				"grid_n": grid_n,
				"maze_ctor": LatticeMazeGenerators.gen_dfs,
				"maze_ctor_kwargs": {},
				"description": f"Basic DFS maze ({grid_n}x{grid_n})",
				"tags": ["dfs", "basic"],
			}
		)

		configs.append(
			{
				"name": "forkless",
				"grid_n": grid_n,
				"maze_ctor": LatticeMazeGenerators.gen_dfs,
				"maze_ctor_kwargs": {"do_forks": False},
				"description": f"DFS without forks ({grid_n}x{grid_n})",
				"tags": ["dfs", "no_forks", "simple_path"],
			}
		)

		# Wilson's algorithm
		configs.append(
			{
				"name": f"basic",
				"grid_n": grid_n,
				"maze_ctor": LatticeMazeGenerators.gen_wilson,
				"maze_ctor_kwargs": {},
				"description": f"Wilson's algorithm ({grid_n}x{grid_n}) - unbiased random maze",
				"tags": ["wilson", "uniform_random"],
			}
		)

		# Percolation with different probabilities
		for p in [0.3, 0.5, 0.7]:
			configs.append(
				{
					"name": f"p{p}",
					"grid_n": grid_n,
					"maze_ctor": LatticeMazeGenerators.gen_percolation,
					"maze_ctor_kwargs": {"p": p},
					"description": f"Pure percolation (p={p}) ({grid_n}x{grid_n})",
					"tags": ["percolation", f"p={p}"],
				}
			)

			configs.append(
				{
					"name": f"p{p}",
					"grid_n": grid_n,
					"maze_ctor": LatticeMazeGenerators.gen_dfs_percolation,
					"maze_ctor_kwargs": {"p": p},
					"description": f"DFS with percolation (p={p}) ({grid_n}x{grid_n})",
					"tags": ["dfs", "percolation", f"p={p}"],
				}
			)

	# Additional specialized configurations
	configs.extend(
		[
			{
				"name": "accessible_cells",
				"grid_n": 10,
				"maze_ctor": LatticeMazeGenerators.gen_dfs,
				"maze_ctor_kwargs": {"accessible_cells": 50},
				"description": "DFS with limited accessible cells (50)",
				"tags": ["dfs", "limited_cells"],
			},
			{
				"name": "accessible_cells_ratio",
				"grid_n": 10,
				"maze_ctor": LatticeMazeGenerators.gen_dfs,
				"maze_ctor_kwargs": {"accessible_cells": 0.6},
				"description": "DFS with 60% accessible cells",
				"tags": ["dfs", "limited_cells", "ratio"],
			},
			{
				"name": "max_tree_depth",
				"grid_n": 10,
				"maze_ctor": LatticeMazeGenerators.gen_dfs,
				"maze_ctor_kwargs": {"max_tree_depth": 10},
				"description": "DFS with max tree depth of 10",
				"tags": ["dfs", "limited_depth"],
			},
			{
				"name": "max_tree_depth_ratio",
				"grid_n": 10,
				"maze_ctor": LatticeMazeGenerators.gen_dfs,
				"maze_ctor_kwargs": {"max_tree_depth": 0.3},
				"description": "DFS with max tree depth 30% of grid size",
				"tags": ["dfs", "limited_depth", "ratio"],
			},
			{
				"name": "start_coord",
				"grid_n": 10,
				"maze_ctor": LatticeMazeGenerators.gen_dfs,
				"maze_ctor_kwargs": {"start_coord": [5, 5]},
				"description": "DFS starting from center of grid",
				"tags": ["dfs", "custom_start"],
			},
			{
				"name": "combined_constraints",
				"grid_n": 15,
				"maze_ctor": LatticeMazeGenerators.gen_dfs,
				"maze_ctor_kwargs": {
					"accessible_cells": 100,
					"max_tree_depth": 25,
					"start_coord": [7, 7],
				},
				"description": "DFS with multiple constraints",
				"tags": ["dfs", "combined_constraints"],
			},
		]
	)

	# Add endpoint options for some configurations
	for deadend_start, deadend_end in [(True, False), (False, True), (True, True)]:
		configs.append(
			{
				"name": f"deadend_start{deadend_start}_end{deadend_end}",
				"grid_n": 8,
				"maze_ctor": LatticeMazeGenerators.gen_dfs,
				"maze_ctor_kwargs": {},
				"endpoint_kwargs": {
					"deadend_start": deadend_start,
					"deadend_end": deadend_end,
					"endpoints_not_equal": True,
				},
				"description": f"DFS with {'deadend start' if deadend_start else ''}{' and ' if deadend_start and deadend_end else ''}{'deadend end' if deadend_end else ''}",
				"tags": ["dfs", "deadend_endpoints"],
			}
		)

	# Add some percolation examples with deadend endpoints
	configs.append(
		{
			"name": "deadends",
			"grid_n": 8,
			"maze_ctor": LatticeMazeGenerators.gen_dfs_percolation,
			"maze_ctor_kwargs": {"p": 0.3},
			"endpoint_kwargs": {
				"deadend_start": True,
				"deadend_end": True,
				"endpoints_not_equal": True,
				"except_on_no_valid_endpoint": False,
			},
			"description": "DFS percolation with deadend endpoints",
			"tags": ["dfs", "percolation", "deadend_endpoints"],
		}
	)

	return configs
