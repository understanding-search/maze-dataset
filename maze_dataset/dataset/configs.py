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
