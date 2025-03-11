"`MAZE_DATASET_CONFIGS` contains some default configs for tests and demos"

import copy
from typing import Mapping

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

	def __iter__(self) -> iter:
		"iterate over the keys"
		return iter(self._configs)

	def keys(self) -> list[str]:
		"return the keys"
		return list(self._configs.keys())

	def items(self) -> list[tuple[str, MazeDatasetConfig]]:
		"return the items"
		return [(k, copy.deepcopy(v)) for k, v in self._configs.items()]

	def values(self) -> list[MazeDatasetConfig]:
		return [copy.deepcopy(v) for v in self._configs.values()]


MAZE_DATASET_CONFIGS: _MazeDatsetConfigsWrapper = _MazeDatsetConfigsWrapper(
	_MAZE_DATASET_CONFIGS_SRC,
)
