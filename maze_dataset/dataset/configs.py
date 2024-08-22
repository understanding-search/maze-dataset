import copy
from typing import Mapping

from muutils.kappa import Kappa

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


def _kappa_md_configs(key: str) -> MazeDatasetConfig:
    return copy.deepcopy(_MAZE_DATASET_CONFIGS_SRC[key])


MAZE_DATASET_CONFIGS: Mapping[str, MazeDatasetConfig] = Kappa(_kappa_md_configs)
