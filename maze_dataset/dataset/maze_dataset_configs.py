from maze_dataset.dataset.maze_dataset import MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators

MAZE_DATASET_CONFIGS: dict[str, MazeDatasetConfig] = {
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
