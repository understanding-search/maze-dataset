"`MazeDatasetConfig`s are used to create a `MazeDataset` via `MazeDataset.from_config(cfg)`"

from maze_dataset.dataset.collected_dataset import (
    MazeDatasetCollection,
    MazeDatasetCollectionConfig,
)
from maze_dataset.dataset.maze_dataset import MazeDataset, MazeDatasetConfig

__all__ = [
    # submodules
    "collected_dataset",
    "configs",
    "dataset",
    "maze_dataset",
    "rasterized",
    # dataset classes
    "MazeDataset",
    "MazeDatasetConfig",
    "MazeDatasetCollection",
    "MazeDatasetCollectionConfig",
]
