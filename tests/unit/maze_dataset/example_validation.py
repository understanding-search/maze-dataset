from pathlib import Path

import pytest

from maze_dataset import MazeDataset, MazeDatasetConfig

DATASET_PATHS: list[Path] = list(Path("docs/examples/datasets").rglob("*.zanj"))

@pytest.mark.parametrize("dataset_path", DATASET_PATHS)
def test_validate_fname(dataset_path: Path):
	dataset: MazeDataset = MazeDataset.read(dataset_path)
	dataset_cfg: MazeDatasetConfig = dataset.cfg
	dataset_fname_path: str = dataset_path.name.removesuffix(".zanj")
	dataset_fname_loaded: str | None = dataset_cfg._fname_loaded
	dataset_fname_new: str = dataset_cfg.to_fname()
	print(f"{dataset_path.as_posix() = }")
	print(f"{dataset_fname_path = }")
	print(f"{dataset_fname_loaded = }")
	print(f"{dataset_fname_new = }")
	print(f"{dataset_cfg.summary() = }")
	print(f"{dataset_cfg = }")

	assert dataset_fname_path == dataset_fname_loaded
	assert dataset_fname_path == dataset_fname_new
