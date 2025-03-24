from pathlib import Path

import pytest
from muutils.json_serialize.util import _FORMAT_KEY

from maze_dataset import MazeDataset, MazeDatasetConfig

DATASET_PATHS: list[Path] = list(Path("docs/examples/datasets").rglob("*.zanj"))


def test_temp():
	cfg = MazeDatasetConfig(
		name="test",
		grid_n=3,
		n_mazes=2,
	)
	print(f"{cfg.fname_loaded = }")
	cfg_ser = cfg.serialize()
	print(f"{list(cfg_ser.keys()) = }")
	print(f"{cfg_ser[_FORMAT_KEY] = }")
	print(f"{cfg_ser['fname'] = }")
	assert "fname_loaded" not in cfg_ser
	cfg_load: MazeDatasetConfig = MazeDatasetConfig.load(cfg_ser)
	print(f"{cfg.diff(cfg_load) = }")
	print(f"{cfg_load.fname_loaded = }")
	assert cfg == cfg_load
	pytest.fail("wip")


@pytest.mark.parametrize("dataset_path", DATASET_PATHS)
def test_validate_fname(dataset_path: Path):
	dataset: MazeDataset = MazeDataset.read(dataset_path)
	dataset_cfg: MazeDatasetConfig = dataset.cfg
	dataset_fname_path: str = dataset_path.name.removesuffix(".zanj")
	datasetfname_loaded: str | None = dataset_cfg.fname_loaded
	dataset_fname_new: str = dataset_cfg.to_fname()
	print(f"{dataset_path.as_posix() = }")
	print(f"{dataset_fname_path = }")
	print(f"{datasetfname_loaded = }")
	print(f"{dataset_fname_new = }")
	print(f"{dataset_cfg.summary() = }")
	print(f"{dataset_cfg = }")

	assert dataset_fname_path == datasetfname_loaded
	assert dataset_fname_path == dataset_fname_new
