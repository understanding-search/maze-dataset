from pathlib import Path
from maze_dataset import MazeDatasetConfig, MazeDataset
from maze_dataset.dataset.configs import _MAZE_DATASET_CONFIGS_SRC


def main():
	for cfg_name, cfg in _MAZE_DATASET_CONFIGS_SRC.items():
		dataset: MazeDataset = MazeDataset.from_config(
			cfg,
			load_local=False,
			save_local=True,
			local_base_path=Path("docs/examples/datasets/"),
		)
		print(dataset.cfg.diff(cfg))
		print(dataset.cfg._serialize_base().keys())
		print(cfg._serialize_base().keys())
		assert cfg_name == dataset.cfg.to_fname(), f"{cfg_name=}, {dataset.cfg.to_fname()=}"
		assert cfg_name == cfg.to_fname(), f"{cfg_name=}, {cfg.to_fname()=}"
		print(f"Dataset {cfg_name} has {len(dataset.mazes)} mazes")


if __name__ == "__main__":
	main()
