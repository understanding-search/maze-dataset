# run with
# python -m cProfile -s cumtime serialization_profiling.py <grid_n> <n_mazes> <serialize_method> > profile.txt

from maze_dataset import (
    MazeDataset,
    MazeDatasetConfig,
)
from maze_dataset.generation.generators import GENERATORS_MAP

def main(grid_n: int, n_mazes: int, serialize_method: str):
	cfg = MazeDatasetConfig(
		name="test",
		grid_n=grid_n,
		n_mazes=n_mazes,
		maze_ctor=GENERATORS_MAP['gen_dfs'],
	)
	dataset = MazeDataset.from_config(cfg)

	ser_result = getattr(dataset, serialize_method)()


if __name__ == "__main__":
	import sys
	main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])