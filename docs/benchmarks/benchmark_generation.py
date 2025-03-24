from typing import Any
from maze_dataset.benchmark.speed import run_benchmark
from maze_dataset.generation.default_generators import DEFAULT_GENERATORS

BENCHMARK_KWARGS: dict[str, dict[str, Any]] = dict(
	test=dict(
		grid_n_vals=[2, 3, 4],
		n_mazes_vals=list(range(1, 5, 2)),
		trials=3,
	),
	default=dict(
		grid_n_vals=[2, 3, 4, 5, 8, 10, 16, 25, 32],
		n_mazes_vals=list(range(1, 12, 2)),
		trials=10,
	),
)

if __name__ == "__main__":
	import argparse

	parser: argparse.ArgumentParser = argparse.ArgumentParser()
	parser.add_argument(
		"analysis",
		type=str,
		choices=BENCHMARK_KWARGS.keys(),
		help="The analysis to run",
	)
	parser.add_argument(
		"-s",
		"--save-path",
		type=str,
		default="tests/_temp/benchmark_generation.jsonl",
		help="The path to save the results",
	)
	parser.add_argument(
		"-q",
		"--quiet",
		action="store_true",
		help="disable verbose output",
	)
	args: argparse.Namespace = parser.parse_args()

	run_benchmark(
		save_path=args.save_path,
		base_configs=DEFAULT_GENERATORS,
		verbose=not args.quiet,
		**BENCHMARK_KWARGS[args.analysis],
	)
