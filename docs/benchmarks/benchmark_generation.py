from maze_dataset.benchmark.speed import run_benchmark
from maze_dataset.generation.default_generators import DEFAULT_GENERATORS

if __name__ == "__main__":
    run_benchmark(
        save_path="tests/_temp/benchmark_generation.jsonl",
        base_configs=DEFAULT_GENERATORS,
        grid_n_vals=[2, 3, 4, 5, 8, 10, 16, 25, 32],
        n_mazes_vals=list(range(1, 12, 2)),
        trials=10,
        verbose=True,
    )
