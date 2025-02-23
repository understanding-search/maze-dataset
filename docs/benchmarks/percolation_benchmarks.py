from pathlib import Path
from typing import Any

SAVE_DIR: Path = Path("docs/benchmarks/percolation_fractions/")


ANALYSIS_KWARGS: dict[str, dict[str, Any]] = dict(
    test=dict(
        n_mazes=16,
        p_val_count=16,
        grid_sizes=[2, 4],
    ),
    small=dict(
        n_mazes=64,
        p_val_count=25,
        grid_sizes=[2, 3, 4, 5, 6],
    ),
    large=dict(
        n_mazes=2048,
        p_val_count=200,
        grid_sizes=[2, 3, 4, 5, 6, 8, 10, 16, 25, 32, 50, 64],
    ),
)

if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "analysis",
        type=str,
        choices=ANALYSIS_KWARGS.keys(),
        help="The analysis to run",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=False,
        help="The number of parallel processes to use",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default=SAVE_DIR.as_posix(),
        help="The directory to save the results",
    )
    args: argparse.Namespace = parser.parse_args()
    kwargs: dict[str, Any] = ANALYSIS_KWARGS[args.analysis]
    save_dir: Path = Path(args.save_dir)

    # import here for speed
    from maze_dataset.benchmark.config_sweep import (
        SweepResult,
        full_percolation_analysis,
        plot_grouped,
    )

    # run analysis and save
    results: SweepResult = full_percolation_analysis(
        **kwargs,
        parallel=args.parallel,
        save_dir=save_dir,
    )

    # plot results
    plot_grouped(
        results,
        save_dir=save_dir,
        show=False,
    )
