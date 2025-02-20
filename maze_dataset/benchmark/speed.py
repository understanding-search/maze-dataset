import random
import timeit
from typing import Sequence

from tqdm import tqdm

from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation.default_generators import DEFAULT_GENERATORS
from maze_dataset.generation.generators import GENERATORS_MAP

_BASE_CFG_KWARGS: dict = dict(
    grid_n=None,
    n_mazes=None,
)

_GENERATE_KWARGS: dict = dict(
    gen_parallel=False,
    pool_kwargs=None,
    verbose=False,
    # do_generate = True,
    # load_local = False,
    # save_local = False,
    # zanj = None,
    # do_download = False,
    # local_base_path = "INVALID",
    # except_on_config_mismatch = True,
    # verbose = False,
)


def time_generation(
    base_configs: list[tuple[str, dict]],
    grid_n_vals: list[int],
    n_mazes_vals: list[int],
    trials: int = 10,
    verbose: bool = False,
) -> dict[str, float]:
    # assemble configs
    configs: list[MazeDatasetConfig] = list()

    for cfg in base_configs:
        for grid_n in grid_n_vals:
            for n_mazes in n_mazes_vals:
                configs.append(
                    MazeDatasetConfig(
                        name="benchmark",
                        grid_n=grid_n,
                        n_mazes=n_mazes,
                        maze_ctor=GENERATORS_MAP[cfg[0]],
                        maze_ctor_kwargs=cfg[1],
                    )
                )

    # shuffle configs (in place) (otherwise progress bar is annoying)
    random.shuffle(configs)

    # time generation for each config
    times: list[dict] = list()
    idx: int = 0
    total: int = len(configs)
    for cfg in tqdm(
        configs,
        desc="Timing generation",
        unit="config",
        total=total,
        disable=verbose,
    ):
        if verbose:
            print(f"Timing generation for config {idx}/{total}\n{cfg}")

        t: float = (
            timeit.timeit(
                stmt=lambda: MazeDataset.generate(cfg, **_GENERATE_KWARGS),
                number=trials,
            )
            / trials
        )

        if verbose:
            print(f"avg time: {t:.3f} s")

        times.append(
            dict(
                cfg_name=cfg.name,
                grid_n=cfg.grid_n,
                n_mazes=cfg.n_mazes,
                maze_ctor=cfg.maze_ctor.__name__,
                maze_ctor_kwargs=cfg.maze_ctor_kwargs,
                trials=trials,
                time=t,
            )
        )

        idx += 1

    return times


def run_benchmark(
    save_path: str,
    base_configs: list[tuple[str, dict]] | None = None,
    grid_n_vals: Sequence[int] = tuple([2, 3, 4, 5, 8, 10, 16, 25, 32]),
    n_mazes_vals: Sequence[int] = tuple(list(range(1, 12, 2))),
    trials: int = 10,
    verbose: bool = True,
):
    import pandas as pd

    if base_configs is None:
        base_configs = DEFAULT_GENERATORS

    times: list[dict] = time_generation(
        base_configs=base_configs,
        grid_n_vals=grid_n_vals,
        n_mazes_vals=n_mazes_vals,
        trials=trials,
        verbose=verbose,
    )

    df: pd.DataFrame = pd.DataFrame(times)

    # print the whole dataframe contents to console as csv
    print(df.to_csv())

    # save to file
    df.to_json(save_path, orient="records", lines=True)

    return df