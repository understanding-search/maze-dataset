"""Benchmarking of how successful maze generation is for various values of percolation

"""


import functools
import json
from typing import Any, Callable, Sequence, TypeVar
from pathlib import Path

import numpy as np
from jaxtyping import Float
import matplotlib.pyplot as plt
from tqdm import tqdm
from muutils.json_serialize import serializable_dataclass, SerializableDataclass, serializable_field
from muutils.dictmagic import dotlist_to_nested_dict, update_with_nested_dict
from muutils.parallel import run_maybe_parallel

from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators

SweepReturnType = TypeVar("SweepReturnType")
ParamType = TypeVar("ParamType")
AnalysisFunc = Callable[[MazeDatasetConfig], SweepReturnType]

def dataset_success_fraction(cfg: MazeDatasetConfig) -> float:
    dataset: MazeDataset = MazeDataset.from_config(
        cfg,
        do_download=False,
        load_local=False,
        save_local=False,
        verbose=False,
    )

    return len(dataset) / cfg.n_mazes

ANALYSIS_FUNCS: dict[str, AnalysisFunc] = dict(
    dataset_success_fraction=dataset_success_fraction,
)

def sweep(
    cfg_base: MazeDatasetConfig,
    param_values: list[ParamType],
    param_key: str,
    analyze_func: Callable[[MazeDatasetConfig], SweepReturnType],
) -> list[SweepReturnType]:
    
    outputs: list[SweepReturnType] = []

    for p in param_values:
        # update the config
        cfg_dict: dict = cfg_base.serialize()
        update_with_nested_dict(
            cfg_dict,
            dotlist_to_nested_dict({param_key: p}),
        )
        cfg_test: MazeDatasetConfig = MazeDatasetConfig.load(cfg_dict)

        outputs.append(analyze_func(cfg_test))

    return outputs



@serializable_dataclass
class SweepResult(SerializableDataclass):
    configs: list[MazeDatasetConfig]
    param_values: list[ParamType]
    result_values: dict[str, list[SweepReturnType]]
    param_key: str
    analyze_func: Callable[[MazeDatasetConfig], SweepReturnType] = serializable_field(
        serialization_fn=lambda f: f.__name__,
        deserialize_fn=ANALYSIS_FUNCS.get,
    )

    def configs_by_name(self) -> dict[str, MazeDatasetConfig]:
        "return configs by name"
        return {cfg.name: cfg for cfg in self.configs}

    def configs_by_key(self) -> dict[str, MazeDatasetConfig]:
        "return configs by the key used in `result_values`, which is the filename of the config"
        return {cfg.to_fname(): cfg for cfg in self.configs}

    def configs_shared(self) -> dict[str, Any]:
        "return key: value pairs that are shared across all configs"
        # we know that the configs all have the same keys, 
        # so this way of doing it is fine
        config_vals: dict[str, set[Any]] = dict()
        for cfg in self.configs:
            for k, v in cfg.serialize().items():
                if k not in config_vals:
                    config_vals[k] = set()
                config_vals[k].add(json.dumps(v))

        shared_vals: dict[str, Any] = dict()

        cfg_ser: dict = self.configs[0].serialize()
        for k, v in config_vals.items():
            if len(v) == 1:
                shared_vals[k] = cfg_ser[k]

        return shared_vals
    
    def configs_differing_keys(self) -> set[str]:
        "return keys that differ across configs"
        shared_vals: dict[str, Any] = self.configs_shared()
        differing_keys: set[str] = set()

        for k in MazeDatasetConfig.__dataclass_fields__.keys():
            if k not in shared_vals:
                differing_keys.add(k)

        return differing_keys

    def get_where(self, key: str, val_check: Callable[[Any], bool]) -> "SweepResult":
        "get a subset of this `Result` where the configs has `key` satisfying `val_check`"
        configs_list: list[MazeDatasetConfig] = [
            cfg for cfg in self.configs if val_check(getattr(cfg, key))
        ]
        configs_keys: set[str] = {cfg.to_fname() for cfg in configs_list}
        result_values: dict[str, np.ndarray] = {
            k: self.result_values[k] for k in configs_keys
        }

        return SweepResult(
            configs=configs_list,
            param_values=self.param_values,
            result_values=result_values,
            param_key=self.param_key,
            analyze_func=self.analyze_func,
        )


    @classmethod
    def analyze(
        cls,
        configs: list[MazeDatasetConfig],
        param_values: list[ParamType],
        param_key: str,
        analyze_func: Callable[[MazeDatasetConfig], SweepReturnType],
        parallel: bool|int = False,
    ) -> "Result":
        """Analyze success rate of maze generation for different percolation values

        # Parameters:
        - `configs : list[MazeDatasetConfig]`
        configs to try
        - `param_values : np.ndarray`
        numpy array of values to try

        # Returns:
        - `Result`
        """
        n_pvals: int = len(param_values)

        result_values_list: list[float] = run_maybe_parallel(
            func = functools.partial(
                sweep,
                param_values=param_values,
                param_key=param_key, #"maze_ctor_kwargs.p",
                analyze_func=analyze_func, #dataset_success_fraction,
            ),
            iterable=configs,
            keep_ordered=True,
            parallel=parallel,
            pbar_kwargs=dict(total=len(configs)),
        )
        result_values: dict[str, Float[np.ndarray, "n_pvals"]] = {
            cfg.to_fname(): np.array(res)
            for cfg, res in zip(configs, result_values_list)
        }
        return cls(
            configs=configs,
            param_values=param_values,
            result_values=result_values,
            param_key=param_key,
            analyze_func=analyze_func,
        )

    def plot(
        self,
        save_path: str = None,
        cfg_keys: list[str] | None = None,
        cmap_name: str | None = "viridis",
        plot_only: bool = False,
        show: bool = True,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot the results of percolation analysis"""
        # set up figure
        if not ax:
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = plt.subplots(1, 1, figsize=(22, 10))

        # plot
        cmap = plt.get_cmap(cmap_name)
        n_cfgs: int = len(self.result_values)
        for i, (ep_cfg_name, result_values) in enumerate(self.result_values.items()):
            ax.plot(
                self.param_values,
                result_values,
                ".-",
                label=self.configs_by_key()[ep_cfg_name].name,
                color=cmap((i + 0.5) / (n_cfgs - 0.5)),
            )

        # repr of config
        cfg_shared: dict = self.configs_shared()
        cfg_repr: str = (
            str(cfg_shared)
            if cfg_keys is None
            else (
                "MazeDatasetConfig("
                + ", ".join(
                    [
                        f"{k}={cfg_shared[k].__name__}"
                        if isinstance(cfg_shared[k], Callable)
                        else f"{k}={cfg_shared[k]}"
                        for k in cfg_keys
                    ]
                )
                + ")"
            )
        )

        # add title and stuff
        if not plot_only:
            ax.set_xlabel(self.param_key)
            ax.set_ylabel(self.analyze_func.__name__)
            ax.set_title(f"{self.param_key} vs {self.analyze_func.__name__}\ncfg_repr")
            ax.grid(True)
            ax.legend(loc="lower center")

        # save and show
        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()

        return ax




DEFAULT_ENDPOINT_KWARGS: list[tuple[str, dict]] = [
    (
        "any",
        dict(deadend_start=False, deadend_end=False, except_on_no_valid_endpoint=False),
    ),
    (
        "deadends",
        dict(deadend_start=True, deadend_end=True, except_on_no_valid_endpoint=False),
    ),
    (
        "deadends_unique",
        dict(
            deadend_start=True,
            deadend_end=True,
            endpoints_not_equal=False,
            except_on_no_valid_endpoint=False,
        ),
    ),
]

def full_analysis(
    n_mazes: int,
    p_val_count: int,
    grid_sizes: list[int],
    ep_kwargs: list[tuple[str, dict]] | None = None,
    generators: Sequence[Callable] = (
        LatticeMazeGenerators.gen_percolation,
        LatticeMazeGenerators.gen_dfs_percolation,
    ),
    save_dir: Path = Path("../docs/benchmarks/percolation_fractions"),
) -> None:
    if ep_kwargs is None:
        ep_kwargs = DEFAULT_ENDPOINT_KWARGS

    total_cfgs: int = len(generators) * len(ep_kwargs)
    cfg_idx: int = 0

    # for i, ((ep_kw_name, ep_kw), gen_func) in enumerate(cfg_augment_product):
    for ep_kw_name, ep_kw in ep_kwargs:
        fig, ax = plt.subplots(1, 1, figsize=(22, 10))
        for gf_idx, gen_func in enumerate(generators):
            cfg_idx += 1
            print(
                f"\n\n# Analyzing {cfg_idx}/{total_cfgs}: endpoint_kwargs '{ep_kw_name}', gen_func={gen_func.__name__}\n\n"
            )
            result: SweepResult = SweepResult.analyze(
                configs=[
                    MazeDatasetConfig(
                        name=f"g{grid_n}-{gen_func.__name__.removeprefix('gen_').removesuffix('olation')}",
                        grid_n=grid_n,
                        n_mazes=n_mazes,
                        maze_ctor=gen_func,
                        maze_ctor_kwargs=dict(),
                        endpoint_kwargs=ep_kw,
                    )
                    for grid_n in grid_sizes
                ],
                param_values=np.linspace(0.0, 1.0, p_val_count),
            )
            ax = result.plot(
                cfg_keys=["n_mazes", "maze_ctor", "endpoint_kwargs"],
                ax=ax,
                show=False,
                cmap_name="Reds" if gf_idx == 0 else "Blues",
            )
        # save the figure
        plt.savefig(save_dir / f"n_{n_mazes}-pvc_{p_val_count}-ep_{ep_kw_name}.svg")
        plt.show()