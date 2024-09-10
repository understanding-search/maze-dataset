"""`MazeDatasetConfig` is where you decide what your dataset should look like, then pass it to `MazeDataset.from_config` to generate or load the dataset.

see [demo_dataset notebook](../../notebooks/demo_dataset)

"""

import copy
import functools
import json
import multiprocessing
import typing
import warnings
from collections import Counter, defaultdict
from typing import Callable

import numpy as np
import tqdm
from jaxtyping import Int
from muutils.json_serialize import (
    JSONitem,
    json_serialize,
    serializable_dataclass,
    serializable_field,
)
from muutils.json_serialize.util import safe_getsource, string_as_lines
from muutils.misc import sanitize_fname, shorten_numerical_to_str, stable_hash
from zanj.loading import LoaderHandler, load_item_recursive, register_loader_handler

from maze_dataset.constants import Coord, CoordTup
from maze_dataset.dataset.dataset import (
    DatasetFilterProtocol,
    GPTDataset,
    GPTDatasetConfig,
    register_dataset_filter,
    register_filter_namespace_for_dataset,
)
from maze_dataset.generation.generators import GENERATORS_MAP
from maze_dataset.maze import LatticeMaze, SolvedMaze

# If `n_mazes>=SERIALIZE_MINIMAL_THRESHOLD`, then the MazeDataset will use `serialize_minimal`.
# Setting to None means that `serialize_minimal` will never be used.
# Set to -1 to make calls to `read` use `MazeDataset._load_legacy`. Used for profiling only.
SERIALIZE_MINIMAL_THRESHOLD: int | None = 100


def set_serialize_minimal_threshold(threshold: int | None) -> None:
    global SERIALIZE_MINIMAL_THRESHOLD
    SERIALIZE_MINIMAL_THRESHOLD = threshold


def _load_maze_ctor(maze_ctor_serialized: str | dict) -> Callable:
    "get the maze constructor from `GENERATORS_MAP`"
    if isinstance(maze_ctor_serialized, dict):
        # this is both the new and old version of the serialization
        return GENERATORS_MAP[maze_ctor_serialized["__name__"]]
    elif isinstance(maze_ctor_serialized, str):
        # this is a version I switched to for a while but now we are switching back
        warnings.warn(
            f"you are loading an old model/config in `_load_maze_ctor()`!!! this should not be happening, please report: "
            + "https://github.com/understanding-search/maze-dataset/issues/new"
        )
        return GENERATORS_MAP[maze_ctor_serialized]
    else:
        raise ValueError(
            f"maze_ctor_serialized is of type {type(maze_ctor_serialized)}, expected str or dict"
        )


EndpointKwargsType = dict[
    typing.Literal[
        "except_when_invalid",
        "allowed_start",
        "allowed_end",
        "deadend_start",
        "deadend_end",
    ],
    bool | None | list[tuple[int, int]],
]
"type hint for `MazeDatasetConfig.endpoint_kwargs`"


@serializable_dataclass(kw_only=True, properties_to_serialize=["grid_shape"])
class MazeDatasetConfig(GPTDatasetConfig):
    """config object which is passed to `MazeDataset.from_config` to generate or load a dataset"""

    grid_n: int

    # not comparing n_mazes is done primarily to avoid conflicts which happen during `from_config` when we have applied filters
    n_mazes: int = serializable_field(compare=False)

    maze_ctor: Callable = serializable_field(
        default=GENERATORS_MAP["gen_dfs"],
        serialization_fn=lambda gen_func: {
            "__name__": gen_func.__name__,
            "__module__": gen_func.__module__,
            "__doc__": string_as_lines(gen_func.__doc__),
            "source_code": safe_getsource(gen_func),
        },
        loading_fn=lambda data: _load_maze_ctor(data["maze_ctor"]),
        assert_type=False,  # TODO: check the type here once muutils supports checking Callable signatures
    )

    maze_ctor_kwargs: dict = serializable_field(
        default_factory=dict,
        serialization_fn=lambda kwargs: kwargs,
        loading_fn=lambda data: (
            dict()
            if data.get("maze_ctor_kwargs", None)
            is None  # this should handle the backwards compatibility
            else data["maze_ctor_kwargs"]
        ),
    )

    endpoint_kwargs: EndpointKwargsType = serializable_field(
        default_factory=dict,
        serialization_fn=lambda kwargs: kwargs,
        loading_fn=lambda data: (
            dict()
            if data.get("endpoint_kwargs", None)
            is None  # this should handle the backwards compatibility
            else {
                k: (
                    # bools and Nones are fine
                    v
                    if (isinstance(v, bool) or v is None)
                    # assume its a CoordList
                    else [tuple(x) for x in v]  # muutils/zanj saves tuples as lists
                )
                for k, v in data["endpoint_kwargs"].items()
            }
        ),
        assert_type=False,
    )

    @property
    def grid_shape(self) -> CoordTup:
        return (self.grid_n, self.grid_n)

    @property
    def grid_shape_np(self) -> Coord:
        return np.array(self.grid_shape)

    @property
    def max_grid_n(self) -> int:
        return max(self.grid_shape)

    def stable_hash_cfg(self) -> int:
        return stable_hash(json.dumps(self.serialize()))

    def to_fname(self) -> str:
        return sanitize_fname(
            f"{self.name}-g{self.grid_n}-n{shorten_numerical_to_str(self.n_mazes)}-a_{self.maze_ctor.__name__.removeprefix('gen_')}-h{self.stable_hash_cfg()%10**5}"
        )

    def summary(self) -> dict:
        """return a summary of the config"""
        # do we run this to make sure it doesn't error?
        super_summary: dict = super().summary()
        self_ser: dict = self.serialize()
        return dict(
            name=self.name,
            fname=self.to_fname(),
            sdc_hash=self.stable_hash_cfg(),
            seed=self.seed,
            seq_len_min=self.seq_len_min,
            seq_len_max=self.seq_len_max,
            applied_filters=self.applied_filters,
            grid_n=self_ser["grid_n"],
            n_mazes=self_ser["n_mazes"],
            maze_ctor_name=self_ser["maze_ctor"]["__name__"],
            maze_ctor_kwargs=self_ser["maze_ctor_kwargs"],
            endpoint_kwargs=self_ser["endpoint_kwargs"],
        )


def _generate_maze_helper(index: int) -> SolvedMaze:
    """helper function for generating mazes in parallel

    > [!CAUTION]
    > don't use this unless generating in parallel!
    """
    # TODO: don't use this unless generating in parallel!
    maze: LatticeMaze = _GLOBAL_WORKER_CONFIG.maze_ctor(
        grid_shape=_GLOBAL_WORKER_CONFIG.grid_shape_np,
        **_GLOBAL_WORKER_CONFIG.maze_ctor_kwargs,
    )
    solution = maze.generate_random_path(**_GLOBAL_WORKER_CONFIG.endpoint_kwargs)
    assert solution is not None, f"{solution = }"
    assert len(solution) > 0, f"{solution = }"
    assert isinstance(solution, np.ndarray), f"{solution = }"
    assert len(solution.shape) == 2, f"{solution = }"
    return SolvedMaze.from_lattice_maze(
        lattice_maze=maze,
        solution=solution,
    )


def _maze_gen_init_worker(config: MazeDatasetConfig):
    """special worker helper

    > [!CAUTION]
    > this makes the generation depend both on whether parallelism is used, and on the number of processes. this is bad!

    """
    # TODO
    global _GLOBAL_WORKER_CONFIG
    _GLOBAL_WORKER_CONFIG = config

    process_id: tuple[int] = multiprocessing.current_process()._identity
    if len(process_id) == 0:
        # no multiprocessing, seed was already set
        pass
    elif len(process_id) == 1:
        # multiprocessing, adjust seed based on process id
        # only set numpy seed, since we do not use other random gens
        np.random.seed(_GLOBAL_WORKER_CONFIG.seed + process_id[0])
    else:
        raise ValueError(
            f"unexpected process id: {process_id}\n{multiprocessing.Process()}"
        )


class MazeDataset(GPTDataset):
    """a maze dataset class. This is a collection of solved mazes, and should be initialized via `MazeDataset.from_config`"""

    def __init__(
        self,
        cfg: MazeDatasetConfig,
        mazes: typing.Sequence[SolvedMaze],
        generation_metadata_collected: dict | None = None,
    ) -> None:
        super().__init__()
        self.cfg: MazeDatasetConfig = cfg
        self.mazes: list[SolvedMaze] = list(mazes)
        self.generation_metadata_collected: dict | None = generation_metadata_collected

    def data_hash(self) -> int:
        return stable_hash(str(tuple([x.serialize() for x in self.mazes])))

    def __getitem__(self, i: int) -> SolvedMaze:
        return self.mazes[i]

    def __deepcopy__(self, memo) -> "MazeDataset":
        return MazeDataset.load(self._serialize_full())

    def as_tokens(
        self,
        maze_tokenizer,  # TODO: MazeTokenizer
        limit: int | None = None,
        join_tokens_individual_maze: bool = False,
    ) -> list[list[str]] | list[str]:
        """return the dataset as tokens according to the passed `maze_tokenizer`

        the `maze_tokenizer` should be either a `MazeTokenizer` or a `MazeTokenizerModular`

        if `join_tokens_individual_maze` is True, then the tokens of each maze are
        joined with a space, and the result is a list of strings.
        i.e.:

            >>> dataset.as_tokens(join_tokens_individual_maze=False)
            [["a", "b", "c"], ["d", "e", "f"]]
            >>> dataset.as_tokens(join_tokens_individual_maze=True)
            ["a b c", "d e f"]
        """
        output: list[list[str]] = [
            maze.as_tokens(maze_tokenizer) for maze in self.mazes[:limit]
        ]
        if join_tokens_individual_maze:
            return [" ".join(tokens) for tokens in output]
        else:
            return output

    def __len__(self) -> int:
        return len(self.mazes)

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, MazeDataset):
            return NotImplemented
        # TODO: compare hashes of data instead of the data itself?
        return self.cfg == other.cfg and self.mazes == other.mazes

    @classmethod
    def generate(
        cls,
        cfg: MazeDatasetConfig,
        gen_parallel: bool = False,
        pool_kwargs: dict | None = None,
        verbose: bool = False,
    ) -> "MazeDataset":
        """generate a maze dataset given a config and some generation parameters"""

        # avoid copying since we dont want to pickle the staticmethod, just load/serialize it to avoid modifying the original config
        cfg_cpy = MazeDatasetConfig.load(cfg.serialize())

        if pool_kwargs is None:
            pool_kwargs = dict()
        mazes: list[SolvedMaze] = list()
        maze_indexes: Int[np.int8, "maze_index"] = np.arange(cfg_cpy.n_mazes)

        solved_mazes: list[SolvedMaze]
        tqdm_kwargs: dict = dict(
            total=cfg_cpy.n_mazes,
            unit="maze",
            desc="generating & solving mazes",
            disable=not verbose,
        )
        # TODO: don't use the global unless generating in parallel!
        if gen_parallel:
            with multiprocessing.Pool(
                **pool_kwargs,
                initializer=_maze_gen_init_worker,
                initargs=(cfg_cpy,),
            ) as pool:
                solved_mazes = list(
                    tqdm.tqdm(
                        pool.imap(
                            _generate_maze_helper,
                            maze_indexes,
                        ),
                        **tqdm_kwargs,
                    )
                )
        else:
            _maze_gen_init_worker(cfg_cpy)
            solved_mazes = list(
                tqdm.tqdm(
                    map(
                        _generate_maze_helper,
                        maze_indexes,
                    ),
                    **tqdm_kwargs,
                )
            )
        # reset seed to default value
        np.random.seed(cfg_cpy.seed)

        return cls(
            cfg=cfg_cpy,
            mazes=solved_mazes,
        )

    @classmethod
    def download(cls, cfg: MazeDatasetConfig, **kwargs) -> "MazeDataset":
        raise NotImplementedError("not implemented yet")

    @classmethod
    def load(cls, data: JSONitem) -> "MazeDataset":
        """load from zanj/json"""
        if data["__format__"] == "MazeDataset:minimal":
            return cls._load_minimal(data)
        elif data["__format__"] == "MazeDataset:minimal_soln_cat":
            return cls._load_minimal_soln_cat(data)
        elif data["__format__"] == "MazeDataset":
            if (
                SERIALIZE_MINIMAL_THRESHOLD == -1
            ):  # Allow access to `_load_legacy` for profiling
                return cls._load_legacy(data)
            return cls._load_full(data)
        else:
            raise KeyError(
                f"`__format__` string {data['__format__']} is not a recognized `MazeDataset` format."
            )

    @classmethod
    def _load_full(cls, data: JSONitem) -> "MazeDataset":
        assert data["__format__"] == "MazeDataset"
        return cls(
            cfg=MazeDatasetConfig.load(data["cfg"]),
            mazes=load_item_recursive(data["mazes"], tuple()),
            generation_metadata_collected=data["generation_metadata_collected"],
        )

    @classmethod
    def _load_minimal(cls, data: JSONitem) -> "MazeDataset":
        assert data["__format__"] == "MazeDataset:minimal"
        return cls(
            cfg=MazeDatasetConfig.load(data["cfg"]),
            generation_metadata_collected=data["generation_metadata_collected"],
            mazes=[
                SolvedMaze(
                    clist,
                    soln[:slen, ...],
                )
                for clist, slen, soln in zip(
                    load_item_recursive(data["maze_connection_lists"], tuple()),
                    load_item_recursive(data["maze_solution_lengths"], tuple()),
                    load_item_recursive(data["maze_solutions"], tuple()),
                    # load_item_recursive(data["maze_endpoints"], tuple()),
                )
            ],
        )

    @classmethod
    def _load_minimal_soln_cat(cls, data: JSONitem) -> "MazeDataset":
        assert data["__format__"] == "MazeDataset:minimal_soln_cat"

        maze_solution_lengths = load_item_recursive(
            data["maze_solution_lengths"], tuple()
        )
        maze_solutions_concat = load_item_recursive(
            data["maze_solutions_concat"], tuple()
        )
        maze_solutions = np.split(
            maze_solutions_concat, np.cumsum(maze_solution_lengths)[:-1], axis=0
        )

        return cls(
            cfg=load_item_recursive(data["cfg"], tuple()),
            generation_metadata_collected=load_item_recursive(
                data["generation_metadata_collected"], tuple()
            ),
            mazes=[
                SolvedMaze(
                    connection_list=clist,
                    solution=soln,
                )
                for clist, soln in zip(
                    load_item_recursive(data["maze_connection_lists"], tuple()),
                    # load_item_recursive(data["maze_endpoints"], tuple()),
                    maze_solutions,
                )
            ],
        )

    @classmethod
    def _load_legacy(cls, data: JSONitem) -> "MazeDataset":
        """Legacy `load` method from <0.5.2. Used exclusively for profiling comparison."""
        assert data["__format__"] == "MazeDataset"
        return cls(
            **{
                key: load_item_recursive(data[key], tuple())
                for key in ["cfg", "mazes", "generation_metadata_collected"]
            }
        )

    def serialize(self) -> JSONitem:
        """serialize to zanj/json"""
        if (
            SERIALIZE_MINIMAL_THRESHOLD is not None
            and len(self) >= SERIALIZE_MINIMAL_THRESHOLD
        ):
            return self._serialize_minimal()
        return self._serialize_full()

    def _serialize_full(self) -> JSONitem:
        return {
            "__format__": "MazeDataset",
            "cfg": json_serialize(self.cfg),
            "mazes": json_serialize(self.mazes),
            "generation_metadata_collected": json_serialize(
                self.generation_metadata_collected
            ),
        }

    def _serialize_minimal(self) -> JSONitem:
        "alternate serialization where metadata is collected and mazes are stored in concatenated form"
        if self.generation_metadata_collected is None:
            filtered_meta = self.filter_by.collect_generation_meta()
        else:
            filtered_meta = self

        max_solution_len: int = max(m.solution.shape[0] for m in filtered_meta.mazes)
        n_mazes: int = len(filtered_meta.mazes)
        grid_n: int = filtered_meta.cfg.grid_n

        maze_connection_lists: np.ndarray = np.empty(
            (n_mazes, 2, grid_n, grid_n), dtype=np.bool_
        )
        # maze_endpoints: np.ndarray = np.empty((n_mazes, 2, 2), dtype=np.int8)
        maze_solution_lengths: np.ndarray = np.empty((n_mazes,), dtype=np.int32)
        maze_solutions: np.ndarray = np.empty(
            (n_mazes, max_solution_len, 2), dtype=np.int8
        )

        for idx, maze in enumerate(filtered_meta.mazes):
            maze_connection_lists[idx] = maze.connection_list
            # maze_endpoints[idx] = np.array([maze.start_pos, maze.end_pos])
            maze_solution_lengths[idx] = maze.solution.shape[0]
            maze_solutions[idx, : maze.solution.shape[0]] = maze.solution

        return dict(
            __format__="MazeDataset:minimal",
            cfg=json_serialize(filtered_meta.cfg),
            generation_metadata_collected=json_serialize(
                filtered_meta.generation_metadata_collected
            ),
            maze_connection_lists=maze_connection_lists,
            # maze_endpoints=maze_endpoints,
            maze_solution_lengths=maze_solution_lengths,
            maze_solutions=maze_solutions,
        )

    def _serialize_minimal_soln_cat(self) -> JSONitem:
        "alternate serialization where metadata is collected, and mazes and their solutions are stored in concatenated form"
        if self.generation_metadata_collected is None:
            filtered_meta = self.filter_by.collect_generation_meta()
        else:
            filtered_meta = self

        maze_solution_lengths: np.ndarray = np.array(
            [m.solution.shape[0] for m in filtered_meta.mazes],
            dtype=np.int32,
        )
        n_mazes: int = len(filtered_meta.mazes)
        grid_n: int = filtered_meta.cfg.grid_n
        total_solution_len: int = np.sum(maze_solution_lengths)

        maze_connection_lists: np.ndarray = np.empty(
            (n_mazes, 2, grid_n, grid_n), dtype=np.bool_
        )
        maze_endpoints: np.ndarray = np.empty((n_mazes, 2, 2), dtype=np.int8)
        maze_solutions_concat: np.ndarray = np.empty(
            (total_solution_len, 2), dtype=np.int8
        )

        solutions_running_idx: int = 0
        for idx, maze in enumerate(filtered_meta.mazes):
            maze_connection_lists[idx] = maze.connection_list
            maze_endpoints[idx] = np.array([maze.start_pos, maze.end_pos])
            soln_len: int = maze.solution.shape[0]
            maze_solution_lengths[idx] = soln_len
            maze_solutions_concat[
                solutions_running_idx : solutions_running_idx + soln_len
            ] = maze.solution
            solutions_running_idx += soln_len

        return dict(
            __format__="MazeDataset:minimal_soln_cat",
            cfg=json_serialize(filtered_meta.cfg),
            generation_metadata_collected=json_serialize(
                filtered_meta.generation_metadata_collected
            ),
            maze_connection_lists=maze_connection_lists,
            maze_endpoints=maze_endpoints,
            maze_solution_lengths=maze_solution_lengths,
            maze_solutions_concat=maze_solutions_concat,
        )

    def update_self_config(self):
        """update the config to match the current state of the dataset (number of mazes, such as after filtering)"""
        self.cfg.n_mazes = len(self.mazes)

    def custom_maze_filter(
        self,
        method: typing.Callable[[SolvedMaze], bool],
        **kwargs,
    ) -> "MazeDataset":
        """filter the dataset using a custom method"""
        output: MazeDataset = MazeDataset(
            cfg=copy.deepcopy(self.cfg),
            mazes=[m for m in self.mazes if method(m, **kwargs)],
        )
        output.cfg.applied_filters.append(
            {
                "name": f"__custom__:{method.__name__}",
                "kwargs": kwargs,
            }
        )
        output.update_self_config()
        return output


# register things with zanj
MazeDatasetConfig._dataset_class = property(lambda self: MazeDataset)
register_loader_handler(
    LoaderHandler(
        check=lambda json_item, path=None, z=None: (
            isinstance(json_item, typing.Mapping)
            and "__format__" in json_item
            and json_item["__format__"].startswith("MazeDataset")
        ),
        load=lambda json_item, path=None, z=None: MazeDataset.load(json_item),
        uid="MazeDataset",
        source_pckg="maze_dataset.generation.maze_dataset",
        desc="MazeDataset",
    )
)


def register_maze_filter(
    method: typing.Callable[[SolvedMaze, typing.Any], bool]
) -> DatasetFilterProtocol:
    """register a maze filter, casting it to operate over the whole list of mazes

    method should be a staticmethod of a namespace class registered with `register_filter_namespace_for_dataset`

    this is a more restricted version of `register_dataset_filter` that removes the need for boilerplate for operating over the arrays
    """

    @functools.wraps(method)
    def wrapper(dataset: MazeDataset, *args, **kwargs):
        # copy and filter
        new_dataset: MazeDataset = copy.deepcopy(
            MazeDataset(
                cfg=dataset.cfg,
                mazes=[m for m in dataset.mazes if method(m, *args, **kwargs)],
            )
        )
        # update the config
        new_dataset.cfg.applied_filters.append(
            dict(name=method.__name__, args=args, kwargs=kwargs)
        )
        new_dataset.update_self_config()
        return new_dataset

    return wrapper


@register_filter_namespace_for_dataset(MazeDataset)
class MazeDatasetFilters:
    "namespace for filters for `MazeDataset`s"

    @register_maze_filter
    @staticmethod
    def path_length(maze: SolvedMaze, min_length: int) -> bool:
        """filter out mazes with a solution length less than `min_length`"""
        return len(maze.solution) >= min_length

    @register_maze_filter
    @staticmethod
    def start_end_distance(maze: SolvedMaze, min_distance: int) -> bool:
        """filter out datasets where the start and end pos are less than `min_distance` apart on the manhattan distance (ignoring walls)"""
        return np.linalg.norm(maze.start_pos - maze.end_pos, 1) >= min_distance

    @register_dataset_filter
    @staticmethod
    def cut_percentile_shortest(
        dataset: MazeDataset,
        percentile: float = 10.0,
    ) -> MazeDataset:
        """cut the shortest `percentile` of mazes from the dataset

        `percentile` is 1-100, not 0-1, as this is what `np.percentile` expects
        """
        lengths: np.ndarray = np.array([len(m.solution) for m in dataset])
        cutoff: int = int(np.percentile(lengths, percentile))

        filtered_mazes: list[SolvedMaze] = [
            m for m in dataset if len(m.solution) > cutoff
        ]
        new_dataset: MazeDataset = MazeDataset(cfg=dataset.cfg, mazes=filtered_mazes)

        return copy.deepcopy(new_dataset)

    @register_dataset_filter
    @staticmethod
    def truncate_count(
        dataset: MazeDataset,
        max_count: int,
    ) -> MazeDataset:
        """truncate the dataset to be at most `max_count` mazes"""
        new_dataset: MazeDataset = MazeDataset(
            cfg=dataset.cfg, mazes=dataset.mazes[:max_count]
        )
        return copy.deepcopy(new_dataset)

    @register_dataset_filter
    @staticmethod
    def remove_duplicates(
        dataset: MazeDataset,
        minimum_difference_connection_list: int | None = 1,
        minimum_difference_solution: int | None = 1,
        _max_dataset_len_threshold: int = 1000,
    ) -> MazeDataset:
        """remove duplicates from a dataset, keeping the **LAST** unique maze

        set minimum either minimum difference to `None` to disable checking

        if you want to avoid mazes which have more overlap, set the minimum difference to be greater

        Gotchas:
        - if two mazes are of different sizes, they will never be considered duplicates
        - if two solutions are of different lengths, they will never be considered duplicates
            TODO: check for overlap?
        """
        if len(dataset) > _max_dataset_len_threshold:
            raise ValueError(
                "this method is currently very slow for large datasets, consider using `remove_duplicates_fast` instead\n",
                "if you know what you're doing, change `_max_dataset_len_threshold`",
            )

        unique_mazes: list[SolvedMaze] = list()

        maze_a: SolvedMaze
        maze_b: SolvedMaze
        for i, maze_a in enumerate(dataset.mazes):
            a_unique: bool = True
            for maze_b in dataset.mazes[i + 1 :]:
                # after all that nesting, more nesting to perform checks
                if (minimum_difference_connection_list is not None) and (
                    maze_a.connection_list.shape == maze_b.connection_list.shape
                ):
                    if (
                        np.sum(maze_a.connection_list != maze_b.connection_list)
                        <= minimum_difference_connection_list
                    ):
                        a_unique = False
                        break

                if (minimum_difference_solution is not None) and (
                    maze_a.solution.shape == maze_b.solution.shape
                ):
                    if (
                        np.sum(maze_a.solution != maze_b.solution)
                        <= minimum_difference_solution
                    ):
                        a_unique = False
                        break

            if a_unique:
                unique_mazes.append(maze_a)

        return copy.deepcopy(
            MazeDataset(
                cfg=dataset.cfg,
                mazes=unique_mazes,
                generation_metadata_collected=dataset.generation_metadata_collected,
            )
        )

    @register_dataset_filter
    @staticmethod
    def remove_duplicates_fast(dataset: MazeDataset) -> MazeDataset:
        """remove duplicates from a dataset"""

        unique_mazes = list(dict.fromkeys(dataset.mazes))
        return copy.deepcopy(
            MazeDataset(
                cfg=dataset.cfg,
                mazes=unique_mazes,
                generation_metadata_collected=dataset.generation_metadata_collected,
            )
        )

    @register_dataset_filter
    @staticmethod
    def strip_generation_meta(dataset: MazeDataset) -> MazeDataset:
        """strip the generation meta from the dataset"""
        new_dataset: MazeDataset = copy.deepcopy(dataset)
        for maze in new_dataset:
            # hacky because it's a frozen dataclass
            maze.__dict__["generation_meta"] = None
        return new_dataset

    @register_dataset_filter
    @staticmethod
    def collect_generation_meta(
        dataset: MazeDataset,
        clear_in_mazes: bool = True,
        inplace: bool = True,
        allow_fail: bool = False,
    ) -> MazeDataset:
        if dataset.generation_metadata_collected is not None:
            return dataset
        else:
            assert (
                dataset[0].generation_meta is not None
            ), "generation meta is not collected and original is not present"
        # if the generation meta is already collected, don't collect it again, do nothing

        new_dataset: MazeDataset
        if inplace:
            new_dataset = dataset
        else:
            new_dataset = copy.deepcopy(dataset)

        gen_meta_lists: dict[bool | int | float | str | CoordTup, Counter] = (
            defaultdict(Counter)
        )
        for maze in new_dataset:
            if maze.generation_meta is None:
                if allow_fail:
                    break
                else:
                    raise ValueError(
                        "generation meta is not present in a maze, cannot collect generation meta"
                    )
            for key, value in maze.generation_meta.items():
                if isinstance(value, (bool, int, float, str)):
                    gen_meta_lists[key][value] += 1

                elif isinstance(value, set):
                    # special case for visited_cells
                    gen_meta_lists[key].update(value)

                elif isinstance(value, (list, np.ndarray)):
                    if isinstance(value, list):
                        try:
                            value = np.array(value)
                        except ValueError:
                            raise ValueError(
                                f"Cannot collect generation meta for {key} as it is a list of type '{str(type(value[0])) = }'",
                                "expected either a basic type (bool, int, float, str), a numpy coord, or a numpy array of coords",
                            )

                    if (len(value.shape) == 1) and (value.shape[0] == maze.lattice_dim):
                        # assume its a single coordinate
                        gen_meta_lists[key][tuple(value)] += 1
                    elif (len(value.shape) == 2) and (
                        value.shape[1] == maze.lattice_dim
                    ):
                        # assume its a list of coordinates
                        gen_meta_lists[key].update([tuple(v) for v in value])
                    else:
                        raise ValueError(
                            f"Cannot collect generation meta for {key} as it is an ndarray of shape {value.shape}",
                            "expected either a coord of shape (2,) or a list of coords of shape (n, 2)",
                        )
                else:
                    raise ValueError(
                        f"Cannot collect generation meta for {key} as it is of type '{str(type(value))}'",
                        "expected either a basic type (bool, int, float, str), a numpy coord, or a numpy array of coords",
                    )

            # clear the data
            if clear_in_mazes:
                # hacky because it's a frozen dataclass
                maze.__dict__["generation_meta"] = None

        new_dataset.generation_metadata_collected = {
            key: dict(value) for key, value in gen_meta_lists.items()
        }

        return new_dataset

        # the code below is for doing some smarter collecting and type checking. Probably will delete.
        """
        collect either the type at the field, or the shape of the field if it is an array
        metadata_types: dict[str, set[type, tuple]] = dict()
        for maze in new_dataset:
            for key, value in maze.generation_meta.items():
                if key not in metadata_types:
                    metadata_types[key] = set()

                if isinstance(value, np.ndarray):
                    metadata_types[key].add(value.shape)
                else:
                    metadata_types[key].add(type(value))

        # figure out what to do for each field
        metadata_actions: dict[str, typing.Callable] = dict()
        for key, key_type in metadata_types.items():
            if all(isinstance(kt, tuple) for kt in key_type):
                if all(kt == (2,) for kt in key_type):
                    # its all coords, do a statcounter on those coords
                    metadata_actions[key] = lambda vals: Counter(tuple(x) for x in vals)
                elif all(
                    (len(kt) == 2) and (kt[1] == 2) 
                    for kt in key_type
                ):
                    # its a list of coords, do a statcounter on those coords
                    metadata_actions[key] = lambda vals: Counter(
                        tuple(x) for x in np.concatenate(vals)
                    )
                else:
                    # its a list of something else, do a counter on those
                    # TODO: throw except here?
                    metadata_actions[key] = Counter
                    
            elif all(kt in (bool, int, float) for kt in key_type):
                # statcounter for numeric types
                metadata_actions[key] = StatCounter
            elif all(kt == str for kt in key_type):
                # counter for string types
                metadata_actions[key] = Counter
            else:
                # counter for everything else
                # TODO: throw except here?
                metadata_actions[key] = Counter
        """
