"""`GPTDatasetConfig` and `GPTDataset` are base classes for datasets
they implement some basic functionality, saving/loading, the `from_config` pipeline, and filtering

> [!NOTE]
> these should probably be moved into a different package, so don't rely on them being here
"""

import functools
import json
import typing
import warnings
from pathlib import Path
from typing import Callable, Type

import numpy as np
import torch
from muutils.json_serialize import (
    JSONitem,
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)
from muutils.misc import sanitize_fname, shorten_numerical_to_str, stable_hash
from muutils.mlutils import DEFAULT_SEED, GLOBAL_SEED, set_reproducibility
from muutils.tensor_utils import DTYPE_MAP
from torch.utils.data import Dataset
from zanj import ZANJ


class FilterInfoMismatchError(ValueError):
    """raised when the filter info in a dataset config does not match the filter info in the dataset"""

    pass


def _dtype_serialization_fn(datatype: torch.dtype | np.dtype) -> str:
    """convert torch dtype to string, while checking that the conversion is reversible"""
    x_str: str = str(datatype)
    assert x_str in DTYPE_MAP, f"unknown dtype {datatype}"
    assert DTYPE_MAP[x_str] == datatype
    return x_str


def _load_applied_filters(
    filters: list[dict[typing.Literal["name", "args", "kwargs"], str | list | dict]],
) -> list[dict[typing.Literal["name", "args", "kwargs"], str | list | dict]]:
    try:
        return [
            dict(
                name=filter_info["name"],
                args=tuple(
                    filter_info["args"]
                ),  # muutils/zanj save tuples as lists, and this causes problems
                kwargs=dict(filter_info["kwargs"]),
            )
            for filter_info in filters
        ]
    except Exception as e:
        raise ValueError(f"failed to load applied filters:\n{filters}") from e


@serializable_dataclass(kw_only=True)
class GPTDatasetConfig(SerializableDataclass):
    """base GPTDatasetConfig class"""

    name: str

    # TODO: get rid of all these things as part of migration to tokenizer-free dataset config
    # --------------------------------------------------
    seq_len_min: int = serializable_field(default=1)
    seq_len_max: int = serializable_field(default=512)
    # --------------------------------------------------

    seed: int | None = serializable_field(default=DEFAULT_SEED)
    applied_filters: list[
        dict[typing.Literal["name", "args", "kwargs"], str | list | dict]
    ] = serializable_field(
        default_factory=list,
        deserialize_fn=_load_applied_filters,
        assert_type=False,  # TODO: check the type here once muutils supports checking Callable signatures
    )

    def __post_init__(self):
        assert self.seq_len_min <= self.seq_len_max
        # if seed set to None, then generate a new random seed
        if self.seed is None:
            self.seed = torch.random.seed() % 2**31

        if (DEFAULT_SEED != self.seed) and (GLOBAL_SEED != self.seed):
            warnings.warn(
                f"in GPTDatasetConfig {self.name=}, {self.seed=} is trying to override {GLOBAL_SEED=} which has already been changed elsewhere from {DEFAULT_SEED=}"
            )

        set_reproducibility(self.seed)

    def summary(self) -> dict:
        """return a summary of the config"""
        # do we run this to make sure it doesn't error?
        self_ser: dict = self.serialize()
        return dict(
            name=self.name,
            seq_len_min=self.seq_len_min,
            seq_len_max=self.seq_len_max,
            seed=self.seed,
            applied_filters=self.applied_filters,
        )

    @classmethod
    @property
    def _dataset_class(cls) -> type:
        raise NotImplementedError("this should be implemented by subclasses!")

    def to_fname(self) -> str:
        """convert config to a filename"""
        self_json_str: str = json.dumps(self.serialize())
        self_json_hash: int = int(abs(stable_hash(self_json_str)) % 1e10)
        warnings.warn(
            f"using fallblack to_fname() method for {self.__class__.__name__}, this should be implemented by subclasses!"
        )
        return sanitize_fname(
            f"f{self.name}-n{shorten_numerical_to_str(len(self))}-h{self_json_hash}"
        )


def _dataset_config_load(*args, **kwargs) -> "GPTDatasetConfig":
    raise NotImplementedError(
        f"this `load` function should be implemented by subclasses! got: {args=}, {kwargs=}"
    )


def _dataset_config_serialize(self, *args, **kwargs) -> JSONitem:
    raise NotImplementedError(
        f"this `serialize` function should be implemented by subclasses! got: {args=}, {kwargs=}"
    )


GPTDatasetConfig.load = _dataset_config_load
GPTDatasetConfig.serialize = _dataset_config_serialize


class GPTDataset(Dataset):
    """wrapper for torch dataset with some extra functionality

    (meaning the functionality should be inherited in downstream classes)

    > [!NOTE]
    > `GPTDatasetConfig` should implement a `to_fname` method that returns a unique filename for the config

    # Requires:
    the following methods should be implemented in subclasses:
        - `__init__(self, cfg: GPTDatasetConfig, **kwargs)`
        initialize the dataset from a given config. kwargs are not passed through, the kwargs should take the actual generated or loaded data (a list of objects or sequences probably)
        - `generate(cls, cfg: GPTDatasetConfig, **kwargs) -> GPTDataset`
        generate the dataset from a given config. kwargs are passed through from `from_config`, and should only contain things that dont belong in the config (i.e. how many threads to use for generation)
        - `serialize(self) -> JSONitem`
        serialize the dataset to a ZANJ-serializable object, including:
            - config
            - data in formats specified by `self.save_formats`
        - `load(cls, data: JSONitem) -> GPTDataset`
        load the dataset from a ZANJ-serializable object
        - `download(cls, cfg: GPTDatasetConfig, **kwargs) -> GPTDataset`
        given a config, try to download a dataset from some source. kwargs are passed through from `from_config`, and should only contain things that dont belong in the config (i.e. some kind of auth token or source url)
        - `__len__(self) -> int`
        return the length of the dataset, required for `torch.utils.data.Dataset`
        - `__getitem__(self, i: int) -> list[str]`
        return the ith item in the dataset, required for `torch.utils.data.Dataset`
        return the ith item in the dataset, required for `torch.utils.data.Dataset`
        -  `update_self_config(self) -> None`
        update the config of the dataset to match the current state of the dataset, used primarily in filtering and validation
        -  decorating the appropriate filter namespace with `register_filter_namespace_for_dataset(your_dataset_class)` if you want to use filters

    # Parameters:
        - `cfg : GPTDatasetConfig`
        config for the dataset, used to generate the dataset
        - `do_generate : bool`
        whether to generate the dataset if it isn't found
        (defaults to `True`)
        - `load_local : bool`
        whether to try finding the dataset locally
        (defaults to `True`)
        - `save_local : bool`
        whether to save the dataset locally if it is generated or downloaded
        (defaults to `True`)
        - `do_download : bool`
        whether to try downloading the dataset
        (defaults to `True`)
        - `local_base_path : Path`
        where to save the dataset
        (defaults to `Path("data/maze_dataset")`)

    # Returns:
        - `GPTDataset`
        the dataset, as you wanted it

    # Implements:
        - `save(self, file_path: str) -> None`
        save the dataset to a file, using ZANJ
        - `read(cls, file_path: str) -> GPTDataset`
        read the dataset from a file, using ZANJ
        get all items in the dataset, in the specified format
        - `filter_by(self)`
        returns a namespace class
        -  `_filter_namespace(self) -> Class`
        returns a namespace class for filtering the dataset, checking that method
        - `_apply_filters_from_config(self) -> None`
        apply filters to the dataset, as specified in the config. used in `from_config()` but only when generating

    """

    _FILTER_NAMESPACE: type = "this isn't a filter namespace! you have to initialize this by registering with `register_filter_namespace_for_dataset`"  # type: ignore

    @classmethod
    def from_config(
        cls,
        cfg: GPTDatasetConfig,
        do_generate: bool = True,
        load_local: bool = True,
        save_local: bool = True,
        zanj: ZANJ | None = None,
        do_download: bool = True,
        local_base_path: Path = Path("data/maze_dataset"),
        except_on_config_mismatch: bool = True,
        allow_generation_metadata_filter_mismatch: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> "GPTDataset":
        """base class for gpt datasets

        priority of loading:
        1. load from local
        2. download
        3. generate

        """

        print_log: Callable = print if verbose else lambda *_a, **_kw: None

        local_base_path = Path(local_base_path)
        fname: Path = Path(f"{cfg.to_fname()}.zanj")
        output: GPTDataset | None = None
        did_load_local: bool = False
        if zanj is None:
            zanj = ZANJ()

        print_log(f"trying to get the dataset '{cfg.to_fname()}'")

        if not (load_local or do_download or do_generate):
            raise ValueError(
                "no way to load dataset! you said not to load local, not to download, and not to generate"
            )

        dataset_path: Path = local_base_path / fname

        # try loading
        if load_local:
            if dataset_path.exists():
                print_log(f"loading dataset from {dataset_path.as_posix()}")
                try:
                    output = cls.read(dataset_path, zanj=zanj)
                    did_load_local = True
                    print_log("load successful!")
                except Exception as e:
                    print_log(f"failed to load dataset: {e}")

        if do_download and output is None:
            print_log("seeing if we can download the dataset...")
            try:
                output = cls.download(cfg, **kwargs)
                print_log("download successful!")
            except NotImplementedError:
                print_log("no download found, or download failed")

        if do_generate and output is None:
            print_log("generating dataset...")
            output = cls.generate(cfg, verbose=verbose, **kwargs)
            # only if we generated it, apply filters
            output = output._apply_filters_from_config()

        # check and save
        if output is None:
            raise ValueError("failed to load dataset!")

        cfg_diff: dict = cfg.diff(output.cfg, of_serialized=True)
        if cfg_diff:
            if except_on_config_mismatch:
                if allow_generation_metadata_filter_mismatch and (
                    cfg_diff
                    == {
                        "applied_filters": {
                            "self": [],
                            "other": [
                                {
                                    "name": "collect_generation_meta",
                                    "args": (),
                                    "kwargs": {},
                                }
                            ],
                        }
                    }
                ):
                    pass
                else:
                    raise ValueError(f"config mismatch: {cfg_diff = }")
            else:
                warnings.warn(f"config mismatch: {cfg_diff = }")

        if save_local and not did_load_local:
            print_log(f"saving dataset to {dataset_path}")
            output.save(dataset_path, zanj=zanj)

        print_log(
            f"Got dataset {output.cfg.name} with {len(output)} items. {output.cfg.to_fname() = }"
        )
        return output

    def save(self, file_path: Path | str, zanj: ZANJ | None = None):
        if zanj is None:
            zanj = ZANJ()
        zanj.save(self.serialize(), file_path)

    # serialization & loading
    @classmethod
    def read(cls, file_path: str, zanj: ZANJ | None = None) -> "GPTDataset":
        if zanj is None:
            zanj = ZANJ()
        return zanj.read(file_path)

    def serialize(self) -> JSONitem:
        raise NotImplementedError()

    def data_hash(self) -> int:
        raise NotImplementedError()

    @classmethod
    def load(cls, data: JSONitem) -> "GPTDataset":
        raise NotImplementedError()

    # generating & downloading
    @classmethod
    def generate(cls, cfg: GPTDatasetConfig, **kwargs) -> "GPTDataset":
        raise NotImplementedError()

    @classmethod
    def download(cls, cfg: GPTDatasetConfig, **kwargs) -> "GPTDataset":
        raise NotImplementedError()

    # filtering
    def update_self_config(self):
        """update the config of the dataset to match the actual data, if needed

        for example, adjust number of mazes after filtering
        """
        pass

    class FilterBy:
        """thanks GPT-4"""

        def __init__(self, dataset: "GPTDataset"):
            self.dataset: "GPTDataset" = dataset

        def __getattr__(self, name: str) -> typing.Callable[..., "GPTDataset"]:
            filter_func: DatasetFilterProtocol = getattr(
                self.dataset._FILTER_NAMESPACE, name
            )

            def wrapped_filter_func(*args, **kwargs):
                return filter_func(self.dataset, *args, **kwargs)

            return wrapped_filter_func

    @property
    def filter_by(self) -> "FilterBy":
        return self.FilterBy(self)

    def _apply_filters_from_config(self):
        """apply filters to the dataset, as specified in the config. used in `from_config()`"""
        output: GPTDataset = self
        # copy the list, and then clear it in the config. we do this because each time we apply a filter it will update config.applied_filters
        applied_filters_old: list[
            dict[typing.Literal["name", "args", "kwargs"], typing.Any]
        ] = self.cfg.applied_filters
        output.cfg.applied_filters = list()
        # apply the filters
        for filter_info in applied_filters_old:
            filter_name: str = filter_info["name"]
            if filter_name not in output._FILTER_NAMESPACE.__dict__:
                if filter_name.startswith("__custom__:"):
                    raise ValueError(
                        f"the dataset {output.cfg.to_fname()} was filtering using a custom filter: '{filter_name}', which we don't know about. add it to MazeDatasetFilters!"
                    )
                else:
                    raise ValueError(
                        f"the dataset {output.cfg.to_fname()} was filtering using an unknown filter: '{filter_name}'"
                    )
            filter_args: list = filter_info["args"] if "args" in filter_info else list()
            filter_kwargs: dict = (
                filter_info["kwargs"] if "kwargs" in filter_info else dict()
            )
            output = getattr(output.filter_by, filter_name)(
                *filter_args, **filter_kwargs
            )

        # update the config, perform checks
        # TODO: some funny business with manually specified filters here?
        output.update_self_config()
        _check_filter_equality(applied_filters_old, output.cfg.applied_filters)
        return output


def _check_filter_equality(
    filters_old: list[
        dict[typing.Literal["name", "args", "kwargs"], str | list | dict]
    ],
    filters_new: list[
        dict[typing.Literal["name", "args", "kwargs"], str | list | dict]
    ],
) -> None:
    try:
        assert len(filters_old) == len(filters_new)

        for filterinfo_new, filterinfo_old in zip(filters_old, filters_new):
            # basic checks
            assert isinstance(filterinfo_new, dict), "filterinfo_new is not a dict"
            assert isinstance(filterinfo_old, dict), "filterinfo_old is not a dict"
            assert all(
                key in filterinfo_new for key in ["name", "args", "kwargs"]
            ), "missing keys in filterinfo_new"
            assert all(
                key in filterinfo_old for key in ["name", "args", "kwargs"]
            ), "missing keys in filterinfo_old"

            # name
            assert (
                filterinfo_new["name"] == filterinfo_old["name"]
            ), "filter names don't match"

            # args
            assert len(filterinfo_new["args"]) == len(
                filterinfo_old["args"]
            ), "filter args of different lengths"
            for arg_new, arg_old in zip(filterinfo_new["args"], filterinfo_old["args"]):
                assert arg_new == arg_old, "filter args don't match"

            # kwargs
            assert len(filterinfo_new["kwargs"]) == len(
                filterinfo_old["kwargs"]
            ), "filter kwargs of different lengths"
            for key in filterinfo_old["kwargs"]:
                assert (
                    key in filterinfo_new["kwargs"]
                ), f"filter kwargs don't match: missing key '{key}'"
                assert (
                    filterinfo_new["kwargs"][key] == filterinfo_old["kwargs"][key]
                ), f"filter kwargs don't match: values for key '{key}' don't match"

    except AssertionError as e:
        raise FilterInfoMismatchError(
            f"config mismatch in applied filters: {filters_new} != {filters_old}"
        ) from e


def register_filter_namespace_for_dataset(
    dataset_cls: Type[GPTDataset],
) -> Callable[[Type], Type]:
    """register the namespace class with the given dataset class"""

    def decorator(filter_namespace_cls: Type) -> Type:
        dataset_cls._FILTER_NAMESPACE = filter_namespace_cls
        filter_namespace_cls._BASE_DATASET = dataset_cls

        return filter_namespace_cls

    return decorator


class DatasetFilterProtocol(typing.Protocol):
    def __call__(
        self,
        dataset: GPTDataset,
        **kwargs,
    ) -> GPTDataset: ...


def register_dataset_filter(
    method: DatasetFilterProtocol,
) -> DatasetFilterProtocol:
    """register a dataset filter, copying the underlying dataset and updating the config

    be sure to return a COPY, not the original?

    method should be a staticmethod of a namespace class registered with `register_filter_namespace_for_dataset`
    """

    @functools.wraps(method)
    def wrapper(dataset: GPTDataset, *args, **kwargs):
        new_dataset = method(dataset, *args, **kwargs)
        # update the config
        new_dataset.cfg.applied_filters.append(
            dict(name=method.__name__, args=args, kwargs=kwargs)
        )
        new_dataset.update_self_config()
        return new_dataset

    return wrapper
