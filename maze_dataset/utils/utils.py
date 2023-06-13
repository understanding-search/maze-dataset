import json
import os
import random
import typing
from itertools import islice
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import torch

from maze_dataset.generation.constants import CoordTup

DEFAULT_SEED: int = 42
GLOBAL_SEED: int = DEFAULT_SEED


def get_device() -> torch.device:
    """Get the torch.device instance on which torch.Tensors should be allocated."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_reproducibility(seed: int = DEFAULT_SEED):
    """
    Improve model reproducibility. See https://github.com/NVIDIA/framework-determinism for more information.

    Deterministic operations tend to have worse performance than nondeterministic operations, so this method trades
    off performance for reproducibility. Set use_deterministic_algorithms to True to improve performance.
    """
    global GLOBAL_SEED

    GLOBAL_SEED = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    # Ensure reproducibility for concurrent CUDA streams
    # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def chunks(it, chunk_size):
    """Yield successive chunks from an iterator."""
    # https://stackoverflow.com/a/61435714
    iterator = iter(it)
    while chunk := list(islice(iterator, chunk_size)):
        yield chunk


def get_checkpoint_paths_for_run(
    run_path: Path,
    extension: typing.Literal["pt", "zanj"],
    checkpoints_format: str = "checkpoints/model.iter_*.{extension}",
) -> list[tuple[int, Path]]:
    """get checkpoints of the format from the run_path

    note that `checkpoints_format` should contain a glob pattern with:
     - unresolved "{extension}" format term for the extension
     - a wildcard for the iteration number
    """

    assert (
        run_path.is_dir()
    ), f"Model path {run_path} is not a directory (expect run directory, not model files)"

    return [
        (int(checkpoint_path.stem.split("_")[-1].split(".")[0]), checkpoint_path)
        for checkpoint_path in sorted(
            Path(run_path).glob(checkpoints_format.format(extension=extension))
        )
    ]


F = TypeVar("F", bound=Callable[..., Any])


def register_method(
    method_dict: dict[str, Callable[..., Any]],
    custom_name: str | None = None,
) -> Callable[[F], F]:
    """Decorator to add a method to the method_dict"""

    def decorator(method: F) -> F:
        if custom_name is None:
            method_name: str = method.__name__
        else:
            method_name = custom_name
            method.__name__ = custom_name
        assert (
            method_name not in method_dict
        ), f"Method name already exists in method_dict: {method_name = }, {list(method_dict.keys()) = }"
        method_dict[method_name] = method
        return method

    return decorator


def corner_first_ndindex(n: int, ndim: int = 2) -> list[CoordTup]:
    """returns an array of indices, sorted by distance from the corner

    this gives the property that `np.ndindex((n,n))` is equal to
    the first n^2 elements of `np.ndindex((n+1, n+1))`

    ```
    >>> corner_first_ndindex(1)
    [(0, 0)]
    >>> corner_first_ndindex(2)
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    >>> corner_first_ndindex(3)
    [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2)]
    ```
    """

    unsorted: list = list(np.ndindex(tuple([n for _ in range(ndim)])))
    return sorted(unsorted, key=lambda x: (max(x), x if x[0] % 2 == 0 else x[::-1]))

    # alternate numpy version from GPT-4:
    """
    # Create all index combinations
    indices = np.indices([n]*ndim).reshape(ndim, -1).T
    # Find the max value for each index
    max_indices = np.max(indices, axis=1)
    # Identify the odd max values
    odd_mask = max_indices % 2 != 0
    # Make a copy of indices to avoid changing the original one
    indices_copy = indices.copy()
    # Reverse the order of the coordinates for indices with odd max value
    indices_copy[odd_mask] = indices_copy[odd_mask, ::-1]
    # Sort by max index value, then by coordinates
    sorted_order = np.lexsort((*indices_copy.T, max_indices))
    return indices[sorted_order]
    """


def pprint_summary(summary: dict):
    print(json.dumps(summary, indent=2))
