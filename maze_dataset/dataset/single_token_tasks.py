import functools
import typing

import numpy as np
from jaxtyping import Float
from maze_dataset import SPECIAL_TOKENS


def get_token_first_index(search_token: str, token_list: list[str]) -> int:
    return token_list.index(search_token)


TaskSetup = typing.NamedTuple(
    "TaskSetup",
    [
        ("prompts", list[list[str]]),
        ("targets", str),
    ],
)


class DLAProtocol(typing.Protocol):
    """should take a dataset's tokens, and return a tuple of (prompts, targets)"""

    def __call__(self, dataset_tokens: list[list[str]], **kwargs) -> TaskSetup:
        ...


class DLAProtocolFixed(typing.Protocol):
    """should take a dataset's tokens, and return a tuple of (prompts, targets)

    this variant signifies it's ready to be used -- no keyword arguments are needed
    """

    def __call__(self, dataset_tokens: list[list[str]]) -> TaskSetup:
        ...


def token_after_fixed_start_token(
    dataset_tokens: list[list[str]],
    start_token: str = SPECIAL_TOKENS.PATH_START,
    offset: int = 1,
) -> TaskSetup:
    """in this task, we simply predict the token after `start_token`

    # Parameters:
     - `dataset_tokens : list[list[str]]`
       list of string-lists
     - `start_token : str`
       token to look for
       (defaults to `SPECIAL_TOKENS.PATH_START`)
     - `offset : int`
       which token to predict:
         1: the token after `start_token`, given everything up to and including `start_token`
         0: the token at `start_token`, given everything up to and **not** including `start_token`
       (defaults to `1`)

    # Returns:
     - `TaskSetup`
       tuple of (prompts, targets)
    """

    prompts: list[list[str]] = list()
    targets: list[str] = list()

    for maze_tokens in dataset_tokens:
        path_start_idx: int = get_token_first_index(start_token, maze_tokens)
        prompt_tokens: list[str] = maze_tokens[: path_start_idx + offset]
        prompts.append(prompt_tokens)
        targets.append(maze_tokens[path_start_idx + offset])

    return TaskSetup(prompts=prompts, targets=targets)


def rand_token_in_range(
    dataset_tokens: list[list[str]],
    start_token: str = SPECIAL_TOKENS.PATH_START,
    end_token: str = SPECIAL_TOKENS.PATH_END,
    start_offset: int = 1,
    end_offset: int = -1,
) -> TaskSetup:
    """predict some random token between (non-inclusive) `start_token` and `end_token`"""
    n_samples: int = len(dataset_tokens)

    prompts: list[list[str]] = list()
    targets: list[str] = list()
    positions_p: Float[np.ndarray, "n_samples"] = np.random.uniform(size=(n_samples,))

    for i, sample_tokens in enumerate(dataset_tokens):
        start_idx: int = (
            get_token_first_index(start_token, sample_tokens) + start_offset
        )
        end_idx: int = get_token_first_index(end_token, sample_tokens) + end_offset

        selected_token_idx: int
        if start_idx < end_idx:
            selected_token_idx = int(positions_p[i] * (end_idx - start_idx) + start_idx)
        else:
            selected_token_idx = start_idx

        prompts.append(sample_tokens[:selected_token_idx])
        targets.append(sample_tokens[selected_token_idx])

    return TaskSetup(prompts=prompts, targets=targets)


LOGIT_ATTRIB_TASKS: dict[str, DLAProtocolFixed] = {
    "path_start": functools.partial(
        token_after_fixed_start_token, start_token=SPECIAL_TOKENS.PATH_START, offset=0
    ),
    "origin_after_path_start": functools.partial(
        token_after_fixed_start_token, start_token=SPECIAL_TOKENS.PATH_START, offset=1
    ),
    "first_path_choice": functools.partial(
        token_after_fixed_start_token, start_token=SPECIAL_TOKENS.PATH_START, offset=2
    ),
    "path_end": functools.partial(
        token_after_fixed_start_token, start_token=SPECIAL_TOKENS.PATH_END, offset=0
    ),
    "final_before_path_end": functools.partial(
        token_after_fixed_start_token, start_token=SPECIAL_TOKENS.PATH_END, offset=-1
    ),
    "rand_path_token": functools.partial(
        rand_token_in_range,
        start_token=SPECIAL_TOKENS.PATH_START,
        end_token=SPECIAL_TOKENS.PATH_END,
        start_offset=1,
        end_offset=-1,
    ),
    "rand_path_token_non_endpoint": functools.partial(
        rand_token_in_range,
        start_token=SPECIAL_TOKENS.PATH_START,
        end_token=SPECIAL_TOKENS.PATH_END,
        start_offset=3,
        end_offset=-2,
    ),
}
