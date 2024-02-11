import functools
import typing

import numpy as np
from jaxtyping import Float
from maze_dataset import SolvedMaze, SPECIAL_TOKENS
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode


def get_token_first_index(search_token: str, token_list: list[str]) -> int:
    return token_list.index(search_token)


TaskSetup = typing.NamedTuple(
    "TaskSetup",
    [
        ("prompts", list[list[str]]),
        ("targets", list[str]),
    ],
)


class TaskCreatorProtocol(typing.Protocol):
    """should take a dataset's tokens, and return a tuple of (prompts, targets)"""

    def __call__(self, dataset_tokens: list[list[str]], **kwargs) -> TaskSetup:
        ...


class TaskCreatorProtocolFixed(typing.Protocol):
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
    all_from_example: bool = True,
) -> TaskSetup:
    """predict some random token between (non-inclusive) `start_token` and `end_token`
    
    if `all_from_example` is `True`, then all possible tokens are selected from the same example
    if `all_from_example` is `False`, then for each example we select a single random token
    """
    n_samples: int = len(dataset_tokens)

    prompts: list[list[str]] = list()
    targets: list[str] = list()
    if all_from_example:
        positions_p: Float[np.ndarray, "n_samples"] = np.random.uniform(size=(n_samples,))

    for i, sample_tokens in enumerate(dataset_tokens):
        # find start and end token indecies
        start_idx: int = (
            get_token_first_index(start_token, sample_tokens) + start_offset
        )
        end_idx: int = get_token_first_index(end_token, sample_tokens) + end_offset

        # decide which token(s) to select
        selected_token_indecies: list[int]
        if start_idx > end_idx:
            selected_token_indecies = [start_idx]
        else:
            if all_from_example:
                selected_token_indecies = list[range(start_idx, end_idx)]
            else:          
                selected_token_indecies = [int(positions_p[i] * (end_idx - start_idx) + start_idx)]

        # add the selected tokens to the prompts and targets
        for selected_token_idx in selected_token_indecies:
            prompts.append(sample_tokens[:selected_token_idx])
            targets.append(sample_tokens[selected_token_idx])

    return TaskSetup(prompts=prompts, targets=targets)

def forking_points(
    dataset_tokens: list[list[str]],
    forks_not_paths: bool = True,
    all_from_example: bool = True,
    maze_tokenizer: MazeTokenizer = MazeTokenizer(TokenizationMode.AOTP_UT_uniform),
) -> TaskSetup:
    """predict tokens from forks
    
    if `forks_not_paths` is `True`, then we give prompts where there is a fork
    if `forks_not_paths` is `False`, then we give prompts where there is only path following (no forks, excluding backtracking)
    """
    assert all_from_example, "'all_from_example==False' not implemented yet"

    dataset_mazes: list[SolvedMaze] = [
        SolvedMaze.from_tokens(tokens, maze_tokenizer) for tokens in dataset_tokens
    ]

    prompts: list[list[str]] = list()
    targets: list[str] = list()

    for tokens, maze in zip(dataset_tokens, dataset_mazes):
        path_start_idx: int = get_token_first_index(SPECIAL_TOKENS.PATH_START, tokens)
        split_positions_in_solution: list[int]
        if forks_not_paths:
            split_positions_in_solution = maze.get_solution_forking_points()
        else:
            split_positions_in_solution = maze.get_solution_path_following_points()

        for soln_split in split_positions_in_solution:
            prompt_split: int = path_start_idx + soln_split
            prompts.append(tokens[:prompt_split])
            targets.append(tokens[prompt_split])

    return TaskSetup(prompts=prompts, targets=targets)


SINGLE_TOKEN_TASKS: dict[str, TaskCreatorProtocolFixed] = {
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
    "forking_choices": functools.partial(
        forking_points, forks_not_paths=True, all_from_example=True
    ),
    "path_following": functools.partial(
        forking_points, forks_not_paths=False, all_from_example=True
    ),
}
