"""Contains `ALL_TOKENIZERS` and supporting limited-use functions.

# `ALL_TOKENIZERS`
A comprehensive collection of all valid `MazeTokenizer2` objects.
This is an overwhelming majority subset of the set of all possible `MazeTokenizer2` objects.
Other tokenizers not contained in `ALL_TOKENIZERS` may be possible to construct, but they are untested and not guaranteed to work.
This collection is in a separate module since it is expensive to compute and will grow more expensive as features are added to `MazeTokenizer2`.

## Use Cases
In general, uses for this module are limited to development of the library and specific research studying many tokenization behaviors.
- Unit testing:
  - Tokenizers to use in unit tests are sampled from `ALL_TOKENIZERS`
- Large-scale tokenizer research:
  - Specific research training models on many tokenization behaviors can use `ALL_TOKENIZERS` as the maximally inclusive collection
  - `ALL_TOKENIZERS` may be subsequently filtered using `MazeTokenizer2.has_element`
For other uses, it's likely that the computational expense can be avoided by using
- `maze_tokenizer.ALL_TOKENIZER_HASHES` for membership checks
- `utils.all_instances` for generating smaller subsets of `MazeTokenizer2` or `TokenizerElement` objects 

# `EVERY_TEST_TOKENIZERS`
A collection of the tokenizers which should always be included in unit tests when test fuzzing is used.
This collection should be expanded as specific tokenizers become canonical or popular.
"""

import random
from functools import cache
from pathlib import Path

import frozendict
import numpy as np
from jaxtyping import Int64

from maze_dataset.tokenization import (
    CoordTokenizers,
    MazeTokenizer2,
    PromptSequencers,
    TokenizerElement,
)
from maze_dataset.utils import all_instances


@cache
def _get_all_tokenizers() -> list[MazeTokenizer2]:
    return all_instances(
        MazeTokenizer2,
        validation_funcs=frozendict.frozendict(
            {
                TokenizerElement: lambda x: x.is_valid(),
                MazeTokenizer2: lambda x: x.is_valid(),
            }
        ),
    )


EVERY_TEST_TOKENIZERS: list[MazeTokenizer2] = [
    MazeTokenizer2(),
    MazeTokenizer2(
        prompt_sequencer=PromptSequencers.AOTP(coord_tokenizer=CoordTokenizers.CTT())
    ),
    # TODO: add more here
]


# TODO: this is pretty bad because it makes it opaque as to when we are actually accessing this massive list of tokenizers


@cache
def all_tokenizers_list() -> list[MazeTokenizer2]:
    """Casts ALL_TOKENIZERS to a list."""
    return list(_get_all_tokenizers())


@cache
def _all_tokenizers_except_every_test_tokenizers() -> list[MazeTokenizer2]:
    """Returns"""
    return list(_get_all_tokenizers().difference(EVERY_TEST_TOKENIZERS))


def sample_all_tokenizers(n: int) -> list[MazeTokenizer2]:
    """Samples `n` tokenizers from `ALL_TOKENIZERS`."""
    return random.sample(all_tokenizers_list(), n)


def sample_tokenizers_for_test(n: int) -> list[MazeTokenizer2]:
    """Returns a sample of size `n` of unique elements from `ALL_TOKENIZERS`,
    always including every element in `EVERY_TEST_TOKENIZERS`.
    """
    if n < len(EVERY_TEST_TOKENIZERS):
        raise ValueError(
            f"`n` must be at least {len(EVERY_TEST_TOKENIZERS)} such that the sample can contain `EVERY_TEST_TOKENIZERS`."
        )
    sample: list[MazeTokenizer2] = random.sample(
        _all_tokenizers_except_every_test_tokenizers(), n - len(EVERY_TEST_TOKENIZERS)
    )
    sample.extend(EVERY_TEST_TOKENIZERS)
    return sample


def save_hashes() -> Int64[np.int64, "tokenizer"]:
    """Computes, sorts, and saves the hashes of every member of `ALL_TOKENIZERS`."""
    hashes_array = np.array(
        [hash(obj) for obj in _get_all_tokenizers()], dtype=np.int64
    )
    sorted_hashes, counts = np.unique(hashes_array, return_counts=True)
    if sorted_hashes.shape[0] != hashes_array.shape[0]:
        collisions = sorted_hashes[counts > 1]
        raise ValueError(
            f"{hashes_array.shape[0] - sorted_hashes.shape[0]} tokenizer hash collisions: {collisions}\nReport error to the developer to increase the hash size or otherwise update the tokenizer hashing algorithm."
        )
    np.save(Path(__file__).parent / "MazeTokenizer2_hashes.npy", sorted_hashes)
    return sorted_hashes
