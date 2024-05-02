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

from typing import Iterable
from functools import cache
import random

from maze_dataset.tokenization import MazeTokenizer2, CoordTokenizers
from maze_dataset.utils import all_instances


ALL_TOKENIZERS: set[MazeTokenizer2] = set(all_instances(MazeTokenizer2))
EVERY_TEST_TOKENIZERS: list[MazeTokenizer2] = [
    MazeTokenizer2(), 
    MazeTokenizer2(coord_tokenizer=CoordTokenizers.CTT())
]


@cache
def all_tokenizers_list() -> list[MazeTokenizer2]:
    """Casts ALL_TOKENIZERS to a list."""
    return list(ALL_TOKENIZERS)


@cache
def all_tokenizers_except_every_test_tokenizers() -> list[MazeTokenizer2]:
    """Returns  """
    return list(ALL_TOKENIZERS.difference(EVERY_TEST_TOKENIZERS))


def sample_all_tokenizers(n: int) -> list[MazeTokenizer2]:
    """Samples `n` tokenizers from `ALL_TOKENIZERS`."""
    return random.sample(all_tokenizers_list(), n)


def sample_tokenizers_for_test(n: int) -> list[MazeTokenizer2]:
    """ Returns a sample of size `n` of unique elements from `ALL_TOKENIZERS`, 
    always including every element in `EVERY_TEST_TOKENIZERS`.
    """
    if n < len(EVERY_TEST_TOKENIZERS):
        raise ValueError(f'`n` must be at least {len(EVERY_TEST_TOKENIZERS)} such that the sample can contain `EVERY_TEST_TOKENIZERS`.')
    sample: list[MazeTokenizer2] = random.sample(all_tokenizers_except_every_test_tokenizers(), n-len(EVERY_TEST_TOKENIZERS))
    sample.extend(EVERY_TEST_TOKENIZERS)
    return sample