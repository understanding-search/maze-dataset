"""Contains `ALL_TOKENIZERS`.

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
"""

from typing import Iterable

from maze_dataset.tokenization import MazeTokenizer2
from maze_dataset.utils import all_instances

ALL_TOKENIZERS: set[MazeTokenizer2] = all_instances(MazeTokenizer2)