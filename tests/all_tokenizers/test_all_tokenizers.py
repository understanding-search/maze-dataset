import itertools
import os
import random
from collections import Counter, namedtuple
from typing import Callable, Iterable, Sequence

import frozendict
import numpy as np
import pytest
from jaxtyping import Int
from pytest import mark, param
from zanj import ZANJ

from maze_dataset import (
    VOCAB,
    VOCAB_LIST,
    ConnectionArray,
    CoordArray,
    CoordTup,
    LatticeMaze,
    LatticeMazeGenerators,
    MazeDataset,
    MazeDatasetConfig,
    SolvedMaze,
    TargetedLatticeMaze,
)
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.tokenization import (
    CoordTokenizers,
    EdgeGroupings,
    EdgePermuters,
    EdgeSubsets,
    MazeTokenizer2,
    PathTokenizers,
    PromptSequencers,
    StepSizes,
    StepTokenizers,
    AdjListTokenizers,
    TokenizerElement,
)
from maze_dataset.tokenization.all_tokenizers import (
    EVERY_TEST_TOKENIZERS,
    _get_all_tokenizers,
    sample_tokenizers_for_test,
    save_hashes,
)
from maze_dataset.tokenization.maze_tokenizer import _load_tokenizer_hashes
from maze_dataset.util import connection_list_to_adj_list
from maze_dataset.utils import all_instances, flatten, manhattan_distance
from maze_dataset.testing_utils import (
    MIXED_MAZES,
    ASCII_MAZES,
    MANUAL_MAZE,
)

# Size of the sample from `all_tokenizers.ALL_TOKENIZERS` to test
NUM_TOKENIZERS_TO_TEST = 100


def test_all_tokenizers():
    all_tokenizers = _get_all_tokenizers()
    assert len(all_tokenizers) > 400
    assert len({hash(mt) for mt in all_tokenizers}) == len(all_tokenizers)


@mark.parametrize(
    "class_",
    [param(c, id=c.__name__) for c in TokenizerElement.__subclasses__()],
)
def test_all_instances_tokenizerelement(class_: type):
    all_vals = all_instances(
        class_,
        validation_funcs=frozendict.frozendict(
            {
                TokenizerElement: lambda x: x.is_valid(),
            }
        ),
    )
    assert len({hash(elem) for elem in all_vals}) == len(all_vals)


def test_all_tokenizer_hashes():
    loaded_hashes = save_hashes()
    assert np.array_equal(_load_tokenizer_hashes(), loaded_hashes)


SAMPLE_MIN: int = len(EVERY_TEST_TOKENIZERS)


@mark.parametrize(
    "n, result",
    [
        param(i, result)
        for i, result in [
            (SAMPLE_MIN - 1, ValueError),
            (SAMPLE_MIN, None),
            (SAMPLE_MIN + 5, None),
            (SAMPLE_MIN + 200, None),
        ]
    ],
)
def test_sample_tokenizers_for_test(n: int, result: type[Exception] | None):
    if isinstance(result, type) and issubclass(result, Exception):
        with pytest.raises(result):
            sample_tokenizers_for_test(n)
        return
    mts: list[MazeTokenizer2] = sample_tokenizers_for_test(n)
    mts_set: set[MazeTokenizer2] = set(mts)
    assert len(mts) == len(mts_set)
    assert set(EVERY_TEST_TOKENIZERS).issubset(mts_set)
    if n > SAMPLE_MIN + 1:
        mts2: list[MazeTokenizer2] = sample_tokenizers_for_test(n)
        assert set(mts2) != mts_set  # Check that succesive samples are different


@mark.parametrize(
    "maze,tokenizer",
    [
        param(maze[0], tokenizer, id=f"{type(maze[0])}{maze[1]}-{tokenizer.name}")
        for maze, tokenizer in itertools.product(
            [(maze, i) for i, maze in enumerate(MIXED_MAZES[:6])],
            sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST),
        )
    ],
)
def test_token_region_delimiters(maze: LatticeMaze, tokenizer: MazeTokenizer2):
    """<PATH_START> and similar token region delimiters should appear at most 1 time, regardless of tokenizer."""
    counts: Counter = Counter(maze.as_tokens(tokenizer))
    assert all([counts[tok] < 2 for tok in VOCAB_LIST[:8]])


@mark.parametrize(
    "tokenizer",
    [
        param(tokenizer, id=tokenizer.name)
        for tokenizer in sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST)
    ],
)
def test_tokenizer_properties(tokenizer: MazeTokenizer2):
    # Just make sure the call doesn't raise exception
    assert len(tokenizer.name) > 5

    assert tokenizer.vocab_size == 4096
    assert isinstance(tokenizer.token_arr, Iterable)
    assert all(isinstance(token, str) for token in tokenizer.token_arr)
    assert tokenizer.token_arr[tokenizer.padding_token_index] == VOCAB.PADDING

    # Just make sure the call doesn't raise exception
    print(tokenizer.summary())


@mark.parametrize(
    "maze,tokenizer",
    [
        param(maze[0], tokenizer, id=f"{tokenizer.name}-maze{maze[1]}")
        for maze, tokenizer in itertools.product(
            [(maze, i) for i, maze in enumerate(MIXED_MAZES[:6])],
            sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST),
        )
    ],
)
def test_encode_decode(maze: LatticeMaze, tokenizer: MazeTokenizer2):
    maze_tok: list[str] = maze.as_tokens(maze_tokenizer=tokenizer)
    maze_encoded: list[int] = tokenizer.encode(maze_tok)
    maze_decoded: LatticeMaze = tokenizer.decode(maze_encoded)
    assert maze_tok == maze_decoded


@mark.parametrize(
    "tokenizer",
    [
        param(tokenizer, id=tokenizer.name)
        for tokenizer in sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST)
    ],
)
def test_zanj_save_read(tokenizer: MazeTokenizer2):
    path = os.path.abspath(
        os.path.join(
            os.path.curdir, "data", "MazeTokenizer2_" + hex(hash(tokenizer)) + ".zanj"
        )
    )
    zanj = ZANJ()
    zanj.save(tokenizer, path)
    assert zanj.read(path) == tokenizer


@mark.parametrize(
    "tokenizer",
    [
        param(tokenizer, id=tokenizer.name)
        for tokenizer in sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST)
    ],
)
def test_is_AOTP(tokenizer: MazeTokenizer2):
    if isinstance(tokenizer.prompt_sequencer, PromptSequencers.AOTP):
        assert tokenizer.is_AOTP()
    else:
        assert not tokenizer.is_AOTP()


@mark.parametrize(
    "tokenizer",
    [
        param(tokenizer, id=tokenizer.name)
        for tokenizer in sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST)
    ],
)
def test_is_UT(tokenizer: MazeTokenizer2):
    if isinstance(tokenizer.prompt_sequencer.coord_tokenizer, CoordTokenizers.UT):
        assert tokenizer.is_UT()
    else:
        assert not tokenizer.is_UT()


_has_elems_type = (
    type[TokenizerElement]
    | TokenizerElement
    | Iterable[type[TokenizerElement] | TokenizerElement]
)


@mark.parametrize(
    "tokenizer, elems, result_func",
    [
        param(
            tokenizer,
            elems_tuple[0],
            elems_tuple[1],
            id=f"{tokenizer.name}-{elems_tuple[0]}",
        )
        for tokenizer, elems_tuple in itertools.product(
            sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST),
            [
                (
                    [PromptSequencers.AOTP()],
                    lambda mt, els: mt.prompt_sequencer == els[0],
                ),
                (PromptSequencers.AOTP(), lambda mt, els: mt.prompt_sequencer == els),
                (
                    [CoordTokenizers.CTT()],
                    lambda mt, els: mt.prompt_sequencer.coord_tokenizer == els[0],
                ),
                (
                    CoordTokenizers.CTT(intra=False),
                    lambda mt, els: mt.prompt_sequencer.coord_tokenizer == els,
                ),
                (
                    [CoordTokenizers.CTT],
                    lambda mt, els: isinstance(
                        mt.prompt_sequencer.coord_tokenizer, els[0]
                    ),
                ),
                (
                    CoordTokenizers._CoordTokenizer,
                    lambda mt, els: isinstance(
                        mt.prompt_sequencer.coord_tokenizer, els
                    ),
                ),
                (
                    [CoordTokenizers.CTT, PathTokenizers.StepSequence],
                    lambda mt, els: isinstance(
                        mt.prompt_sequencer.coord_tokenizer, els[0]
                    )
                    and isinstance(mt.prompt_sequencer.path_tokenizer, els[1]),
                ),
                # ((a for a in [CoordTokenizers.CTT, PathTokenizers.Coords]),
                #  lambda mt, els: isinstance(mt.coord_tokenizer, list(els)[0]) and isinstance(mt.path_tokenizer, list(els)[1])
                #  ),
                (
                    [CoordTokenizers.CTT, PathTokenizers.StepSequence(post=False)],
                    lambda mt, els: isinstance(
                        mt.prompt_sequencer.coord_tokenizer, els[0]
                    )
                    and mt.prompt_sequencer.path_tokenizer == els[1],
                ),
                (
                    [
                        CoordTokenizers.CTT,
                        PathTokenizers.StepSequence,
                        PromptSequencers.AOP(),
                    ],
                    lambda mt, els: isinstance(
                        mt.prompt_sequencer.coord_tokenizer, els[0]
                    )
                    and isinstance(mt.prompt_sequencer.path_tokenizer, els[1])
                    and mt.prompt_sequencer == els[2],
                ),
            ],
        )
    ],
)
def test_has_element(
    tokenizer: MazeTokenizer2,
    elems: _has_elems_type,
    result_func: Callable[[MazeTokenizer2, _has_elems_type], bool],
):
    assert tokenizer.has_element(elems) == result_func(tokenizer, elems)


@mark.parametrize(
    "tokenizer",
    [
        param(tokenizer, id=tokenizer.name)
        for tokenizer in sample_tokenizers_for_test(NUM_TOKENIZERS_TO_TEST)
    ],
)
def test_is_tested_tokenizer(tokenizer: MazeTokenizer2):
    assert tokenizer.is_tested_tokenizer()
