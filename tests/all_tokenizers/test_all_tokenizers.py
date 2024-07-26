import itertools
import os
from collections import Counter
from typing import Callable, Iterable

import numpy as np
import pytest
from pytest import mark, param
from zanj import ZANJ

from maze_dataset import VOCAB, VOCAB_LIST, LatticeMaze
from maze_dataset.maze.lattice_maze import SolvedMaze
from maze_dataset.testing_utils import MIXED_MAZES
from maze_dataset.token_utils import equal_except_adj_list_sequence, tokens_between
from maze_dataset.tokenization import (
    CoordTokenizers,
    EdgeGroupings,
    EdgePermuters,
    MazeTokenizerModular,
    PathTokenizers,
    PromptSequencers,
    StepSizes,
    StepTokenizers,
    AdjListTokenizers,
    _TokenizerElement,
)
from maze_dataset.tokenization.all_tokenizers import (
    EVERY_TEST_TOKENIZERS,
    MAZE_TOKENIZER_MODULAR_DEFAULT_VALIDATION_FUNCS,
    get_all_tokenizers,
    sample_tokenizers_for_test,
    save_hashes,
)
from maze_dataset.tokenization.maze_tokenizer import _load_tokenizer_hashes
from maze_dataset.utils import all_instances

# Size of the sample from `all_tokenizers.ALL_TOKENIZERS` to test
# get from env, or set to default value of 100
_os_env_num_tokenizers: str = os.getenv("NUM_TOKENIZERS_TO_TEST", "100")
NUM_TOKENIZERS_TO_TEST: int | None = (
    int(_os_env_num_tokenizers) if _os_env_num_tokenizers.isdigit() else None
)
print(f"{NUM_TOKENIZERS_TO_TEST = }")

ALL_TOKENIZERS: list[MazeTokenizerModular] = get_all_tokenizers()
SAMPLED_TOKENIZERS: list[MazeTokenizerModular] = sample_tokenizers_for_test(
    NUM_TOKENIZERS_TO_TEST
)

SAMPLED_MAZES: list[SolvedMaze] = MIXED_MAZES[:6]


@pytest.fixture(scope="session")
def save_tokenizer_hashes():
    save_hashes()


def test_all_tokenizers():
    assert len(ALL_TOKENIZERS) > 400


@mark.parametrize(
    "class_",
    [param(c, id=c.__name__) for c in _TokenizerElement.__subclasses__()],
)
def test_all_instances_tokenizerelement(class_: type):
    all_vals = list(
        all_instances(
            class_, validation_funcs=MAZE_TOKENIZER_MODULAR_DEFAULT_VALIDATION_FUNCS
        )
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
    mts: list[MazeTokenizerModular] = sample_tokenizers_for_test(n)
    mts_set: set[MazeTokenizerModular] = set(mts)
    assert len(mts) == len(mts_set)
    assert set(EVERY_TEST_TOKENIZERS).issubset(mts_set)
    if n > SAMPLE_MIN + 1:
        mts2: list[MazeTokenizerModular] = sample_tokenizers_for_test(n)
        assert set(mts2) != mts_set  # Check that succesive samples are different


@mark.parametrize(
    "tokenizer",
    [param(tokenizer, id=tokenizer.name) for tokenizer in SAMPLED_TOKENIZERS],
)
def test_token_region_delimiters(tokenizer: MazeTokenizerModular):
    """<PATH_START> and similar token region delimiters should appear at most 1 time, regardless of tokenizer."""
    for maze in SAMPLED_MAZES:
        counts: Counter = Counter(maze.as_tokens(tokenizer))
        assert all([counts[tok] < 2 for tok in VOCAB_LIST[:8]])


@mark.parametrize(
    "tokenizer",
    [param(tokenizer, id=tokenizer.name) for tokenizer in SAMPLED_TOKENIZERS],
)
def test_token_stability(tokenizer: MazeTokenizerModular):
    """Tests consistency of tokenizations over multiple method calls."""
    for maze in SAMPLED_MAZES:
        tokens1: list[str] = maze.as_tokens(tokenizer)
        tokens2: list[str] = maze.as_tokens(tokenizer)
        if (
            tokenizer.has_element(EdgeGroupings.ByLeadingCoord, EdgePermuters.RandomCoords)
            or tokenizer.has_element(AdjListTokenizers.AdjListCardinal, EdgePermuters.RandomCoords)
        ):
            # In this case, the adjlist is expected to have different token counts over multiple calls
            # Exclude that region from the test
            non_adjlist1 = tokens1[:tokens1.index(VOCAB.ADJLIST_START)]
            non_adjlist1.extend(tokens1[tokens1.index(VOCAB.ADJLIST_END):])
            non_adjlist2 = tokens2[:tokens2.index(VOCAB.ADJLIST_START)]
            non_adjlist2.extend(tokens2[tokens2.index(VOCAB.ADJLIST_END):])
            assert non_adjlist1 == non_adjlist2
        else:
            assert equal_except_adj_list_sequence(tokens1, tokens2)


@mark.parametrize(
    "tokenizer",
    [param(tokenizer, id=tokenizer.name) for tokenizer in SAMPLED_TOKENIZERS],
)
def test_tokenizer_properties(tokenizer: MazeTokenizerModular):
    # Just make sure the call doesn't raise exception
    assert len(tokenizer.name) > 5

    assert tokenizer.vocab_size == 4096
    assert isinstance(tokenizer.token_arr, Iterable)
    assert all(isinstance(token, str) for token in tokenizer.token_arr)
    assert tokenizer.token_arr[tokenizer.padding_token_index] == VOCAB.PADDING

    # Just make sure the call doesn't raise exception
    print(tokenizer.summary())


@mark.parametrize(
    "tokenizer",
    [param(tokenizer, id=tokenizer.name) for tokenizer in SAMPLED_TOKENIZERS],
)
def test_encode_decode(tokenizer: MazeTokenizerModular):
    for maze in SAMPLED_MAZES:
        maze_tok: list[str] = maze.as_tokens(maze_tokenizer=tokenizer)
        maze_encoded: list[int] = tokenizer.encode(maze_tok)
        maze_decoded: LatticeMaze = tokenizer.decode(maze_encoded)
        assert maze_tok == maze_decoded


@mark.parametrize(
    "tokenizer",
    [param(tokenizer, id=tokenizer.name) for tokenizer in SAMPLED_TOKENIZERS],
)
def test_zanj_save_read(tokenizer: MazeTokenizerModular):
    path = os.path.abspath(
        os.path.join(
            os.path.curdir,
            "data",
            "MazeTokenizerModular_" + hex(hash(tokenizer)) + ".zanj",
        )
    )
    zanj = ZANJ()
    zanj.save(tokenizer, path)
    assert zanj.read(path) == tokenizer


@mark.parametrize(
    "tokenizer",
    [param(tokenizer, id=tokenizer.name) for tokenizer in SAMPLED_TOKENIZERS],
)
def test_is_AOTP(tokenizer: MazeTokenizerModular):
    if isinstance(tokenizer.prompt_sequencer, PromptSequencers.AOTP):
        assert tokenizer.is_AOTP()
    else:
        assert not tokenizer.is_AOTP()


@mark.parametrize(
    "tokenizer",
    [param(tokenizer, id=tokenizer.name) for tokenizer in SAMPLED_TOKENIZERS],
)
def test_is_UT(tokenizer: MazeTokenizerModular):
    if isinstance(tokenizer.prompt_sequencer.coord_tokenizer, CoordTokenizers.UT):
        assert tokenizer.is_UT()
    else:
        assert not tokenizer.is_UT()


_has_elems_type = (
    type[_TokenizerElement]
    | _TokenizerElement
    | Iterable[type[_TokenizerElement] | _TokenizerElement]
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
            SAMPLED_TOKENIZERS,
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
                    StepSizes.Singles,
                    lambda mt, els: isinstance(
                        mt.prompt_sequencer.path_tokenizer.step_size, els
                    ),
                ),
                (
                    StepTokenizers.Coord,
                    lambda mt, els: any(
                        isinstance(step_tok, els)
                        for step_tok in mt.prompt_sequencer.path_tokenizer.step_tokenizers
                    ),
                ),
                (
                    [CoordTokenizers.CTT()],
                    lambda mt, els: mt.prompt_sequencer.coord_tokenizer == els[0],
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
    tokenizer: MazeTokenizerModular,
    elems: _has_elems_type,
    result_func: Callable[[MazeTokenizerModular, _has_elems_type], bool],
):
    assert tokenizer.has_element(elems) == result_func(tokenizer, elems)


@mark.parametrize(
    "tokenizer",
    [param(tokenizer, id=tokenizer.name) for tokenizer in SAMPLED_TOKENIZERS],
)
def test_is_tested_tokenizer(tokenizer: MazeTokenizerModular, save_tokenizer_hashes):
    assert tokenizer.is_tested_tokenizer()
