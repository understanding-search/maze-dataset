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
    EdgePermuters,
    MazeTokenizer2,
    PathTokenizers,
    PromptSequencers,
    StepSizes,
    StepTokenizers,
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
from maze_dataset.utils import all_instances

# TODO: this needs to be cleaned up, and duplicated functionality in `test_tokenizer.py`
NUM_TOKENIZERS_TO_TEST = 100
GRID_N = 5
N_MAZES = 5
CFG: MazeDatasetConfig = MazeDatasetConfig(
    name="test",
    grid_n=GRID_N,
    n_mazes=N_MAZES,
    maze_ctor=LatticeMazeGenerators.gen_dfs,
)
MAZE_DATASET: MazeDataset = MazeDataset.from_config(
    CFG,
    do_download=False,
    load_local=False,
    do_generate=True,
    save_local=False,
    verbose=True,
    gen_parallel=False,
)
LATTICE_MAZES: list[LatticeMaze] = [
    LatticeMazeGenerators.gen_dfs(np.array([GRID_N, GRID_N])) for _ in range(N_MAZES)
]
_PATHS = [maze.generate_random_path() for maze in LATTICE_MAZES]
TARGETED_MAZES: list[TargetedLatticeMaze] = [
    TargetedLatticeMaze.from_lattice_maze(maze, path[0], path[-1])
    for maze, path in zip(LATTICE_MAZES, _PATHS)
]
# MIXED_MAZES alternates the maze types, so you can slice a contiguous subset and still get all types
# MIXED_MAZES alternates the maze types, so you can slice a contiguous subset and still get all types
MIXED_MAZES: list[LatticeMaze | TargetedLatticeMaze | SolvedMaze] = [
    x
    for x in itertools.chain.from_iterable(
        itertools.zip_longest(MAZE_DATASET.mazes, TARGETED_MAZES, LATTICE_MAZES)
    )
]

_MANUAL_MAZE = namedtuple(
    "_MANUAL_MAZE", ["tokens", "ascii", "straightaway_footprints"]
)


_ASCII_MAZES: dict[str, tuple[str, list[str]]] = dict(
    small_3x3=_MANUAL_MAZE(
        tokens="<ADJLIST_START> (2,0) <--> (2,1) ; (0,0) <--> (0,1) ; (0,0) <--> (1,0) ; (0,2) <--> (1,2) ; (1,0) <--> (2,0) ; (0,2) <--> (0,1) ; (2,2) <--> (2,1) ; (1,1) <--> (2,1) ; <ADJLIST_END> <ORIGIN_START> (0,0) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (0,0) (1,0) (2,0) (2,1) <PATH_END>",
        ascii=[
            "#######",
            "#S    #",
            "#X### #",
            "#X# # #",
            "#X# ###",
            "#XXE  #",
            "#######",
        ],
        straightaway_footprints=np.array(
            [
                [0, 0],
                [2, 0],
                [2, 1],
            ]
        ),
    ),
    big_10x10=_MANUAL_MAZE(
        tokens="<ADJLIST_START> (8,2) <--> (8,3) ; (3,7) <--> (3,6) ; (6,7) <--> (6,8) ; (4,6) <--> (5,6) ; (9,5) <--> (9,4) ; (3,3) <--> (3,4) ; (5,1) <--> (4,1) ; (2,6) <--> (2,7) ; (8,5) <--> (8,4) ; (1,9) <--> (2,9) ; (4,1) <--> (4,2) ; (0,8) <--> (0,7) ; (5,4) <--> (5,3) ; (6,3) <--> (6,4) ; (5,0) <--> (4,0) ; (5,3) <--> (5,2) ; (3,1) <--> (2,1) ; (9,1) <--> (9,0) ; (3,5) <--> (3,6) ; (5,5) <--> (6,5) ; (7,1) <--> (7,2) ; (0,1) <--> (1,1) ; (7,8) <--> (8,8) ; (3,9) <--> (4,9) ; (4,6) <--> (4,7) ; (0,6) <--> (0,7) ; (3,4) <--> (3,5) ; (6,0) <--> (5,0) ; (7,7) <--> (7,6) ; (1,6) <--> (0,6) ; (6,1) <--> (6,0) ; (8,6) <--> (8,7) ; (9,9) <--> (9,8) ; (1,8) <--> (1,9) ; (2,1) <--> (2,2) ; (9,2) <--> (9,3) ; (5,9) <--> (6,9) ; (3,2) <--> (2,2) ; (0,8) <--> (0,9) ; (5,6) <--> (5,7) ; (2,3) <--> (2,4) ; (4,5) <--> (4,4) ; (8,9) <--> (8,8) ; (9,6) <--> (8,6) ; (3,7) <--> (3,8) ; (8,0) <--> (7,0) ; (6,1) <--> (6,2) ; (0,1) <--> (0,0) ; (7,3) <--> (7,4) ; (9,4) <--> (9,3) ; (9,6) <--> (9,5) ; (8,7) <--> (7,7) ; (5,2) <--> (5,1) ; (0,0) <--> (1,0) ; (7,2) <--> (7,3) ; (2,5) <--> (2,6) ; (4,9) <--> (5,9) ; (5,5) <--> (5,4) ; (5,6) <--> (6,6) ; (7,8) <--> (7,9) ; (1,7) <--> (2,7) ; (4,6) <--> (4,5) ; (1,1) <--> (1,2) ; (3,1) <--> (3,0) ; (1,5) <--> (1,6) ; (8,3) <--> (8,4) ; (9,9) <--> (8,9) ; (8,5) <--> (7,5) ; (1,4) <--> (2,4) ; (3,0) <--> (4,0) ; (3,3) <--> (4,3) ; (6,9) <--> (6,8) ; (1,0) <--> (2,0) ; (6,0) <--> (7,0) ; (8,0) <--> (9,0) ; (2,3) <--> (2,2) ; (2,8) <--> (3,8) ; (5,7) <--> (6,7) ; (1,3) <--> (0,3) ; (9,7) <--> (9,8) ; (7,5) <--> (7,4) ; (1,8) <--> (2,8) ; (6,5) <--> (6,4) ; (0,2) <--> (1,2) ; (0,7) <--> (1,7) ; (0,3) <--> (0,2) ; (4,3) <--> (4,2) ; (5,8) <--> (4,8) ; (9,1) <--> (8,1) ; (9,2) <--> (8,2) ; (1,3) <--> (1,4) ; (2,9) <--> (3,9) ; (4,8) <--> (4,7) ; (0,5) <--> (0,4) ; (8,1) <--> (7,1) ; (0,3) <--> (0,4) ; (9,7) <--> (9,6) ; (7,6) <--> (6,6) ; (1,5) <--> (0,5) ; <ADJLIST_END> <ORIGIN_START> (6,2) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (6,2) (6,1) (6,0) (5,0) (4,0) (3,0) (3,1) (2,1) <PATH_END>",
        ascii=[
            "#####################",
            "#   #       #       #",
            "# # # # ### # # #####",
            "# #   #   #   # #   #",
            "# ####### ##### # # #",
            "# #E      #     # # #",
            "###X# ########### # #",
            "#XXX# #           # #",
            "#X##### ########### #",
            "#X#     #         # #",
            "#X# ######### ### # #",
            "#X#         #   # # #",
            "#X######### # # ### #",
            "#XXXXS#     # #     #",
            "# ########### #######",
            "# #         #   #   #",
            "# # ####### ### # ###",
            "# # #       #   #   #",
            "# # # ####### ##### #",
            "#   #               #",
            "#####################",
        ],
        straightaway_footprints=np.array(
            [
                [6, 2],
                [6, 0],
                [3, 0],
                [3, 1],
                [2, 1],
            ]
        ),
    ),
    longer_10x10=_MANUAL_MAZE(
        tokens="<ADJLIST_START> (8,2) <--> (8,3) ; (3,7) <--> (3,6) ; (6,7) <--> (6,8) ; (4,6) <--> (5,6) ; (9,5) <--> (9,4) ; (3,3) <--> (3,4) ; (5,1) <--> (4,1) ; (2,6) <--> (2,7) ; (8,5) <--> (8,4) ; (1,9) <--> (2,9) ; (4,1) <--> (4,2) ; (0,8) <--> (0,7) ; (5,4) <--> (5,3) ; (6,3) <--> (6,4) ; (5,0) <--> (4,0) ; (5,3) <--> (5,2) ; (3,1) <--> (2,1) ; (9,1) <--> (9,0) ; (3,5) <--> (3,6) ; (5,5) <--> (6,5) ; (7,1) <--> (7,2) ; (0,1) <--> (1,1) ; (7,8) <--> (8,8) ; (3,9) <--> (4,9) ; (4,6) <--> (4,7) ; (0,6) <--> (0,7) ; (3,4) <--> (3,5) ; (6,0) <--> (5,0) ; (7,7) <--> (7,6) ; (1,6) <--> (0,6) ; (6,1) <--> (6,0) ; (8,6) <--> (8,7) ; (9,9) <--> (9,8) ; (1,8) <--> (1,9) ; (2,1) <--> (2,2) ; (9,2) <--> (9,3) ; (5,9) <--> (6,9) ; (3,2) <--> (2,2) ; (0,8) <--> (0,9) ; (5,6) <--> (5,7) ; (2,3) <--> (2,4) ; (4,5) <--> (4,4) ; (8,9) <--> (8,8) ; (9,6) <--> (8,6) ; (3,7) <--> (3,8) ; (8,0) <--> (7,0) ; (6,1) <--> (6,2) ; (0,1) <--> (0,0) ; (7,3) <--> (7,4) ; (9,4) <--> (9,3) ; (9,6) <--> (9,5) ; (8,7) <--> (7,7) ; (5,2) <--> (5,1) ; (0,0) <--> (1,0) ; (7,2) <--> (7,3) ; (2,5) <--> (2,6) ; (4,9) <--> (5,9) ; (5,5) <--> (5,4) ; (5,6) <--> (6,6) ; (7,8) <--> (7,9) ; (1,7) <--> (2,7) ; (4,6) <--> (4,5) ; (1,1) <--> (1,2) ; (3,1) <--> (3,0) ; (1,5) <--> (1,6) ; (8,3) <--> (8,4) ; (9,9) <--> (8,9) ; (8,5) <--> (7,5) ; (1,4) <--> (2,4) ; (3,0) <--> (4,0) ; (3,3) <--> (4,3) ; (6,9) <--> (6,8) ; (1,0) <--> (2,0) ; (6,0) <--> (7,0) ; (8,0) <--> (9,0) ; (2,3) <--> (2,2) ; (2,8) <--> (3,8) ; (5,7) <--> (6,7) ; (1,3) <--> (0,3) ; (9,7) <--> (9,8) ; (7,5) <--> (7,4) ; (1,8) <--> (2,8) ; (6,5) <--> (6,4) ; (0,2) <--> (1,2) ; (0,7) <--> (1,7) ; (0,3) <--> (0,2) ; (4,3) <--> (4,2) ; (5,8) <--> (4,8) ; (9,1) <--> (8,1) ; (9,2) <--> (8,2) ; (1,3) <--> (1,4) ; (2,9) <--> (3,9) ; (4,8) <--> (4,7) ; (0,5) <--> (0,4) ; (8,1) <--> (7,1) ; (0,3) <--> (0,4) ; (9,7) <--> (9,6) ; (7,6) <--> (6,6) ; (1,5) <--> (0,5) ; <ADJLIST_END> <ORIGIN_START> (6,2) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (6,2) (6,1) (6,0) (5,0) (4,0) (3,0) (3,1) (2,1) (2,2) (2,3) (2,4) (1,4) (1,3) (0,3) (0,4) (0,5) (1,5) (1,6) (0,6) (0,7) (0,8) <PATH_END>",
        ascii=[
            "#####################",
            "#   #  XXXXX#XXXXE  #",
            "# # # #X###X#X# #####",
            "# #   #XXX#XXX# #   #",
            "# #######X##### # # #",
            "# #XXXXXXX#     # # #",
            "###X# ########### # #",
            "#XXX# #           # #",
            "#X##### ########### #",
            "#X#     #         # #",
            "#X# ######### ### # #",
            "#X#         #   # # #",
            "#X######### # # ### #",
            "#XXXXS#     # #     #",
            "# ########### #######",
            "# #         #   #   #",
            "# # ####### ### # ###",
            "# # #       #   #   #",
            "# # # ####### ##### #",
            "#   #               #",
            "#####################",
        ],
        straightaway_footprints=np.array(
            [
                [6, 2],
                [6, 0],
                [3, 0],
                [3, 1],
                [2, 1],
                [2, 4],
                [1, 4],
                [1, 3],
                [0, 3],
                [0, 5],
                [1, 5],
                [1, 6],
                [0, 6],
                [0, 8],
            ]
        ),
    ),
)


def test_all_tokenizers():
    ALL_TOKENIZERS = _get_all_tokenizers()
    assert len(ALL_TOKENIZERS) > 400
    assert len(_get_all_tokenizers()) == len(ALL_TOKENIZERS)
    assert len({hash(mt) for mt in ALL_TOKENIZERS}) == len(ALL_TOKENIZERS)


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


@mark.parametrize(
    "ep,maze",
    [
        param(tokenizer, maze, id=f"{tokenizer.name}-maze[{i}]")
        for (i, maze), tokenizer in itertools.product(
            enumerate(MIXED_MAZES[:6]),
            all_instances(
                EdgePermuters._EdgePermuter,
                frozendict.frozendict({TokenizerElement: lambda x: x.is_valid()}),
            ),
        )
    ],
)
def test_edge_permuters(ep: EdgePermuters._EdgePermuter, maze: LatticeMaze):
    edges: ConnectionArray = connection_list_to_adj_list(maze.connection_list)
    edges_copy = np.copy(edges)
    old_shape = edges.shape
    permuted: ConnectionArray = ep._permute(edges)
    match ep:
        case EdgePermuters.RandomCoord():
            assert permuted.shape == old_shape
            assert edges is permuted
            i = 0
            while np.array_equal(permuted, edges_copy) and i < 5:
                # Permute again in case for small mazes the random selection happened to not change anything
                permuted: ConnectionArray = ep._permute(permuted)
                i += 1
            assert not np.array_equal(permuted, edges_copy)
        case EdgePermuters.BothCoords():
            new_shape = old_shape[0] * 2, *old_shape[1:]
            n = old_shape[0]
            assert permuted.shape == new_shape
            assert np.array_equal(permuted[:n, ...], edges_copy)
            assert np.array_equal(permuted[:n, 0, :], permuted[n:, 1, :])
            assert np.array_equal(permuted[:n, 1, :], permuted[n:, 0, :])
            assert edges is not permuted


def test_all_tokenizer_hashes():
    loaded_hashes = save_hashes()
    assert np.array_equal(_load_tokenizer_hashes(), loaded_hashes)


@mark.parametrize(
    "pt,manual_maze",
    [
        param(tokenizer, maze_kv[1], id=f"{tokenizer.name}-{maze_kv[0]}")
        for maze_kv, tokenizer in itertools.product(
            _ASCII_MAZES.items(),
            random.sample(
                all_instances(
                    PathTokenizers._PathTokenizer,
                    frozendict.frozendict({TokenizerElement: lambda x: x.is_valid()}),
                ),
                min(
                    3, NUM_TOKENIZERS_TO_TEST
                ),  # TODO: Get rid of "3" when reinstantiating all `StepTokenizer` leaf classes
            ),
        )
    ],
)
def test_path_tokenizers(pt: PathTokenizers._PathTokenizer, manual_maze: _MANUAL_MAZE):
    solved_maze: SolvedMaze = SolvedMaze.from_ascii("\n".join(manual_maze.ascii))
    match type(pt.step_size):
        case StepSizes.Singles:
            footprint_inds = range(solved_maze.solution.shape[0])
        case StepSizes.Straightaways:
            swy_coordtup_set: set[CoordTup] = set(
                tuple(c) for c in manual_maze.straightaway_footprints
            )
            footprint_inds: list[int] = [
                i
                for i, c in enumerate(solved_maze.solution)
                if tuple(c) in swy_coordtup_set
            ]
        case StepSizes.Forks:
            footprint_inds = solved_maze.get_solution_forking_points(
                always_include_endpoints=True
            )[0]
        case StepSizes.ForksAndStraightaways:
            swy_step_inds: list[int] = StepSizes.Straightaways()._step_single_indices(
                solved_maze
            )
            footprint_inds: Int[np.ndarray, "footprint_index"] = np.concatenate(
                (
                    solved_maze.get_solution_forking_points(
                        always_include_endpoints=True
                    )[0],
                    swy_step_inds,
                )
            )
            footprint_inds, _ = np.unique(footprint_inds, axis=0, return_index=True)
    _helper_test_path_tokenizers(
        pt,
        solved_maze,
        footprint_inds,
    )


def _helper_test_path_tokenizers(
    pt: PathTokenizers._PathTokenizer,
    maze: SolvedMaze,
    footprint_inds: Sequence[int],
):
    ct: CoordTokenizers._CoordTokenizer = CoordTokenizers.UT()
    path_toks: list[str] = pt.to_tokens(maze, ct)
    path_toks_set: set[str] = set(path_toks)
    footprint_inds: Int[np.ndarray, "footprint_index"] = np.array(footprint_inds)
    footprints: Int[np.ndarray, "footprint_index row_col=2"] = maze.solution[
        footprint_inds
    ]
    if StepTokenizers.Coord() in pt.step_tokenizers:
        non_steps: set[CoordTup] = set(tuple(c) for c in maze.solution) - set(
            tuple(c) for c in footprints
        )
        assert all([ct.to_tokens(coord)[0] in path_toks_set for coord in footprints])
        assert all([ct.to_tokens(coord)[0] not in path_toks_set for coord in non_steps])
    if StepTokenizers.Distance() in pt.step_tokenizers:
        distances: list[int] = footprint_inds[1:] - footprint_inds[:-1]
        assert (
            len(
                Counter(getattr(VOCAB, f"I_{d:03}") for d in distances)
                - Counter(path_toks)
            )
            == 0
        )
    # TODO: Uncomment tests when restoring full breadth of TokenizerElements
    # if StepTokenizers.Cardinal() in pt.step_tokenizers:
    #     c = Counter(path_toks)
    #     assert c[VOCAB.PATH_NORTH] + c[VOCAB.PATH_SOUTH] + c[VOCAB.PATH_EAST] + c[VOCAB.PATH_WEST] == len(footprint_inds)-1
    # if StepTokenizers.Relative() in pt.step_tokenizers:
    #     c = Counter(path_toks)
    #     assert c[VOCAB.PATH_LEFT] + c[VOCAB.PATH_RIGHT] + c[VOCAB.PATH_FORWARD] + c[VOCAB.PATH_BACKWARD] == len(footprint_inds)-1


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


def _helper_test_path_tokenizers(
    pt: PathTokenizers.PathTokenizer,
    maze: SolvedMaze,
    footprint_inds: Sequence[int],
):
    ct: CoordTokenizers.CoordTokenizer = CoordTokenizers.UT()
    path_toks: list[str] = pt.to_tokens(maze, ct)
    path_toks_set: set[str] = set(path_toks)
    footprint_inds: Int[np.ndarray, "footprint_index"] = np.array(footprint_inds)
    footprints: Int[np.ndarray, "footprint_index row_col=2"] = maze.solution[
        footprint_inds
    ]
    if StepTokenizers.Coord() in pt.step_tokenizers:
        non_steps: set[CoordTup] = set(tuple(c) for c in maze.solution) - set(
            tuple(c) for c in footprints
        )
        assert all([ct.to_tokens(coord)[0] in path_toks_set for coord in footprints])
        assert all([ct.to_tokens(coord)[0] not in path_toks_set for coord in non_steps])
    if StepTokenizers.Distance() in pt.step_tokenizers:
        distances: list[int] = footprint_inds[1:] - footprint_inds[:-1]
        assert (
            len(
                Counter(getattr(VOCAB, f"I_{d:03}") for d in distances)
                - Counter(path_toks)
            )
            == 0
        )
    # TODO: Uncomment tests when restoring full breadth of TokenizerElements
    # if StepTokenizers.Cardinal() in pt.step_tokenizers:
    #     c = Counter(path_toks)
    #     assert c[VOCAB.PATH_NORTH] + c[VOCAB.PATH_SOUTH] + c[VOCAB.PATH_EAST] + c[VOCAB.PATH_WEST] == len(footprint_inds)-1
    # if StepTokenizers.Relative() in pt.step_tokenizers:
    #     c = Counter(path_toks)
    #     assert c[VOCAB.PATH_LEFT] + c[VOCAB.PATH_RIGHT] + c[VOCAB.PATH_FORWARD] + c[VOCAB.PATH_BACKWARD] == len(footprint_inds)-1


@mark.parametrize(
    "pt,manual_maze",
    [
        param(tokenizer, maze_kv[1], id=f"{tokenizer.name}-{maze_kv[0]}")
        for maze_kv, tokenizer in itertools.product(
            _ASCII_MAZES.items(),
            random.sample(
                all_instances(
                    PathTokenizers.PathTokenizer,
                    frozendict.frozendict({TokenizerElement: lambda x: x.is_valid()}),
                ),
                min(
                    3, NUM_TOKENIZERS_TO_TEST
                ),  # TODO: Get rid of "3" when reinstantiating all `StepTokenizer` leaf classes
            ),
        )
    ],
)
def test_path_tokenizers(pt: PathTokenizers.PathTokenizer, manual_maze: _MANUAL_MAZE):
    solved_maze: SolvedMaze = SolvedMaze.from_ascii("\n".join(manual_maze.ascii))
    match type(pt.step_size):
        case StepSizes.Singles:
            footprint_inds = range(solved_maze.solution.shape[0])
        case StepSizes.Straightaways:
            swy_coordtup_set: set[CoordTup] = set(
                tuple(c) for c in manual_maze.straightaway_footprints
            )
            footprint_inds: list[int] = [
                i
                for i, c in enumerate(solved_maze.solution)
                if tuple(c) in swy_coordtup_set
            ]
        case StepSizes.Forks:
            footprint_inds = solved_maze.get_solution_forking_points(
                always_include_endpoints=True
            )[0]
        case StepSizes.ForksAndStraightaways:
            swy_step_inds: list[int] = StepSizes.Straightaways()._step_single_indices(
                solved_maze
            )
            footprint_inds: Int[np.ndarray, "footprint_index"] = np.concatenate(
                (
                    solved_maze.get_solution_forking_points(
                        always_include_endpoints=True
                    )[0],
                    swy_step_inds,
                )
            )
            footprint_inds, _ = np.unique(footprint_inds, axis=0, return_index=True)
    _helper_test_path_tokenizers(
        pt,
        solved_maze,
        footprint_inds,
    )


@mark.parametrize(
    "ep,maze",
    [
        param(tokenizer, maze, id=f"{tokenizer.name}-maze[{i}]")
        for (i, maze), tokenizer in itertools.product(
            enumerate(MIXED_MAZES[:6]),
            all_instances(
                EdgePermuters.EdgePermuter,
                frozendict.frozendict({TokenizerElement: lambda x: x.is_valid()}),
            ),
        )
    ],
)
def test_edge_permuters(ep: EdgePermuters.EdgePermuter, maze: LatticeMaze):
    edges: ConnectionArray = connection_list_to_adj_list(maze.connection_list)
    edges_copy = np.copy(edges)
    old_shape = edges.shape
    permuted: ConnectionArray = ep._permute(edges)
    match ep:
        case EdgePermuters.RandomCoord():
            assert permuted.shape == old_shape
            assert edges is permuted
            i = 0
            while np.array_equal(permuted, edges_copy) and i < 5:
                # Permute again in case for small mazes the random selection happened to not change anything
                permuted: ConnectionArray = ep._permute(permuted)
                i += 1
            assert not np.array_equal(permuted, edges_copy)
        case EdgePermuters.BothCoords():
            new_shape = old_shape[0] * 2, *old_shape[1:]
            n = old_shape[0]
            assert permuted.shape == new_shape
            assert np.array_equal(permuted[:n, ...], edges_copy)
            assert np.array_equal(permuted[:n, 0, :], permuted[n:, 1, :])
            assert np.array_equal(permuted[:n, 1, :], permuted[n:, 0, :])
            assert edges is not permuted


@mark.parametrize(
    "es,maze",
    [
        param(tokenizer, maze, id=f"{tokenizer.name}-maze[{i}]")
        for (i, maze), tokenizer in itertools.product(
            enumerate(MIXED_MAZES[:6]),
            all_instances(
                EdgeSubsets.EdgeSubset,
                frozendict.frozendict({TokenizerElement: lambda x: x.is_valid()}),
            ),
        )
    ],
)
def test_edge_subsets(es: EdgeSubsets.EdgeSubset, maze: LatticeMaze):
    edges: ConnectionArray = es._get_edges(maze)
    n: int = maze.grid_n
    match type(es):
        case EdgeSubsets.AllLatticeEdges:
            assert_shape: tuple = (4 * n * (n - 1), 2, 2)
        case EdgeSubsets.ConnectionEdges:
            if not es.walls:
                assert_shape: tuple = (np.count_nonzero(maze.connection_list), 2, 2)
            else:
                assert_shape: tuple = (
                    2 * n * (n - 1) - np.count_nonzero(maze.connection_list),
                    2,
                    2,
                )
    assert edges.dtype == np.int8
    assert assert_shape == tuple(edges.shape)
    assert assert_shape == tuple(
        np.unique(edges, axis=0).shape
    )  # All edges are unique (swapping leading/trailing coords is considered different)
    assert np.array_equal(
        manhattan_distance(edges), np.array([1] * assert_shape[0], dtype=np.int8)
    )


@mark.parametrize(
    "tok_elem,es,maze",
    [
        param(tok_elem, es, maze, id=f"{tok_elem.name}-{es.name}-maze[{i}]")
        for (i, maze), tok_elem, es in itertools.product(
            enumerate(MIXED_MAZES[:6]),
            all_instances(
                EdgeGroupings.EdgeGrouping,
                frozendict.frozendict(
                    {
                        TokenizerElement: lambda x: x.is_valid(),
                        # Add a condition to prune the range space that doesn't affect functionality being tested
                        EdgeGroupings.ByLeadingCoord: lambda x: x.intra
                        and x.connection_token_ordinal == 1,
                    }
                ),
            ),
            all_instances(
                EdgeSubsets.EdgeSubset,
                frozendict.frozendict({TokenizerElement: lambda x: x.is_valid()}),
            ),
        )
    ],
)
def test_edge_subsets(
    tok_elem: EdgeGroupings.EdgeGrouping, es: EdgeSubsets.EdgeSubset, maze: LatticeMaze
):
    edges: ConnectionArray = es._get_edges(maze)
    n: int = maze.grid_n
    groups: Sequence[ConnectionArray] = tok_elem._group_edges(edges)
    match type(tok_elem):
        case EdgeGroupings.Ungrouped:
            assert_shape = edges.shape[0], 1, 2, 2
            assert tuple(groups.shape) == assert_shape
        case EdgeGroupings.ByLeadingCoord:
            assert len(groups) == np.unique(edges[:, 0, :], axis=0).shape[0]
            assert sum(g.shape[0] for g in groups) == edges.shape[0]
            trailing_coords: list[CoordArray] = [g[:, 1, :] for g in groups]
            # vector_diffs is the position vector difference between the trailing coords of each group
            # These are stacked into a single array since we don't care about maintaining group separation
            vector_diffs: CoordArray = np.stack(
                list(flatten([np.diff(g[:, 1, :], axis=0) for g in groups], 1))
            )
            if tok_elem.shuffle_group:
                allowed_diffs = {(1, -1), (1, 1), (0, 2), (2, 0)}
                # The set of all 2D vectors between any 2 coords adjacent to a central coord
                allowed_diffs = allowed_diffs.union(
                    {(-d[0], -d[1]) for d in allowed_diffs}
                )
            else:
                # If vector_diffs are lexicographically sorted, these are the only possible values. Any other value indicates an error in sorting
                allowed_diffs = {(1, -1), (1, 1), (0, 2), (2, 0)}
            assert all(
                tuple(diff) in allowed_diffs for diff in np.unique(vector_diffs, axis=0)
            )
