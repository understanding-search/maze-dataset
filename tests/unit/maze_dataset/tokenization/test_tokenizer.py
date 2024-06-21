import itertools
import frozendict
import re
import random
from collections import namedtuple, Counter
from itertools import product
from typing import Iterable

import numpy as np
from pytest import mark, param
from muutils.mlutils import GLOBAL_SEED

from maze_dataset import (
    Coord,
    CoordTup,
    LatticeMaze,
    MazeDataset,
    MazeDatasetConfig,
    SolvedMaze,
    TargetedLatticeMaze,
    VOCAB,
)
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.plotting.print_tokens import color_maze_tokens_AOTP
from maze_dataset.tokenization import (
    AdjListTokenizers,
    EdgeGroupings,
    EdgePermuters,
    EdgeSubsets,
    CoordTokenizers,
    MazeTokenizer,
    MazeTokenizer2,
    PathTokenizers,
    PromptSequencers,
    StepTokenizers,
    TargetTokenizers,
    TokenizationMode,
    TokenizerElement,
)
from maze_dataset.util import equal_except_adj_list_sequence
from maze_dataset.testing_utils import (
    MIXED_MAZES,
    MAZE_DATASET,
    ASCII_MAZES,
)
from maze_dataset.utils import all_instances


def test_tokenizer():
    cfg: MazeDatasetConfig = MazeDatasetConfig(
        name="test",
        grid_n=5,
        n_mazes=3,
        maze_ctor=LatticeMazeGenerators.gen_dfs,
    )
    # to create a dataset, just call MazeDataset.from_config
    dataset: MazeDataset = MazeDataset.from_config(
        cfg,
        do_download=False,
        load_local=False,
        do_generate=True,
        save_local=False,
        verbose=True,
        gen_parallel=False,
    )

    for mode in (
        TokenizationMode.AOTP_UT_rasterized,
        TokenizationMode.AOTP_UT_uniform,
        TokenizationMode.AOTP_CTT_indexed,
    ):
        tokenizer: MazeTokenizer = MazeTokenizer(
            tokenization_mode=mode, max_grid_size=100
        )

        assert tokenizer.name == f"maze_tokenizer-{mode.name}-g{100}"

        if mode == TokenizationMode.AOTP_CTT_indexed:
            # TODO: fix these asserts
            assert tokenizer.node_strings_map is not None
            # assert len(tokenizer.node_strings_map) == 100  # `tokenizer.node_strings_map` is a `Kappa` which has no length
            assert 100 < tokenizer.vocab_size < 200
        elif mode in (
            TokenizationMode.AOTP_UT_rasterized,
            TokenizationMode.AOTP_UT_uniform,
        ):
            assert tokenizer.node_strings_map is None
            assert tokenizer.vocab_size > 10000

        assert isinstance(tokenizer.token_arr, Iterable)
        assert all(isinstance(token, str) for token in tokenizer.token_arr)
        assert len(tokenizer.token_arr) == tokenizer.vocab_size

        print(tokenizer.summary())

        for maze in dataset:
            # clear the cache here so we test if it works fine on the next loop
            tokenizer.clear_cache()

            maze_tok = maze.as_tokens(maze_tokenizer=tokenizer)

            maze_encoded = tokenizer.encode(maze_tok)
            maze_decoded = tokenizer.decode(maze_encoded)

            assert maze_tok == maze_decoded

            # you can view the tokens directly
            print("\nRaw tokens:\n")
            print(" ".join(maze_tok))

            maze_recovered = SolvedMaze.from_tokens(maze_tok, maze_tokenizer=tokenizer)

            assert (maze.connection_list == maze_recovered.connection_list).all()

            # or color and print them in various formats
            print("\nColored tokens, raw html:\n")
            print(color_maze_tokens_AOTP(maze_tok, fmt="html"))
            print("\nColored tokens, raw latex:\n")
            print(color_maze_tokens_AOTP(maze_tok, fmt="latex"))
            print("\nColored tokens, terminal:\n")
            print(color_maze_tokens_AOTP(maze_tok, fmt="terminal"))


@mark.parametrize(
    "maze_ascii, tok_mode, tokens",
    [
        param(
            ASCII_MAZES[maze_ascii_key][1],  # maze_ascii
            tok_mode,  # tok_mode
            ASCII_MAZES[maze_ascii_key][0],  # tokens
            id=f"{tok_mode.name}_{maze_ascii_key}",
        )
        for maze_ascii_key, tok_mode in product(
            ["small_3x3", "big_10x10"],
            [
                TokenizationMode.AOTP_UT_uniform,
                TokenizationMode.AOTP_UT_rasterized,
                TokenizationMode.AOTP_CTT_indexed,
            ],
        )
    ],
)
def test_maze_to_tokens_roundtrip(
    maze_ascii: list[str],
    tok_mode: TokenizationMode,
    tokens: str,
):
    if tok_mode == TokenizationMode.AOTP_CTT_indexed:
        # The hardcoded `tokens` assumes a UT tokenizer.
        # Here we modify `tokens` to match what a `AOTP_CTT_indexed` tokenizer would produce.
        tokens = re.sub(r"\(([0-9]),([0-9])\)", r"(\1 , \2)", tokens)
        tokens = re.sub(r"\(([0-9]+ ,)", r"( \1", tokens)
        tokens = re.sub(r"(, [0-9]+)\)", r"\1 )", tokens)
    tokens_original_split: list[str] = tokens.split()

    # join into a single string, and get a maze out
    ascii_str: str = "\n".join(maze_ascii)
    maze: SolvedMaze = SolvedMaze.from_ascii(ascii_str)
    # init tokenizer
    tokenizer: MazeTokenizer = MazeTokenizer(tokenization_mode=tok_mode)

    # maze as tokens
    tokens_from_maze: list[str] = maze.as_tokens(tokenizer)

    # maze round trip
    maze_roundtrip: SolvedMaze = SolvedMaze.from_tokens(tokens_from_maze, tokenizer)
    tokens_roundtrip: list[str] = maze_roundtrip.as_tokens(tokenizer)

    # check that the mazes and tokens are all equivalent
    assert maze == maze_roundtrip
    assert equal_except_adj_list_sequence(tokens_original_split, tokens_from_maze)
    assert equal_except_adj_list_sequence(tokens_original_split, tokens_roundtrip)


@mark.parametrize(
    "tok_mode, max_grid_size, result",
    [
        param(
            tok_mode,
            max_grid_size,
            MazeTokenizer(tokenization_mode=tok_mode, max_grid_size=max_grid_size),
            id=f"{tok_mode}-{max_grid_size}",
        )
        for tok_mode, max_grid_size in [
            (TokenizationMode.AOTP_CTT_indexed, None),
            (TokenizationMode.AOTP_UT_rasterized, None),
            (TokenizationMode.AOTP_UT_uniform, None),
            (TokenizationMode.AOTP_CTT_indexed, 5),
        ]
    ],
)
def test_to_legacy_tokenizer(
    tok_mode: TokenizationMode, max_grid_size: int | None, result: MazeTokenizer
):
    assert tok_mode.to_legacy_tokenizer(max_grid_size) == result


# MazeTokenizer2 tests
# =====================

# Backwards compatibility tests
# =============================


@mark.parametrize(
    "maze,legacy_tokenizer",
    [
        param(maze[0], tok_spec, id=f"{tok_spec.value}-maze{maze[1]}")
        for maze, tok_spec in itertools.product(
            [(maze, i) for i, maze in enumerate(MIXED_MAZES)],
            [tok_mode for tok_mode in TokenizationMode],
        )
    ],
)
def test_to_tokens_backwards_compatible(
    maze: SolvedMaze, legacy_tokenizer: TokenizationMode
):
    tokenizer: MazeTokenizer2 = MazeTokenizer2.from_legacy(legacy_tokenizer)
    toks: list[str] = maze.as_tokens(tokenizer)
    toks2: list[str] = tokenizer.to_tokens(maze)
    toks_legacy: list[str] = maze.as_tokens(legacy_tokenizer)

    try:
        assert equal_except_adj_list_sequence(toks, toks_legacy)
        assert equal_except_adj_list_sequence(toks2, toks_legacy)
    except AssertionError as e:
        raise AssertionError(
            "Tokens from `as_tokens` and `to_tokens` should be equal to tokens from `as_tokens` with the legacy tokenizer.\n"
            f"{len(toks) = }, {len(toks2) = }, {len(toks_legacy) = }\n"
            f"{toks = }\n{toks2 = }\n{toks_legacy = }",
        ) from e


@mark.parametrize(
    "coords, legacy_tok_mode",
    [
        param(
            coords,
            tok_mode,
            id=f"{tok_mode.value}-coords(type={type(coords[0])},len={len(coords)})",
        )
        for tok_mode, coords in itertools.product(
            [tok_mode for tok_mode in TokenizationMode],
            [
                *[[maze.start_pos] for maze in MAZE_DATASET.mazes[:2]],
                [maze.start_pos for maze in MAZE_DATASET.mazes],
                *[[tuple(maze.start_pos)] for maze in MAZE_DATASET.mazes[:2]],
                [tuple(maze.start_pos) for maze in MAZE_DATASET.mazes],
            ],
        )
    ],
)
def test_coords_to_strings_backwards_compatible(
    coords: list[Coord, CoordTup], legacy_tok_mode: TokenizationMode
):
    tokenizer: MazeTokenizer2 = MazeTokenizer2.from_legacy(legacy_tok_mode)
    legacy_tokenizer = MazeTokenizer(tokenization_mode=legacy_tok_mode)
    strings: list[str] = tokenizer.coords_to_strings(coords)
    strings_legacy: list[str] = legacy_tokenizer.coords_to_strings(coords)
    assert strings == strings_legacy


@mark.parametrize(
    "maze,tok_mode",
    [
        param(maze[0], tok_spec, id=f"{tok_spec.value}-maze{maze[1]}")
        for maze, tok_spec in itertools.product(
            [(maze, i) for i, maze in enumerate(MIXED_MAZES)],
            [tok_mode for tok_mode in TokenizationMode],
        )
    ],
)
def test_from_tokens_backwards_compatible(
    maze: LatticeMaze, tok_mode: TokenizationMode
):
    tokenizer = MazeTokenizer2.from_legacy(tok_mode)
    toks = maze.as_tokens(tok_mode)
    # Equality test of `as_tokens` output done in a separate unit test
    maze_legacy: LatticeMaze = LatticeMaze.from_tokens(toks, tok_mode)
    maze: LatticeMaze = LatticeMaze.from_tokens(toks, tokenizer)
    assert maze == maze_legacy


# General functionality tests
# ===========================


@mark.parametrize(
    "el, result",
    [
        param(elem, result, id=elem.name)
        for elem, result in [
            (CoordTokenizers.CTT(), True),
            (CoordTokenizers.CTT(intra=True), True),
            (CoordTokenizers.UT(), True),
            (AdjListTokenizers.AdjListCoord(), True),
            # (AdjListTokenizers.AdjListCoord(post=True), True), # TODO: this breaks collecting tests
            (TargetTokenizers.Unlabeled(post=True), True),
            (PathTokenizers.StepSequence(), True),
            (
                PathTokenizers.StepSequence(step_tokenizers=(StepTokenizers.Coord(),)),
                True,
            ),
            (
                PathTokenizers.StepSequence(
                    step_tokenizers=(
                        StepTokenizers.Coord(),
                        StepTokenizers.Coord(),
                    )
                ),
                False,
            ),
            (PromptSequencers.AOP(), True),
            (PromptSequencers.AOP(path_tokenizer=PathTokenizers.StepSequence()), True),
            (
                PromptSequencers.AOP(
                    path_tokenizer=PathTokenizers.StepSequence(
                        step_tokenizers=(StepTokenizers.Coord(),)
                    )
                ),
                True,
            ),
            (
                PromptSequencers.AOP(
                    path_tokenizer=PathTokenizers.StepSequence(
                        step_tokenizers=(
                            StepTokenizers.Coord(),
                            StepTokenizers.Coord(),
                        )
                    )
                ),
                True,
            ),
        ]
    ],
)
def test_tokenizer_element_is_valid(el: TokenizerElement, result: bool):
    assert el.is_valid() == result


@mark.parametrize(
    "tokenizer, result",
    [
        param(tokenizer, result, id=str(tokenizer))
        for tokenizer, result in [
            (MazeTokenizer2(), True),
            (MazeTokenizer2.from_legacy(TokenizationMode.AOTP_CTT_indexed), True),
            (MazeTokenizer2(prompt_sequencer=PromptSequencers.AOP()), False),
        ]
    ],
)
def test_is_legacy_equivalent(tokenizer: MazeTokenizer2, result: bool):
    assert tokenizer.is_legacy_equivalent() == result

random.seed(GLOBAL_SEED)
@mark.parametrize(
    "tok_elem,maze",
    [
        param(tok_elem, maze, id=f"{tok_elem.name}-maze[{i}]")
        for (i, maze), tok_elem in itertools.product(
            enumerate(MAZE_DATASET),
            random.sample(all_instances(
                AdjListTokenizers._AdjListTokenizer,
                frozendict.frozendict({TokenizerElement: lambda x: x.is_valid(),}),
            ), 20),
        )
    ],
)
def test_adjlist_tokenizers(tok_elem: AdjListTokenizers._AdjListTokenizer, maze: LatticeMaze):
    toks: list[str] = tok_elem.to_tokens(maze, CoordTokenizers.UT())
    tok_counter: Counter = Counter(toks)
    n: int = maze.grid_n
    edge_count: int = 1  # To be updated in match/case blocks
    group_count: int = 1  # To be updated in match/case blocks

    match tok_elem.edge_subset:
        case EdgeSubsets.AllLatticeEdges():
            edge_count *= n*(n - 1)*2
        case EdgeSubsets.ConnectionEdges(walls=False):
            edge_count *= np.count_nonzero(maze.connection_list)
        case EdgeSubsets.ConnectionEdges(walls=True):
            edge_count *= n*(n - 1)*2 - np.count_nonzero(maze.connection_list)
        case _:
            raise NotImplementedError(f'`match` case missing for {tok_elem.edge_subset=}')
    
    match tok_elem.edge_permuter:
        case EdgePermuters.BothCoords():
            edge_count *= 2
            if tok_elem.edge_subset == EdgeSubsets.ConnectionEdges(walls=True):
                ...
            else:
                ...
        case EdgePermuters.RandomCoords() | EdgePermuters.SortedCoords():
            edge_count *= 1
            group_count = None  # Group count is stochastic

    match type(tok_elem.edge_grouping):
        # TODO: Get group count without relying on `pre` or `post` tokens
        case EdgeGroupings.Ungrouped:
            group_count = edge_count  # Override all above cases
            pass
        case EdgeGroupings.ByLeadingCoord:
            if group_count is not None:
                group_count *= 1
            if tok_elem.edge_grouping.intra:
                assert tok_counter[VOCAB.ADJLIST_INTRA] == edge_count
        case _:
            raise NotImplementedError(f'`match` case missing for {tok_elem.edge_grouping=}')

    match type(tok_elem):
        case AdjListTokenizers.AdjListCoord:
            pass
        case AdjListTokenizers.AdjListCardinal:
            assert tok_counter[VOCAB.PATH_NORTH] + tok_counter[VOCAB.PATH_SOUTH] + tok_counter[VOCAB.PATH_EAST] + tok_counter[VOCAB.PATH_WEST] == edge_count

    if group_count is not None:
        if tok_elem.pre:
            assert tok_counter[VOCAB.ADJLIST_PRE] == group_count
        if tok_elem.post:
            assert tok_counter[VOCAB.ADJACENCY_ENDLINE] == group_count

    assert tok_counter[VOCAB.CONNECTOR] + tok_counter[VOCAB.ADJLIST_WALL] == edge_count