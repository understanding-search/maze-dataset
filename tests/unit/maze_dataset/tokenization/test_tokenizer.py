from collections import Counter
from itertools import product
from typing import Iterable

from pytest import mark, param

from maze_dataset import MazeDataset, MazeDatasetConfig, SolvedMaze
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.plotting.print_tokens import color_maze_tokens_AOTP
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode


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
        TokenizationMode.AOTP_indexed,
    ):
        tokenizer: MazeTokenizer = MazeTokenizer(
            tokenization_mode=mode, max_grid_size=100
        )

        assert tokenizer.name == f"maze_tokenizer-{mode.name}-g{100}"

        if mode == TokenizationMode.AOTP_indexed:
            # TODO: fix these asserts
            # assert tokenizer.node_strings_map is None
            pass
        else:
            # assert tokenizer.node_strings_map is not None
            # assert len(tokenizer.node_strings_map) == 100
            assert tokenizer.vocab_size > 100

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


_ASCII_MAZES: dict[str, tuple[str, list[str]]] = dict(
    small_3x3=(
        "<ADJLIST_START> (2,0) <--> (2,1) ; (0,0) <--> (0,1) ; (0,0) <--> (1,0) ; (0,2) <--> (1,2) ; (1,0) <--> (2,0) ; (0,2) <--> (0,1) ; (2,2) <--> (2,1) ; (1,1) <--> (2,1) ; <ADJLIST_END> <ORIGIN_START> (0,0) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (0,0) (1,0) (2,0) (2,1) <PATH_END>",
        [
            "#######",
            "#S    #",
            "#X### #",
            "#X# # #",
            "#X# ###",
            "#XXE  #",
            "#######",
        ],
    ),
    big_10x10=(
        "<ADJLIST_START> (8,2) <--> (8,3) ; (3,7) <--> (3,6) ; (6,7) <--> (6,8) ; (4,6) <--> (5,6) ; (9,5) <--> (9,4) ; (3,3) <--> (3,4) ; (5,1) <--> (4,1) ; (2,6) <--> (2,7) ; (8,5) <--> (8,4) ; (1,9) <--> (2,9) ; (4,1) <--> (4,2) ; (0,8) <--> (0,7) ; (5,4) <--> (5,3) ; (6,3) <--> (6,4) ; (5,0) <--> (4,0) ; (5,3) <--> (5,2) ; (3,1) <--> (2,1) ; (9,1) <--> (9,0) ; (3,5) <--> (3,6) ; (5,5) <--> (6,5) ; (7,1) <--> (7,2) ; (0,1) <--> (1,1) ; (7,8) <--> (8,8) ; (3,9) <--> (4,9) ; (4,6) <--> (4,7) ; (0,6) <--> (0,7) ; (3,4) <--> (3,5) ; (6,0) <--> (5,0) ; (7,7) <--> (7,6) ; (1,6) <--> (0,6) ; (6,1) <--> (6,0) ; (8,6) <--> (8,7) ; (9,9) <--> (9,8) ; (1,8) <--> (1,9) ; (2,1) <--> (2,2) ; (9,2) <--> (9,3) ; (5,9) <--> (6,9) ; (3,2) <--> (2,2) ; (0,8) <--> (0,9) ; (5,6) <--> (5,7) ; (2,3) <--> (2,4) ; (4,5) <--> (4,4) ; (8,9) <--> (8,8) ; (9,6) <--> (8,6) ; (3,7) <--> (3,8) ; (8,0) <--> (7,0) ; (6,1) <--> (6,2) ; (0,1) <--> (0,0) ; (7,3) <--> (7,4) ; (9,4) <--> (9,3) ; (9,6) <--> (9,5) ; (8,7) <--> (7,7) ; (5,2) <--> (5,1) ; (0,0) <--> (1,0) ; (7,2) <--> (7,3) ; (2,5) <--> (2,6) ; (4,9) <--> (5,9) ; (5,5) <--> (5,4) ; (5,6) <--> (6,6) ; (7,8) <--> (7,9) ; (1,7) <--> (2,7) ; (4,6) <--> (4,5) ; (1,1) <--> (1,2) ; (3,1) <--> (3,0) ; (1,5) <--> (1,6) ; (8,3) <--> (8,4) ; (9,9) <--> (8,9) ; (8,5) <--> (7,5) ; (1,4) <--> (2,4) ; (3,0) <--> (4,0) ; (3,3) <--> (4,3) ; (6,9) <--> (6,8) ; (1,0) <--> (2,0) ; (6,0) <--> (7,0) ; (8,0) <--> (9,0) ; (2,3) <--> (2,2) ; (2,8) <--> (3,8) ; (5,7) <--> (6,7) ; (1,3) <--> (0,3) ; (9,7) <--> (9,8) ; (7,5) <--> (7,4) ; (1,8) <--> (2,8) ; (6,5) <--> (6,4) ; (0,2) <--> (1,2) ; (0,7) <--> (1,7) ; (0,3) <--> (0,2) ; (4,3) <--> (4,2) ; (5,8) <--> (4,8) ; (9,1) <--> (8,1) ; (9,2) <--> (8,2) ; (1,3) <--> (1,4) ; (2,9) <--> (3,9) ; (4,8) <--> (4,7) ; (0,5) <--> (0,4) ; (8,1) <--> (7,1) ; (0,3) <--> (0,4) ; (9,7) <--> (9,6) ; (7,6) <--> (6,6) ; (1,5) <--> (0,5) ; <ADJLIST_END> <ORIGIN_START> (6,2) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (6,2) (6,1) (6,0) (5,0) (4,0) (3,0) (3,1) (2,1) <PATH_END>",
        [
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
    ),
)


@mark.parametrize(
    "maze_ascii, tok_mode, tokens",
    [
        param(
            _ASCII_MAZES[maze_ascii_key][1],  # maze_ascii
            tok_mode,  # tok_mode
            _ASCII_MAZES[maze_ascii_key][0],  # tokens
            id=f"{tok_mode.name}_{maze_ascii_key}",
        )
        for maze_ascii_key, tok_mode in product(
            ["small_3x3", "big_10x10"],
            [TokenizationMode.AOTP_UT_uniform, TokenizationMode.AOTP_UT_rasterized],
        )
    ],
)
def test_maze_to_tokens_roundtrip(
    maze_ascii: list[str],
    tok_mode: TokenizationMode,
    tokens: str,
):
    tokens_original_split: list[str] = tokens.split()

    def get_token_regions(toks: list[str]) -> tuple[list[str], list[str]]:
        adj_list_start, adj_list_end = toks.index("<ADJLIST_START>") + 1, tokens.index(
            "<ADJLIST_END>"
        )
        adj_list = toks[adj_list_start:adj_list_end]
        non_adj_list = toks[:adj_list_start] + toks[adj_list_end:]
        return adj_list, non_adj_list

    # join into a single string, and get a maze out
    ascii_str: str = "\n".join(maze_ascii)
    maze: SolvedMaze = SolvedMaze.from_ascii(ascii_str)
    # init tokenizer
    tokenizer: MazeTokenizer = MazeTokenizer(tokenization_mode=tok_mode)

    # maze as tokens
    tokens_from_maze: list[str] = maze.as_tokens(tokenizer)
    adj_list, non_adj_list = get_token_regions(tokens_from_maze)

    # maze round trip
    maze_roundtrip: SolvedMaze = SolvedMaze.from_tokens(tokens_from_maze, tokenizer)
    tokens_roundtrip: list[str] = maze_roundtrip.as_tokens(tokenizer)
    adj_list_rt, non_adj_list_rt = get_token_regions(tokens_roundtrip)

    # regions from original tokens
    adj_list_orig, non_adj_list_orig = get_token_regions(tokens_original_split)

    # check that the maze works
    assert maze == maze_roundtrip

    # check that the counters match
    counter_original: Counter = Counter(tokens_original_split)
    counter_from_maze: Counter = Counter(tokens_from_maze)
    counter_roundtrip: Counter = Counter(tokens_roundtrip)

    assert counter_original == counter_from_maze
    assert counter_original == counter_roundtrip

    # check that the token regions match
    assert non_adj_list_orig == non_adj_list
    assert non_adj_list_rt == non_adj_list
