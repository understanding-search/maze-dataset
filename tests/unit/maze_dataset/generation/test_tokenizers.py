"""
Test that the wrapped tokenizer loaded provides the same outputs
as the original tokenizer (i.e. just using the token map in cfg)

We may want a separate set of tests for different tokenization schemes
"""
import torch
from pytest import mark, param
from transformer_lens import HookedTransformer

from maze_dataset.dataset.maze_dataset import MazeDatasetConfig
from maze_dataset.generation.generators import get_maze_with_solution
from maze_dataset.generation.lattice_maze import SolvedMaze
from maze_dataset.training.config import BaseGPTConfig, ConfigHolder


def test_tokenization_encoding():
    # Check that wrapped tokenizer __call__ returns the same as original tokenizer
    solved_maze: SolvedMaze = get_maze_with_solution("gen_dfs", (3, 3))

    # Need to generate a config to extract the token map >.<
    # TODO: Part of https://github.com/AISC-understanding-search/maze-transformer/issues/77
    cfg = MazeDatasetConfig(name="testing_maze", grid_n=3, n_mazes=1)
    node_token_map = cfg.node_token_map

    # Adjacency List Tokenization
    maze_str_tokens = solved_maze.as_tokens(node_token_map)

    # Manual Tokenization
    token_to_index = {token: i for i, token in enumerate(cfg.token_arr)}
    maze_tokens = [token_to_index[token] for token in maze_str_tokens]

    # WrappedTokenizer
    # Initialized with a configholder - tokenizer will eventually be a string
    cfg_holder = ConfigHolder(
        train_cfg=None,
        dataset_cfg=cfg,
        model_cfg=None,
    )

    tokenizer_out = cfg_holder.tokenizer(maze_str_tokens)["input_ids"]
    assert torch.all(
        torch.tensor(tokenizer_out).flatten() == torch.tensor(maze_tokens)
    ), "Tokenization mismatch"


def test_to_ascii():
    # Check that the ascii encoding works for multiple different inputs
    maze_str_tokens = """<ADJLIST_START> (1,1) <--> (2,1) ; (2,0) <--> (1,0) ; (0,1) <--> (0,0) ;
    (2,2) <--> (2,1) ; (2,0) <--> (2,1) ; (0,2) <--> (1,2) ; (0,0) <--> (1,0) ; (0,2) <--> (0,1) ;
    <ADJLIST_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (0,0) (1,0) (2,0) (2,1) <PATH_END>""".split()

    target = [
        "#######",
        "#     #",
        "# ### #",
        "# # # #",
        "# # ###",
        "#     #",
        "#######",
    ]

    # Need to generate a config to extract the token map >.<
    cfg = MazeDatasetConfig(name="testing_maze", grid_n=5, n_mazes=1)
    cfg_holder = ConfigHolder(
        train_cfg=None,
        dataset_cfg=cfg,
        model_cfg=None,
    )

    # Try with string tokens
    assert (
        cfg_holder.tokenizer.to_ascii(maze_str_tokens).splitlines() == target
    ), "ASCII encoding from string tokens failed"
    # And with token ids
    token_ids = cfg_holder.tokenizer.encode(maze_str_tokens)
    assert (
        cfg_holder.tokenizer.to_ascii(token_ids).splitlines() == target
    ), "ASCII encoding from token ids failed"


def test_tokenizer_inside_hooked_transformer():
    # Need to generate a config to extract the token map >.<
    cfg = MazeDatasetConfig(name="testing_maze", grid_n=3, n_mazes=1)

    # Adjacency List Tokenization
    maze_str_tokens = """<ADJLIST_START> (1,1) <--> (2,1) ; (2,0) <--> (1,0) ; (0,1) <--> (0,0) ;
    (2,2) <--> (2,1) ; (2,0) <--> (2,1) ; (0,2) <--> (1,2) ; (0,0) <--> (1,0) ; (0,2) <--> (0,1) ;
    <ADJLIST_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (0,0) (1,0) (2,0) (2,1) <PATH_END>""".split()

    #! Can I initalise this from the config hodler directly by using the nano model cfg
    # refactored on 2023-03-27 15:50 to do just that
    cfg_holder = ConfigHolder(
        train_cfg=None,
        dataset_cfg=cfg,
        model_cfg=BaseGPTConfig(
            name="for test_tokenizer_inside_hooked_transformer",
            act_fn="relu",
            d_model=5,
            d_head=1,
            n_layers=1,
        ),
    )

    hktransformer: HookedTransformer = cfg_holder.create_model()

    token_ids = hktransformer.to_tokens("".join(maze_str_tokens), prepend_bos=False)

    # -- Test Simple Tokenization --
    # Manual Tokenization
    vocab_map = {k: v for v, k in enumerate(cfg.token_arr)}
    maze_tokens = [vocab_map[token] for token in maze_str_tokens]

    assert torch.all(
        token_ids.flatten().cpu() == torch.tensor(maze_tokens)
    ), "Simple tokenization encoding inside HookedTransformer failed"

    # Test casting back to string tokens
    str_tokens = hktransformer.to_str_tokens(token_ids)
    assert (
        str_tokens == maze_str_tokens
    ), "Simple tokenization decoding inside HookedTransformer failed"

    # -- Test Batched Tokenization --
    maze_str_tokens_2 = """<ADJLIST_START> (1,1) <--> (2,1) ; (2,0) <--> (1,0) ;
    (0,1) <--> (0,0) ; (0,2) <--> (0,1) ; <ADJLIST_END> <TARGET_START> (1,0) <TARGET_END>
    <PATH_START> (0,0) (1,0) <PATH_END>""".split()

    batched_tokens = [" ".join(maze_str_tokens), " ".join(maze_str_tokens_2)]

    # Manual Tokenization
    padded_str_2 = ["<PADDING>"] * (
        len(maze_str_tokens) - len(maze_str_tokens_2)
    ) + maze_str_tokens_2
    maze_tokens_2 = [vocab_map[token] for token in padded_str_2]
    batched_tokens_manual = [maze_tokens, maze_tokens_2]

    # WrappedTokenizer use
    token_ids_2 = hktransformer.to_tokens(batched_tokens, prepend_bos=False)

    assert torch.all(
        token_ids_2.cpu() == torch.tensor(batched_tokens_manual)
    ), "Batched tokenization encoding inside HookedTransformer failed"


# Padding Tests
PAD_PLACEHOLDER = -1
test_data = [
    param([1, 2, 3], [PAD_PLACEHOLDER, PAD_PLACEHOLDER, 1, 2, 3], id="short"),
    param([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], id="max_length"),
    param([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6], id="too_long"),
]


@mark.parametrize("inp,expected", test_data)
def test_pad_sequence_param(inp, expected):
    # Initialized with a configholder - tokenizer will eventually be a string
    cfg = MazeDatasetConfig(name="testing_maze", grid_n=3, n_mazes=1)
    cfg_holder = ConfigHolder(
        train_cfg=None,
        dataset_cfg=cfg,
        model_cfg=None,
    )

    # Pad token id is chosen when the tokenizer is initialized
    expected = [
        x if x != PAD_PLACEHOLDER else cfg_holder.tokenizer.pad_token_id
        for x in expected
    ]

    # Need to go to string representation to pad
    inp = cfg_holder.tokenizer.decode(inp)
    result = cfg_holder.tokenizer(
        inp, padding="max_length", truncation=True, max_length=5, return_tensors="pt"
    )["input_ids"][0]

    assert torch.equal(result, torch.tensor(expected))


def test_manual_tokenizer():
    """tests setting the kwargs for a pretrained tokenizer, instead of getting HuggingMazeTokenizer

    this is mostly just testing to make sure it doesnt crash lol
    """

    cfg: ConfigHolder = ConfigHolder(
        train_cfg=None,
        dataset_cfg=MazeDatasetConfig(name="testing_maze", grid_n=3, n_mazes=1),
        model_cfg=None,
        pretrainedtokenizer_kwargs=dict(
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
        ),
    )

    tok = cfg.tokenizer

    assert tok.bos_token == "<bos>"
    assert tok.eos_token == "<eos>"
    assert tok.pad_token == "<pad>"
