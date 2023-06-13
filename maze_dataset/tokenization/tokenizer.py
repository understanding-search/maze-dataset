# Avoid circular import from training/config.py
from typing import Union  # need Union as "a" | "b" doesn't work

import torch
from muutils.tensor_utils import ATensor, NDArray
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding

from maze_dataset.dataset.dataset import GPTDatasetConfig
from maze_dataset.generation.constants import SPECIAL_TOKENS
from maze_dataset.generation.lattice_maze import LatticeMaze
from maze_dataset.plotting.plot_maze import MazePlot

# if TYPE_CHECKING:
#     from maze_transformer.training.config import ConfigHolder

# pylint: disable=unused-import, abstract-method


class HuggingMazeTokenizer(PreTrainedTokenizer):
    """extension of PreTrainedTokenizer for mazes"""

    vocab: dict[str, int]  # map of token_ids to strings

    bos_token: str = SPECIAL_TOKENS["adj_list_start"]
    eos_token: str = SPECIAL_TOKENS["path_end"]
    pad_token: str = SPECIAL_TOKENS["padding"]
    unk_token: str = "<UNK>"

    vocab_size: int = 0
    additional_special_tokens: list[str] = [
        x for x in SPECIAL_TOKENS.values() if x not in [SPECIAL_TOKENS["padding"]]
    ]

    # Overwrite class attributes
    padding_side = "left"
    truncation_side = "left"  #! strange choice, but it's what we did in pad_sequence

    name_or_path = "maze_tokenizer"

    # TODO: this should just take seq_len_max and max grid n
    def __init__(
        self,
        cfg: Union["ConfigHolder", "GPTDatasetConfig", None] = None,
        token_arr: list[str] | None = None,
        seq_len_max: int | None = None,
        **kwargs,
    ) -> None:
        """takes either a cfg, or a token_arr and seq_len_max. also, kwargs are passed to super `PreTrainedTokenizer`"""

        if cfg is None:
            assert token_arr is not None
            assert seq_len_max is not None
        else:
            assert token_arr is None
            assert seq_len_max is None
            # Avoid isinstance() because of circular import
            if type(cfg).__name__ == "ConfigHolder":
                cfg = cfg.dataset_cfg

            seq_len_max = cfg.seq_len_max
            token_arr = cfg.token_arr

        super().__init__(max_len=seq_len_max, **kwargs)
        # We are having to do evil things here
        vocab: dict[str, int] = {token: i for i, token in enumerate(token_arr)}
        vocab[self.unk_token] = len(vocab)
        self.vocab: dict[str, int] = vocab

        self.added_tokens_encoder: dict[str, int] = vocab
        self.added_tokens_decoder: dict[int, str] = {
            i: token for token, i in vocab.items()
        }

        self.unique_no_split_tokens = token_arr
        self._create_trie(self.unique_no_split_tokens)

        # IDs specified during construction
        self.bos_token_id: int = self.added_tokens_encoder[self.bos_token]
        self.eos_token_id: int = self.added_tokens_encoder[self.eos_token]
        self.pad_token_id: int = self.added_tokens_encoder[self.pad_token]

    def __call__(self, text, **kwargs) -> BatchEncoding:
        """
        Tokenizer will take a list of strings and encode each
        I.e. a single example should be a continuous string
            "a b c d e f" not ["a", "b", "c", "d", "e", "f"]
        """
        try:
            return super().__call__(text, **kwargs)
        except (NotImplementedError, ValueError) as e:
            raise NotImplementedError(
                f"Caught an error during tokenization - probably because you are trying to encode a token not present in the tokenizer's vocabulary",
                f"text: '{text}'",
            )

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        assert len(kwargs) == 0, f"kwargs not supported: {kwargs}"

        return text.split(" ")

    def _convert_token_to_id(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        else:
            raise ValueError(f"Token not in vocab: '{token}'")

    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | ATensor,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> list[str]:
        if isinstance(sequences, torch.Tensor) and sequences.ndim == 1:
            # Because the slow tokenizer behaves differently to fast ones...
            sequences = sequences.unsqueeze(-1)
        return super().batch_decode(sequences, skip_special_tokens, **kwargs)

    def to_ascii(
        self,
        sequence: list[int | str] | ATensor,
        start_post=None,
        end_pos=None,
    ) -> NDArray:
        # Sequence should be a single maze (not batch)
        if isinstance(sequence, list) and isinstance(sequence[0], str):
            str_sequence = sequence  # already decoded
        else:
            # remove padding
            sequence = torch.tensor(sequence)
            assert sequence.ndim == 1, f"Expected 1D sequence, got {sequence.ndim}D"
            sequence = sequence[sequence != self.pad_token_id]
            str_sequence = self.batch_decode(sequence)

        lattice_maze = LatticeMaze.from_tokens(str_sequence)
        return MazePlot(lattice_maze).to_ascii()
