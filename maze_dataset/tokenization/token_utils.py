"""a whole bunch of utilities for tokenization"""

import warnings

from maze_dataset.constants import SPECIAL_TOKENS
from maze_dataset.tokenization.maze_tokenizer import TokenizationMode, is_UT

# filtering things from a prompt or generated text
# ==================================================


def remove_padding_from_token_str(token_str: str) -> str:
    token_str = token_str.replace(f"{SPECIAL_TOKENS.PADDING} ", "")
    token_str = token_str.replace(f"{SPECIAL_TOKENS.PADDING}", "")
    return token_str


def tokens_between(
    tokens: list[str],
    start_value: str,
    end_value: str,
    include_start: bool = False,
    include_end: bool = False,
    except_when_tokens_not_unique: bool = False,
) -> list[str]:
    if start_value == end_value:
        raise ValueError(
            f"start_value and end_value cannot be the same: {start_value = } {end_value = }"
        )
    if except_when_tokens_not_unique:
        if (tokens.count(start_value) != 1) or (tokens.count(end_value) != 1):
            raise ValueError(
                "start_value or end_value is not unique in the input tokens:",
                f"{tokens.count(start_value) = } {tokens.count(end_value) = }"
                f"{start_value = } {end_value = }",
                f"{tokens = }",
            )
    else:
        if (tokens.count(start_value) < 1) or (tokens.count(end_value) < 1):
            raise ValueError(
                "start_value or end_value is not present in the input tokens:",
                f"{tokens.count(start_value) = } {tokens.count(end_value) = }",
                f"{start_value = } {end_value = }",
                f"{tokens = }",
            )

    start_idx: int = tokens.index(start_value) + int(not include_start)
    end_idx: int = tokens.index(end_value) + int(include_end)

    assert start_idx < end_idx, "Start must come before end"

    return tokens[start_idx:end_idx]


def get_adj_list_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens, SPECIAL_TOKENS.ADJLIST_START, SPECIAL_TOKENS.ADJLIST_END
    )


def get_path_tokens(tokens: list[str], trim_end: bool = False) -> list[str]:
    """The path is considered everything from the first path coord to the path_end token, if it exists."""
    if SPECIAL_TOKENS.PATH_START not in tokens:
        raise ValueError(
            f"Path start token {SPECIAL_TOKENS.PATH_START} not found in tokens:\n{tokens}"
        )
    start_idx: int = tokens.index(SPECIAL_TOKENS.PATH_START) + int(trim_end)
    end_idx: int | None = None
    if trim_end and (SPECIAL_TOKENS.PATH_END in tokens):
        end_idx = tokens.index(SPECIAL_TOKENS.PATH_END)
    return tokens[start_idx:end_idx]


def get_context_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens,
        SPECIAL_TOKENS.ADJLIST_START,
        SPECIAL_TOKENS.PATH_START,
        include_start=True,
        include_end=True,
    )


def get_origin_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens,
        SPECIAL_TOKENS.ORIGIN_START,
        SPECIAL_TOKENS.ORIGIN_END,
        include_start=False,
        include_end=False,
    )


def get_target_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens,
        SPECIAL_TOKENS.TARGET_START,
        SPECIAL_TOKENS.TARGET_END,
        include_start=False,
        include_end=False,
    )


def get_tokens_up_to_path_start(
    tokens: list[str],
    include_start_coord: bool = True,
    tokenization_mode: TokenizationMode = TokenizationMode.AOTP_UT_uniform,
) -> list[str]:
    warnings.warn(
        "`get_tokens_up_to_path_start` assumes a unique token (UT) type tokenizer when `include_start_coord=True`. "
        "This method is deprecated for a tokenizer-agnostic function in a future release.",
        PendingDeprecationWarning,
    )
    path_start_idx: int = tokens.index(SPECIAL_TOKENS.PATH_START) + 1
    if include_start_coord:
        if is_UT(tokenization_mode):
            return tokens[: path_start_idx + 1]
        elif tokenization_mode == TokenizationMode.AOTP_CTT_indexed:
            return tokens[: path_start_idx + 5]
        else:
            raise ValueError(f"Invalid tokenization mode: {tokenization_mode}")
    else:
        return tokens[:path_start_idx]
