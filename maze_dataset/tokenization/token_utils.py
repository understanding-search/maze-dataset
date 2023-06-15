from typing import Any, Iterable, Literal

from maze_dataset.constants import SPECIAL_TOKENS, CoordTup

WhenMissing = Literal["except", "skip", "include"]


def tokens_between(
    tokens: list[str],
    start_value: str,
    end_value: str,
    include_start: bool = False,
    include_end: bool = False,
) -> list[str]:
    start_idx = tokens.index(start_value) + int(not include_start)
    end_idx = tokens.index(end_value) + int(include_end)

    assert start_idx < end_idx, "Start must come before end"

    return tokens[start_idx:end_idx]


def get_adj_list_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens, SPECIAL_TOKENS["adj_list_start"], SPECIAL_TOKENS["adj_list_end"]
    )


def get_path_tokens(tokens: list[str], trim_end: bool = False) -> list[str]:
    """The path is considered everything from the first path coord to the path_end token, if it exists."""
    if SPECIAL_TOKENS["path_start"] not in tokens:
        raise ValueError(
            f"Path start token {SPECIAL_TOKENS['path_start']} not found in tokens:\n{tokens}"
        )
    start_idx: int = tokens.index(SPECIAL_TOKENS["path_start"]) + int(trim_end)
    end_idx: int | None = None
    if trim_end and (SPECIAL_TOKENS["path_end"] in tokens):
        end_idx = tokens.index(SPECIAL_TOKENS["path_end"])
    return tokens[start_idx:end_idx]


def get_context_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens,
        SPECIAL_TOKENS["adj_list_start"],
        SPECIAL_TOKENS["path_start"],
        include_start=True,
        include_end=True,
    )


def get_origin_token(tokens: list[str]) -> str:
    return tokens_between(
        tokens, SPECIAL_TOKENS["origin_start"], SPECIAL_TOKENS["origin_end"]
    )[0]


def get_target_token(tokens: list[str]) -> str:
    return tokens_between(
        tokens, SPECIAL_TOKENS["target_start"], SPECIAL_TOKENS["target_end"]
    )[0]


def get_tokens_up_to_path_start(
    tokens: list[str], include_start_coord: bool = True
) -> list[str]:
    path_start_idx: int = tokens.index(SPECIAL_TOKENS["path_start"]) + 1
    if include_start_coord:
        return tokens[: path_start_idx + 1]
    else:
        return tokens[:path_start_idx]


def apply_mapping(
    iter: Iterable[Any],
    mapping: dict[Any, Any],
    when_missing: WhenMissing = "skip",
) -> list[Any]:
    """Given a list and a mapping, apply the mapping to the list"""
    output: list = list()
    for item in iter:
        if item in mapping:
            output.append(mapping[item])
            continue
        match when_missing:
            case "skip":
                continue
            case "include":
                output.append(item)
            case "except":
                raise ValueError(f"item {item} is missing from mapping {mapping}")
            case _:
                raise ValueError(f"invalid value for {when_missing = }")
    return output


def tokens_to_coords(
    tokens: list[str],
    maze_data_cfg,  # TODO: cannot type this right now because importing MazeDatasetConfig causes a circular import
    when_noncoord: WhenMissing = "skip",
) -> list[str | CoordTup]:
    return apply_mapping(tokens, maze_data_cfg.token_node_map, when_noncoord)


def coords_to_tokens(
    coords: list[str | CoordTup],
    maze_data_cfg,  # TODO: cannot type this right now because importing MazeDatasetConfig causes a circular import
    when_noncoord: WhenMissing = "skip",
) -> list[str]:
    return apply_mapping(coords, maze_data_cfg.node_token_map, when_noncoord)


def remove_padding_from_token_str(token_str: str) -> str:
    token_str = token_str.replace(f"{SPECIAL_TOKENS['padding']} ", "")
    token_str = token_str.replace(f"{SPECIAL_TOKENS['padding']}", "")
    return token_str
