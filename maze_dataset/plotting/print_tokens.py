from typing import Sequence
import html

import numpy as np
from jaxtyping import UInt8
from IPython.display import display, HTML
import matplotlib
from maze_dataset.constants import SPECIAL_TOKENS

from maze_dataset.tokenization import tokens_between
from maze_dataset.tokenization.token_utils import get_adj_list_tokens, get_origin_tokens, get_path_tokens, get_target_tokens

RGBArray = UInt8[np.ndarray, "n 3"]

_DEFAULT_TEMPLATE: str = '<span style="color: black; background-color: rgb{clr}">&nbsp{tok}&nbsp</span>'

def color_tokens_rgb(
        tokens: list[str], 
        colors: RGBArray,
        template: str = _DEFAULT_TEMPLATE,
    ):
    output: list[str] = [
        template.format(
            tok=html.escape(tok), 
            clr=tuple(np.array(clr, dtype=np.uint8)),
        )
        for tok, clr in zip(tokens, colors)
    ]
    return ' '.join(output)

def color_tokens_cmap(
        tokens: list[str], 
        weights: Sequence[float],
        cmap: str|matplotlib.colors.Colormap = "Blues",
    ):
    assert len(tokens) == len(weights)
    weights = np.array(weights)

    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    colors: RGBArray = cmap(weights)[:, :3] * 255

    return color_tokens_rgb(tokens, colors)

# these colors are to match those from the original understanding-search talk at the conclusion of AISC 2023
_MAZE_TOKENS_DEFAULT_COLORS: dict[tuple[str, str], tuple[int, int, int]] = {
    (SPECIAL_TOKENS.ADJLIST_START, SPECIAL_TOKENS.ADJLIST_END): (234, 209, 220), # pink
    (SPECIAL_TOKENS.ORIGIN_START, SPECIAL_TOKENS.ORIGIN_END): (217, 210, 233), # purple
    (SPECIAL_TOKENS.TARGET_START, SPECIAL_TOKENS.TARGET_END): (207, 226, 243), # blue    
    (SPECIAL_TOKENS.PATH_START, SPECIAL_TOKENS.PATH_END): (217, 234, 211), # green
}

def color_maze_tokens_AOTP(
    tokens: list[str],
) -> str:
    
    output: list[str] = [
        " ".join(tokens_between(
            tokens, start_tok, end_tok, include_start=True, include_end=True
        ))
        for start_tok, end_tok in _MAZE_TOKENS_DEFAULT_COLORS.keys()
    ]

    colors: RGBArray = np.array(list(_MAZE_TOKENS_DEFAULT_COLORS.values()), dtype=np.uint8)
    
    return color_tokens_rgb(output, colors)

def display_html(html: str):
    display(HTML(html))


def display_color_tokens_rgb(
    tokens: list[str], 
    colors: RGBArray,
    template: str = _DEFAULT_TEMPLATE,
) -> None:
    html: str = color_tokens_rgb(tokens, colors, template)
    display_html(html)

def display_color_tokens_cmap(
    tokens: list[str], 
    weights: Sequence[float],
    cmap: str|matplotlib.colors.Colormap = "Blues",
) -> None:
    html: str = color_tokens_cmap(tokens, weights, cmap)
    display_html(html)

def display_color_maze_tokens_AOTP(
    tokens: list[str],
) -> None:
    html: str = color_maze_tokens_AOTP(tokens)
    display_html(html)