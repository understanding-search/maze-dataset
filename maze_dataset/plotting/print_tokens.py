import html
from typing import Literal, Sequence

import matplotlib
import numpy as np
from IPython.display import HTML, display
from jaxtyping import UInt8

from maze_dataset.constants import SPECIAL_TOKENS
from maze_dataset.tokenization import tokens_between

RGBArray = UInt8[np.ndarray, "n 3"]

FormatType = Literal["html", "latex", "terminal", None]

TEMPLATES: dict[FormatType, str] = {
    "html": '<span style="color: black; background-color: rgb({clr})">&nbsp{tok}&nbsp</span>',
    "latex": "\\colorbox[RGB]{{ {clr} }}{{ \\texttt{{ {tok} }} }}",
    "terminal": "\033[30m\033[48;2;{clr}m{tok}\033[0m",
}

_COLOR_JOIN: dict[FormatType, str] = {
    "html": ",",
    "latex": ",",
    "terminal": ";",
}


def _escape_tok(
    tok: str,
    fmt: FormatType,
) -> str:
    if fmt == "html":
        return html.escape(tok)
    elif fmt == "latex":
        return tok.replace("_", "\\_").replace("#", "\\#")
    elif fmt == "terminal":
        return tok
    else:
        raise ValueError(f"Unexpected format: {fmt}")


def color_tokens_rgb(
    tokens: list,
    colors: Sequence[Sequence[int]],
    fmt: FormatType = "html",
    template: str | None = None,
    clr_join: str | None = None,
) -> str:
    """tokens will not be escaped if `fmt` is None"""
    # process format
    if fmt is None:
        assert template is not None
        assert clr_join is not None
    else:
        assert template is None
        assert clr_join is None
        template = TEMPLATES[fmt]
        clr_join = _COLOR_JOIN[fmt]

    # put everything together
    output = [
        template.format(
            clr=clr_join.join(map(str, map(int, clr))),
            tok=_escape_tok(tok, fmt),
        )
        for tok, clr in zip(tokens, colors)
    ]

    return " ".join(output)


def color_tokens_cmap(
    tokens: list[str],
    weights: Sequence[float],
    cmap: str | matplotlib.colors.Colormap = "Blues",
    fmt: FormatType = "html",
    template: str | None = None,
    labels: bool = False,
):
    assert len(tokens) == len(weights), f"{len(tokens)} != {len(weights)}"
    weights = np.array(weights)
    # normalize weights to [0, 1]
    weights_norm = matplotlib.colors.Normalize()(weights)

    if isinstance(cmap, str):
        cmap = matplotlib.colormaps.get_cmap(cmap)

    colors: RGBArray = cmap(weights_norm)[:, :3] * 255

    output: str = color_tokens_rgb(
        tokens=tokens,
        colors=colors,
        fmt=fmt,
        template=template,
    )

    if labels:
        if fmt != "terminal":
            raise NotImplementedError("labels only supported for terminal")
        # align labels with the tokens
        output += "\n"
        for tok, weight in zip(tokens, weights):
            # 2 decimal points, left-aligned and trailing spaces to match token length
            weight_str: str = f"{weight:.1f}"
            # omit if longer than token
            if len(weight_str) > len(tok):
                weight_str = " " * len(tok)
            else:
                weight_str = weight_str.ljust(len(tok))
            output += f"{weight_str} "

    return output


# colors roughly made to be similar to visual representation
_MAZE_TOKENS_DEFAULT_COLORS: dict[tuple[str, str], tuple[int, int, int]] = {
    (SPECIAL_TOKENS.ADJLIST_START, SPECIAL_TOKENS.ADJLIST_END): (
        217,
        210,
        233,
    ),  # purple
    (SPECIAL_TOKENS.ORIGIN_START, SPECIAL_TOKENS.ORIGIN_END): (217, 234, 211),  # green
    (SPECIAL_TOKENS.TARGET_START, SPECIAL_TOKENS.TARGET_END): (234, 209, 220),  # red
    (SPECIAL_TOKENS.PATH_START, SPECIAL_TOKENS.PATH_END): (207, 226, 243),  # blue
}


def color_maze_tokens_AOTP(
    tokens: list[str],
    fmt: FormatType = "html",
    template: str | None = None,
) -> str:
    output: list[str] = [
        " ".join(
            tokens_between(
                tokens, start_tok, end_tok, include_start=True, include_end=True
            )
        )
        for start_tok, end_tok in _MAZE_TOKENS_DEFAULT_COLORS.keys()
    ]

    colors: RGBArray = np.array(
        list(_MAZE_TOKENS_DEFAULT_COLORS.values()), dtype=np.uint8
    )

    return color_tokens_rgb(
        tokens=output,
        colors=colors,
        fmt=fmt,
        template=template,
    )


def display_html(html: str):
    display(HTML(html))


def display_color_tokens_rgb(
    tokens: list[str],
    colors: RGBArray,
) -> None:
    html: str = color_tokens_rgb(tokens, colors, fmt="html")
    display_html(html)


def display_color_tokens_cmap(
    tokens: list[str],
    weights: Sequence[float],
    cmap: str | matplotlib.colors.Colormap = "Blues",
) -> None:
    html: str = color_tokens_cmap(tokens, weights, cmap)
    display_html(html)


def display_color_maze_tokens_AOTP(
    tokens: list[str],
) -> None:
    html: str = color_maze_tokens_AOTP(tokens)
    display_html(html)
