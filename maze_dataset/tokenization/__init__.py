"""turning a maze into text

- `MazeTokenizerModular` is the new recommended way to do this as of 1.0.0
- legacy `TokenizationMode` enum and `MazeTokenizer` class for supporting existing code
- a whole lot of helper classes and functions

"""

from maze_dataset.tokenization.maze_tokenizer import (
    AdjListTokenizers,
    CoordTokenizers,
    EdgeGroupings,
    EdgePermuters,
    EdgeSubsets,
    MazeTokenizer,
    MazeTokenizerModular,
    PathTokenizers,
    PromptSequencers,
    StepSizes,
    StepTokenizers,
    TargetTokenizers,
    TokenizationMode,
    _TokenizerElement,
    get_tokens_up_to_path_start,
)

__all__ = [
    # submodules
    "all_tokenizers",
    "maze_tokenizer",
    "save_hashes",
    # modular maze tokenization components
    "TokenizationMode",
    "_TokenizerElement",
    "MazeTokenizerModular",
    "PromptSequencers",
    "CoordTokenizers",
    "AdjListTokenizers",
    "EdgeGroupings",
    "EdgePermuters",
    "EdgeSubsets",
    "TargetTokenizers",
    "StepSizes",
    "StepTokenizers",
    "PathTokenizers",
    # helpers
    "coord_str_to_tuple",
    "get_tokens_up_to_path_start",
    # old tokenizer
    "MazeTokenizer",
]
