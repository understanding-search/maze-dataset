"""generate and save the hashes of all supported tokenizers

calls `maze_dataset.tokenization.all_tokenizers.save_hashes()`

Usage:

To save to the default location (inside package, `maze_dataset/tokenization/MazeTokenizerModular_hashes.npy`):
```bash
python -m maze_dataset.tokenization.save_hashes
```

to save to a custom location:
```bash
python -m maze_dataset.tokenization.save_hashes /path/to/save/to.npy
```

to check hashes shipped with the package:
```bash
python -m maze_dataset.tokenization.save_hashes --check
```

"""

from pathlib import Path

import numpy as np
from muutils.spinner import SpinnerContext

import maze_dataset.tokenization.all_tokenizers as all_tokenizers
from maze_dataset.tokenization.maze_tokenizer import (
    _load_tokenizer_hashes,
    get_all_tokenizer_hashes,
)

if __name__ == "__main__":
    # parse args
    # ==================================================
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="generate and save the hashes of all supported tokenizers"
    )

    parser.add_argument("path", type=str, nargs="?", help="path to save the hashes to")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="disable progress bar and spinner"
    )
    parser.add_argument(
        "--parallelize", "-p", action="store_true", help="parallelize the computation"
    )
    parser.add_argument(
        "--check",
        "-c",
        action="store_true",
        help="save to temp location, then compare to existing",
    )

    args: argparse.Namespace = parser.parse_args()

    if not args.check:
        # write new hashes
        # ==================================================
        all_tokenizers.save_hashes(
            path=args.path,
            verbose=not args.quiet,
            parallelize=args.parallelize,
        )

    else:
        # check hashes only
        # ==================================================

        # set up path
        if args.path is not None:
            raise ValueError("cannot use --check with a custom path")
        temp_path: Path = Path("tests/_temp/tok_hashes.npz")
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        # generate and save to temp location
        returned_hashes: np.ndarray = all_tokenizers.save_hashes(
            path=temp_path,
            verbose=not args.quiet,
            parallelize=args.parallelize,
        )

        # load saved hashes
        with SpinnerContext(
            spinner_chars="square_dot",
            update_interval=0.5,
            message="loading saved hashes...",
        ):
            read_hashes: np.ndarray = np.load(temp_path)["hashes"]
            read_hashes_pkg: np.ndarray = _load_tokenizer_hashes()
            read_hashes_wrapped: np.ndarray = get_all_tokenizer_hashes()

        # compare
        with SpinnerContext(
            spinner_chars="square_dot",
            update_interval=0.01,
            message="checking hashes: ",
            format_string="\r{spinner} ({elapsed_time:.2f}s) {message}{value}        ",
            format_string_when_updated=True,
        ) as sp:
            sp.update_value("returned vs read")
            assert np.array_equal(returned_hashes, read_hashes)
            sp.update_value("returned vs _load_tokenizer_hashes")
            assert np.array_equal(returned_hashes, read_hashes_pkg)
            sp.update_value("returned vs get_all_tokenizer_hashes()")
            assert np.array_equal(read_hashes, read_hashes_wrapped)
