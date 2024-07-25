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



"""

import maze_dataset.tokenization.all_tokenizers as all_tokenizers

if __name__ == "__main__":
    import argparse

    """
    def save_hashes(
    path: Path | None = None,
    verbose: bool = False,
    parallelize: bool|int = True,
) -> Int64[np.int64, "tokenizers"]:
"""

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

    args: argparse.Namespace = parser.parse_args()

    all_tokenizers.save_hashes(
        path=args.path,
        verbose=not args.quiet,
        parallelize=args.parallelize,
    )
