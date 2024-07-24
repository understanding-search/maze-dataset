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
    import sys

    all_tokenizers._GET_ALL_TOKENIZERS_SHOW_SPINNER = True

    if len(sys.argv) == 1:
        all_tokenizers.save_hashes(verbose=True)
    elif len(sys.argv) == 2:
        all_tokenizers.save_hashes(sys.argv[1], verbose=True)
    else:
        raise ValueError("Too many arguments")
