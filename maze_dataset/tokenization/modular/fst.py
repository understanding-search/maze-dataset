"""to check if a tokenizer is one of our "approved" ones, we store this in a fst set using `rust_fst`

this file handles the creation of this fst file, which we ship to the user

this file relies on importing `get_all_tokenizers` and thus `MazeTokenizerModular`.
as such, loading this file for validating a tokenizer is the separate `maze_dataset.tokenization.modular.fst_load`
module, since we need to be able to import that from `maze_dataset.tokenization.modular.maze_tokenizer_modular` and
we cannot circularly import

"""

from rust_fst import Set as FstSet

from maze_dataset.tokenization.modular.all_tokenizers import get_all_tokenizers
from maze_dataset.tokenization.modular.fst_load import MMT_FST_PATH

from muutils.spinner import SpinnerContext


def save_all_tokenizers_fst() -> FstSet:
	"""get all the tokenizers, save an fst file at `MMT_FST_PATH` and return the set"""
	with SpinnerContext(
		message="getting all tokenizers",
	):
		all_tokenizers: list = get_all_tokenizers()

	with SpinnerContext(
		message="sorting tokenizer names",
	):
		all_tokenizers_names_sorted: list[str] = sorted([
			tokenizer.name 
			for tokenizer in all_tokenizers
		])

	# construct an fst set and save it
	# we expect it to be 1.6kb or so
	with SpinnerContext(
		message="constructing and saving tokenizers fst set",
	):
		tok_set: FstSet = FstSet.from_iter(all_tokenizers_names_sorted, path=MMT_FST_PATH)

	print(f"tokenizers fst set saved to {MMT_FST_PATH}, size: {MMT_FST_PATH.stat().st_size} bytes")

	return tok_set


if __name__ == "__main__":
	save_all_tokenizers_fst()
