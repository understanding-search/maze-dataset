"""to check if a tokenizer is one of our "approved" ones, we store this in a fst set using `rust_fst`

this file handles the creation of this fst file, which we ship to the user

this file relies on importing `get_all_tokenizers` and thus `MazeTokenizerModular`.
as such, loading this file for validating a tokenizer is the separate `maze_dataset.tokenization.modular.fst_load`
module, since we need to be able to import that from `maze_dataset.tokenization.modular.maze_tokenizer_modular` and
we cannot circularly import

"""

from typing import ContextManager

from muutils.spinner import SpinnerContext, NoOpContextManager
from rust_fst import Set as FstSet

from maze_dataset.tokenization.modular.all_tokenizers import get_all_tokenizers
from maze_dataset.tokenization.modular.fst_load import (
	MMT_FST_PATH,
	check_tokenizer_in_fst,
	get_tokenizers_fst,
)


def save_all_tokenizers_fst(verbose: bool = True) -> FstSet:
	"""get all the tokenizers, save an fst file at `MMT_FST_PATH` and return the set"""

	sp: type[ContextManager] = SpinnerContext if verbose else NoOpContextManager

	with sp(message="getting all tokenizers"):
		all_tokenizers: list = get_all_tokenizers()

	with sp(message="sorting tokenizer names"):
		all_tokenizers_names_sorted: list[str] = sorted(
			[tokenizer.name for tokenizer in all_tokenizers]
		)

	# construct an fst set and save it
	# we expect it to be 1.6kb or so
	with sp(message="constructing and saving tokenizers fst set"):
		tok_set: FstSet = FstSet.from_iter(
			all_tokenizers_names_sorted,
			path=MMT_FST_PATH.as_posix(),
		)

	print(
		f"tokenizers fst set saved to {MMT_FST_PATH}, size: {MMT_FST_PATH.stat().st_size} bytes"
	)

	return tok_set


def check_tokenizers_fst(verbose: bool = True) -> FstSet:
	"regen all tokenizers, check they are in the pre-existing fst set"

	sp: type[ContextManager] = SpinnerContext if verbose else NoOpContextManager

	with sp(message="getting all tokenizers from scratch"):
		all_tokenizers: list = get_all_tokenizers()

	with sp(message="load the pre-existing tokenizers fst set"):
		get_tokenizers_fst()

	import tqdm

	for tok in tqdm.tqdm(
		all_tokenizers,
		total=len(all_tokenizers),
		desc="checking tokenizers in fst",
		disable=not verbose,
	):
		assert check_tokenizer_in_fst(tok.name, do_except=True)


if __name__ == "__main__":
	import argparse
	arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="save the tokenizers fst set"
	)
	arg_parser.add_argument(
		"-c", "--check",
		action="store_true",
		help="check that all tokenizers are in the pre-existing fst set",
	)
	arg_parser.add_argument(
		"-q", "--quiet",
		action="store_true",
		help="don't show spinners and progress bars"
	)
	args: argparse.Namespace = arg_parser.parse_args()

	if args.check:
		check_tokenizers_fst(verbose=not args.quiet)
	else:
		save_all_tokenizers_fst(verbose=not args.quiet)
