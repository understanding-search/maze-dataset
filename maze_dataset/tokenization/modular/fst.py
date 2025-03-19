"""to check if a tokenizer is one of our "approved" ones, we store this in a fst set using `rust_fst`

this file handles the creation of this fst file, which we ship to the user

this file relies on importing `get_all_tokenizers` and thus `MazeTokenizerModular`.
as such, loading this file for validating a tokenizer is the separate `maze_dataset.tokenization.modular.fst_load`
module, since we need to be able to import that from `maze_dataset.tokenization.modular.maze_tokenizer_modular` and
we cannot circularly import

"""

import functools
from typing import ContextManager

import tqdm
from muutils.misc.numerical import shorten_numerical_to_str
from muutils.parallel import run_maybe_parallel
from muutils.spinner import NoOpContextManager, SpinnerContext
from rust_fst import Set as FstSet

from maze_dataset.tokenization.modular.all_tokenizers import get_all_tokenizers
from maze_dataset.tokenization.modular.fst_load import (
	MMT_FST_PATH,
	check_tokenizer_in_fst,
	get_tokenizers_fst,
)


def _get_tokenizer_name(tokenizer) -> str:  # noqa: ANN001
	return tokenizer.name


def save_all_tokenizers_fst(
	verbose: bool = True, parallel: bool | int = False
) -> FstSet:
	"""get all the tokenizers, save an fst file at `MMT_FST_PATH` and return the set"""
	sp: type[ContextManager] = SpinnerContext if verbose else NoOpContextManager

	with sp(message="getting all tokenizers"):
		all_tokenizers: list = get_all_tokenizers()

	n_tokenizers: int = len(all_tokenizers)

	all_tokenizers_names: list[str] = run_maybe_parallel(
		func=_get_tokenizer_name,
		iterable=all_tokenizers,
		parallel=parallel,
		pbar=tqdm.tqdm,
		pbar_kwargs=dict(
			total=n_tokenizers, desc="get name of each tokenizer", disable=not verbose
		),
	)

	assert n_tokenizers == len(all_tokenizers_names)
	print(
		f"# got {shorten_numerical_to_str(n_tokenizers)} ({n_tokenizers}) tokenizers names"
	)

	with sp(message="sorting tokenizer names"):
		all_tokenizers_names_sorted: list[str] = sorted(all_tokenizers_names)

	# construct an fst set and save it
	# we expect it to be 1.6kb or so
	with sp(message="constructing and saving tokenizers fst set"):
		tok_set: FstSet = FstSet.from_iter(
			all_tokenizers_names_sorted,
			path=MMT_FST_PATH.as_posix(),
		)

	print(
		f"# tokenizers fst set saved to {MMT_FST_PATH}, size: {MMT_FST_PATH.stat().st_size} bytes"
	)

	return tok_set


def check_tokenizers_fst(verbose: bool = True, parallel: bool | int = False) -> FstSet:
	"regen all tokenizers, check they are in the pre-existing fst set"
	sp: type[ContextManager] = SpinnerContext if verbose else NoOpContextManager

	with sp(message="getting all tokenizers from scratch"):
		all_tokenizers: list = get_all_tokenizers()

	with sp(message="load the pre-existing tokenizers fst set"):
		get_tokenizers_fst()

	n_tokenizers: int = len(all_tokenizers)

	all_tokenizers_names: list[str] = run_maybe_parallel(
		func=_get_tokenizer_name,
		iterable=all_tokenizers,
		parallel=parallel,
		pbar=tqdm.tqdm,
		pbar_kwargs=dict(
			total=n_tokenizers, desc="get name of each tokenizer", disable=not verbose
		),
	)

	assert n_tokenizers == len(all_tokenizers_names)
	print(
		f"# got {shorten_numerical_to_str(n_tokenizers)} ({n_tokenizers}) tokenizers names"
	)

	check_tokenizer_in_fst__do_except = functools.partial(
		check_tokenizer_in_fst, do_except=True
	)

	run_maybe_parallel(
		func=check_tokenizer_in_fst__do_except,
		iterable=all_tokenizers_names,
		parallel=parallel,
		pbar=tqdm.tqdm,
		pbar_kwargs=dict(
			total=len(all_tokenizers),
			desc="checking tokenizers in fst",
			disable=not verbose,
		),
	)

	print("# all tokenizers are in the pre-existing fst set!")


if __name__ == "__main__":
	import argparse

	arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="save the tokenizers fst set"
	)
	arg_parser.add_argument(
		"-c",
		"--check",
		action="store_true",
		help="check that all tokenizers are in the pre-existing fst set",
	)
	arg_parser.add_argument(
		"-q",
		"--quiet",
		action="store_true",
		help="don't show spinners and progress bars",
	)
	arg_parser.add_argument(
		"-p",
		"--parallel",
		action="store",
		nargs="?",
		type=int,
		const=True,
		default=False,
		help="Control parallelization. will run in serial if nothing specified, use all cpus if flag passed without args, or number of cpus if int passed.",
	)
	args: argparse.Namespace = arg_parser.parse_args()

	if args.check:
		check_tokenizers_fst(verbose=not args.quiet, parallel=args.parallel)
	else:
		save_all_tokenizers_fst(verbose=not args.quiet, parallel=args.parallel)
