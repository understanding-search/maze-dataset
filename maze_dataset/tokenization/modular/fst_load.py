"""to check if a tokenizer is one of our "approved" ones, look in an fst set we made with `rust_fst`

this file handles the creation of this fst file, which we ship to the user

this file relies on importing `get_all_tokenizers` and thus `MazeTokenizerModular`.
as such, loading this file for validating a tokenizer is the separate `maze_dataset.tokenization.modular.fst_load`
module, since we need to be able to import that from `maze_dataset.tokenization.modular.maze_tokenizer_modular` and
we cannot circularly import

thanks to https://github.com/rozbb for suggesting doing this instead of storing a whole bunch of hashes like we were doing before

"""

from functools import cache
from pathlib import Path

from rust_fst import Set as FstSet  # type: ignore[import-untyped]

MMT_FST_PATH: Path = Path(__file__).parent / "MazeTokenizerModular_tested.fst"


@cache
def get_tokenizers_fst() -> FstSet:
	"""(cached) load the tokenizers fst set from `MMT_FST_PATH`"""
	return FstSet(MMT_FST_PATH.as_posix())


def check_tokenizer_in_fst(tokenizer_name: str, do_except: bool = False) -> bool:
	"""check if a tokenizer is in the fst set

	prints nearest matches if `do_except` is `True` and the tokenizer is not found
	"""
	search_0: list[str] = list(get_tokenizers_fst().search(tokenizer_name, 0))
	in_fst: bool = len(search_0) == 1 and search_0[0] == tokenizer_name

	if do_except and not in_fst:
		search_1: list[str] | None = None
		search_2: list[str] | None = None
		try:
			search_1 = list(get_tokenizers_fst().search(tokenizer_name, 1))
			search_2 = list(get_tokenizers_fst().search(tokenizer_name, 2))
		except Exception:  # noqa: BLE001, S110
			# the only thing failing here is getting possible match tokenizers, so it's fine to just ignore the errors
			pass

		err_msg: str = (
			f"Tokenizer `{tokenizer_name}` not found in the list of tested tokenizers, and {do_except = }. We found the following matches based on edit distance:"
			f"\nedit dist 0 (should be empty?): {search_0}"
			+ (f"\nedit dist 1: {search_1}" if search_1 is not None else "")
			+ (f"\nedit dist 2: {search_2}" if search_2 is not None else "")
		)
		raise ValueError(err_msg)

	return in_fst
