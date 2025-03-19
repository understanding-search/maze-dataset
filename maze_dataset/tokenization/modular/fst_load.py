from functools import cache
from pathlib import Path

from rust_fst import Set as FstSet

MMT_FST_PATH: Path = Path(__file__).parent / "MazeTokenizerModular_tested.fst"


@cache
def get_tokenizers_fst() -> FstSet:
	"""(cached) load the tokenizers fst set from `MMT_FST_PATH`"""
	return FstSet(MMT_FST_PATH)


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
