from rust_fst import Set as FstSet

from maze_dataset.tokenization.modular.all_tokenizers import get_all_tokenizers
from maze_dataset.tokenization.modular.fst_load import MMT_FST_PATH


def save_all_tokenizers_fst() -> FstSet:
	"""get all the tokenizers, save an fst file at `MMT_FST_PATH` and return the set"""
	# get all the tokenizer names
	all_tokenizers_names_sorted: list[str] = sorted(
		[tokenizer.name for tokenizer in get_all_tokenizers()]
	)

	# construct an fst set and save it
	# we expect it to be 1.6kb or so
	tok_set: FstSet = FstSet.from_iter(all_tokenizers_names_sorted, path=MMT_FST_PATH)
	return tok_set
