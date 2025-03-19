import abc
import base64
import hashlib
import random
import warnings
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import (
	Any,
	Callable,
	Iterable,
	Literal,
	Mapping,
	Sequence,
	TypedDict,
	TypeVar,
	overload,
)

import numpy as np
from jaxtyping import Bool, Int, UInt32, UInt64
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)
from muutils.json_serialize.util import _FORMAT_KEY
from muutils.kappa import Kappa
from muutils.misc import empty_sequence_if_attr_false, flatten
from muutils.misc.sequence import WhenMissing
from zanj.loading import load_item_recursive

# from maze_dataset import SolvedMaze
from maze_dataset.constants import (
	SPECIAL_TOKENS,
	VOCAB,
	VOCAB_LIST,
	VOCAB_TOKEN_TO_INDEX,
	ConnectionArray,
	ConnectionList,
	Coord,
	CoordTup,
)
from maze_dataset.generation import numpy_rng
from maze_dataset.maze.lattice_maze import LatticeMaze, SolvedMaze
from maze_dataset.token_utils import (
	TokenizerPendingDeprecationWarning,
	_coord_to_strings_indexed,
	_coord_to_strings_UT,
	connection_list_to_adj_list,
	coords_to_strings,
	get_cardinal_direction,
	get_relative_direction,
	is_connection,
	strings_to_coords,
	tokens_between,
)
from maze_dataset.utils import corner_first_ndindex, lattice_connection_array


class TokenError(ValueError):
	"""error for tokenization"""

	pass