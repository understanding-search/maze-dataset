import numpy as np


def pack_custom_bits(arr: np.ndarray, num_bits: int) -> np.ndarray:
	"""Packs a 1D numpy array of nonnegative integers into a compact byte array, using only `num_bits` per integer in a vectorized manner.

	# Parameters:
	- `arr : np.ndarray`
		A one-dimensional numpy array of nonnegative integers. Each element must be less than 2**num_bits.
	- `num_bits : int`
		The number of bits to represent each integer.

	# Returns:
	- `np.ndarray`
		A numpy array of type np.uint8 containing the packed bits.

	# Raises:
	- `ValueError` : if the input array is not 1D or if any element is out of range.

	# Usage:

	```python
	>>> import numpy as np
	>>> data = np.array([5, 7, 3], dtype=np.uint64)
	>>> packed = pack_custom_bits(data, 3)
	>>> unpacked = unpack_custom_bits(packed, data.shape[0], 3)
	>>> np.array_equal(data, unpacked)
	True
	```
	"""
	if arr.ndim != 1:
		raise ValueError("Input array must be one-dimensional")
	if np.any(arr >= (1 << num_bits)) or np.any(arr < 0):
		raise ValueError("Array elements must be in the range [0, 2**num_bits)")

	n: int = arr.shape[0]
	# Create a (n, num_bits) array where each row is the little‑endian bit representation
	# of the corresponding integer.
	bits: np.ndarray = (
		(arr[:, None] >> np.arange(num_bits, dtype=arr.dtype)) & 1
	).astype(np.uint8)
	# Flatten to a 1D bit array
	bits_flat: np.ndarray = bits.reshape(-1)
	# Pack bits into bytes. np.packbits packs each group of 8 bits where the first element becomes the MSB.
	packed: np.ndarray = np.packbits(bits_flat)
	return packed


def unpack_custom_bits(packed: np.ndarray, n: int, num_bits: int) -> np.ndarray:
	"""Unpacks a compact byte array into a 1D numpy array of integers,
	assuming each integer is represented using `num_bits` bits in little‑endian order.

	# Parameters:
	 - `packed : np.ndarray`
	     A numpy array of type np.uint8 containing the packed bits.
	 - `n : int`
	     The number of integers originally packed.
	 - `num_bits : int`
	     The number of bits used per integer.

	# Returns:
	 - `np.ndarray`
	     A numpy array of type np.uint64 containing the unpacked integers.

	# Usage:

	```python
	>>> import numpy as np
	>>> data = np.array([5, 7, 3], dtype=np.uint64)
	>>> packed = pack_custom_bits(data, 3)
	>>> unpacked = unpack_custom_bits(packed, data.shape[0], 3)
	>>> np.array_equal(data, unpacked)
	True
	```
	"""
	total_bits: int = n * num_bits
	# Unpack bits from the byte array and slice off any padding bits.
	bits_flat: np.ndarray = np.unpackbits(packed)[:total_bits]
	# Reshape into (n, num_bits)
	bits: np.ndarray = bits_flat.reshape(n, num_bits)
	# Each bit position has weight 2**i for little‑endian ordering.
	powers: np.ndarray = 1 << np.arange(num_bits, dtype=np.uint64)
	values: np.ndarray = (bits * powers).sum(axis=1, dtype=np.uint64)
	return values


# Example usage:
if __name__ == "__main__":
	data: np.ndarray = np.array([5, 7, 3], dtype=np.uint64)
	packed_data: np.ndarray = pack_custom_bits(data, 3)
	unpacked_data: np.ndarray = unpack_custom_bits(packed_data, data.shape[0], 3)
	print("Original data:", data)
	print("Packed data (in bytes):", packed_data)
	print("Unpacked data:", unpacked_data)
