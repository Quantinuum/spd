# NumPy Backend

This backend uses Python/NumPy-native data structures for sparse-Pauli state.

## Layout

- [`sparse_pauli.py`](sparse_pauli.py): concrete `SparsePauliOp` and `SparsePauliGradientOp`
- [`kernels.py`](kernels.py): backend math kernels and factories
- [`utils.py`](utils.py): bit-packing and string helpers

## Representation

- `SparsePauliOp`: `dict[packed_pauli, coeff]`
- `SparsePauliGradientOp`: `dict[packed_pauli, (coeff, grad)]`

This backend favors readability and direct Python manipulation, which also makes it a useful reference implementation for backend conformance tests.

For the NumPy backend, stored `coeff` and `grad` values are real-valued only.
Precision is selected globally within the backend as either single precision
(`float32`) or double precision (`float64`).

Runner-level `max_num_str` is enforced here as an upper bound by tightening the
effective truncation threshold and then trimming any remaining ties if needed.
