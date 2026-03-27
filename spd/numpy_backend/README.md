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
