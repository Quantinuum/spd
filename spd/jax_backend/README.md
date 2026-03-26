# JAX Backend

This backend uses JAX arrays for sparse-Pauli state and JIT-compiled kernels for the heavy transformations.

## Layout

- [`sparse_pauli.py`](sparse_pauli.py): concrete `SparsePauliOp` and `SparsePauliGradientOp`
- [`kernels.py`](kernels.py): backend math kernels and factories
- [`utils.py`](utils.py): packing and formatting helpers

## Representation

- `SparsePauliOp`: `(xz_array, c_array)`
- `SparsePauliGradientOp`: `(xz_array, c_array, grad_c_array)`

The custom classes in `sparse_pauli.py` are registered as JAX pytrees so they can continue to flow through `jit`-compiled kernels while exposing a clearer object interface.
