# JAX Backend

This backend uses JAX arrays for sparse-Pauli state and JIT-compiled kernels for the heavy transformations.

## Layout

- [`sparse_pauli.py`](sparse_pauli.py): concrete `SparsePauliOp` and `SparsePauliGradientOp`
- [`kernels.py`](kernels.py): backend math kernels and factories
- [`utils.py`](utils.py): packing and formatting helpers

## Representation

- `SparsePauliOp`: `(xz_array, c_array)`
- `SparsePauliGradientOp`: `(xz_array, c_array, grad_c_array)`

For the JAX backend, `c_array` and `grad_c_array` are stored as real-valued
arrays only. Precision is selected globally within the backend as either
single precision (`float32`) or double precision (`float64`).

The runner-facing `max_num_str` limit is applied after the existing JIT merge
path returns. JAX rounds that limit up to the next power of two and then caps
the final slice size with `min(new_size, max_num_str)`.

The custom classes in `sparse_pauli.py` are registered as JAX pytrees so they can continue to flow through `jit`-compiled kernels while exposing a clearer object interface.
