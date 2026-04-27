# JAX Backend

This backend uses JAX arrays for sparse-Pauli state and JIT-compiled kernels for the heavy transformations.

## Layout

- [`sparse_pauli.py`](sparse_pauli.py): concrete `SparsePauliOp` and `SparsePauliGradientOp`
- [`kernels.py`](kernels.py): public backend entrypoints plus shared low-level math helpers
- [`algorithms/`](algorithms/): strategy-local forward/backward orchestration
- [`utils.py`](utils.py): packing and formatting helpers

## Representation

- `SparsePauliOp`: `(xz_array, c_array)` plus `lexsorted` metadata
- `SparsePauliGradientOp`: `(xz_array, c_array, grad_c_array)` plus `lexsorted` metadata

For the JAX backend, `c_array` and `grad_c_array` are stored as real-valued
arrays only. Precision is selected globally within the backend as either
single precision (`float32`) or double precision (`float64`).

`SparsePauliOp.lexsort()` returns a lexicographically sorted copy with
`lexsorted=True`. `SparsePauliOp.dot(...)` uses a JAX-native matching path and
only requires one sorted haystack internally. `SparsePauliGradientOp.to_spo()`
preserves the primal coefficients and the `lexsorted` flag.

`SparsePauliOp.get_pauli_weight_distribution()` returns the squared coefficient
mass, `|c|^2`, grouped by Pauli weight. `get_pauli_weight_counts()` returns the
number of stored Pauli strings at each weight.

The default JAX algorithm is `stack_sort_merge`. The alternate
`search_update_merge` algorithm remains available when lexicographically
sorted long-lived storage is preferred.

Algorithm selection is currently an internal JAX-backend setting:

```python
import spd.jax_backend as jax_backend

jax_backend.set_algorithm("stack_sort_merge")
jax_backend.set_algorithm("search_update_merge")
```

When using the higher-level runners, advanced users can combine this with a
reusable configured `BackendAdapter`:

```python
import spd

backend = spd.BackendAdapter.from_name("jax", packbit=32, precision="single")
backend.module.set_algorithm("stack_sort_merge")
```

The runner-facing `max_num_str` limit is applied after the active JIT path
returns. JAX rounds that limit up to the next power of two and then caps the
final slice size with `min(new_size, max_num_str)`. Because the two algorithms
use different internal ordering and truncation strategies, they may retain
different subsets near the `max_num_str` boundary.

`create_op(...)` currently returns lexsorted storage. Other operations may
conservatively clear that metadata when sorted output is not guaranteed.

The custom classes in `sparse_pauli.py` are registered as JAX pytrees so they can continue to flow through `jit`-compiled kernels while exposing a clearer object interface.
