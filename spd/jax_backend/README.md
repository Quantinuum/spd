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
sorted long-lived storage is preferred. It applies `max_num_str` caps by
selecting the largest live coefficients with a top-k step, then lexsorts the
retained rows before returning storage. The experimental
`search_update_merge_donate` algorithm uses the same top-k search/update/merge
logic, but donates full JIT steps after storage reaches `max_num_str`.

Algorithm selection is currently an internal JAX-backend setting:

```python
import spd.jax_backend as jax_backend

jax_backend.set_algorithm("stack_sort_merge")
jax_backend.set_algorithm("search_update_merge")
jax_backend.set_algorithm("search_update_merge_donate")
```

When using the higher-level runners, advanced users can combine this with a
reusable configured `BackendAdapter`:

```python
import spd

backend = spd.BackendAdapter.from_name("jax", packbit=32, precision="single")
backend.module.set_algorithm("stack_sort_merge")
```

The runner-facing `max_num_str` limit is normalized to a JAX-friendly power of
two. The selected algorithms choose live retained rows by coefficient magnitude
before the final slice, so representation order should not change which
operator is represented at the `max_num_str` boundary. The
`search_update_merge` implementation keeps this behavior in the
`forward_search_update_merge_top_k_jitted` and
`backward_search_update_merge_top_k_jitted` helpers.

`create_op(...)` currently returns lexsorted storage. Other operations may
conservatively clear that metadata when sorted output is not guaranteed.

The custom classes in `sparse_pauli.py` are registered as JAX pytrees so they can continue to flow through `jit`-compiled kernels while exposing a clearer object interface.

## Static channels

`CreateZero`, `ResetZero`, and `Discard` support expectation evaluation and
rotation gradients on JAX CPU and accelerator arrays. Channel transforms and
coefficient-gradient transposes are implemented in `channels.py`; they do not
fall back to NumPy computation. Compact execution uses functional inverse
rotation reconstruction, retaining zero coordinates at cutoff zero and avoiding
buffer donation so native checkpoints remain valid. This path is independent
of the ordinary rotation algorithm setting and uses no per-gate cache.

Snapshots retain native objects within the default 4 GiB CPU and 1 GiB
per-device budgets. Device eviction creates host NumPy arrays; CPU-origin
snapshots convert only when spilled to disk. Restoration preserves device,
precision, and active-index metadata. See [static channels](../../docs/static_channels.md)
for the full contract and [benchmark design](../../docs/static_channel_benchmarks.md)
for proposed performance measurements.
