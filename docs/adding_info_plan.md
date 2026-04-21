# Plan: Add Truncation Info To Forward And Backward Runs

## Summary

Add truncation-error tracking during both forward and backward runs.
Track one history entry for every executed non-skipped gate. Each entry stores:

- `num_truncated`
- `truncated_l1_norm`
- `truncated_l2_norm`

The later public API should always return an `info` object instead of making the
return shape depend on a flag.

## Intended `info` shape

Use a plain dictionary with these keys:

- `history`
- `num_steps_tracked`
- `total_num_truncated`
- `total_truncated_l1_norm`
- `total_truncated_l2_norm`

Each entry in `history` is a dictionary with:

- `num_truncated`
- `truncated_l1_norm`
- `truncated_l2_norm`

The order of `history` is the order of executed non-skipped gates in that run.
Clifford gates are included with zero truncation values. Skipped operations are
not included.

## Implementation direction

- Extend backend rotation-step interfaces so they return truncation metadata
  together with the updated state.
- Compute truncation statistics at the exact truncation site inside backend code.
- Count truncation from both:
  - threshold truncation from `trunc_val`
  - extra removals caused by `max_num_str`
- Use coefficient magnitudes for the norms:
  - `l1 = sum(abs(c_i))`
  - `l2 = sqrt(sum(abs(c_i)^2))`
- Keep simulation numerics unchanged. Only add bookkeeping.

## Backend scope

The exact truncation logic lives in backend code, so this change is expected to
touch:

- `spd/numpy_backend/kernels.py`
- `spd/jax_backend/kernels.py`
- JAX algorithm modules used by the forward/backward rotation steps

Runner-side collection is expected in:

- `spd/run_circuit.py`
- `spd/backend_adapter.py`

## Tests and docs

Add tests for both NumPy and JAX backends that verify:

- zero truncation gives zero-valued entries
- nonzero truncation records correct counts and norms
- `max_num_str`-driven removals are included
- skipped operations do not create entries
- Clifford gates create zero-error entries
- the final `info` object has the expected shape

Also update the README and add an example showing how to inspect truncation
history.
