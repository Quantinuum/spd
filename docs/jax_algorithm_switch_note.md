# JAX Algorithm Switch Summary

## Outcome
The JAX backend now supports two internal forward/backward algorithms behind a
stable public backend surface:

- `search_update_merge`
- `stack_sort_merge`

The default algorithm is now `stack_sort_merge`.

## Final Structure
- `spd/jax_backend/kernels.py`
  - public JAX backend entrypoints
  - module-level algorithm selection
  - shared low-level helpers
- `spd/jax_backend/algorithms/search_update_merge.py`
  - lexicographically sorted search/update/merge orchestration
- `spd/jax_backend/algorithms/stack_sort_merge.py`
  - legacy stack/sort/merge orchestration

## Implemented Behavior

### `search_update_merge`
- keeps the long-lived stored JAX `spo` lexicographically sorted
- generates conjugated partner candidates
- finds partners through binary-search matching
- updates matched rows directly
- inserts missing partners
- filters by coefficient magnitude
- applies `max_num_str` caps with top-k coefficient selection in
  `forward_search_update_merge_top_k_jitted` and
  `backward_search_update_merge_top_k_jitted`
- returns lexicographically sorted storage

### `stack_sort_merge`
- keeps the previous split-conjugate / stack / sort / deduplicate behavior
- still reorders by descending `|c|` before truncation
- remains available as a fallback and reference path

## Important Semantics
- The two JAX algorithms intentionally do not share the same internal ordering.
- Both selected algorithms now apply `max_num_str` caps by coefficient
  magnitude before returning live storage.
- `search_update_merge` restores lexicographic storage after choosing the
  retained top-coefficient terms.
- Public `SparsePauliOp` / `SparsePauliGradientOp` semantics remain unchanged.

## Gradient Support
- Forward and backward dispatch both route through the selected algorithm.
- `search_update_merge` backward includes fused `grad_i` computation from the
  discovered `P/Q` pairs.
- Targeted forward-only and backward-only parity tests cover both algorithms.

## Validation
The switch work included:
- direct kernel parity tests against NumPy
- selected forward-only JAX dual-algorithm matrix tests
- selected backward-only JAX dual-algorithm matrix tests
- capped top-k parity tests against `stack_sort_merge`
- JAX precision checks
- runner-level conformance checks

## Current User Control
Algorithm selection is currently controlled at the JAX backend layer:

```python
import spd.jax_backend as jax_backend

jax_backend.set_algorithm("search_update_merge")
jax_backend.set_algorithm("stack_sort_merge")
```

The higher-level runner APIs do not expose algorithm selection as a dedicated
keyword argument.

## Status
- [x] Internal algorithm selection added
- [x] `search_update_merge` implemented for forward
- [x] `search_update_merge` implemented for backward
- [x] `stack_sort_merge` restored as the default
- [x] Algorithm-specific code moved into `algorithms/`
- [x] Focused forward/backward matrix tests added
- [x] Benchmark comparison performed

## Follow-Up
Remaining deferred items are tracked in
[`docs/todo_plan.md`](todo_plan.md).
