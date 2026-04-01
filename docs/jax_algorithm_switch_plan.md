# JAX Backend Algorithm Switch Plan

## Goal
Enable switching between multiple JAX forward/backward/merge algorithms without changing public runner APIs.

Current default algorithm:
- Conjugation split into two branches
- Stack/concatenate rows
- Lexicographic sort
- Deduplicate via grouped sum
- Sort by coefficient magnitude
- Truncate

Planned alternative algorithm:
- Keep separate arrays
- Sort one side
- Binary-search from the other side
- Update matches in-place (or scatter-style)
- Combine residuals + truncate

## Non-Goals (for first implementation)
- No changes to `run_circuit.py` public signatures
- No backend behavior changes by default
- No immediate deletion of existing algorithm code

## Proposed Structure

### 1) Add internal algorithm id
Use an internal name like:
- `stack_sort_merge` (default; current behavior)
- `search_update_merge` (future)

### 2) Create strategy modules
Suggested files:
- `spd/jax_backend/algorithms/stack_sort_merge.py`
- `spd/jax_backend/algorithms/search_update_merge.py`

Each strategy should expose the same internal contract:
- `forward_step(spo, xzk, theta, trunc_val, max_num_str)`
- `backward_step(spgo, xzk, theta, trunc_val, max_num_str)`

### 3) Keep public kernel entrypoints stable
`kernels.py` keeps:
- `conjugated_pauli_forward(...)`
- `conjugated_pauli_backward(...)`

These functions dispatch to selected strategy and remain the only call targets for `BackendAdapter`.

### 4) Configuration location
Initial suggestion:
- keep a module-level setting in `jax_backend` (defaulting to `stack_sort_merge`)
- optionally route from `BackendAdapter.from_name(..., algorithm=...)` later

## Migration Steps
1. Extract current stack/sort/merge logic into `stack_sort_merge.py`.
2. Add thin dispatcher in `kernels.py`.
3. Add unit tests asserting identical outputs to today for default strategy.
4. Implement `search_update_merge.py`.
5. Add parity tests (forward/backward/gradients) between two strategies.
6. Add benchmark script comparing runtime and memory.

## Safety/Correctness Checklist
- Same truncation semantics
- Same `max_num_str` semantics
- Same deterministic ordering (or explicitly documented ordering differences)
- Same gradient output shape/value conventions
- Works when valid count is 0 (no `log2(0)` / overslice issues)

## Test Matrix To Run
- `tests/test_run_pytket_circuit_e2e.py`
- `tests/test_rotation.py`
- `tests/test_backend_adapter.py`
- `tests/test_backend_conformance.py`
- full suite before merging

## Notes
- Keep legacy/prototype helpers in `legacy.py` until the new strategy is proven.
- Only remove old paths after parity tests are stable.
