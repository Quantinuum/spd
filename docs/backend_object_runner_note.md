# Backend Object Runner Summary

## Outcome
The public runner helpers now support two usage styles:

### Simple path
Pass `backend_name=...` and let SPD construct the backend internally.

### Advanced path
Construct a `BackendAdapter` once, configure it as needed, and pass it through
`backend=...` across forward and backward calls.

## Motivation
This keeps the common workflow simple while avoiding a growing list of
backend-specific top-level keyword arguments.

It also gives advanced users a clean way to:
- reuse one configured backend across multiple calls
- keep packbit / precision choices in one place
- configure JAX algorithm selection before running

## Implemented API

Supported public entry points now accept `backend=None`:
- `run_pytket_circuit(...)`
- `init_gradient_spo(...)`
- `run_pytket_backward_from_spgo(...)`
- `run_pytket_circuit_backward(...)`
- `run_openqasm_file(...)`
- `run_openqasm_str(...)`
- `run_openqasm_backward_from_spgo(...)`
- `run_openqasm_file_backward(...)`
- `run_openqasm_str_backward(...)`

`backend_name=` remains supported for backward compatibility and the simple
workflow.

When `backend` is provided:
- it must be a `BackendAdapter`
- it takes precedence over `backend_name=...`
- its own `name`, `packbit`, and `precision` settings are used

## Example Workflow

```python
import spd

backend = spd.BackendAdapter.from_name("jax", packbit=32, precision="single")
backend.module.set_algorithm("search_update_merge")

exp_val, final_spo = spd.run_pytket_circuit(
    circ,
    [0],
    trunc_val=1e-12,
    max_num_str=1000,
    backend=backend,
)

initial_spgo = spd.init_gradient_spo(
    final_spo,
    basis="0",
    backend=backend,
)

grads, final_spgo = spd.run_pytket_backward_from_spgo(
    circ,
    initial_spgo,
    trunc_val=1e-12,
    max_num_str=1000,
    backend=backend,
)
```

## Package Surface
- `BackendAdapter` is now exported from `spd`
- the same configured backend can be reused across forward and backward flows
- runner-side padded sizing and JAX `max_num_str` normalization now follow the
  passed backend object

## Validation
This change is covered by focused end-to-end tests for:
- configured backend reuse across forward and backward
- invalid `backend=` rejection
- continued compatibility with the original `backend_name=` workflow

## Status
- [x] `BackendAdapter` exported from `spd`
- [x] `backend=` support added to public runner helpers
- [x] `backend=` support added to `init_gradient_spo(...)`
- [x] configured backend reuse tested
- [x] example script added

## Follow-Up
Remaining deferred items are tracked in
[`docs/todo_plan.md`](todo_plan.md).
