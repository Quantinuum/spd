# Backend Object Execution Summary

## Outcome
The public execution helpers now support two usage styles:

### Simple path
Create an initial SPO/SPGO and let SPD infer the backend from that state.

### Advanced path
Construct a `BackendAdapter` once, configure it as needed, and pass it through
`backend=...` across `evolve(...)`, `init_gradient_spo(...)`, and
`backpropagate(...)`.

## Motivation
This keeps the common workflow explicit and state-based.

It also gives advanced users a clean way to:
- reuse one configured backend across multiple calls
- keep packbit / precision choices in one place
- configure JAX algorithm selection before running

## Implemented API

Supported public entry points now accept `backend=None`:
- `evolve(...)`
- `init_gradient_spo(...)`
- `backpropagate(...)`

When `backend` is provided:
- it must be a `BackendAdapter`
- it must match the backend of the provided SPO or SPGO
- its own `name`, `packbit`, and `precision` settings are reused

## Example Workflow

```python
import spd

backend = spd.BackendAdapter.from_name("jax", packbit=32, precision="single")
backend.module.set_algorithm("search_update_merge")
initial_spo = backend.create_initial_spo({"Z": 1.0})

final_spo = spd.evolve(
    initial_spo,
    circ,
    trunc_val=1e-12,
    max_num_str=1000,
    backend=backend,
)
exp_val = final_spo.get_expectation_value()

initial_spgo = spd.init_gradient_spo(
    final_spo,
    basis="0",
    backend=backend,
)

final_spgo, grads = spd.backpropagate(
    initial_spgo,
    circ,
    trunc_val=1e-12,
    max_num_str=1000,
    backend=backend,
)
```

## Package Surface
- `BackendAdapter` is now exported from `spd`
- the same configured backend can be reused across forward and backward flows
- execution helpers validate that the provided backend matches the input state

## Validation
This change is covered by focused end-to-end tests for:
- configured backend reuse across forward and backward
- invalid `backend=` rejection
- backend inference from SPO / SPGO objects

## Status
- [x] `BackendAdapter` exported from `spd`
- [x] `backend=` support added to public runner helpers
- [x] `backend=` support added to `init_gradient_spo(...)`
- [x] configured backend reuse tested
- [x] example script added

## Follow-Up
Remaining deferred items are tracked in
[`docs/todo_plan.md`](todo_plan.md).
