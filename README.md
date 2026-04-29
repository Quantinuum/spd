# spd

Sparse-Pauli dynamics for static quantum circuits.

SPD takes a circuit, evolves a sparse Pauli operator (SPO), computes expectation values, and can backpropagate a sparse Pauli gradient operator (SPGO) through the same circuit.

## What Is In This Repo

- [`spd/`](./spd): package code
- [`examples/`](./examples): example scripts
- [`tests/`](./tests): test suite
- [`docs/`](./docs): notes and design docs

Useful entry points:

- [`spd/run_circuit.py`](./spd/run_circuit.py): public workflow helpers such as `create_spo`, `evolve`, `init_gradient_spo`, and `backpropagate`
- [`spd/backend_adapter.py`](./spd/backend_adapter.py): backend selection and configuration
- [`spd/pytket_frontend.py`](./spd/pytket_frontend.py): `pytket` frontend

Recommended examples:

- [`examples/run_simple_circuit_1.py`](./examples/run_simple_circuit_1.py): smallest forward workflow, including truncation info
- [`examples/gradient/run_tfi_gs_1d.py`](./examples/gradient/run_tfi_gs_1d.py): forward + backward workflow inside an optimization loop
- [`examples/run_with_backend_adapter.py`](./examples/run_with_backend_adapter.py): reusable configured backend

## Installation

Base install:

```bash
pip install -e .
```

If you want to run the `pytket` examples:

```bash
pip install -e .[pytket]
```

If you want to run this on the GPU with JAX,
you need to first install
```bash
pip install "jax[cuda12]"
```


## Current Scope

- Frontends:
  - built-in OpenQASM 2
  - `pytket`
- Backends:
  - NumPy
  - JAX
- Circuit model:
  - static circuits
  - no mid-circuit measurement or feedforward

## Main Workflow

The usual workflow is:

1. create an SPO with `spd.create_spo(...)`
2. evolve it with `spd.evolve(...)`
3. read an expectation value with `final_spo.get_expectation_value(...)`
4. build the terminal gradient object with `spd.init_gradient_spo(...)`
5. run reverse propagation with `spd.backpropagate(...)`

## Minimal Forward Example

This matches the structure of [`examples/run_simple_circuit_1.py`](./examples/run_simple_circuit_1.py).

```python
from pytket.circuit import Circuit

import spd

circ = Circuit(3)
circ.Rz(0.5, 0)
circ.Rx(0.5, 1)
circ.ZZPhase(0.25, 0, 2)
circ.measure_all()

trunc_val = 3e-5
max_num_str = int(1e6)

initial_spo = spd.create_spo({"IZI": 1.0})
final_spo, info = spd.evolve(
    initial_spo,
    circ,
    trunc_val=trunc_val,
    max_num_str=max_num_str,
)

exp_val = final_spo.get_expectation_value()

print("expectation value:", exp_val)
print("final SPO size:", final_spo.get_size())
print("tracked truncation steps:", info["num_steps_tracked"])
```

## Forward + Backward Example

This is the core workflow used in [`examples/gradient/run_tfi_gs_1d.py`](./examples/gradient/run_tfi_gs_1d.py).

```python
import spd

backend = spd.BackendAdapter.from_name("jax", packbit=32, precision="double")
backend.module.set_algorithm("stack_sort_merge")

initial_spo = spd.create_spo(ham_dict, backend=backend)
final_spo, forward_info = spd.evolve(
    initial_spo,
    circ,
    trunc_val=trunc_val,
    max_num_str=max_num_str,
    backend=backend,
)

exp_val = final_spo.get_expectation_value(basis=basis)

initial_spgo = spd.init_gradient_spo(
    final_spo,
    basis=basis,
    backend=backend,
)
final_spgo, raw_grads, backward_info = spd.backpropagate(
    initial_spgo,
    circ,
    trunc_val=trunc_val,
    max_num_str=max_num_str,
    backend=backend,
)

backward_spo = final_spgo.to_spo()
overlap = initial_spo.dot(backward_spo)
```

In the TFI example, `raw_grads` are then combined into parameter gradients for the optimizer.
For overlap diagnostics, `to_spo()` extracts the primal SPO from the backward
object and `dot(...)` compares matching Pauli-string coefficients. If you want a
quantity that should be close to `1`, normalize that overlap in user code.

## L2 Gradient Support

For `loss_type="l2_difference"`, `spd.init_gradient_spo(...)` builds the
terminal gradient object on the support of the current `spo`.

If you need the union support of the current and target operators, use the
backend helper `init_gradient_from_l2_difference_union(...)` directly.

## Input To `create_spo`

`create_spo(...)` accepts two common forms:

- a Pauli-string dictionary such as `{"IZI": 1.0, "ZZI": -0.5}`
- a list of qubit indices such as `[0, 2]`, together with `system_size=...`

Examples:

```python
spo_1 = spd.create_spo({"Z": 1.0})
spo_2 = spd.create_spo([0, 2], system_size=3)
```

## Backends

If you do nothing, SPD uses the NumPy backend.

If you want a reusable configured backend:

```python
import spd

backend = spd.BackendAdapter.from_name("jax", packbit=32, precision="single")
backend.module.set_algorithm("stack_sort_merge")
```

Then pass `backend=backend` to `create_spo`, `evolve`, `init_gradient_spo`, and `backpropagate`.

## OpenQASM Note

SPD also includes a built-in OpenQASM 2 frontend. If you need it, see [`spd/openqasm_frontend.py`](./spd/openqasm_frontend.py) and [`examples/open_qasm/run_openqasm_file.py`](./examples/open_qasm/run_openqasm_file.py).

## Rotation Convention

SPD uses

`exp(-i * theta * P / 2)`

for a Pauli rotation generated by `P`.

For `pytket`, this means the frontend converts from

`exp(-i * param * pi * P / 2)`

to `theta = param * pi`.

## Run Tests

```bash
pytest tests
```

## Assumptions

- The examples above use `pytket` when they build circuits in Python.
- The backward example assumes `ham_dict`, `circ`, `basis`, `trunc_val`, and `max_num_str` already exist, just like in [`examples/gradient/run_tfi_gs_1d.py`](./examples/gradient/run_tfi_gs_1d.py).
