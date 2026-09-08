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
- [`examples/run_randomized.py`](./examples/run_randomized.py): unbiased Randomized SPD propagation and error bars
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

## Variational Circuits

`VariationalCircuit` keeps a pytket circuit together with the mapping from its
rotation gates to optimizer parameters. Repeated parameter indices represent
shared parameters. `parameter_gradients(...)` sums their gate gradients and
applies pytket's phase-to-angle conversion.

```python
from spd.ansatz import tfi_1d_hva

ansatz = tfi_1d_hva(params, system_size=12, basis="+")

final_spo, _ = spd.evolve(initial_spo, ansatz.circuit, trunc_val, max_num_str)
initial_spgo = spd.init_gradient_spo(final_spo, basis="+")
_, gate_grads, _ = spd.backpropagate(
    initial_spgo,
    ansatz.circuit,
    trunc_val,
    max_num_str,
)
parameter_grads = ansatz.parameter_gradients(gate_grads)
```

The metadata follows parameterized rotation commands in pytket command order.
Do not rebase or structurally modify the circuit after constructing the
`VariationalCircuit`. Fixed rotations use parameter index `-1`. The optional
gate factors describe affine relations such as `Rx(params[k] / 2, qubit)`.

Stable periodic TFI HVA generators are available as `tfi_1d_hva`,
`tfi_2d_hva`, and `tfi_3d_hva` in `spd.ansatz`. Their interaction terms are
scheduled as disjoint brickwork layers separated by barriers. Odd periodic
dimensions are rejected because they cannot be split into two disjoint
even/odd bond coverings. See `examples/variational_tfi.py` for a complete
forward and backward calculation.


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


## [WIP] Randomized Sparse Pauli Dynamics

The NumPy backend also provides Randomized Sparse Pauli Dynamics (R-SPD). It
replaces deterministic truncation with unbiased pivotal randomized truncation
after exact propagation and merging. One run evolves a sampled SPO whose
persistent support is at most the Pauli-string budget `pauli_budget`.

```python
import spd

result = spd.run_ensemble(
    initial_spo,
    circuit,
    pauli_budget=100,
    runs=200,
    master_seed=7,
    basis="0",
)

print(result.mean)
print(result.sample_variance)
print(result.standard_error)
print(result.estimates)
```

The child seeds and every per-run randomized-truncation record are retained in
the result. `spd.evolve_randomized(...)` evolves a sampled SPO without
evaluating an expectation, and `spd.run_randomized_spd(...)` returns one
estimator sample from one seeded run.

In randomized mode, Pauli rotations use zero coefficient threshold and a
transient capacity of `2 * pauli_budget`. Equal Pauli strings are merged before
`spd.pivotal_truncate(...)` is called. No deterministic epsilon pruning or
top-k pass is applied.

Current R-SPD scope is NumPy, forward propagation, and expectation estimation.
JAX, gradients, backpropagation, and noise analysis are not supported by this
API.

The first prototype names `population_size`, `num_populations`,
`run_population`, and `pivotal_compress` remain compatibility aliases. New code
should use the SPD-native terms above: Pauli-string budget, run/realization,
sampled SPO, and randomized truncation. “Trajectory” is reserved for a method
that follows one Pauli string at a time.

## [WIP] Deterministic-backbone residual-corrected R-SPD

The NumPy backend also provides a separate residual-corrected estimator. A
deterministic top-`K_d` backbone is never randomized. Pivotal truncation acts
only on a sampled correction for the backbone's signed discarded residuals.

```python
result = spd.run_residual_corrected_ensemble(
    initial_spo,
    circuit,
    backbone_budget=1_000,
    correction_budget=10_000,
    runs=32,
    master_seed=7,
    basis="0",
)

print(result.backbone_estimate)
print(result.correction_mean)
print(result.mean)
print(result.standard_error)
```

`spd.evolve_residual_corrected(...)` returns the final deterministic backbone,
sampled correction, and per-gate diagnostics. `spd.run_residual_corrected_spd(...)`
returns one estimator sample. The ensemble API recomputes the deterministic
backbone in every independently seeded run; this is the correctness-first
execution layout.

The default `numerical_zero_tolerance=1e-12` applies a declared numerical-zero
policy. If it removes a non-exact value, the API emits one aggregate warning
and records the removed count and norms. The estimator is then unbiased only
up to that policy. Set `numerical_zero_tolerance=0.0` to remove exact zeros only
and retain the formal unbiasedness guarantee.

The residual-corrected API is NumPy-only, forward-only, and expectation-only.
It does not change direct R-SPD. The feasibility results and design rationale
are documented in
[`docs/deterministic_backbone_residual_correction_handoff.md`](./docs/deterministic_backbone_residual_correction_handoff.md).
