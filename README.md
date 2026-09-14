# spd

State-of-the-art GPU code for sparse-Pauli dynamics (SPD), with a NumPy CPU backend and a legacy JAX backend.

SPD takes a circuit, evolves a sparse Pauli operator (SPO), computes expectation values, and can backpropagate a sparse Pauli gradient operator (SPGO) through the same circuit.

## Installation

For CPU execution with NumPy:

```bash
pip install -e .
```

For NVIDIA GPU execution, **Triton is the recommended backend**. It requires
Linux, a supported NVIDIA GPU and driver, and CUDA-enabled PyTorch:

```bash
pip install -e '.[triton]'
```

The `triton` extra installs PyTorch and Triton; Triton execution is GPU-only.
See the [Triton backend guide](./spd/triton_backend/README.md) for capabilities,
persistent storage, and validation results.

Add `pytket` to run the Python circuit examples:

```bash
pip install -e '.[pytket]'         # CPU
pip install -e '.[triton,pytket]'  # NVIDIA GPU
```

JAX remains available as a legacy backend for existing workflows. To use it on
an NVIDIA GPU, install its CUDA dependencies and select `backend_name="jax"`
explicitly:

```bash
pip install 'jax[cuda12]'
```

Without an explicit backend, `spd.create_spo` selects Triton when its dependencies
and NVIDIA CUDA are available, otherwise NumPy. Existing states keep their backend
through forward and backward execution.

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
- [`examples/gradient/run_tfi_gs.py`](./examples/gradient/run_tfi_gs.py): 1D/2D/3D variational TFI optimization

## Current Scope

- Frontends:
  - built-in OpenQASM 2
  - `pytket`
- Backends:
  - Triton: recommended NVIDIA GPU backend, including forward, diagnostics, gradients, and analysis
  - NumPy: default CPU backend
  - JAX: legacy CPU/GPU backend
- Circuit model:
  - static circuits
  - no mid-circuit measurement or feedforward

## GPU benchmark

On the published 2D Ising scaling workload, SPD Triton measured **8.3–15.9×
faster than the published cuPauliProp GPU results** across 36–324 qubits.
Triton used one A100-SXM4-80GB; the published GPU reference identifies an
A100-SXM-64GB. The figure compares three-trial Triton medians with digitized
published curves; Triton compilation is excluded.

![Triton compared with the published MonoProp scaling benchmark](./benchmarks/monoprop/pauli_scaling_comparison.png)

In the separate 12×12 fixed-observable test, the final step took **0.834 s** for
Triton versus **11.287 s** for cuPauliProp on the same local GPU. All 28 term
counts match the published GPU run, with expectation differences below 6e-16.
These are forward-plus-expectation timings, without gradient or truncation-history
calculation. MonoProp uses a different truncation rule and is faster at the
largest scaling sizes; the comparison does not establish a universal speedup.
See the [benchmark setup, tables, memory, and reproducible data](./benchmarks/monoprop/README.md).

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

## Light-cone pruning

For local observables, opt into geometric light-cone pruning:

```python
final_spo, info = spd.evolve(
    initial_spo, circ, trunc_val, max_num_str,
    pruning="light-cone",
)
print(info["pruning"])  # total, retained, and pruned gate counts

initial_spgo = spd.init_gradient_spo(final_spo, basis="0")
final_spgo, gate_grads, backward_info = spd.backpropagate(
    initial_spgo, circ, trunc_val, max_num_str,
)
```

Planning happens once per forward call, from the actual input SPO. Gates outside
its reverse causal cone are completely skipped. Fixed-depth nearest-neighbor
brickwork circuits and local observables are the main use case. Physical qubit
numbering and packed storage widths stay unchanged. `pruning=None` is the default;
`"light-cone"` is currently the only supported method.

For a fixed multi-step evolution, a single whole-circuit call scans only the
initial observable. For example, with a `CircuitIR` layer:

```python
from spd.circuit_ir import CircuitIR

full_circuit = CircuitIR(layer.system_size, layer.operations * 28)
final_spo, info = spd.evolve(
    initial_spo, full_circuit, trunc_val, max_num_str,
    pruning="light-cone", progress=False,
)
```

Calling `evolve` separately for each step rebuilds the plan from each evolving
SPO. On Triton, support is reduced on the GPU and only the packed support mask
and a validity flag are copied to the host. The scan still grows with the input
term count; it avoids transferring the full state.
Use separate calls when intermediate results are needed, or when rebuilding
cones from truncated states saves enough gate work to justify the scans. Measure
which approach helps the workload. A whole-circuit cone can retain more gates than
per-step cones seeded from truncated states; cheap planning alone does not
promise a speedup for a deep circuit whose cone fills the system. The
[12×12 per-step benchmark](benchmarks/light_cone/monoprop_gpu_support_12x12/README.md)
measured 2.17× speedup with GPU support reduction; whole-circuit pruning retained
more late gates on that workload.

The forward result carries its internal plan through `init_gradient_spo`.
Backward automatically follows that plan and requires the same circuit, including
angles. Omitted rotation gradients are zero in their original slots, so
`VariationalCircuit.parameter_gradients` still works. The returned SPGO is the
backward result for the **reduced circuit**; its adjoint coefficients need not
match an unpruned run. With truncation/caps, forward and backward use the retained
operations' numerical policy. History keeps the original executable-gate slots,
with zero truncation entries for omitted gates.

Skipping is exact at the circuit-algebra level, but skipped rotations do not
apply a cutoff or term cap. An initially sub-threshold coefficient can therefore
survive if every gate is pruned. There is no initial cleanup pass. When all gates
are pruned, the mathematical operator is unchanged.

Use the unmodified forward result with the public gradient initializer. The
record describes one forward call, not arbitrary subsequent algebra or manual
array edits. Another `evolve` call replaces the record and derives a new cone
from its input; this handles growing support during stepwise evolution. Triton's
`evolve_step(..., pruning="light-cone")` also supports the feature, accepting a
`CircuitIR` or an operation sequence. Use the same circuit representation in the
matching backward call. Pruned noise analysis is not yet supported and raises
`NotImplementedError`.

## Forward + Backward Example

This is the core workflow used in [`examples/gradient/run_tfi_gs.py`](./examples/gradient/run_tfi_gs.py).

```python
import spd

backend = spd.BackendAdapter.from_name("triton", precision="double")
# Use "numpy" above to run this example on CPU.

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

In the TFI example, `VariationalCircuit.parameter_gradients(...)` combines
`raw_grads` into parameter gradients for the optimizer.
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

`spd.create_spo(...)` automatically uses Triton on an available NVIDIA CUDA GPU
when the optional dependencies are installed, and NumPy otherwise. Explicit
selection overrides this choice:

```python
cpu_spo = spd.create_spo({"Z": 1.0}, backend_name="numpy")
gpu_spo = spd.create_spo({"Z": 1.0}, backend_name="triton")  # requires CUDA
```

For a reusable configuration, use
`spd.BackendAdapter.from_name("triton", precision="double")` and pass
`backend=backend` to the workflow helpers. JAX is available with `"jax"`.
Explicit Triton requests raise an error when CUDA is unavailable.

`evolve`, `init_gradient_spo`, and `backpropagate` infer the backend from their
input state. Forward and backward evolution preserve their inputs by default.
Triton also supports `in_place=True` in `evolve`, `backpropagate`, and
`backpropagate_noise_analysis` to retain mutable storage across circuit calls;
NumPy and JAX raise `NotImplementedError` for this option.

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
- The backward example assumes `ham_dict`, `circ`, `basis`, `trunc_val`, and `max_num_str` already exist, just like in [`examples/gradient/run_tfi_gs.py`](./examples/gradient/run_tfi_gs.py).
