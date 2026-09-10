# Tests

This directory contains regression, conformance, and end-to-end tests for the SPD package.

## Native Triton forward path

`test_backend_semantic_contract.py` records the parity milestone's boundary
semantics across the available implementations. Its strict expected failure
documents the existing JAX stack-sort equality-diagnostics bug; see the
[Triton compatibility contract](../spd/triton_backend/COMPATIBILITY.md).

The Triton backend has a smaller API than NumPy/JAX; passing the repository suite
does not mean it implements their Clifford, SPGO, diagnostics, or arithmetic APIs.
Its implemented forward scope is exercised by:

- `test_triton_backend.py`: randomized multiword rotations against NumPy, caps,
  cancellation, empty states, operation order, and unsupported operations.
- `test_triton_invariants.py`: the same known rotation cases used by NumPy/JAX,
  independent dense matrix conjugation for all 64 three-qubit generators,
  deliberately colliding hash keys, compaction boundaries, uniqueness, inverse
  recovery, norm conservation, measurements, and cutoff-adjacent values.
- `test_triton_jax_conformance.py`: every retained coefficient after multiple
  lattice steps, against both JAX algorithms in single and double precision.
- `test_backend_imports.py`: public import compatibility and native import
  isolation from the JAX GPU allocator.

The numerical Triton tests require an NVIDIA GPU plus PyTorch/Triton and skip
when unavailable; the public import compatibility check can run without CUDA.
Run from the repository root:

```sh
python -m pytest tests/test_triton_backend.py tests/test_triton_invariants.py tests/test_triton_jax_conformance.py tests/test_backend_imports.py -q
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest tests/test_triton_invariants.py -k 'collision or compaction' -q
```

Python line coverage does not measure compiled Triton kernel branches. The
adversarial storage cases and CUDA memory checker complement the numerical
oracles. See [the Triton README](../spd/triton_backend/README.md) for the algorithm,
scope differences, and proposed SPGO extension.

## Test Categories

- [`test_pauli_product.py`](test_pauli_product.py): low-level Pauli multiplication checks
- [`test_rotation.py`](test_rotation.py): rotation-kernel behavior on known cases
- [`test_clifford.py`](test_clifford.py): one- and two-qubit Clifford transformations
- [`test_sparse_pauli_op_string.py`](test_sparse_pauli_op_string.py): string rendering and simple object behavior
- [`test_backend_conformance.py`](test_backend_conformance.py): semantic agreement between NumPy and JAX backends
- [`test_backend_adapter.py`](test_backend_adapter.py): adapter-level execution dispatch
- [`test_jax_algorithm_switch_forward.py`](test_jax_algorithm_switch_forward.py): JAX algorithm-switch forward tests, including the default-algorithm assertion and explicit legacy-path coverage
- [`test_jax_algorithm_switch_backward.py`](test_jax_algorithm_switch_backward.py): focused parity test for the `search_update_merge` backward path
- [`test_jax_forward_algorithm_matrix.py`](test_jax_forward_algorithm_matrix.py): selected forward-only parity matrix that runs JAX under both forward algorithms, including capped top-k behavior
- [`test_jax_backward_algorithm_matrix.py`](test_jax_backward_algorithm_matrix.py): selected backward-only parity matrix that runs JAX under both backward-capable algorithms, including capped top-k behavior
- [`test_openqasm_frontend.py`](test_openqasm_frontend.py): built-in OpenQASM parsing into the internal IR
- [`test_openqasm_pytket_compat.py`](test_openqasm_pytket_compat.py): semantic IR and execution compatibility checks between the built-in OpenQASM frontend and the `pytket` OpenQASM importer
- [`test_pytket_frontend.py`](test_pytket_frontend.py): frontend parsing from `pytket` into the internal IR
- [`test_run_openqasm_e2e.py`](test_run_openqasm_e2e.py): end-to-end execution through the built-in OpenQASM path
- [`test_run_pytket_circuit_e2e.py`](test_run_pytket_circuit_e2e.py): end-to-end `evolve(...)` / `backpropagate(...)` tests
- [`test_truncation_info.py`](test_truncation_info.py): focused truncation-info checks for both backends

## Backend Conformance

[`test_backend_conformance.py`](test_backend_conformance.py) compares NumPy and JAX at the semantic level rather than by internal storage layout.

Current coverage includes:

- `create_op`
- `create_measurement_op`
- `init_gradient_spo`
- OSE and L2 gradient initialization semantics
- split backward coverage via `init_gradient_spo(...)` plus `backpropagate(...)`
- a simple `conjugate_pauli_rot_forward` rotation case
- runner-level `max_num_str` behavior for both backends

Design notes:

- NumPy and JAX store `SPO` / `SPGO` differently, so tests normalize both into Pauli-string keyed dictionaries before comparison.
- Rotation outputs are compared with tolerance because the JAX path defaults to `float32`, while NumPy may keep higher precision.
- `init_gradient_spo(...)` is the canonical initializer.
- `init_gradient_from_l2_difference(...)` keeps the current-SPO support, while
  `init_gradient_from_l2_difference_union(...)` keeps the union support.
- These tests protect interface cleanup around the backend, adapter, and runner layers.

OpenQASM compatibility note:

- `pytket.qasm` may canonicalize imported OpenQASM gate order, especially within commuting layers.
- Compatibility fixtures should therefore use a canonical source ordering and avoid encoding presentation-only order as part of the expected behavior.
- The built-in-OpenQASM-vs-`pytket` compatibility tests are intended to validate lowered semantics and execution agreement, not literal preservation of source formatting.
- The larger 8-qubit file-based compatibility test runs on `numpy` only to keep default test time under control; both backends are still covered by the smaller OpenQASM string-based execution checks.

## Fixtures And Helpers

- [`conftest.py`](conftest.py): shared backend fixtures
- [`helpers.py`](helpers.py): normalization and assertion helpers used across multiple files

## Triton milestone 2

`test_triton_backward.py` checks primal/adjoint sequences against NumPy and both
JAX algorithms, independent dense finite differences, gradient-only support,
truncation diagnostics, caps, and collision/compaction stress.
`test_triton_clifford.py` checks all nine gates in both directions and precisions
against dense matrices and NumPy, including packed-word boundaries.
The semantic contract now exercises Triton backward and diagnostic calls.
Terminal loss and public runner tests were added in milestone 3 below.

```sh
python -m pytest -q
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest tests/test_triton_backward.py tests/test_triton_clifford.py -q
```

## Triton milestone 3

`test_triton_runner.py` covers terminal basis/OSE adjoints against JAX and
coefficient finite differences; 2x2/4x4 TFI energies, entropies and parameter
gradients against both JAX algorithms; independent dense two-layer TFI finite
differences; capped mixed Clifford/rotation circuits; IR, pickle round trips,
empty states, precision and backend inference. Subprocess tests run all four
TFI optimizer modes and assert JAX does not initialize a GPU backend in the
Triton Adam path. Progress-disabled execution is checked to skip reporting
reductions while retaining diagnostics.

The mixed capped inverse test permits a one-row discarded-count difference at
one specific cancellation: JAX may leave a roundoff-sized nonzero primal where
Triton obtains zero. Norms and state/gradient values must still agree. This is
separate from the documented JAX stack cutoff-equality bug.

## Triton milestone 4

The `all_backend` fixture runs shared arithmetic, complex Pauli-product and
OpenQASM runner cases against NumPy, JAX and Triton. Noise cases also include
Triton, including dense finite differences. `test_triton_algebra_analysis.py`
adds sparse-join collision/compaction stress, empty and zero support, L2/OSE
finite differences, packed-prefix translation with nonzero suffixes, weight
histograms, complex multiword phases, rebase, and abstract/export checks.

Validation: **1,077 passed, 1 expected failure**; new GPU memory checks:
**77 passed, 0 errors**. See the [milestone report](../spd/triton_backend/MILESTONE4.md).
