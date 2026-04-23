# Tests

This directory contains regression, conformance, and end-to-end tests for the SPD package.

## Test Categories

- [`test_pauli_product.py`](test_pauli_product.py): low-level Pauli multiplication checks
- [`test_rotation.py`](test_rotation.py): rotation-kernel behavior on known cases
- [`test_clifford.py`](test_clifford.py): one- and two-qubit Clifford transformations
- [`test_sparse_pauli_op_string.py`](test_sparse_pauli_op_string.py): string rendering and simple object behavior
- [`test_backend_conformance.py`](test_backend_conformance.py): semantic agreement between NumPy and JAX backends
- [`test_backend_adapter.py`](test_backend_adapter.py): adapter-level execution dispatch
- [`test_jax_algorithm_switch_forward.py`](test_jax_algorithm_switch_forward.py): JAX algorithm-switch forward tests, including the default-algorithm assertion and explicit legacy-path coverage
- [`test_jax_algorithm_switch_backward.py`](test_jax_algorithm_switch_backward.py): focused parity test for the `search_update_merge` backward path
- [`test_jax_forward_algorithm_matrix.py`](test_jax_forward_algorithm_matrix.py): selected forward-only parity matrix that runs JAX under both forward algorithms while keeping backward coverage separate
- [`test_jax_backward_algorithm_matrix.py`](test_jax_backward_algorithm_matrix.py): selected backward-only parity matrix that runs JAX under both backward-capable algorithms
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
- These tests protect interface cleanup around the backend, adapter, and runner layers.

OpenQASM compatibility note:

- `pytket.qasm` may canonicalize imported OpenQASM gate order, especially within commuting layers.
- Compatibility fixtures should therefore use a canonical source ordering and avoid encoding presentation-only order as part of the expected behavior.
- The built-in-OpenQASM-vs-`pytket` compatibility tests are intended to validate lowered semantics and execution agreement, not literal preservation of source formatting.
- The larger 8-qubit file-based compatibility test runs on `numpy` only to keep default test time under control; both backends are still covered by the smaller OpenQASM string-based execution checks.

## Fixtures And Helpers

- [`conftest.py`](conftest.py): shared backend fixtures
- [`helpers.py`](helpers.py): normalization and assertion helpers used across multiple files
