# Tests

This directory contains regression, conformance, and end-to-end tests for the SPD package.

## Test Categories

- [`test_pauli_product.py`](test_pauli_product.py): low-level Pauli multiplication checks
- [`test_rotation.py`](test_rotation.py): rotation-kernel behavior on known cases
- [`test_clifford.py`](test_clifford.py): one- and two-qubit Clifford transformations
- [`test_sparse_pauli_op_string.py`](test_sparse_pauli_op_string.py): string rendering and simple object behavior
- [`test_backend_conformance.py`](test_backend_conformance.py): semantic agreement between NumPy and JAX backends
- [`test_backend_adapter.py`](test_backend_adapter.py): adapter-level execution dispatch
- [`test_pytket_frontend.py`](test_pytket_frontend.py): frontend parsing from `pytket` into the internal IR
- [`test_run_pytket_circuit_e2e.py`](test_run_pytket_circuit_e2e.py): end-to-end forward/backward runner tests

## Backend Conformance

[`test_backend_conformance.py`](test_backend_conformance.py) compares NumPy and JAX at the semantic level rather than by internal storage layout.

Current coverage includes:

- `create_op`
- `create_measurement_op`
- `get_size`
- `get_norm_square`
- `get_expectation_value`
- `init_gradient_spo`
- `create_gradient_spo` compatibility alias semantics
- OSE and L2 gradient initialization semantics
- split backward runner coverage via `run_pytket_backward_from_spgo`
- a simple `conjugated_pauli_forward` rotation case
- runner-level `max_num_str` behavior for both backends

Design notes:

- NumPy and JAX store `SPO` / `SPGO` differently, so tests normalize both into Pauli-string keyed dictionaries before comparison.
- Rotation outputs are compared with tolerance because the JAX path defaults to `float32`, while NumPy may keep higher precision.
- `init_gradient_spo(...)` is the canonical initializer; `create_gradient_spo(...)` is covered only for backward compatibility.
- These tests protect interface cleanup around the backend, adapter, and runner layers.

## Fixtures And Helpers

- [`conftest.py`](conftest.py): shared backend fixtures
- [`helpers.py`](helpers.py): normalization and assertion helpers used across multiple files
