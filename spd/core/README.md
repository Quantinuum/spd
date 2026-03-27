# Core Interfaces

This directory defines small abstract interfaces for the sparse-Pauli domain objects.

## Purpose

[`sparse_pauli.py`](sparse_pauli.py) introduces:

- `BaseSparsePauliOp`
- `BaseSparsePauliGradientOp`

These are intentionally minimal. They describe intrinsic object behavior only:

- size
- norm
- expectation value
- Pauli-weight inspection
- readable string rendering

## Boundary

The core layer does not implement backend math kernels.

- `SPO` / `SPGO` are the domain objects
- backend `kernels.py` files implement transformations that take and return those objects

This keeps the object model explicit while still allowing NumPy and JAX to use different internal representations.
