# Low-Level API Scope Note (JAX Backend)

## Why this note
Some users may want direct access to `SparsePauliOp` / `xz_array` operations without running full circuit workflows (for example, manual Pauli products or custom operator updates).

## Direction
Keep `kernels.py` focused on runner-critical internals, and define a small, explicit low-level public surface separately (for example `ops.py` / `low_level.py`).

## Candidate low-level public functions
- `create_op`
- `create_measurement_op`
- `pauli_product_uint`
- `pauli_product_batched_second_uint` (optional; mainly for symmetry/tests)
- `merge_` (optional; only if we want supported manual composition)

## Boundary
- `kernels.py`: runtime internals used by `run_circuit` / adapter path
- low-level module: stable, documented primitives for power users
- `legacy.py`: archived/experimental/reference code; not part of supported API

## Ongoing Guidance
With algorithm switching now in place, keep the same separation:
- runtime dispatch in kernels
- small curated low-level API for direct operator manipulation
