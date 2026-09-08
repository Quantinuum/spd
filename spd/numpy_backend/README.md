# NumPy Backend

This backend uses Python/NumPy-native data structures for sparse-Pauli state.

## Layout

- [`sparse_pauli.py`](sparse_pauli.py): concrete `SparsePauliOp` and `SparsePauliGradientOp`
- [`kernels.py`](kernels.py): backend math kernels and factories
- [`utils.py`](utils.py): bit-packing and string helpers

## Representation

- `SparsePauliOp`: `dict[packed_pauli, coeff]`
- `SparsePauliGradientOp`: `dict[packed_pauli, (coeff, grad)]`

This backend favors readability and direct Python manipulation, which also makes it a useful reference implementation for backend conformance tests.

`SparsePauliOp.dot(...)` computes the coefficient overlap on matching Pauli
rows. `inner_product(...)` remains as a compatibility alias. For backward-flow
diagnostics, `SparsePauliGradientOp.to_spo()` extracts the primal SPO.

For the NumPy backend, stored `coeff` and `grad` values are real-valued only.
Precision is selected globally within the backend as either single precision
(`float32`) or double precision (`float64`).

`SparsePauliOp.get_pauli_weight_distribution()` returns the squared coefficient
mass, `|c|^2`, grouped by Pauli weight. `get_pauli_weight_counts()` returns the
number of stored Pauli strings at each weight.

Runner-level `max_num_str` is enforced here as an upper bound by tightening the
effective truncation threshold and then trimming any remaining ties if needed.

## Randomized Sparse Pauli Dynamics

The separate API in [`../randomized.py`](../randomized.py) implements unbiased
Randomized Sparse Pauli Dynamics (R-SPD) for this backend. One run evolves a
sampled SPO at Pauli-string budget `K`; it is not a population of walkers or a
single-Pauli trajectory. The implementation deliberately does not change the
deterministic kernels. Every Pauli rotation calls the existing forward kernel
with `trunc_val=0` and transient capacity `2K`, removes only exact zeros, then
applies standalone pivotal randomized truncation from
[`../randomized_truncation.py`](../randomized_truncation.py) if the merged
support exceeds `K`. The historical `spd.compression` import path remains a
compatibility alias for the first prototype.
