# Fixed-index static channels

SPD supports `CreateZero(i)`, `ResetZero(i)`, and `Discard(i)` in `CircuitIR`.
NumPy implements expectation evaluation and rotation gradients for these
operations. JAX and Triton channel execution raises `NotImplementedError`.
Select `backend_name="numpy"` explicitly when creating a channel observable.

## Circuit semantics

`system_size` defines the fixed index universe `0, ..., system_size - 1`.
Indices with a `CreateZero` instruction start inactive; every other index is an
ordinary, initially active input. Circuits without creation retain their
all-active input convention.

| Operation | State action | Activity requirement and result |
| --- | --- | --- |
| `CreateZero(i)` | Introduce index i in state zero | Requires inactive i; activates it |
| `ResetZero(i)` | Trace out i's state and replace it with state zero | Requires active i; keeps it active |
| `Discard(i)` | Trace out i | Requires active i; deactivates it |

Static validation permits at most one creation per index. Gates, resets, and
discards require active indices, and every index must be in range. Recreation
after discard is unsupported. Use reset for repeated replacement of an active
index. Barriers represented by `SkippedOperation` have no quantum operands and
do not change activity, including full-register barriers around creations.

Ordinary inputs retain the existing evaluation convention: all-zero by default,
or all-plus with `basis='+'`. Explicit creation always prepares zero.

## Execution and code ownership

```text
CircuitIR → run_circuit → BackendAdapter → selected backend kernels
                 │
                 └─ CheckpointStore
```

| Responsibility | Location |
| --- | --- |
| Operation definitions and static activity validation | `spd/circuit_ir.py` |
| Execution order, index translation, contraction grouping, checkpoint lifetime | `spd/run_circuit.py` |
| Backend capability checks and dispatch | `spd/backend_adapter.py` |
| NumPy channel transformations and coefficient-gradient transposes | `spd/numpy_backend/kernels.py` |
| SPO active-index metadata and validation | `spd/core/sparse_pauli.py` |
| Snapshot storage in host memory/disk and cleanup | `spd/checkpoints.py` |

The runner uses the common operation loop for unitary and channel operations.
Backend kernels receive current column positions and logical widths. The runner
translates original circuit indices and attaches the resulting index metadata.
Checkpoint storage is independent of the IR and backend mathematics.

## Observable representation

`create_spo(..., system_size=n, active_qubits=[...])` creates a compact observable.
Column k represents original index `active_qubits[k]`. Pauli keys must have that
many columns; measurement-index lists still refer to original indices. Metadata
validates uniqueness, range, and agreement with stored width.

- `active_qubits=None` means the full canonical ordering.
- `active_qubits=[]` means a scalar, for example
  `create_spo({'': 2.}, system_size=8, active_qubits=[])`.
- A zero operator on a nonempty register retains that register's mapping.

Execution begins on the circuit's final active outputs and orders their columns
by increasing original index. Explicit mappings must describe exactly these
outputs. A full-width observable is accepted if every omitted output column is
identity in every stored term; nonidentity on discarded outputs is rejected.
Activity is never inferred from nonidentity support.

For a selected column, write an observable as `I⊗A + X⊗B + Y⊗C + Z⊗D`, with
placement determined by its index. Reverse observable propagation applies:

| Circuit operation | Observable adjoint |
| --- | --- |
| Creation | Return `A + D` and remove the column |
| Reset | Return `I⊗(A + D)`, retaining the column |
| Discard | Insert an identity column |

These transformations have no factor of two and no normalization. Creation
merges terms and physically removes columns. Packed storage shrinks to the word
count required by the current width; a scalar has zero packed columns.
`info['active_widths']` reports the initial width and each reverse instruction's
logical output width. Term count (`get_size()`) and active width are distinct.

## Backend channel interface

A channel-capable backend implements and exports these five functions:

```python
reindex_spo(spo, num_qubits, columns)
contract_zero_forward(spo, num_qubits, columns, remove_columns)
contract_zero_backward(spgo, checkpoint, num_qubits, columns, remove_columns)
insert_identity_forward(spo, num_qubits, column)
insert_identity_backward(spgo, num_qubits, column)
```

`reindex_spo` reorders selected columns and checks that omitted columns contain
only identity. Contraction applies the zero-state expectation to every column
in `columns`; `remove_columns` is the subset whose columns are removed.

| Operation | Contracted columns | Removed columns |
| --- | --- | --- |
| `CreateZero(i)` | Current column of i | Current column of i |
| `ResetZero(i)` | Current column of i | None |

Discard dispatches to identity insertion. The backward functions reconstruct
the primal operator and apply the transpose of the coefficient transformation.

Reset is mathematically equivalent to removing its column by contraction and
then inserting identity at that position. The NumPy implementation fuses these
steps: it writes identity directly into the retained column while merging terms.
This avoids an intermediate operator and an additional unpack/repack pass, and
lets independent creations and resets share one batched transformation. No
performance comparison has been measured. A backend can implement reset using
the two constituent steps internally while satisfying the same interface.
That decomposition still needs only one pre-contraction checkpoint; its gradient
pass applies the identity-insertion transpose before the contraction transpose.

The adapter checks available entry points. Unsupported backends fail before
execution or checkpoint allocation. Backend implementations must support
zero-column scalars and preserve the gradient coordinates required for exact
cancellation. Legacy circuits retain their existing backend dispatch.

## Gradients and checkpoints

Use the existing execution sequence with numeric rotation angles:

```python
final, info = evolve(observable, circuit, 0., 10000, progress=False)
gradient = init_gradient_spo(final)
_, angle_gradients, _ = backpropagate(
    gradient, circuit, 0., 10000, progress=False,
)
```

The runner groups consecutive creation/reset operations on distinct indices,
allowing intervening barriers. It saves one complete SPO before the entire group
in reverse observable propagation, then applies the composite contraction.
Thus `Barrier; CreateZero(i); Barrier; CreateZero(j); Barrier` needs one snapshot.
A gate, discard, non-barrier skipped operation, or repeated reset index ends a
group. Gates are never reordered to enlarge a group.

The snapshot contains all coefficients, Pauli rows, and active-index metadata,
including zero coefficients and terms killed by contraction. During gradient
propagation, the runner loads that group's snapshot once. Each original row
receives the matching reduced gradient if all contracted axes are I/Z, or zero
if any is X/Y. All original rows are restored so preceding unitary gradients
have the required operator. Exactly cancelled merged coordinates are retained:
a zero coefficient can still have a nonzero derivative.

Unitary segments retain the existing reverse-reconstruction algorithm without a
per-gate forward cache. Discard requires no complete-SPO checkpoint. Grouping
reduces disk writes and intermediate allocations; width and truncation histories
retain per-instruction entries, while progress displays a group as one step.

### Native-object retention and memory budgets

```python
evolve(
    observable, circuit, 0., 10000,
    checkpoint_memory_budget_bytes=4 * 1024**3,          # CPU RAM: 4 GiB
    checkpoint_device_memory_budget_bytes=1024**3,      # Each GPU: 1 GiB
)
```

Budgets are per evaluation. CPU-origin snapshots stay as native objects in CPU
RAM. Device-origin snapshots stay as native device objects in VRAM. Saving an
in-budget snapshot retains a reference without serialization, copying, or device
transfer. NumPy uses only the CPU budget.

```text
CPU origin:                  CPU object → serialize to disk
GPU origin: device object → CPU object → serialize to disk
```

Each memory tier evicts its oldest arrival when space is needed. Device eviction
converts the complete snapshot to a CPU representation; CPU eviction serializes
that representation directly to disk. A snapshot larger than a tier's budget
bypasses that tier without evicting its smaller resident snapshots. Zero disables
retention in the selected tier; an exact fit remains resident. Inspection does
not change eviction order.

The CPU budget counts estimated resident object storage, including NumPy SPO
dictionaries, keys, scalar coefficients, and metadata. Shared objects are counted
once within a snapshot and conservatively counted again across snapshots.
Device budgets count retained allocations independently for each device. These
are retention limits, not total process/device memory limits: live operators,
restored snapshots, transfer/serialization buffers, and allocator reservations
are outside the budgets. Concurrent evaluations have independent budgets.

Snapshots are read-only while retained. The NumPy execution path constructs a
new operator at contractions and does not mutate saved coefficients or metadata
in subsequent forward/backward operations. Code using the internal store must
respect the same ownership contract; retaining a reference does not protect it
from arbitrary external mutation.

Disk storage uses an isolated temporary directory created only on the first
spill. `checkpoint_directory=path` chooses its parent. Backward uses `take(key)`
to consume just the required snapshot, removing its cache entry and any disk
file. Native snapshots are used directly; host or disk snapshots are restored to
the original device as necessary. Consumed objects are never promoted back into
the cache. Internal `load(key)` supports non-consuming, read-only inspection.

A tape is one-shot and backward requires the same circuit, angles, backend,
precision, cutoff, and term cap. Budgets do not affect this matching requirement.
Forward, transfer, serialization, and restoration errors clean up owned storage;
an entered backward pass also closes its tape on completion or failure. Keep
the forward result or initialized gradient operator alive until backward ends.
For expectation-only execution, call `spd.close_channel_checkpoints(final)` for
immediate cleanup. Abandoned tapes release references and have temporary-directory
cleanup. Live tapes are not portable serialized executions; `save_strings=True`
is unsupported.

### Backend size and transfer helpers

`CheckpointStore` owns retention, eviction, and disk serialization. A small
`CheckpointBackend` bundle supplies backend-specific operations:

- `size(state)`: estimate resident bytes for a native or host representation.
- `device_of(state)`: return a device identifier, or `None` for CPU.
- `to_host(state)`: transfer a complete device snapshot into host storage.
- `from_host(state, device)`: restore it to the original device.

NumPy supplies `checkpoint_size`; its host conversions are identity operations.
Future device backends export `create_checkpoint_backend(state)` to capture
restoration context once per evaluation, without retaining the input snapshot.
Transfers must preserve coefficients, Pauli rows, precision, and active-index
metadata and finish before the original device reference is released. Helpers
must not mutate inputs or retain hidden snapshot references.

The generic device-tier policy is tested with a simulated backend. Actual
JAX/Triton channel execution and their transfer helpers remain unimplemented.
Disk encoding uses pickle on CPU representations only.

## Rotation parameters and units

`PauliRotation.theta` uses radians in `exp(-i * theta * P / 2)`.
`backpropagate` returns derivatives with respect to these radian angles.
`VariationalCircuit` accepts either pytket circuits or direct `CircuitIR` and
maps those derivatives to optimizer parameters:

| Circuit supplied to `VariationalCircuit` | Native rotation parameter | Angle conversion |
| --- | --- | --- |
| pytket | Half-turn parameter t | `theta = pi * t` |
| `CircuitIR` | Radian angle theta | No conversion |

`gate_parameter_factors` gives the derivative of each native rotation parameter
with respect to its optimizer parameter. Consequently, the chain-rule multiplier
is `pi * factor` for pytket and `factor` for direct IR. Shared parameter indices
sum their contributions; index `-1` marks a fixed rotation. Numeric angles must
already be populated. Symbolic binding and a single rotation depending on
multiple optimizer parameters are unsupported.

## Exact execution, approximation, and limitations

Channel operations apply no coefficient cutoff or term cap. Their truncation
diagnostics are zero. With cutoff zero and a sufficiently large cap, expectation
values and supported rotation gradients are exact up to floating-point error.
With approximation enabled, unitary segments use the existing truncation and
reverse-reconstruction rules. Their gradients are approximate and are not
guaranteed derivatives of the discontinuous truncated program. Checkpoints do
not recover information previously removed by unitary truncation.

Channel gradients support `basis_expectation` without an OSE term. Channel or
compact execution does not support pruning, noise susceptibility, in-place
execution, or cyclic translation of compact operators. Metadata is preserved
through execution, copies, supported scalar algebra, SPGO-to-SPO conversion,
and SPO serialization, including JAX pytree metadata. Changing coefficients or
doing algebra after forward does not create a valid new gradient tape.

General noise channels, adaptive control, postselection, physical allocation,
a full MERA ansatz, general IR export, and Guppy integration are outside scope.

## pytket import

The supported import subset is verified with pytket 2.18.1:

- `created_qubits` become creations at the circuit input.
- `discarded_qubits` become discards at the circuit output.
- In-circuit `Reset` becomes `ResetZero`.
- `add_qubit` alone denotes an ordinary input, not preparation.

Qubits must use canonical `q[0..n-1]` indices. Implicit permutations must first be
materialized. Measurements in channel circuits and conditional commands are
rejected. Numeric `PauliExpBox` lowers to a rotation; this does not provide
symbolic parameter binding. Direct IR supports internal creations independently
of a pytket round trip.

## Examples and validation

From the repository root:

```sh
python -m examples.functionality.static_channels
JAX_PLATFORMS=cpu python -m pytest -q tests/test_checkpoints.py tests/test_static_channels.py tests/test_create_spo.py tests/test_circuit_ir.py tests/test_variational_circuit.py
JAX_PLATFORMS=cpu python -m pytest -q tests
```

Set `PYTHONPATH` to the worktree's absolute path when running the full suite so
subprocess examples import the same checkout. Triton tests require a supported
GPU environment.

The creation example uses eight fixed indices and three sublayers. Reverse
propagation contracts through widths `8 → 4 → 2 → 0`, using three checkpoints.
The reset example prepares a correlated pair, resets one index, and rotates it
again. Both rotations share a parameter with factors 1 and 2; analytic
expectation and derivative assertions verify the result.

Focused tests compare against independent dense calculations and finite
differences. They cover activity validation, output mappings, entanglement,
shared parameters, cancellation, killed terms affecting earlier gradients,
memory/disk/mixed checkpoint storage, barrier-separated contraction groups,
packed-width boundaries, native ownership, simulated device transfers, cleanup, and approximation behavior.
