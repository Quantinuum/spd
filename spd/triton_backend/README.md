# Native GPU sparse Pauli dynamics

The next implementation milestones and chosen reference semantics are recorded
in [the compatibility contract](COMPATIBILITY.md).

Forward and backward gate evolution of real sparse Pauli observables on NVIDIA GPUs. PyTorch owns
GPU tensors and supplies allocation, reductions, and optional top-k; Triton
compiles the custom kernels in [kernels.py](kernels.py). Python dispatches
rotations but never loops over the observable's coefficients.

```sh
pip install -e '.[triton,pytket]'
python examples/benchmark_2d_obc_xx_z_stepwise.py --backend triton
```

```python
from spd.triton_backend import create_op, evolve_step
from spd.circuit_ir import PauliRotation

state = create_op({"ZI": 1.0}, precision="double")
operations = [PauliRotation("XXPhase", "XX", -0.08)]
state = evolve_step(state, operations, trunc_val=2**-18)
print(state.get_expectation_value("Z"))
keys, coefficients = state.to_host()
```

## Representation and pair update

For q qubits, a row contains W = 2 ceil(q/32) packed words: X masks followed by
Z masks, with qubit zero in the most significant bit. I/X/Y/Z correspond to
(x,z) = (0,0)/(1,0)/(1,1)/(0,1). Tensors store the words as int32; their bits
are interpreted as uint32. A separate array holds float32 or float64 coefficients.
`create_op` combines keys made identical by identity padding; subsequent
rotations preserve uniqueness. All-ones Pauli words are valid keys.

For O = sum_P c_P P and a generator G, evolution is
`O' = exp(+i theta G/2) O exp(-i theta G/2)`. Commuting rows are unchanged before
truncation. Every anticommuting row has exactly one possible partner:

```
Q_key = P_key XOR G_key
```

The XOR gives the key, not its phase. Population counts of the packed masks
compute commutation parity and the sign s_P defined by `i G P = s_P Q`.
Then, with a missing partner treated as coefficient zero:

```
c'_P = cos(theta) c_P - s_P sin(theta) c_Q
c'_Q = cos(theta) c_Q + s_P sin(theta) c_P
```

For example, `X Z = -i Y`, so rotating Z by X gives `cos(theta) Z + sin(theta) Y`.
This closed pair structure removes the need for a general duplicate-summing
algorithm: distinct input rows cannot create the same missing partner.

## What happens for each rotation

1. **Build the index.** Allocate a power-of-two table with at least 2N slots
   (load factor at most 1/2), filled with -1. `build_index` inserts input row
   indices using integer atomic compare-and-swap and linear probing. The table
   stores indices, not coefficients. The build completes before lookups begin.
2. **Find and update partners.** `rotate` handles 128 input rows per program.
   Each anticommuting row probes for its XOR partner, checking *every packed
   word* before accepting a match. Hash collisions cannot merge different
   Paulis. Each existing row computes its own updated coefficient; it emits
   a second row only if its partner was absent.
3. **Threshold and compact in the same kernel.** Apply `abs(c') > trunc_val`
   after combining the pair's contributions. A block prefix sum assigns local
   output positions; one integer atomic addition reserves that block's output
   range. There are no floating-point coefficient atomics. Row order can vary
   with scheduling, while the mathematical mapping remains unchanged.
4. **Read the count.** Copy one scalar to the CPU and expose the live output
   prefix. Output buffers have capacity 2N; slicing retains that backing
   allocation, but the next launch scans only the live rows. If the size cap
   binds, PyTorch top-k selects the largest magnitudes on the GPU.

The common path has expected O(NW) work and O(NW) temporary storage, assuming
short hash probes. Adversarial collisions can make probing much worse. A binding
size cap adds top-k work. There is still one host synchronization per rotation;
this is not a fully GPU-resident circuit loop. Inputs are preserved. Key-buffer
addresses use 64-bit arithmetic; input row counts must be below 2^30.

Gate masks and trigonometric scalars are cached. Kernels specialize on word
width and precision, rather than the live row count or angle. Circuit operations
run in reverse order for Heisenberg evolution. Lazy backend imports prevent
unused JAX initialization from reserving memory in a native-only process.

## Why the measured speedup is large

Both implementations execute on the GPU. The important change is the work done
per gate. The existing JAX `stack_sort_merge` concatenates candidates, lexsorts
packed keys, performs segmented sums, and sorts coefficient magnitudes even
when the size cap is inactive. Its wrapper also uses padded rows, several host
synchronizations, and discarded-norm diagnostics.

The native path replaces those global sorts and merges with expected-linear
partner lookup and combines update/filter/compaction into one kernel. It scans
live rows and transfers one count. It currently omits per-gate discarded-norm
reporting, which is additional work done by the reference. These are reasons
supported by source inspection; we have not profiled or ablated their individual
contributions, so no percentage of the speedup is attributed to a single cause.
Avoiding shape recompilation improves first-run time, but does **not** explain
our warmed timing comparison: both paths warm each input before measurement.

On the H100, default benchmark step 14 took 0.681 s versus 83.153 s for JAX
(122×); the native run completed all 23 steps, ending with 221.9 million rows.
This measures these implementations on this workload, not an inherent limit of
JAX. The same algorithm could be exposed through a JAX custom GPU kernel.
[Full measurements and limitations](../../benchmarks/README.md).

## Test coverage and current scope

Triton does **not** have feature/coverage parity with the complete NumPy/JAX
backends. Public adapter/runner execution and the TFI gradient workflow are
supported; the remaining functionality is listed below.

| Area | Triton validation |
|---|---|
| Pauli signs/rotations | Shared NumPy/JAX rotation examples; independent dense 3-qubit matrix conjugation for all 64 generators |
| Packing and precision | float32/float64; randomized 3/33/65/121-qubit evolution; identity padding |
| Sparse storage | Existing/missing partners; forced hash collision chains; block boundaries; duplicate-output checks; input preservation |
| Truncation | Combined contributions before cutoff, equality, cancellation, empty output, top-k and ties |
| Physical invariants | Norm conservation and rotation/inverse recovery without truncation |
| Measurements | X/Z expectations and aliases, norm, host conversion |
| Circuit evolution | Reverse order; multi-step coefficient comparison with both JAX algorithms |
| Large workload | 23 native default steps; exact counts and expectation/norm agreement through 14 reference steps |
| Backward gates | Coefficients, adjoints, parameter gradients, gradient-only support, threshold/cap diagnostics; NumPy/JAX and dense finite differences |
| Clifford gates | H/S/Sdg/X/Y/Z/CX/CY/CZ, both directions and precisions; dense matrices and cross-word placements |
| Terminal losses / public runner | Basis/OSE adjoints, regularization, energy/OSE/parameter gradients, progress control, IR/pytket and serialization |
| Not implemented | L2 losses, arithmetic/translation, weight/product utilities and noise susceptibility |

Forward-only baseline validation on the H100: **679 repository tests passed**,
including **122 Triton/import-focused cases**. CUDA Compute Sanitizer memcheck
reported **0 errors** on the 8 collision/compaction stress cases.
[Suite log](../../benchmarks/results/validation_expanded.txt) ·
[Memory-check log](../../benchmarks/results/cuda_memcheck.txt).

The earlier 79-test result included existing backend regression tests; it was
not 79 Triton-specific tests. Test results are functional evidence, not a Triton
instruction/branch coverage percentage. Full coefficients were compared on the
small conformance problems, not on the 221.9-million-row final state.

Run from the repository root:

```sh
python -m pytest tests/test_triton_backend.py tests/test_triton_invariants.py tests/test_triton_jax_conformance.py tests/test_backend_imports.py -q
```

The strict forward cutoff matches JAX; NumPy retains equality. Equal-magnitude
cap ties and output order are unspecified. Cross-framework bitwise equality is
not promised. Unsupported circuit operations fail explicitly.

## SPGO and diagnostic APIs (milestone 2)

All four entry points below are exported from `spd.triton_backend`:

| Path | Function and implementation | Returns |
|---|---|---|
| Fast forward rotation | `conjugate_pauli_rotation` in [__init__.py](__init__.py) | State only |
| Fast sequence | `evolve_step` in [__init__.py](__init__.py); calls `conjugate_pauli_rotation` | State only |
| Forward with diagnostics | `conjugate_pauli_rot_forward` in [operations.py](operations.py) | `(state, live_count, diagnostics)` |
| Backward with diagnostics | `conjugate_pauli_rot_backward` in [operations.py](operations.py) | `(state, live_count, angle_gradient, diagnostics)` |

The rotation paths share the same [Triton kernel](kernels.py), specialized using
compile-time flags. There is currently no diagnostic-free backward API.
The fast sequence helper supports Pauli rotations and skipped operations;
Clifford gates currently use their separate gate APIs.

"No performance degradation" refers to the updated state-only path compared
with the original state-only path. Diagnostics still add measurable overhead:
about 12.5% at one million input rows and 2% at ten million in the measured
workload. Keep the two paths separate and compare equivalent diagnostic settings;
see [measurement details and limitations](MILESTONE2.md).

`SparsePauliGradientOp` adds a contiguous adjoint array to the same packed keys
and primal coefficients. `create_gradient_op({"ZI": (1., .2)})` constructs one;
`to_spo()` exposes its primal state and `to_host()` returns `(keys, c, g)`.
Direct array construction assumes unique keys, as does the fast SPO path.

```python
from spd import triton_backend as gpu

state = gpu.create_op({"ZI": 1.})
state, count, info = gpu.conjugate_pauli_rot_forward(state, "XX", .2, 1e-8, 1000)
adjoint = gpu.create_gradient_op({"ZI": (1., .2), "YX": (.1, -.3)})
adjoint, count, dtheta, info = gpu.conjugate_pauli_rot_backward(
    adjoint, "XX", .2, 1e-8, 1000)
adjoint = gpu.conjugate_CX_backward(adjoint, 0, 1)
```

Backward rotation shares the hash lookup and rotates both `c` and `g` by
`-theta`. Before rotating, each existing anticommuting pair contributes
`s_P * (c_P*g_Q - c_Q*g_P)` to the angle gradient, counted only when the
partner index is larger. Triton reduces block partials; PyTorch reduces them
into the returned scalar. Missing partners have zero primal and adjoint.
This implements SPD's truncated backward algorithm, including its lost support;
it does not differentiate hard selection or recover discarded forward rows.

Backward keeps meaningful `(c,g)` rows with `abs(c) >= cutoff`, including
zero-primal/nonzero-adjoint rows at cutoff zero. Forward keeps `abs(c) > cutoff`.
Both apply an optional exact top-k cap by primal magnitude. Compaction and top-k
apply the same indices to keys, primal, and adjoint arrays.

Diagnostic calls return `num_str_truncated`, `truncated_l1_norm`, and
`truncated_l2_norm`. Block reductions count discarded nonzero primal terms once;
cap losses are summed directly, avoiding subtraction of nearly equal norms.
These reductions use float64 even for single-precision states. They require
additional work and host readback. The existing `conjugate_pauli_rotation` and
`evolve_step` state-only APIs retain their fast specialization: compile-time
flags remove adjoint and diagnostic instructions. Benchmark these APIs separately.

Clifford gates transform packed masks and signs in one pass, without hashing
or sorting. Backward applies the inverse gate and the same sign to both arrays.
CY is also a single fused pass. Gate calls preserve input arrays.

`set_precision` controls subsequent construction (initial default: double).
Existing states retain their dtype. Packing is 32-bit. Standard rotations accept
Pauli strings or packed generators.

See [milestone 2 validation and performance](MILESTONE2.md). The earlier forward
122× result must not be assumed for backward evolution.


## Public execution and the TFI example (milestone 3)

```python
import spd
from spd.ansatz import tfi_2d_hva

backend = spd.BackendAdapter.from_name("triton", precision="double")
ansatz = tfi_2d_hva([.13, -.21], system_size_x=2, system_size_y=2)
initial = spd.create_spo({"ZZII": -1., "XIII": -3.1}, backend=backend)
final, forward_info = spd.evolve(
    initial, ansatz.circuit, 1e-8, 1000, backend=backend, progress=False)
terminal = spd.init_gradient_spo(
    final, basis="+", lambda_ose=.1, alpha=2., backend=backend)
_, gate_gradients, backward_info = spd.backpropagate(
    terminal, ansatz.circuit, 1e-8, 1000, backend=backend, progress=False)
parameter_gradients = ansatz.parameter_gradients(gate_gradients)
cost = final.get_expectation_value("+") + .1 * final.get_OSE(alpha=2.)
```

The public runner uses the diagnostic gate APIs and returns per-gate history
and total discarded norms. `progress=False` skips per-gate norm/OSE reductions
and printing; it does not disable truncation diagnostics. The runner retains
`progress=True` as its default for existing callers. The TFI example defaults
to quiet gate execution and offers `--progress` to enable those reductions.

Public caps round upward to a power of two, matching JAX (1000 becomes 1024).
Direct gate APIs still enforce exact caps. Public construction pads qubit width
to 32-bit words; it does not pad the row count. Passing `system_size` permits
empty observables. Adapter precision defaults to single; the TFI example uses
double. Backend inference for Triton states does not import JAX or copy GPU
coefficient arrays to the host.

Basis adjoints are one for strings contributing to the selected X/Z expectation
and zero otherwise, including explicitly stored zero-primal rows. OSE uses
`p = c**2 / sum(c**2)` and the reference `1e-12` epsilon convention. Its adjoint
is `2*c/sum(c**2) * (dOSE/dp - sum(p*dOSE/dp))`; regularization adds this to the
basis adjoint without changing primal coefficients. All array work stays on GPU
in [losses.py](losses.py). Positive finite alpha is supported; zero-norm entropy
reports zero, while requesting its undefined OSE gradient raises `ValueError`.

From the repository root, run the unified example in dimension 2:

```sh
python examples/gradient/run_tfi_gs.py 2 2 0 --linear-system-size 2 --method eval_only --backend triton --trunc-val 0 --max-num-str 4096 --lambda-ose .1 --alpha 2
python examples/gradient/run_tfi_gs.py 2 2 2 --linear-system-size 2 --method adam --backend triton --trunc-val 0 --max-num-str 4096 --lambda-ose .1 --alpha 2
```

Evaluation, Adam, L-BFGS and basinhopping retain the example's output files.
Run directories include the backend name; metadata records requested/effective
caps and `native_hash` for Triton's algorithm. `--algorithm` applies only to JAX.
The storage estimate describes live arrays at the cap, not peak memory.
The Triton Adam path restricts JAX/Optax to CPU with double precision before
optimizer initialization, leaving GPU allocation to PyTorch/Triton.

[Milestone 3 validation and performance](MILESTONE3.md) records reference checks,
independent derivatives and measured costs. L2 losses and the remaining parity
inventory are milestone 4; broader performance comparisons are milestone 5.
