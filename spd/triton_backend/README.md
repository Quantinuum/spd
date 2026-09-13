# Triton GPU backend

Triton is SPD's recommended NVIDIA GPU backend. It supports forward evolution,
per-gate diagnostics/history, coefficient adjoints and angle gradients, exact
Clifford gates, terminal losses, and the common algebra/analysis APIs. Persistent
storage is owned by each SPO/SPGO and retained across gates.

## Installation and selection

Run from the repository root on Linux with a supported NVIDIA GPU and driver:

```sh
pip install -e '.[triton,pytket]'
```

The `triton` extra installs `torch>=2.8` and `triton>=3.4`; `pytket` is optional
for built-in OpenQASM/IR workflows. Execution requires CUDA-enabled PyTorch and
an available NVIDIA GPU. This backend cannot execute on CPU.

`spd.create_spo(...)` selects Triton automatically when PyTorch, Triton, and
NVIDIA CUDA are available; otherwise it selects NumPy. Plain `import spd` does
not probe CUDA or import either GPU backend. Explicit `backend_name="triton"`
or a configured adapter selects this backend without CPU fallback. Existing
states determine the backend of subsequent evolution and gradient calls.
JAX remains available explicitly for legacy workflows.

```python
import spd
from spd.circuit_ir import CircuitIR, PauliRotation

initial = spd.create_spo({"ZI": 1.0}, backend_name="triton", precision="double")
circuit = CircuitIR(2, (PauliRotation("XXPhase", "XX", -0.08),))
final, forward_info = spd.evolve(initial, circuit, 1e-8, 1024, progress=False)
energy = final.get_expectation_value("Z")
terminal = spd.init_gradient_spo(final, basis="Z")
adjoint, gate_gradients, backward_info = spd.backpropagate(
    terminal, circuit, 1e-8, 1024, progress=False)
keys, coefficients = final.to_host()
```

## Storage and gate execution

For q qubits, each row contains `2 * ceil(q / 32)` packed words: X masks followed
by Z masks, with qubit zero in the most significant bit. CUDA tensors store
int32 words interpreted as uint32 bits, real float32/float64 coefficients, and
optional adjoints. `create_op` combines keys made identical by identity padding.
All-ones words are valid keys. Explicit tensor constructors require unique keys.

[Persistent storage](persistent.py) maintains row buffers and a hash index that
compares every packed word. Rotations look up XOR partners, update each pair,
and append missing partners. Hash collisions cannot merge different Paulis.
The index is reused across rotations with incremental insertion; capacity growth
and occasional compaction rebuild it. Dead slots are excluded from live counts,
observations, and diagnostics. Binding caps select by primal magnitude on GPU.
Exact Cliffords transform keys/signs in one pass and rebuild the changed index.

For `O' = exp(+i theta G/2) O exp(-i theta G/2)`, define `i G P = s_P Q` for
an anticommuting pair. The update is:

```text
c'_P = cos(theta) c_P - s_P sin(theta) c_Q
c'_Q = cos(theta) c_Q + s_P sin(theta) c_P
```

Missing partners have coefficient zero. No floating-point coefficient atomics
or per-gate global key sorting are needed. Gate metadata is cached; kernels
specialize on packed width and precision rather than angle or live row count.
There is still host synchronization for row counts and reductions; execution is
not a fully GPU-resident circuit loop. Input row counts must remain below 2^30.

`evolve_step` and public `spd.evolve` process circuit operations in reverse order
for Heisenberg evolution. Public backward execution traverses the corresponding
reverse evolution. Direct `apply_in_place` calls execute the single operation
given, so callers control their ordering.

## Functional and in-place APIs

| Entry point | Result |
|---|---|
| `conjugate_pauli_rotation` | State-only forward rotation |
| `evolve_step` | State-only sequence using persistent storage |
| `conjugate_pauli_rot_forward` | `(state, live_count, diagnostics)` |
| `conjugate_pauli_rot_backward` | `(state, live_count, angle_gradient, diagnostics)` |
| `spd.evolve` | Final SPO and diagnostic history/totals |
| `spd.backpropagate` | Final SPGO, gate gradients, and diagnostic history/totals |
| `spd.backpropagate_noise_analysis` | Backward execution with circuit-aligned noise analysis |

Functional gate APIs preserve their inputs. Public forward/backward loops copy
once, apply gates on that private storage, and compact on return. State-only
specializations omit truncation diagnostic reductions. Public `progress=False`
skips printing and reporting-only norm/OSE reductions, but retains truncation
history. The original standalone per-gate kernels remain available as benchmark
references; they do not describe storage lifetime in the public circuit loops.

For explicit mutation:

```python
work = initial.copy()
work, info = spd.evolve(work, circuit, 1e-8, 1024, progress=False, in_place=True)
# Further circuits can reuse work's capacity and index.
```

`evolve`, `backpropagate`, and `backpropagate_noise_analysis` accept
`in_place=True` for Triton only. They mutate the supplied object and retain
storage on return. NumPy/JAX raise `NotImplementedError`. Execution errors can
leave earlier gates applied.

`copy()` creates independent SPO/SPGO storage. `apply_in_place(operation,
trunc_val, max_num_str)` returns `(self, live_count, angle_gradient_or_None,
diagnostics)`; `diagnostics=False` omits truncation reductions while retaining
SPGO angle gradients. `compact()` explicitly materializes compact storage.

Access to `xz_array`, `c_array`, or `grad_c_array` exports compact writable tensor
views, invalidates the reusable index, and protects those aliases on the next
mutation by detaching storage. Host conversion and serialization materialize
arrays as needed. Scalar observations and noise-history reductions read storage
directly. Avoid exporting arrays inside a gate loop if reuse matters.

Terminal gradient initialization can share primal buffers with its source SPO.
First mutation of the SPGO detaches them to preserve the SPO, even with
`in_place=True`; merely deleting the SPO does not clear that sharing flag.
`to_spo()` can likewise share primal buffers with alias protection. An explicit
consuming ownership-transfer initializer is a
[deferred variational-pipeline TODO](../../docs/triton_variational_pipeline_todo.md).
Compacting before initialization can release spare forward capacity, but does
not remove the protective backward copy.

## Numerical semantics

- Forward rotations keep `abs(c) > cutoff`; backward uses `abs(c) >= cutoff` on
  meaningful `(c, g)` support. Gradient-only rows survive at cutoff zero and can
  grow through backward rotations.
- Caps rank by primal magnitude; the same selected rows apply to keys,
  coefficients, and adjoints. Direct gate APIs enforce exact caps. Public runner
  caps round upward to a power of two, matching JAX (1000 becomes 1024).
- Exact Cliffords H/S/Sdg/X/Y/Z/CX/CY/CZ apply no rotation cutoff or cap.
  Backward applies inverse Cliffords and transforms both channels.
- Backward rotates primal and adjoint by `-theta` and computes the angle gradient
  from the pre-update pairs. It implements SPD's truncated backward algorithm;
  it does not differentiate hard selection or recover discarded forward support.
- Diagnostics report discarded counts, L1 norm, and L2 norm, with float64
  reductions even for single-precision states. Cap losses are summed directly.
- Row order and equal-magnitude cap ties are unspecified. NumPy retains forward
  cutoff equality; Triton/JAX use a strict forward cutoff. Cross-backend bitwise
  equality is not promised.

The public adapter defaults to single precision; direct backend construction
initially defaults to double. Existing states retain their precision.
Public construction pads qubit width to packed words, not the row count.

Terminal losses support basis expectation, OSE regularization, and L2 difference.
Basis adjoints include explicitly stored zero-primal rows. OSE supports positive
finite alpha; zero-norm entropy is zero, while its undefined gradient raises
`ValueError`. L2 support rules are described below.

## Algebra and analysis

These operations run on demand and add no work to the fast rotation path.

| Operation | API |
|---|---|
| Coefficient inner product | `a.dot(b)`, `a.inner_product(b)` |
| SPO/SPGO arithmetic | `a + b`, `a - b`, `scalar * a`, `a * scalar`, `sum(states)` |
| L2 adjoint on current support | `init_gradient_from_l2_difference(a, target)` or public `init_gradient_spo(..., loss_type="l2_difference", target_spo=target)` |
| L2 adjoint on union support | `init_gradient_from_l2_difference_union(a, target)` |
| Pauli weight | `get_pauli_weight_distribution()`, `get_pauli_weight_counts()` and existing aliases |
| Physical-site translation | `state.translate(shift, system_size)` |
| Complex Pauli multiplication | `pauli_product_uint(xz1, c1, xz2, c2)`, `pauli_product_batched_second_uint(xz1, c1, keys, coefficients)` |
| Depolarizing susceptibility | `get_depolarizing_susceptibility(spgo, qubits)` and one-/two-qubit wrappers |
| Circuit-aligned noise gradients | Public `spd.backpropagate_noise_analysis(..., progress=False)` |
| Readable state | `str(state)`, `repr(state)` |

Sparse alignment uses the existing hash-table builder and an exact-key lookup
kernel in [auxiliary_kernels.py](auxiliary_kernels.py). A match gives the other
operand's row index; absent rows get zero coefficients. Addition emits each left
row with its combined value, then unmatched right rows. Matching writes are
unique, so no floating-point atomic accumulation or global key sort is needed.
PyTorch compacts the resulting arrays. Dot multiplies matched coefficients and
reduces them. Inputs must have matching packed widths, device and precision;
explicit constructors/factories continue to require unique packed keys.

**Arithmetic semantics:** addition removes only exact zero sums, matching
JAX. SPGO retains rows with either nonzero primal or nonzero adjoint. This differs
from NumPy's near-zero addition rule. Scalar multiplication follows the existing
reference near-zero scalar rule (`abs(scalar) <= 1e-8` after precision conversion),
returning an empty state with the same width. Other operations preserve inputs.

L2 uses `g = 2*(c - target_c)`. The restricted initializer includes only nonzero
current primal support; the union initializer also includes nonzero target-only
rows with `c=0`. Explicit zero rows are excluded from both support definitions.
Public L2 initialization supports OSE regularization on the restricted support.
No gradient is implied through hard support selection.

Translation rotates the first `system_size` sites to the right, accepts negative
and wrapped shifts, and preserves suffix bits. It moves packed words directly,
without unpacking a row into a full bit array; primal and adjoint arrays can be
shared because their values do not change. Weight counts include explicitly
stored zero-primal rows, with zero mass in the distribution, and exclude any
nonexistent padding rows. Only the small weight histograms are copied to CPU.
Readable strings intentionally copy state arrays to CPU and omit rows whose
primal and adjoint magnitudes are both at most 1e-6, like the reference format.

Pauli products accept NumPy packed arrays or CUDA int32/uint32 tensors and return
CUDA int32 packed keys plus complex coefficients. A population-count kernel
computes the exact phase in `{1, -i, -1, i}`; float64/complex128 inputs retain double
precision. Complex products do not change the real-coefficient SPO/SPGO contract.

Noise susceptibility is `-sum(c*g)` over rows nonidentity on any requested site,
with each row counted once even for two sites. A fused kernel forms block partials
and reduces them to the returned scalar. Circuit noise analysis retains the
existing operation-aligned output, skipped-operation zeros and gate ordering.
`progress=False` skips reporting-only norm/OSE reductions but retains noise and
truncation diagnostics. OpenQASM IR, pytket rebase and serialization work through
the shared runner.

## CUDA allocator and memory

Importing `spd.triton_backend` enables PyTorch's `expandable_segments:True` when
neither `PYTORCH_CUDA_ALLOC_CONF` nor `PYTORCH_ALLOC_CONF` is set. This process-wide
setting uses PyTorch's private allocator-settings helper, validated with PyTorch
2.8, and does not itself initialize CUDA. Explicit environment configuration
wins. To opt out, start a fresh process with:

```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False python your_script.py
```

Live tensors, buffer capacity, hash indices, temporary workspace, and allocator
reservation are different memory costs. In-place execution avoids the initial
working copy when storage is exclusive; it does not eliminate growth/compaction
workspace or shared-buffer detachment. Allocator reservation can exceed live
allocation substantially.

## Validation and performance

The completed persistent integration passed **1,203 tests with 1 expected JAX
failure** on the H100 GPU validation node. Coverage includes mixed rotations and
Cliffords, wide keys, binding caps, cutoff boundaries, gradient-only growth,
immutable inputs, aliasing, history, noise analysis, and independent NumPy/JAX
and dense-gradient comparisons. These results precede the automatic backend
selection change; CPU checks of selection do not replace GPU kernel validation.

The [acceptance report](../../docs/triton_persistent_acceptance.md) records
correctness, runtime, and allocated/reserved memory separately for forward,
diagnostics, and backward, with reproducible benchmark scripts and
[durable results](../../benchmarks/results/persistent_integration_stage4.json).
For example, TFI 11x11 at 23 steps measured 10.246 s state-only and 10.831 s with
diagnostics; historical main diagnostics measured 74.713 s. These are workload-
and environment-specific samples, not a universal speedup. Backward measurements
use smaller TFI support and the same-terminal original per-gate reference.

Full arrays match on conformance cases; the final 221.9-million-row TFI check
compares counts and scalars, not every coefficient. Equal-magnitude top-k ties
can lead to different capped AFH histories. See the report for these limits and
the reserved-memory investigation.

Run from the repository root on a GPU node:

```sh
python -m pytest tests -q
python benchmarks/benchmark_persistent_integration.py --diagnostics
python benchmarks/benchmark_persistent_backward.py
```

See the [compatibility inventory](COMPATIBILITY.md) for the common backend
contract. JAX-specific sorting APIs, PyTrees, and donation are not public Triton
features. Optional arithmetic/index/layout/padding experiments and multi-GPU
execution are outside this integration.
