# Native GPU sparse Pauli dynamics

Forward evolution of real sparse Pauli observables on NVIDIA GPUs. PyTorch owns
GPU tensors and supplies allocation, reductions, and optional top-k; Triton
compiles the two custom kernels in [kernels.py](kernels.py). Python dispatches
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
backends. It is an independent forward API, not a complete BackendAdapter.

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
| Not implemented | SPGO/backpropagation, Clifford dispatch, discarded-norm diagnostics, OSE, arithmetic/translation, full public runner integration |

Validation on the H100 after this expansion: **679 repository tests passed**,
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

## Extending this to SPGO

Yes: reuse the packed keys, hash index, partner lookup, and compaction, and carry
both a coefficient c and its adjoint g per row. Existing SPD backward evolution
rotates **both arrays by -theta**. For a canonical pair with `G P = i Q`, the
parameter-gradient contribution before that reverse rotation is
`c_Q g_P - c_P g_Q`. Compute it once per pair (or sum both orientations and
halve), then reduce block partials to one scalar. The hash lookup can serve all
these computations. No general key sort is required.

The hard part is backward semantics, not adding another coefficient array.
Existing backward code retains meaningful `(c,g)` rows with `abs(c) >= cutoff`;
in particular, a zero coefficient with a nonzero gradient can survive when the
cutoff is zero. Copying the forward keep mask would incorrectly delete it.
Caps, lost-support behavior, terminal gradient initialization, and discarded-norm
reporting also need to match the chosen reference. This would reproduce SPD's
existing truncated backward algorithm; it would not automatically make hard
thresholding differentiable or reconstruct discarded forward terms.

Before enabling SPGO, port the existing backward conformance/finite-difference
checks (finite differences away from cutoff/cap transitions), including zero-coefficient/nonzero-gradient rows, caps and cutoff
boundaries, basis/OSE/L2 terminal losses, and multi-step gradients. The additional
array traffic and gradient reduction change performance, so the forward 122×
result should not be assumed for backward evolution. SPGO is not implemented yet.
