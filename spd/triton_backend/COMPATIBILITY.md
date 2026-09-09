# Triton compatibility contract and parity milestones

Milestone 1, based on commit `1a8cedc`. This document defines the implementation
target; it does not claim pending features already work. Preserve the native
forward implementation as the performance baseline.

## Reference and approved decisions

Match public SPD APIs and numerical meaning, not backend storage layout. Use
JAX hard-cutoff evolution as the primary reference, both JAX algorithms for
ordinary conformance, and NumPy/dense matrices as independent checks. Private
sorting kernels, donation, PyTrees, padding, and JAX algorithm switches are not
Triton requirements.

**User-approved cap policy:** the future public runner rounds `max_num_str`
upward to the next power of two, like JAX. Thus 1,000,000 has an effective cap
of 1,048,576. This affects term selection; it does not require padded storage.
Direct backend rotation calls retain an exact positive integer cap. The existing
standalone API can still accept `None`. Record requested/effective caps in example
metadata. This distinction is necessary for comparable gradient evaluations.

| Case | Observed references | Triton target |
|---|---|---|
| Forward cutoff equality | NumPy keeps equality; JAX drops it | Strict `abs(c) > cutoff`, unchanged |
| Backward support | JAX keeps meaningful `(c,g)` with `abs(c) >= cutoff` | Same; retain `c=0,g!=0` at cutoff zero |
| Tied cap magnitudes | Selection depends on ordering/algorithm | Largest magnitudes and exact effective count; tied keys unspecified |
| Row count | JAX `get_size()` includes padding capacity | Count stored live rows; compare semantic supports |
| Equality diagnostics | JAX stack drops the row but omits its discarded norms; search accounts for it | Correct accounting, as in search |
| Zero generated candidates | NumPy can count these as truncated; JAX excludes them | Exclude zero primal coefficients from diagnostics |

Tied cap selections can affect later trajectories. Tests requiring identical
support must avoid ambiguous ties or allow their effect. Bitwise identity and
identical optimizer iterates are not required. Row order remains unspecified.

## Public inventory

| Surface | Required functionality | Current state | Milestone |
|---|---|---|---|
| Construction/configuration | SPO, SPGO, `create_op`, `create_measurement_op`, `set_precision`, 32-bit packing utilities | Implemented | M2 |
| Rotation kernels | Standard `conjugate_pauli_rot_forward/backward` signatures and tuples | Implemented; fast helper preserved | M2 |
| Clifford gates | H, S, Sdg, X, Y, Z, CX, CY, CZ in both directions | Implemented | M2 |
| Measurements | Size/norm/expectation, OSE and aliases, SPGO `to_spo` | Size/norm/expectation and SPGO to_spo; OSE pending | M2–M3 |
| Terminal gradients | Basis expectation, OSE initialization/regularization, `init_gradient_spo` | Missing | M3 |
| Public execution | `BackendAdapter.from_name('triton')`, `create_spo`, `evolve`, `init_gradient_spo`, `backpropagate` | Missing | M3 |
| Diagnostics/history | Discarded count/L1/L2 and existing history/total fields | Gate diagnostics implemented; history pending | M2–M3 |
| L2 losses | Restricted-support and union initializers; public `loss_type='l2_difference'` | Missing | M4 |
| Arithmetic | SPO dot/inner product, add/subtract/scalar multiply; SPGO add/scalar multiply/reverse add | Missing | M4 |
| Analysis/utilities | Weight distributions/counts and aliases, translation, readable strings, single/batched Pauli products | Missing | M4 |
| Noise analysis | General/one-/two-qubit susceptibility and `backpropagate_noise_analysis` | Missing | M4 |
| Frontends/serialization | IR/pytket/OpenQASM, rebase, `save_strings`, parameter-gradient mapping | Shared frontend exists; execution pending | M3–M4 |

Cover common NumPy/JAX module exports and abstract object methods, including
`get_pauli_weight_count`, `get_Pauli_weight_distribution`, `get_OSE`, and
arithmetic aliases. Backend-specific array layouts/constructors may differ.
Support Linux/NVIDIA, 32-bit packed words, and float32/float64 real states.
Pauli product utilities must still support their documented complex phases.
Do not register success-returning stubs to satisfy an interface check.

## Numerical and execution rules

- Preserve Heisenberg signs and supplied gate order. Forward reverses circuit
  operations; backward uses circuit order. Cliffords emit no parameter gradient.
  Barriers/measurements retain existing skipped-operation behavior.
- Construct pairs from input support, combine both contributions, then truncate.
  Backward rotates both primal and adjoint arrays by `-theta`; compute parameter
  gradients before that rotation and count each pair once.
- Reproduce SPD's truncated backward algorithm. This is not differentiation
  through hard support selection or recovery of discarded forward terms.
  Check finite differences without truncation or away from support transitions;
  compare with references when truncation is active.
- Rank caps by primal magnitude, not gradient magnitude. Gradient-only rows at
  cutoff zero are meaningful but need not survive a binding primal cap.
- Diagnostics count unique nonzero post-rotation primal coefficients removed
  by threshold or cap, once each. L1 sums magnitudes; L2 is root-sum-of-squares.
  Exclude padding/zero candidates and gradient magnitude. Public calls return
  real diagnostics; preserve a separate state-only fast path without fake zeros.
- Preserve history fields and return tuples. `sum_truncated_l2_norm` sums gate
  L2 values; `total_truncated_l2_norm` is their root-sum-of-squares.
- Explicit precision controls new states; existing states retain their dtype.
  Public adapter defaults remain single precision; TFI uses double. Preserve
  the standalone double default when no configuration has been supplied.
- Return measurements/gradients usable as Python/NumPy scalars. Keep internal
  arrays on GPU. Inference must neither call `np.asarray` on GPU tensors nor
  import JAX solely to recognize Triton states.
- Empty states retain qubit width. Size/norm/dot/expectation and weight maps
  return zero/empty results. Define zero-norm OSE as zero for progress reporting;
  requesting an OSE gradient at zero norm raises a clear `ValueError`, since
  normalization is undefined. This degenerate convention is explicit because
  reference behavior varies with padding. Nonzero-norm OSE and its derivative
  follow the existing `1e-12` epsilon convention, for alpha=1 and positive alpha!=1.
- Preserve input states. Reject incompatible devices/widths/types. Arithmetic
  must merge equal keys and follow the existing cancellation/near-zero tests.
  No presentation-only sort is required.

## First complete workflow: current 2D TFI example

The current checkout has the unified
[`run_tfi_gs.py`](../../examples/gradient/run_tfi_gs.py), with dimension `2`,
not the earlier `run_tfi_gs_2d.py`. It uses `tfi_2d_hva` and
`VariationalCircuit.parameter_gradients`. Preserve shared parameter mapping and
pytket's pi conversion; do not restore the old hand-grouped gradients.

M3 must add `--backend triton` and make this intended command work:

```sh
python examples/gradient/run_tfi_gs.py 2 2 0 --linear-system-size 2 --method eval_only --backend triton --trunc-val 0 --max-num-str 4096
```

The example needs OSE even when lambda_ose=0, actual diagnostics, and regularized
terminal gradients when lambda_ose!=0. Make algorithm selection, memory estimates,
metadata, and run naming backend-aware. SciPy stays on CPU; the Optax/Adam path
must not inadvertently reserve JAX GPU memory during Triton evolution.

M3 acceptance:

1. Fixed-parameter 2x2/4x4 cases match energy, OSE, all parameter gradients, and
   diagnostics against JAX, for alpha=1/2 and lambda=0/>0.
2. Independent dense/finite-difference checks on small untruncated systems
   validate signs and parameter mapping.
3. Truncated/capped comparisons use the same effective cap; check gradient-only
   support and separate documented reference exceptions.
4. `eval_only` and short optimizer runs finish with finite outputs and the
   existing output files. Every optimizer iterate need not match.

## Acceptance and commit boundaries

M1 supplies this inventory, policies, and executable boundary cases in
[`test_backend_semantic_contract.py`](../../tests/test_backend_semantic_contract.py).
Reference-only backward/diagnostic cases must be enabled for Triton as capabilities
land. One strict expected failure records the pre-existing JAX stack equality-
diagnostics bug; fixing that reference is a separate change.

M2 delivers SPGO, forward/backward gates, diagnostics, coefficient/adjoint
conformance, and GPU memory checks. M3 delivers the TFI workflow. M4 completes
the inventory and shared conformance matrix. M5 measures forward, backward,
objective/gradient time, and peak GPU memory at equal precision/effective caps/
diagnostics settings, and rechecks the large forward benchmark.

Commit completed milestones independently. Existing untracked benchmark artifacts
are outside the committed baseline and remain untouched by this step.

M1 validation on the H100: **150 passed, 1 expected failure** in 46.07 seconds.
The expected failure is the documented existing JAX stack diagnostics issue.

```sh
python -m pytest tests/test_backend_semantic_contract.py tests/test_triton_backend.py tests/test_triton_invariants.py tests/test_triton_jax_conformance.py tests/test_backend_imports.py -q
```

M2 implementation and measured costs: [validation report](MILESTONE2.md).
