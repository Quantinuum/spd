# Generic persistent-storage integration

## Verified starting point

Worktree: `/teamspace/studios/this_studio/spd-persistent-main`.
Branch: `integrate/persistent-storage`; initially clean at `4a84b0f`, the
cherry-pick of `350828a`. The persistent module and tests match that source commit.
The user reports baseline testing complete; it was not repeated as a separate
baseline stage. No applicable AGENTS.md files were found.

## Review and commit stages

Pause after each stage for user review and commit instruction. Merge/push require
user instruction; the user authorized committing stage 3 and pushing the
integration branch as a backup. No merge into main is authorized.

1. **Standalone forward sequence (approved).** Route `evolve_step`
   through basic persistent storage, support mixed exact Cliffords, preserve the
   per-gate forward implementation as an independent reference. Check ownership,
   gate order, cutoff/top-k, wide keys, empty/zero rows and input immutability.
2. **Public forward diagnostics/history (approved).** Extend pair updates with optional
   discarded-count/L1/L2 reductions. Retain SPO-owned storage across
   the public runner's operation loop. Preserve progress, history aggregation,
   save-strings behavior, requested/effective caps, and public return types.
   Observation access must not reconstruct storage or expose mutable snapshots.
3. **Backward and analysis (approved for commit).** Extend storage with adjoints and support defined by
   meaningful `(c,g)` pairs. Preserve inclusive backward cutoff, primal-ranked
   caps, gradient-only support growth, exact inverse Cliffords and pre-rotation
   angle gradients. Integrate ordinary and noise-analysis backward loops;
   validate terminal basis/OSE/L2 initializers and supported analysis methods.
4. **Complete correctness/performance acceptance.** Run the existing relevant
   conformance suites plus representative TFI/AFH energy-and-gradient workloads.
   Compare keys, coefficients, adjoints, gate/parameter gradients and diagnostics.
   Report runtime and peak allocated/reserved memory separately for state-only
   forward, diagnostic forward and backward. Compile on small disposable inputs;
   avoid full-workload warm-up repeats. Record workload sizes, precision, caps,
   cutoff and comparison limitations.

No optional local arithmetic, row selection, inverted index, fingerprints,
layout/padding changes or multi-GPU experiments are included. The options branch's
`docs/triton_persistent_generalization_experiment.md` was inspected as historical
context; its forward-only large-workload timings are not backward evidence.

## Stage 1 implementation

`evolve_step` creates one private `_Storage`, applies operations in reverse order,
and materializes once on return. Pauli rotations retain the baseline's pair
ownership, exact full-key lookup, strict cutoff and per-gate top-k behavior.
Cliffords reuse the exact bit/sign kernel in place on privately owned rows, then
rebuild the index because keys may have changed. They do not apply cutoff/cap.
Clifford-only sequences preserve explicit zero rows. Skipped-only calls return
the original object. Direct per-gate APIs and the diagnostic/backward runner are
unchanged in this stage.

The original persistent tests now compare against an explicit per-gate reference,
avoiding a circular comparison after public dispatch changed. New tests cover all
nine Clifford types in mixed sequences at 3/65/121 qubits, float32/float64, binding
caps, empty/invalid Clifford calls, zero-row preservation, and one index build
across twenty rotations on fixed support. An old unsupported-Clifford assertion
now checks an unsupported object instead.

Validation on H100, Torch 2.8.0+cu128, Triton 3.4.0:

```sh
python -m pytest tests/test_triton_persistent.py tests/test_triton_backend.py tests/test_triton_invariants.py tests/test_triton_clifford.py -q
```

**251 passed in 5.07 s**. `git diff --check` also passed.

### Forward smoke measurements

Reproduce with `python benchmarks/benchmark_persistent_integration.py`.
Raw results: `benchmarks/results/persistent_integration_stage1.json`.
Actual example Hamiltonian/ansatz builders, 8-site periodic 1D models, 3 layers,
float64, cutoff 1e-4, exact cap 4096. Fixed seeded parameters and original frontend
gate order. TFI has 65 operations and 200 output terms; AFH has 120 operations
including Neel preparation and 4096 output terms. These are small smoke workloads,
not the final scale/performance acceptance or energy-and-gradient evaluation.

One synchronized timed pass per path, after compiling gate kernels on tiny
inputs. Shared gate tensors are prepared before either measurement. Input copying,
allocation, truncation, index maintenance and materialization are timed. Energy
and host comparisons run afterward. GPU runs are sequential; allocator peaks are
reset for each measurement, unused cached blocks released, and shared live caches
are included in total peaks. An initial trial with asymmetric cold gate-tensor
caches was discarded and the benchmark corrected before the recorded run.

| Forward workload | Reference ms | Persistent ms | Reference peak allocated bytes | Persistent peak allocated bytes |
|---|---:|---:|---:|---:|
| TFI | 5.236 | 4.511 | 69,120 | 71,168 |
| AFH, binding cap | 43.896 | 14.098 | 527,360 | 664,064 |

Peak reserved memory is 2,097,152 bytes for both paths on both workloads. Incremental
allocated peaks above the pre-call live allocation are respectively 16,896/18,944
bytes for TFI and 376,832/513,536 bytes for AFH (reference/persistent). Persistent
storage uses more peak memory in these small cases. Single-sample timings do not
establish a stable speedup or predict large-support performance.

**Correctness:** identical key sets and exactly equal coefficients for both
workloads; energy differences are zero for TFI and 4.44e-16 for AFH, from reduction
order. Diagnostics and backward have not been integrated or measured in stage 1.


### Large-workload sanity check (approved)

On the same H100 with float64, public `evolve_step` matches the experiment
branch's generic persistent timings closely for the larger cases:

| Workload | Historical seconds | Current seconds |
|---|---:|---:|
| 11x11 TFI, 23 timesteps | 9.3985–9.6182 | 9.5416 |
| TFI final timestep | 2.3108–2.3938 | 2.3392 |
| 6x6x6 AFH, 20M cap, cutoff 1e-5 | 14.4424 | 14.5844 |
| 6x6x6 AFH, uncapped, cutoff 3e-4 | 0.6080 | 0.6747 |

The uncapped AFH sample is 10.96% slower; the capped sample is 0.98% slower.
Stage 1 rebuilds the index after each exact Clifford, including X, whereas the
historical AFH harness materialized before its X preparation gates. This is a
possible contributor, not a measured attribution.

TFI uses the original OBC XX+Z circuit, h=3.044382, dt=0.04, 341 rotations per
step, center Z observable, cutoff 2^-18 and nonbinding cap 1e9. All 23 counts
match, ending at 221,899,620 terms; maximum expectation error is 3.33e-16 and
L2 norm error 1.11e-16. Peak allocated/reserved memory is 50,914,283,008 /
70,428,655,616 bytes, matching the historical peaks to 3072 allocated bytes.

AFH uses the original two-layer, nine-term local Hamiltonian workload with
4320 rotations and 108 X preparation gates. Seed-0 scale-1 parameters and circuit
hash match the saved logs exactly. Final counts match at 20,000,000 capped and
2,265,180 uncapped. Energy errors are 1.73e-17 / 6.94e-18, and norm-square errors
are zero. Current allocated/reserved peaks are 7,340,987,904 / 8,543,797,248 bytes
capped and 1,094,670,848 / 1,623,195,648 bytes uncapped. Historical AFH logs lack
GPU peaks, so memory parity cannot be established. Both AFH runs report zero
allocation retries and OOMs.

Each workload ran once in its own process, sequentially, after tiny disposable
compilation inputs; no full-workload warm-up repeats. Timings include private
storage, per-gate truncation/top-k, exact Cliffords and materialization. Scalar
observations and host logging are excluded. Historical TFI used a full warm pass
per timestep, so cache/allocator preparation differs. These are single-sample
forward sanity checks, comparing counts and scalars rather than every coefficient.

As requested, detailed scripts, extracted references and logs remain temporary:
`/tmp/spd-stage1-performance-ffye4747/README.md` and `summary.json`. Historical
references are `persistent_storage_20260910/candidate{1,2}`,
`persistent_options_20260910/baseline_{final,repeat}`, and
`persistent_afh6_{capped,uncapped}_20260912` under the experiment branch's
`benchmarks/results/`. TFI postprocessing initially mishandled the saved initial
expectation and norm convention; corrected comparisons recovered the completed
23 timed steps from the log without repeating the workload.


## Stage 2: SPO-owned storage and public forward diagnostics/history

Stage 1 was approved and committed as `3abb844`. Revised stage 2 was committed
as `f03cce5`, including the requested `spd.evolve(..., in_place=True)` option. It replaces the earlier `_ForwardSession` design with
the agreed SPO/SPGO ownership model; no separate execution context is needed.

### Ownership and implementation

- SPO owns `_Storage`; SPGO uses the same storage abstraction with an adjoint
  channel. `copy()` creates independent storage. `apply_in_place()` mutates an
  SPO through a forward IR gate and returns the existing runner's four-value
  tuple. SPGO in-place backward evolution is reserved for stage 3.
- Public forward execution explicitly branches on the backend, copies the
  Triton input once, and retains that object's storage/index across gates.
  It compacts once on return. Empty/skipped-only evolution preserves input
  identity and physical storage. Direct diagnostic rotations copy once too.
- Compact array properties remain compatible. Export marks buffers shared and
  invalidates the index because external callers can retain or edit them.
  Subsequent in-place mutation detaches. Internal scalar observations, backend
  inference, terminal initialization and noise susceptibility use storage
  directly. Other existing algebra/analysis operations can materialize through
  the compact properties. Pickles contain logical arrays, not cached indices,
  and the reader accepts the previous pickle fields.
- Basis/OSE initialization shares primal buffers without consuming the SPO or
  duplicating its keys/coefficients. Only adjoints and loss-computation workspace
  are allocated. `to_spo()` keeps its compact shared-view API. Compaction retains
  rows with either nonzero coefficients or nonzero adjoints. A subsequent
  mutation protects shared inputs; backward lifetime/ownership optimization
  remains stage 3 work.
- The pair-update kernel has a compile-time diagnostic specialization counting
  each discarded combined output once, with float64 L1/squared-L2 reductions.
  Cap diagnostics reduce removed coefficients from the same top-k selection
  used for retention, avoiding cancellation from subtracting retained norms.
  The state-only specialization omits these reductions. Exact Cliffords, gate
  order, strict cutoff, public cap rounding and shared history handling remain.

### Correctness

H100 80GB, Torch 2.8.0+cu128, Triton 3.4.0:

- Broad conformance: **404 passed, 1 expected failure in 76.02 s**. The expected
  failure is the existing JAX stack cutoff-equality diagnostic issue.
- Ownership, runner and algebra/analysis suite: **140 passed in 79.96 s**.
- `git diff --check` passed.
- Final focused ownership/diagnostics/backward validation after the last alias
  and skipped-state fixes: **63 passed in 13.58 s**.

The diagnostic tests cover mixed rotations/all nine Cliffords, 3/65/121 qubits,
float32/float64, cutoff equality, binding caps, reactivation, per-gate results,
small discarded norms beside large retained coefficients, and one persistent
storage object across a gate sequence. Ownership tests cover independent copy,
constructor/export aliases, externally modified keys and stale indices, shared
terminal initialization without copying/compaction, gradient-only rows,
`to_spo()`, skipped-only storage preservation, and old/new pickle formats.
Existing runner tests exercise TFI energy/angle gradients against JAX and dense
finite differences using the unchanged backward implementation.

```sh
python -m pytest tests/test_triton_backend.py tests/test_triton_invariants.py tests/test_triton_clifford.py tests/test_triton_algebra_analysis.py tests/test_triton_jax_conformance.py tests/test_backend_semantic_contract.py tests/test_triton_persistent.py tests/test_triton_persistent_diagnostics.py tests/test_triton_storage_ownership.py -q
python -m pytest tests/test_triton_storage_ownership.py tests/test_triton_persistent_diagnostics.py tests/test_triton_backward.py -q
```

### Small forward/diagnostic benchmark

Reproduce with `python benchmarks/benchmark_persistent_integration.py --diagnostics`.
Raw results: `benchmarks/results/persistent_integration_stage2.json`.
Eight-site periodic TFI/AFH, three layers, float64, cutoff 1e-4, nonbinding cap
65536. Tiny disposable compilation, one measured pass per path, observations
excluded; copying and final materialization included. Reference runs the pre-persistent per-gate algorithm retained in the current
working tree: `conjugate_pauli_rotation` for state-only and
`operations._rotation(..., backward=False)` for diagnostics. Their Triton
`kernels.py` is unchanged from main commit `c9a2b004d2598f1741c04619dfa30865e86a9aad`,
but these small reference runs use the current state wrappers and validation;
they are **not measurements of a separate main checkout**. Persistent
diagnostics uses the current public runner. The large table below instead uses
measurements from an actual main archive for its Main diagnostics column.

| Workload/path | Reference ms | Persistent ms | Reference peak allocated bytes | Persistent peak allocated bytes |
|---|---:|---:|---:|---:|
| TFI state-only | 6.214 | 5.163 | 109,056 | 111,104 |
| TFI diagnostics/history | 8.493 | 6.979 | 110,080 | 111,104 |
| AFH state-only | 17.132 | 14.926 | 2,615,808 | 2,856,448 |
| AFH diagnostics/history | 20.676 | 20.519 | 2,624,512 | 2,856,448 |

Reserved peaks: 2,097,152 bytes for TFI, 4,194,304 for AFH in every path.
Key sets and coefficients match exactly; diagnostic histories and aggregates
pass 1e-12 tolerances. These single short samples are sensitive to CPU overhead.

### Large performance and main comparison

Raw scripts, six runs, initialization probes and `summary.json` remain in
`/tmp/spd-stage2-owned-large-cfmoi3hf`. The actual main baseline is reused from
`/tmp/spd-stage2-large-bx8ep4dn`, main commit
`c9a2b004d2598f1741c04619dfa30865e86a9aad`; main was not needlessly rerun.
Same workloads as the earlier requested sanity check: TFI 11x11, 23 timesteps;
AFH 6x6x6 at cutoff 3e-4 uncapped or 1e-5 with exact cap 20,000,000.
Diagnostic timing uses the shared operation loop with that exact cap, avoiding
public cap rounding. Only tiny disposable inputs are compiled first, with no
full-workload warm-up. Timings exclude observations and terminal initialization.

| Workload | State-only | Diagnostics | Overhead | Main diagnostics |
|---|---:|---:|---:|---:|
| TFI 11×11, 23 steps | 9.67 s | 10.84 s | +12.1% | 74.71 s |
| AFH 6×6×6, uncapped | 0.689 s | 0.898 s | +30.2% | 2.505 s |
| AFH 6×6×6, exact 20M cap | 14.62 s | 16.62 s | +13.7% | 29.56 s |

State-only and Diagnostics are the revised SPO-owned implementation. Overhead
is Diagnostics / State-only − 1, calculated from unrounded timings. Main
diagnostics reuses the same actual-main measurements as the previous table.
Revised diagnostic times are within 1% of the previous session samples
(10.742 / 0.900 / 16.589 s respectively). These are sanity measurements, not
statistical estimates.

The uncapped state-only entry uses the short repeat, 0.689 s. The first sample
was anomalously slow at 0.987 s; both are retained in the temporary directory,
with the repeat under `uncapped-repeat/`. Peak memory and final row count were
unchanged. The earlier session-era state-only sample was 0.654 s, so the smaller
overhead percentage alone does not demonstrate an improvement in diagnostics.

| Workload | State peak allocated GB | Diagnostic peak allocated GB | Main diagnostic peak allocated GB |
|---|---:|---:|---:|
| TFI | 50.914 | 50.914 | 51.714 |
| AFH uncapped | 1.095 | 1.095 | 0.948 |
| AFH capped | 7.341 | 7.552 | 7.057 |

Decimal GB. Reserved peaks (state/diagnostic/main) are respectively
79.530/66.150/70.701 GB for TFI, 1.497/1.497/1.309 GB for uncapped AFH, and
8.376/7.957/7.789 GB for capped AFH. TFI state reserved memory increased from
70.429 GB in the prior session-era check despite essentially unchanged allocated
peak. Reserved allocator cache depends on allocation history; no allocation
retries or OOMs occurred in these runs.

Against actual main, TFI and uncapped AFH discarded counts match exactly at
every gate, with maximum L1 errors 1.42e-14 / 5.68e-14 and L2 errors
3.47e-18 / 5.55e-17. Energies/norms agree to roundoff and final counts match.
Capped AFH does **not** have exact history equality: 875 discarded-count entries
differ (maximum 11,471); maximum L1/L2 differences are 0.103023 / 8.318e-6;
final energy/norm2 differences are 3.167e-5 / 7.234e-7. The previous investigation
identified exact top-k boundary ties at the first binding cap: 126,833 equal
candidates competing for 120,845 places, with different retained tied keys.
Same-input diagnostics matched at the first divergent gate and its successor.
Tied ordering is unspecified, so later support trajectories can differ. The
revised run also differs from the earlier session in 40 discarded-count entries
(maximum 12), while final energy/norm2 agree to roundoff. This is not reported
as exact diagnostic equality. Strict tests retain unambiguous binding caps.

### Backward boundary and memory

The large diagnostic outputs were passed to public `init_gradient_spo` for basis
expectation, with and without lambda_ose=0.13. All probes confirm identical key
and primal-coefficient pointers in SPO and SPGO. There is no temporary full
primal copy and no ownership-consuming API change.

| Workload | Retained adjoint GB | Basis extra peak GB | Basis + OSE extra peak GB |
|---|---:|---:|---:|
| TFI, 221,899,620 rows | 1.775 | 3.772 | 10.651 |
| AFH uncapped, 2,265,180 rows | 0.018 | 0.039 | 0.109 |
| AFH capped, 20,000,000 rows | 0.160 | 0.340 | 0.960 |

Extra peaks are above pre-initialization allocated memory and include temporary
loss tensors. Basis initialization took 11.86 / 0.29 / 1.48 ms respectively;
basis + OSE took 51.23 / 27.57 / 29.80 ms. These are one-shot boundary probes,
not full backward benchmarks. OSE still has substantial arithmetic workspace:
sharing the primal eliminates cloning, but does not eliminate that workspace.

Persistent backward/coefficient adjoints, angle gradients and analysis-loop
integration remain stage 3. Current backward correctness is covered by existing
tests; stage 2 makes no new full-backward runtime or peak-memory claims.

Stage 2 final API addition: `spd.evolve` accepts keyword-only `in_place=False`.
With `True`, Triton mutates the supplied object and retains capacity/index
between circuit calls; shared tensors detach before mutation. NumPy/JAX reject
this mode with `NotImplementedError`. The default still copies once and compacts
on return. Saving strings materializes through serialization. The added tests
compare consecutive in-place calls with functional calls and check unsupported
backends explicitly.

Final in-place API validation: ownership, diagnostics and runner suites: **96 passed in 76.73 s**. `git diff --check` passed.


## Stage 3: persistent backward and analysis

Stage 2 is committed as `f03cce5`. Stage 3 was approved for commit and remote
backup, including both backward in-place entry points.

### Implementation and API behavior

The existing generic pair-update kernel now has a compile-time backward
specialization. A stable pair owner reads both incoming coefficient and adjoint
channels, reduces the pre-rotation angle contribution once, then applies the
inverse rotation to both channels. Dead rows have both channels zeroed and can
be reactivated. Forward remains its own compile-time specialization.

Backward retention preserves the existing inclusive primal cutoff and meaningful
`(c,g)` support. At cutoff zero, nonzero adjoints with zero primal coefficients
survive and can create partner rows. A binding top-k cap ranks primal magnitudes
but assigns dead slots a score below every live row, so dead zeros cannot steal
places from live gradient-only rows. This is a correctness mask, not an optional
row-selection/indexing experiment. Temporary cap scores are released before
output copying and diagnostic workspace allocation. Discarded-count/L1/L2
accounting remains based on primal coefficients; angle gradients are computed
before truncation/capping as in the original implementation.

`SparsePauliGradientOp.apply_in_place()` selects backward evolution from the
adjoint channel. Exact inverse Cliffords transform both channels and rebuild the
index; rotations retain it. Public `backpropagate` and
`backpropagate_noise_analysis` copy once, retain storage across gates, and compact
on return. Their defaults remain input-preserving. At the user's request, both backward
entry points now accept `in_place=False`: `True` mutates the supplied Triton
SPGO and retains its storage on return. NumPy/JAX raise `NotImplementedError`
when this mode is requested. Skipped-only backward returns the original SPGO.
Direct diagnostic backward rotations use a private single-gate copy; the old
`operations._rotation(..., backward=True)` remains an independent reference.

Noise susceptibility reads storage after each gate, without exporting compact
arrays. Existing basis/OSE/L2 initialization, scalar observations, algebra,
translation, serialization and analysis APIs remain supported. SPGO initialization
shares the SPO's primal buffers; public backward makes one private working copy,
including adjoints, to preserve its input. It does not consume either caller
object or promise zero-copy backward evolution.

### Validation

- Initial existing backward/diagnostic/ownership suite: **66 passed in 27.04 s**.
- Mixed persistent backward, existing backward, runner, algebra/analysis and
  Clifford suites: **270 passed in 91.93 s**.
- Broader forward/JAX/semantic/persistent conformance, including the new
  backward/noise tests: **224 passed, 1 expected failure in 75.07 s**.
  The expected failure is the existing JAX cutoff-equality diagnostic issue.
- Final focused backward/forward-diagnostic/ownership suite, including the
  forward-only entry-point guard and cap workspace lifetime adjustment:
  **69 passed in 6.11 s**. `git diff --check` passed.

New tests compare every gate with the retained per-gate implementation for
float32/float64 and 3/65/121-qubit keys, including all nine Cliffords, binding
caps, inclusive cutoff, dead-owner reactivation, pre-update angle gradients,
gradient-only support growth, preservation of shared terminal primals, skipped
operations and storage lifecycle. Further tests compare full mixed-gate noise
histories and backward execution with truncation diagnostics disabled. Existing
tests supply independent NumPy/JAX comparisons, dense finite differences,
terminal-loss checks and the example optimizer paths.

### Backward runtime and memory

Small benchmark: `python benchmarks/benchmark_persistent_backward.py`.
Raw results: `benchmarks/results/persistent_integration_stage3_backward.json`.
Eight-site TFI/AFH with three layers, float64, cutoff 1e-4, cap 65536, basis
expectation plus 0.13 * OSE(alpha=2). These use a separate seeded parameter set
from the stage 2 forward benchmark. Terminal supports are 136 / 32,603 rows.

Larger scripts/results remain in `/tmp/spd-stage3-backward-large`:
TFI 11x11 uses **10 forward steps**, with 994,285 terminal rows, cutoff 2^-18;
AFH 6x6x6 uses cutoff 3e-4, with 2,265,180 terminal rows. Both use basis expectation
plus 0.13 * OSE(alpha=2), and nonbinding caps. This is not a measurement of
backpropagating the 23-step TFI or exact-20M AFH workload.

Every row below measures one backward pass on the **same terminal SPGO** for
both implementations. Compile first on tiny disposable inputs, prepare gate
metadata, then measure once per path. Timings include the backward loop,
diagnostics/history, copying and final compaction; exclude forward construction,
terminal initialization and host comparisons. GPU measurements run sequentially.
The reference is the original per-gate backward kernel retained in the current
worktree, with current wrappers—not a separate main checkout.

| Workload | Original per-gate backward | Persistent backward | Original peak allocated MB | Persistent peak allocated MB |
|---|---:|---:|---:|---:|
| TFI 8 sites, 3 layers | 10.139 ms | 10.401 ms | 0.119 | 0.120 |
| AFH 8 sites, 3 layers | 25.824 ms | 16.969 ms | 4.450 | 5.097 |
| TFI 11x11, 10 steps | 0.807 s | 0.468 s | 247.971 | 279.973 |
| AFH 6x6x6, uncapped | 1.970 s | 1.428 s | 845.969 | 968.318 |

Decimal MB. Reserved peaks (reference/persistent) are 2.097/2.097 MB,
6.291/6.291 MB, 274.727/297.796 MB, and 1033.896/1012.924 MB respectively.
Allocated peaks include the shared terminal inputs retained during both paths;
raw JSON also records incremental peaks above those inputs.

Keys, coefficients and adjoints match **exactly** in all four workloads.
Discarded counts match at every gate. Maximum angle-gradient error is 8.88e-16
across the small cases and 1.39e-16 / 5.55e-17 for the larger TFI/AFH cases.
The larger cases' maximum L1/L2 diagnostic errors are 5.55e-17 / 1.08e-19 and
7.11e-15 / 1.39e-17 respectively. Final backward supports are 16 / 24 / 89,945 /
384,943 rows. These comparisons validate backward results on shared terminal
inputs; independent end-to-end energy/gradient checks are in the existing tests.

### Forward regression measurements

The existing small forward benchmark was rerun after extending the kernel.
Raw results: `benchmarks/results/persistent_integration_stage3_forward.json`.
Workloads and measurement protocol match the stage 2 small benchmark.

| Workload/path | Original per-gate ms | Persistent ms | Original peak allocated bytes | Persistent peak allocated bytes |
|---|---:|---:|---:|---:|
| TFI state-only | 6.298 | 5.178 | 109,056 | 111,104 |
| TFI diagnostics | 10.551 | 7.355 | 110,080 | 111,104 |
| AFH state-only | 16.204 | 16.986 | 2,615,808 | 2,856,448 |
| AFH diagnostics | 22.572 | 16.882 | 2,624,512 | 2,856,448 |

Reserved peaks remain 2,097,152 bytes for TFI and 4,194,304 for AFH.
Keys and coefficients match exactly, energies agree to roundoff, and histories
pass the existing strict comparison. Allocated peaks match stage 2. Short
one-shot timings fluctuate; the AFH state-only sample shows no speedup and is
not presented as evidence of a timing guarantee.

### Limits and next acceptance stage

Persistent backward exchanges some additional live memory for index/buffer reuse.
The larger cases use about 13–14% more peak allocated memory than the original
per-gate backward path while taking about 42% / 28% less time. The small TFI case
shows no speedup. These one-shot samples establish sanity, not statistical timing
bounds. Forward state-only and forward diagnostic results are reported separately
in stage 2; backward gains must not be attributed to those paths.

The stage 2 TFI state-only reserved-memory increase remains an allocation-history
question: unchanged peak allocated memory does not prove identical headroom.
Stage 4 acceptance should include allocator tracing of that workload before any
claim of equivalent memory headroom, along with consolidated forward/diagnostic/
backward performance checks. OSE initialization still uses arithmetic workspace;
this stage introduces no optional local arithmetic, alternate indexing, layout,
padding or multi-GPU experiments. Top-k boundary ties remain unspecified.


### Follow-up: public backward in-place mode and complete pipeline memory

Both `backpropagate` and `backpropagate_noise_analysis` now accept keyword-only
`in_place=False`. When requested, Triton mutates the supplied SPGO, retains its
storage on return, and detaches shared buffers before mutation. NumPy/JAX raise
`NotImplementedError`. Defaults and skipped-only identity remain unchanged.
New tests cover repeated public calls, both analysis modes, rejected backends,
and preservation of the originating SPO. The persistent-backward suite passed
**34 tests in 5.31 s** after this addition.

For compact same-support basis/OSE initialization, an SPO of X bytes and an SPGO
of Y bytes share their primal buffers: retained combined storage is roughly Y,
with roughly Y-X additional adjoint bytes. This is not a peak-memory guarantee:
loss workspace, spare capacity, dead slots and the retained index also matter.
L2 initialization can construct different support and is not covered by that
simple sharing formula.

The initializer marks primal buffers shared. Therefore the first backward mutation
still makes one private working copy, even with `in_place=True`. The implementation
uses a conservative sharing flag, not lifetime tracking: deleting the original
SPO alone does not remove that copy. No consuming initialization API was added.

Complete forward → basis+0.13*OSE(alpha=2) initialization → backward measurements
are in `/tmp/spd-stage3-inplace-memory`, including scripts, phase metrics, raw
arrays and comparisons. Same TFI 11x11 **10-step** and uncapped AFH 6x6x6 workloads
as the preceding backward benchmark. Separate processes, tiny compilation only,
one measured pipeline per mode, no cache clearing between phases. Peaks include
all three phases and initialization workspace. The source SPO remains alive.

| Workload | Functional peak allocated MB | In-place peak allocated MB | In-place + compact before initialization MB |
|---|---:|---:|---:|
| TFI 11x11, 10 steps | 279.622 | 327.790 | 271.667 |
| AFH 6x6x6, uncapped | 1094.671 | 1207.420 | 1094.670 |

Reserved peaks for the same three modes: TFI 299.893 / 423.625 / 297.796 MB;
AFH 1497.367 MB in all modes. Decimal MB throughout.

TFI forward peak alone falls from 207.978 to 183.779 MB and forward copies from
ten to one. However, leaving forward capacity untrimmed increases storage held
across initialization and the backward copy. One explicit `spo.compact()` before
initialization restores compact primal storage. Then the total TFI peak falls
about 2.8%; AFH total peak is unchanged because forward dominates (its backward
peak falls from 968.318 to 950.196 MB). Initialization makes zero primal copies
and backward makes one private copy in every measured mode.

Keys and coefficients match exactly in all modes, discarded counts match at every
gate, and adjoints/angles/diagnostic norms agree to roundoff. Removing the remaining
handoff copy requires explicit ownership transfer/consumption or lifetime-aware
sharing. Such a change must define what happens to the source SPO and exposed
views; it cannot silently consume the current public input.

The consuming variational-pipeline feasibility result and deferred TODO are
preserved in [triton_variational_pipeline_todo.md](triton_variational_pipeline_todo.md).
This follow-up is outside stage 4 acceptance.
