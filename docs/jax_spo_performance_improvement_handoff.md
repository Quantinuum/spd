# JAX SPO Performance Improvement Handoff

## Purpose

This file turns the performance exploration into separate implementation tasks.
Each task is intended to be started in its own Codex thread.

The main goals are:

- reduce JAX GPU runtime,
- reduce peak device memory,
- keep current sparse-Pauli semantics where a task is marked exact,
- make approximate alternatives explicit,
- avoid changing several parts of the design at once.

No proposal in this file has been implemented as part of this handoff.

## Project Rule About `kernels.py`

Do not change `spd/jax_backend/kernels.py` without asking the user first.

Every task below includes a `kernels.py` field:

- `No`: the task should not need that file.
- `Maybe`: stop and ask if the implementation reaches that file.
- `Yes`: ask before starting the code change.

Do not invent a complicated workaround merely to avoid `kernels.py`.

## Current Execution Map

```text
pytket / OpenQASM
        |
        v
backend-independent circuit IR
        |
        v
BackendAdapter
        |
        v
run_circuit.evolve / run_circuit.backpropagate
        |
        v
JAX algorithm dispatch
        |
        +-- stack_sort_merge
        +-- search_update_merge
        +-- search_update_merge_donate
        |
        v
shared JAX kernels and SparsePauliOp / SparsePauliGradientOp
```

Main files:

- `spd/run_circuit.py`: public forward and backward loops.
- `spd/backend_adapter.py`: operation dispatch and generator packing.
- `spd/jax_backend/sparse_pauli.py`: JAX SPO and SPGO objects.
- `spd/jax_backend/kernels.py`: shared low-level operations and algorithm selection.
- `spd/jax_backend/algorithms/stack_sort_merge.py`: split, sort, merge path.
- `spd/jax_backend/algorithms/search_update_merge.py`: sorted search/update path.
- `spd/jax_backend/algorithms/search_update_merge_donate.py`: donated capped path.
- `examples/benchmark_jax_memory_donation.py`: current memory benchmark.

Related notes:

- [`jax_memory_efficiency_handoff.md`](jax_memory_efficiency_handoff.md)
- [`jax_donated_search_update_merge_working_note.md`](jax_donated_search_update_merge_working_note.md)
- [`jax_next_algorithm_proposal.md`](jax_next_algorithm_proposal.md)
- [`multi_gpu_tpu_spo_handoff.md`](multi_gpu_tpu_spo_handoff.md)
- [`tfi_symmetry_breaking_findings.md`](tfi_symmetry_breaking_findings.md)

## Current Representation And Memory Model

Let:

```text
N = physical system size
W = ceil(N / 32)
M = stored row capacity
p = coefficient bytes, 4 for single and 8 for double precision
```

The packed X/Z row contains `2W` `uint32` values.

```text
XZ bytes per row   = 8W
SPO bytes per row  = 8W + p
SPGO bytes per row = 8W + 2p
```

For 64 qubits in double precision:

```text
SPO row  = 24 bytes
SPGO row = 32 bytes
```

At `M = 33,554,432`:

```text
stored SPO  ~= 0.75 GiB
stored SPGO ~= 1.00 GiB
```

The stored state is not the main peak. The current rotation algorithms create
large `2M` intermediates.

## Current Bottleneck

### `stack_sort_merge`

One rotation does roughly this:

1. create two `M`-row branches,
2. concatenate them into `2M` rows,
3. lexicographically sort all keys,
4. merge duplicates,
5. sort all coefficients by magnitude,
6. apply threshold and cap truncation,
7. slice the result.

### `search_update_merge`

One rotation does roughly this:

1. keep the input state lexicographically sorted,
2. compute one partner key per row,
3. binary-search partners in the stored state,
4. update existing rows,
5. create `M` insertion slots,
6. concatenate base and insertion slots into `2M` rows,
7. run coefficient `top_k`,
8. create masks and masked arrays of length `2M`,
9. lexsort all `2M` rows,
10. return `2M` rows to Python before slicing.

The search path avoids duplicate segment reduction, but still rebuilds and
sorts the full candidate state.

### Donated search/update

The donated path keeps the `2M` work inside one JIT and returns `M` rows when
the stored state is already at the cap. This enables input/output aliasing and
better buffer scheduling.

## Existing Evidence

The checked-in H100 measurements for 64 qubits, double precision, and
`M = 33,554,432` include:

| Workload | Current search | Donated full step |
| --- | ---: | ---: |
| Gradient-evaluation GPU peak | about 17.1 GB | about 6.8 GB |
| Gradient-evaluation live JAX peak | about 7.85 GB | about 4.87 GB |
| Gradient-evaluation runtime | about 7.60 s | about 7.05 s |
| Isolated backward runtime | about 2.58 s | about 1.86 s |

See [`jax_memory_efficiency_handoff.md`](jax_memory_efficiency_handoff.md) and
[`jax_donated_search_update_merge_working_note.md`](jax_donated_search_update_merge_working_note.md).

The current examples still default to `stack_sort_merge`. Their shared CLI
does not expose `search_update_merge_donate`.

The local development environment used during the exploration had JAX 0.8.1
and CPU only. No new GPU result was produced during that exploration.

## Correctness Invariants

Exact tasks must preserve these rules unless the task explicitly changes the
public contract:

1. A live row has a physically valid X/Z key and aligned coefficient.
2. A padded or invalid row has zero primal and gradient coefficients.
3. Hard cutoff uses `abs(c) > trunc_val` for live terms.
4. Cap truncation keeps the largest live coefficient magnitudes.
5. Value and gradient arrays remain aligned with the same X/Z rows.
6. Search/update output remains lexicographically sorted when its metadata says so.
7. Backward parameter gradients match the NumPy/reference path.
8. Truncation count, L1 norm, and L2 norm describe the terms actually removed.
9. Tie handling at the cap is deterministic or is documented and tested.
10. A donated input must be treated as consumed.

## Benchmark Protocol For All Performance Tasks

Use one process per large benchmark variant.

Measure separately:

- first-call compile plus execution time,
- warmed steady-state execution time,
- peak live JAX bytes,
- total GPU memory reserved,
- memory after the operation,
- compiled input, output, alias, and temporary sizes when available.

Always block on a result before recording runtime.

For memory:

- record JAX and `jaxlib` versions,
- record GPU model, CUDA version, and allocator variables,
- run with the normal allocator settings,
- also run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
- use XProf memory traces when possible,
- do not use `nvidia-smi` reservation alone as live-memory evidence.

Use at least these workload cases:

1. one forward step while growing,
2. one forward step at the cap,
3. one backward step at the cap,
4. a full forward/init-gradient/backward evaluation,
5. a real TFI or AFH circuit,
6. `lambda_ose=0`,
7. nonzero `lambda_ose`,
8. exact zero-angle rotations,
9. generic nonzero rotations.

Correctness checks should include:

- normalized term dictionaries,
- expectation values,
- backward gradients,
- stored size and live count,
- truncation statistics,
- single- and double-precision cases where supported.

## Task Dependency Summary

```text
P0 profiling and telemetry
 |
 +--> E6 skip unnecessary top-k
 +--> E7 gather-first capped output
 +--> A3 delta merge decision
 +--> A4 compaction redesign
 +--> A6 hash-table engine

E1 expose donated algorithm --> production donated trials

A1 fixed-capacity state --> A2 compiled gate blocks

A4 exact cutoff selection --> A6 hash-table engine --> D1 multi-device row sharding

N1 MPO and N2 stochastic propagation are independent backend experiments.
```

The tasks below are ordered by recommended priority, not by task ID.

---

## Task P0: Representative GPU Profile And SPO Telemetry

### Type

Measurement only. No semantic change.

### Goal

Produce a reliable profile of the real workload and measure the properties
that determine which merge design is appropriate.

### Why

The existing benchmarks establish that donation helps. They do not yet show:

- how many rows anticommute at each gate,
- how often a partner already exists,
- how many genuinely new rows are inserted,
- how often the cap is active,
- whether `top_k`, partner search, or lexsort dominates each phase,
- how zero and tiny angles affect the real workflow.

The insertion fraction is required before choosing a delta-merge design.

### Suggested scope

- Extend benchmark-only scripts.
- Add optional algorithm telemetry without changing normal public output.
- Capture XProf traces on a representative GPU.
- Record compilation separately from steady-state execution.

Useful counters per rotation:

```text
input capacity
input live count
anti-commuting count
partner hit count
missing partner count
nonzero insertion count
post-threshold count
cap active
returned capacity
```

### Likely files

- `examples/benchmark_jax_memory_donation.py`
- a new focused benchmark under `examples/`
- algorithm modules only if optional scalar counters are needed
- a new results note under `docs/`

### `kernels.py`

Maybe. Keep telemetry in algorithm modules if simple. Ask before changing
`kernels.py`.

### Acceptance criteria

- One documented command reproduces each profile.
- Results identify growth and capped phases separately.
- Memory numbers distinguish live bytes from allocator reservation.
- The report records software and hardware versions.
- No production result changes.

### Suggested thread request

> Implement Task P0 from `docs/jax_spo_performance_improvement_handoff.md`: add representative GPU profiling and per-step SPO telemetry. Keep production semantics unchanged. Do not modify `kernels.py` without asking me first.

---

## Task P1: Reproducible JAX Performance Environment

### Type

Infrastructure and documentation.

### Goal

Make benchmark results comparable across machines and sessions.

### Why

The project currently accepts a broad JAX version range. XLA buffer scheduling,
sort lowering, donation, and Pallas APIs can change between versions.

### Suggested scope

- Record `jax`, `jaxlib`, Python, CUDA, driver, and GPU versions in benchmark output.
- Record relevant XLA allocator environment variables.
- Document the reference H100 environment.
- Consider a separate benchmark environment file or lock file.
- Consider JAX persistent compilation cache only for startup time, not as a
  substitute for steady-state runtime work.

### Likely files

- `examples/benchmark_jax_*.py`
- `README.md` or a benchmark-specific document
- environment or dependency metadata if the user wants it

### `kernels.py`

No.

### Acceptance criteria

- Every performance result contains enough environment data to reproduce it.
- Compile time and warmed runtime are reported separately.
- The normal library dependency policy is not narrowed without discussion.

### Suggested thread request

> Implement Task P1 from `docs/jax_spo_performance_improvement_handoff.md`: make the JAX benchmark environment reproducible and record all compiler, allocator, CUDA, and GPU metadata. Do not change algorithm behavior.

---

## Task E1: Expose The Donated Search Algorithm In Main Examples

### Type

Exact. Low-risk integration.

### Goal

Allow the main gradient scripts to select the already implemented
`search_update_merge_donate` algorithm.

### Why

The strongest checked-in memory result uses whole-step donation. The algorithm
is already registered and tested, but the shared gradient CLI only offers
`stack_sort_merge` and `search_update_merge`.

### Suggested scope

- Add `search_update_merge_donate` to the shared algorithm choices.
- Keep the default unchanged for the first patch unless the user explicitly
  asks to change it.
- Document that the input state is consumed at capped donated steps.
- Record the chosen algorithm in existing run metadata.
- Add or extend runner-level smoke tests.

### Likely files

- `examples/gradient/run_utils.py`
- `examples/README.md`
- relevant example tests
- possibly `spd/jax_backend/README.md`

### `kernels.py`

No. The algorithm is already registered.

### Acceptance criteria

- Main gradient scripts accept `--algorithm search_update_merge_donate`.
- Metadata records the selected algorithm.
- Existing algorithm choices still work.
- A small forward/backward example matches `search_update_merge`.
- The documentation explains the donation ownership rule.

### Suggested thread request

> Implement Task E1 from `docs/jax_spo_performance_improvement_handoff.md`: expose `search_update_merge_donate` in the main gradient examples and documentation. Keep the current default unless I explicitly approve changing it.

---

## Task E2: Remove Accidental Device-To-Host Dtype Materialization

### Type

Exact. Small cleanup.

### Goal

Inspect JAX array dtype without converting the coefficient array to NumPy.

### Why

`run_circuit._infer_backend_name_and_precision` currently calls
`np.asarray(state.c_array).dtype`. On GPU, converting a JAX array to NumPy can
materialize the array on the host. Only dtype metadata is needed.

### Suggested scope

- Use JAX array dtype metadata directly.
- Check both SPO and SPGO paths.
- Confirm NumPy backend behavior remains unchanged.
- Add a regression test that does not require a real GPU.

### Likely files

- `spd/run_circuit.py`
- `tests/test_backend_adapter.py` or a focused runner test

### `kernels.py`

No.

### Acceptance criteria

- Backend inference does not call `np.asarray` on a full JAX coefficient array.
- Backend and precision inference results remain unchanged.
- Existing runner tests pass.

### Suggested thread request

> Implement Task E2 from `docs/jax_spo_performance_improvement_handoff.md`: remove the full JAX-array NumPy conversion used only for dtype inference. Keep behavior identical and add a focused regression test.

---

## Task E3: Cache Lowered Circuits And Packed Rotation Generators

### Type

Exact. Orchestration improvement.

### Goal

Avoid rebuilding the pytket circuit IR and packing Pauli generators during
every optimizer evaluation.

### Why

The gradient examples rebuild circuits with the same topology on every
evaluation. Only angle values change. The backend also converts Pauli strings
to packed arrays one gate at a time.

This is secondary for very large SPOs, but can matter during small and growing
phases and is needed for whole-block JIT work.

### Suggested design

Separate static operation structure from dynamic angles:

```text
packed generator array
operation kind array
qubit/control/target metadata
dynamic theta vector
```

Do not add a broad public API before a small internal design is proven.

### Likely files

- `spd/circuit_ir.py`
- `spd/pytket_frontend.py`
- `spd/backend_adapter.py`
- `spd/run_circuit.py`
- gradient example setup code
- frontend and runner tests

### `kernels.py`

No.

### Acceptance criteria

- Repeated evaluations reuse static packed generators.
- Changing theta values does not rebuild the circuit topology.
- Current public workflows still work.
- A benchmark shows lower Python/dispatch overhead for small states.

### Suggested thread request

> Implement Task E3 from `docs/jax_spo_performance_improvement_handoff.md`: prototype cached lowered circuits and packed Pauli generators for repeated optimizer evaluations. Keep the public API simple and preserve current semantics.

---

## Task E4: Configurable Progress Diagnostics

### Type

Exact when diagnostics are enabled. User-visible execution option.

### Goal

Avoid full-state norm and OSE reductions after every gate when the user does
not need them.

### Why

The operation loop computes norm and OSE for progress output after every
non-skipped operation. Existing synthetic tests suggest this is not the main
peak-memory cause, but it is still repeated full-array work and host
synchronization.

### Suggested scope

- Add a diagnostics policy such as `every_step`, `every_n`, or `none`.
- Keep current behavior as the initial default unless discussed.
- Preserve truncation accounting even when progress diagnostics are disabled.
- Avoid complicated logging abstractions.

### Likely files

- `spd/run_circuit.py`
- public API tests
- README and examples

### `kernels.py`

No.

### Acceptance criteria

- Users can disable norm/OSE progress reductions.
- Final SPO/SPGO and gradients do not change.
- Truncation info remains available.
- Benchmark reports the runtime difference on a realistic loop.

### Suggested thread request

> Implement Task E4 from `docs/jax_spo_performance_improvement_handoff.md`: make per-gate norm/OSE progress diagnostics configurable without changing mathematical results or truncation accounting.

---

## Task E5: Exact Public Cap Instead Of Mandatory Power-Of-Two Rounding

### Type

Exact cap-policy change. Needs careful compatibility review.

### Goal

Allow `max_num_str` to remain the exact user-requested cap.

### Why

The public JAX runner rounds the cap upward:

```text
100,000    -> 131,072
1,000,000  -> 1,048,576
30,000,000 -> 33,554,432
```

The extra rows increase resident and temporary memory. The low-level selected
algorithms already accept static non-power-of-two cap values.

### Suggested design

- Keep geometric internal growth buckets if they help compilation reuse.
- Treat the final user cap separately from internal bucket sizes.
- The last bucket may be the exact cap.
- Define `get_size`, live count, and stored capacity clearly.

### Likely files

- `spd/run_circuit.py`
- algorithm wrappers
- tests for runner cap behavior
- memory-estimate helpers and documentation

### `kernels.py`

Maybe. Ask before changing it.

### Acceptance criteria

- Returned live terms never exceed the requested cap.
- Exact top-magnitude selection is preserved.
- Non-power-of-two caps pass forward and backward tests.
- Benchmark compares exact and rounded caps on GPU.
- Any public behavior change is documented.

### Suggested thread request

> Implement Task E5 from `docs/jax_spo_performance_improvement_handoff.md`: separate the exact user `max_num_str` cap from internal growth buckets. Preserve exact top-magnitude selection and ask before modifying `kernels.py`.

---

## Task E6: Skip `top_k` When It Selects The Entire Candidate Array

### Type

Exact. High-priority algorithm cleanup.

### Goal

Avoid coefficient ranking during growth when the cap cannot remove any row.

### Why

The search path uses:

```text
k = min(max_num_str, 2M)
top_k(scores, k)
```

When `max_num_str >= 2M`, `k == 2M`. The code ranks every candidate and then
performs a lexsort. Only threshold filtering is needed in this case.

The condition is static for a compiled input shape and static cap.

### Suggested scope

- Add a static uncapped-growth branch.
- Apply hard threshold directly.
- Preserve exact truncation statistics.
- Keep output lexsorted.
- Cover forward and backward.

### Likely files

- `spd/jax_backend/algorithms/search_update_merge.py`
- donated algorithm only if it reuses a changed helper
- forward/backward algorithm tests
- benchmark script

### `kernels.py`

No.

### Acceptance criteria

- `top_k` is absent from the compiled growth branch when it cannot remove rows.
- Results and step info match the old path.
- GPU benchmark covers at least two growing sizes.
- Capped behavior remains unchanged.

### Suggested thread request

> Implement Task E6 from `docs/jax_spo_performance_improvement_handoff.md`: skip `top_k` in `search_update_merge` when it would select the full `2M` candidate array. Preserve hard cutoff, step info, sorting, and forward/backward parity.

---

## Task E7: Gather Selected Rows Before Lexsort

### Type

Exact. Highest-priority new algorithm experiment.

### Goal

At the cap, convert `2M` scalar candidates directly into `M` selected rows and
lexsort only those `M` rows.

### Current flow

```text
2M scores
-> M top indices
-> 2M selected mask
-> 2M masked XZ/value/gradient arrays
-> lexsort 2M rows
-> return 2M rows
-> slice M rows
```

### Proposed flow

```text
M base arrays + M candidate arrays
-> 2M scalar scores
-> M top indices
-> gather M XZ/value/gradient rows
-> zero/PAD invalid selected slots
-> lexsort M rows
-> return M rows
```

### Important details

- A top index below `M` refers to an updated base row.
- A top index at or above `M` refers to an insertion row.
- If fewer than `M` rows pass `trunc_val`, selected `-inf` slots must become
  PAD/zero rows.
- Forward and backward must use the same structural selection.
- Tie behavior must match the old stable `top_k` behavior or be explicitly
  defined.
- Step-info computation must not reintroduce large `2M` masked XZ arrays.
- The optimization barrier used for truncation-stat scheduling may still be
  needed.

### Likely files

- `spd/jax_backend/algorithms/search_update_merge.py`
- `spd/jax_backend/algorithms/search_update_merge_donate.py`
- focused benchmark scripts
- forward, backward, truncation, and donation tests

### `kernels.py`

No, unless a shared low-level gather helper becomes necessary. Ask first in
that case.

### Acceptance criteria

- The capped compiled function returns `M`, not `2M`, XZ/value rows.
- Lexsort input length is `M` at the cap.
- Input donation aliases matching outputs where expected.
- Exact forward/backward results match the old search path.
- Truncation statistics match.
- H100 peak memory and runtime improve over the current donated path.

### Suggested thread request

> Implement Task E7 from `docs/jax_spo_performance_improvement_handoff.md`: prototype gather-first capped selection for search/update. Select from `2M` scalar scores, gather `M` rows, lexsort only `M`, preserve exact forward/backward semantics and truncation stats, and do not touch `kernels.py` without asking.

---

## Task E8: Exact Zero-Angle And Non-Branching Rotation Fast Paths

### Type

Exact for recognized angles.

### Goal

Avoid merge and sort when a Pauli rotation cannot create a second branch.

### Priority case: `theta = 0`

Forward:

- the state is unchanged,
- no partner lookup or merge is required,
- truncation info is zero.

Backward:

- the propagated SPGO is unchanged,
- the parameter gradient may be nonzero,
- compute the gradient with partner lookup only,
- do not build `2M` candidates.

### Other exact cases

When `sin(theta) == 0`, no new key is created. Anti-commuting coefficients may
change sign when `cos(theta) == -1`.

Clifford angles with `cos(theta) == 0` map anti-commuting keys one-to-one and
may permit a separate no-branch implementation. Treat this as a second stage.

### Why

Several gradient workflows initialize whole parameter families at exactly
zero. Current frontends still emit those rotations, and current JAX paths run
the full merge.

### Likely files

- algorithm modules
- possibly `spd/backend_adapter.py` or frontend normalization
- forward/backward tests at exact special angles
- gradient example benchmark

### `kernels.py`

Maybe. Ask before changing it. Do not create a complicated duplicate gradient
kernel merely to avoid a simple shared change.

### Acceptance criteria

- Exact zero forward returns an unchanged operator.
- Exact zero backward returns an unchanged SPGO and the correct gradient.
- No merge or lexsort appears in the zero-angle compiled path.
- Results match NumPy at `0`, `pi`, and selected Clifford angles.
- A free-fermion initialization benchmark shows the effect.

### Suggested thread request

> Implement Task E8 from `docs/jax_spo_performance_improvement_handoff.md`: add exact zero-angle forward and gradient-only backward fast paths, then evaluate other non-branching angles. Preserve gradients and ask before changing `kernels.py`.

---

## Task E9: Same-Support OSE Gradient Initialization

### Type

Exact. One-off gradient-workflow optimization.

### Goal

Avoid a full general merge when adding the OSE terminal gradient to the basis
or L2 terminal gradient on identical support.

### Why

For nonzero `lambda_ose`, the current initializer computes two SPGOs derived
from the same SPO and combines them through general SPGO addition. General
addition concatenates, sorts, and merges rows.

When support and primal coefficients are already aligned, only the gradient
coefficient arrays need elementwise addition.

This can be important in the symmetry-breaking examples, where
`lambda_ose` is commonly nonzero.

### Suggested design

- Build the final `SparsePauliGradientOp` directly on the existing SPO support.
- Reuse X/Z and primal coefficient buffers.
- Add only gradient coefficients.
- Preserve the `lexsorted` flag.
- Keep general SPGO addition unchanged.

### Likely files

- `spd/jax_backend/kernels.py`
- NumPy backend only if a shared semantic cleanup is desired
- backend conformance and gradient-init tests
- gradient-evaluation benchmark with nonzero `lambda_ose`

### `kernels.py`

Yes. Ask before implementation.

### Acceptance criteria

- Nonzero-`lambda_ose` gradient initialization performs no general SPO merge.
- Terminal gradient values match the old initializer.
- X/Z and primal coefficients remain aligned.
- JAX/NumPy conformance tests pass.
- Peak memory is measured inside a full gradient evaluation.

### Suggested thread request

> Implement Task E9 from `docs/jax_spo_performance_improvement_handoff.md`: optimize same-support OSE gradient initialization by combining gradient coefficients elementwise. This task requires `kernels.py`; ask me before editing it.

---

## Task E10: Gradient Collection And Object-Lifetime Cleanup

### Type

Exact. Workflow cleanup.

### Goal

Reduce Python/device scalar overhead and retained memory during repeated
objective evaluations.

### Current issues

- Backpropagation appends one JAX scalar object per parameterized gate.
- Example code converts many device scalars through Python and NumPy slices.
- Assigning the returned final SPGO to `_` still retains it until the function
  returns.
- Old final SPO/SPGO references can retain buffers between phases.

### Suggested scope

- Stack or aggregate raw gradients once, rather than performing many scalar
  host transfers.
- Consider backend-side segment reduction for shared parameters.
- Explicitly release unused large state objects after their final use.
- Keep donation aliasing rules clear.
- Do not hide lifetime-sensitive behavior behind magic cleanup.

### Likely files

- `spd/run_circuit.py`
- gradient example `combine_grads` helpers
- example tests and benchmark modes

### `kernels.py`

No for basic collection and lifetime cleanup. Maybe for backend-side parameter
aggregation; ask first.

### Acceptance criteria

- Gradient values remain identical.
- Host transfer count is reduced.
- Retained memory after one objective evaluation is reduced or explained.
- Peak memory and after-run memory are reported separately.

### Suggested thread request

> Implement Task E10 from `docs/jax_spo_performance_improvement_handoff.md`: reduce raw-gradient scalar transfers and clean up large object lifetimes in the gradient workflow. Preserve exact gradients and donation ownership.

---

## Task E11: Precision And Mixed-Precision Study

### Type

Numerical experiment before any default change.

### Goal

Measure whether single or mixed precision can reduce runtime and memory without
breaking the target workloads.

### Memory expectation at 64 qubits

```text
double SPO row  = 24 bytes
single SPO row  = 20 bytes
double SPGO row = 32 bytes
single SPGO row = 24 bytes
```

Packing means single precision does not halve total state memory. It saves
about 17% for SPO storage and 25% for SPGO storage at 64 qubits, with larger
savings in coefficient-heavy intermediates.

### Suggested experiments

- float32 state and reductions,
- float32 state with float64 scalar diagnostics,
- float32 forward with a double reference,
- workload-specific tolerances at `trunc_val` values from `1e-6` to `1e-14`.

Do not propose float32 for `1e-14` truncation without strong evidence.

### Likely files

- benchmark scripts
- precision tests
- documentation first
- implementation files only after a result is chosen

### `kernels.py`

Maybe. Ask before any mixed-precision kernel change.

### Acceptance criteria

- Report energy, gradient, truncation, and support differences.
- Report runtime and live memory on the target GPU.
- No default precision changes without user approval.

### Suggested thread request

> Implement Task E11 from `docs/jax_spo_performance_improvement_handoff.md`: run a single- and mixed-precision accuracy/performance study for representative SPD workloads. Do not change defaults or `kernels.py` without discussing the results first.

---

## Task A1: Fixed-Capacity State With Explicit Live Count

### Type

Exact architectural change.

### Goal

Keep array shapes stable across many gates and stop reading the next dynamic
slice size on the host every step.

### Proposed representation

```text
xz_array[capacity, words]
c_array[capacity]
grad_c_array[capacity]       # SPGO only
live_count
capacity
```

### Suggested policy

- Keep PAD/zero rows beyond `live_count`.
- Grow through geometric capacity buckets.
- Keep the exact user cap separate from buckets.
- Shrink only with hysteresis, for example when live count falls far below
  capacity.
- Expose or document the difference between stored capacity and live terms.

### Why

Current power-of-two shape changes cause recompilation and allocator churn.
The host reads `new_size`, chooses a static slice size, and calls separate slice
JITs after each rotation.

### Risks

- Processing too much padding can slow early steps.
- Public `get_size()` semantics may need a decision.
- Fixed capacity must not make PAD rows mathematically live.
- Donation and aliases must remain valid.

### Likely files

- `spd/jax_backend/sparse_pauli.py`
- algorithm modules
- `spd/run_circuit.py`
- tests for live count and capacity
- documentation

### `kernels.py`

Maybe. Ask before changing it.

### Acceptance criteria

- Repeated steps in one bucket do not recompile because of live-count changes.
- No host read is required solely to choose every output shape.
- PAD/zero invariants hold.
- Results match the current exact path.
- Benchmarks report compute spent on padding versus compilation savings.

### Suggested thread request

> Implement Task A1 from `docs/jax_spo_performance_improvement_handoff.md`: prototype fixed-capacity JAX SPO/SPGO storage with explicit live count and hysteretic growth/shrink. Preserve exact semantics and ask before changing `kernels.py`.

---

## Task A2: Compile Consecutive Rotation Blocks With `lax.scan`

### Type

Exact architectural change. Depends on stable shapes.

### Goal

Move consecutive gate execution inside one compiled JAX computation.

### Dependencies

- Task A1 or another fixed-shape state design.
- Task E3 is helpful for packed static circuit data.

### Why

The current runner calls one compiled step per operation and repeatedly
synchronizes scalar results. A fixed-shape scan can:

- keep the state on device,
- reduce Python dispatch,
- enable whole-block donation,
- accumulate gradient scalars and truncation statistics on device,
- reuse buffers across iterations.

### Suggested first scope

- One block containing only Pauli rotations.
- Fixed number of operations.
- Packed generator matrix and theta vector.
- Return final state, gradient vector, and aggregate statistics.
- Keep Clifford operations outside the first prototype.

### Risks

- Sort operations may limit fusion.
- A large scan body may increase compile time.
- Returning full per-step diagnostics can retain extra memory.
- Carry shape must remain fixed.

### Likely files

- a new JAX algorithm or execution module
- `spd/run_circuit.py`
- cached circuit representation
- focused tests and benchmarks

### `kernels.py`

Maybe. Ask before changing it.

### Acceptance criteria

- One block executes through one compiled loop.
- Final state and gradients match stepwise execution.
- Compile time and steady-state time are reported separately.
- The benchmark distinguishes dispatch savings from merge-kernel savings.

### Suggested thread request

> Implement Task A2 from `docs/jax_spo_performance_improvement_handoff.md`: prototype fixed-shape consecutive Pauli rotations inside `jax.lax.scan`, using Task A1-style state. Preserve exact results and ask before editing `kernels.py`.

---

## Task A3: Sorted Delta Merge Prototype

### Type

Exact experimental algorithm.

### Goal

Sort only genuinely new partner rows and merge them into the sorted base.

### Dependency

Task P0 must first measure missing-partner and insertion fractions.

### Proposed flow

1. update matched base coefficients,
2. identify missing partner rows,
3. sort insertion rows,
4. merge sorted base and sorted insertions,
5. compact only when required.

### Decision rule

Proceed only if real workloads usually have a small insertion fraction.

If missing insertions are near `M / 2`, the delta is not small. A custom
two-way merge may then offer little benefit over optimized accelerator sorting.

### Prototype stages

1. benchmark a pure two-way merge of already sorted arrays,
2. compare it with concatenate plus lexsort,
3. test several delta fractions,
4. integrate only if the primitive result is favorable.

### Likely files

- new benchmark script
- a new algorithm module if the primitive wins
- algorithm parity tests
- update to `jax_next_algorithm_proposal.md` or a new result note

### `kernels.py`

Maybe. Ask before adding a shared merge kernel there.

### Acceptance criteria

- Primitive benchmark covers delta ratios such as 1%, 10%, 25%, and 50%.
- Integrated output is exactly lexsorted and deduplicated.
- Forward and backward parity pass.
- The algorithm is rejected cleanly if GPU data does not support it.

### Suggested thread request

> Implement Task A3 from `docs/jax_spo_performance_improvement_handoff.md`: use Task P0 telemetry to decide whether a sorted delta merge is justified, benchmark the primitive across insertion ratios, and integrate only if GPU results support it. Ask before changing `kernels.py`.

---

## Task A4: Exact Selection Without Sorted `top_k` Output

### Type

Exact algorithm research.

### Goal

Find the exact magnitude cutoff for the largest `K` terms without fully
ordering all selected coefficients.

### Why

SPD only needs to know which rows survive. It does not need coefficients in
descending order once search storage is restored to lexicographic order.
`top_k` returns selected values in ranked order, which may do extra work.

### Candidate designs

1. Multi-pass radix selection on positive floating-point magnitude bit patterns.
2. Histogram high bits, locate the boundary bucket, then refine only that bucket.
3. Exact cutoff plus deterministic index tie breaking.

### Required semantics

- keep exactly the correct number of largest live terms,
- respect hard threshold first,
- define deterministic ties,
- compute exact removed count, L1, and L2 statistics,
- avoid a full selected mask if possible.

### Prototype rule

Start as a standalone scalar-selection benchmark. Do not combine it with a
hash-table rewrite in the same first task.

### Likely files

- new benchmark script
- new experimental algorithm/helper module
- truncation and tie tests

### `kernels.py`

Maybe. Ask before adding the primitive there.

### Acceptance criteria

- Exact selection matches stable `top_k` on random and adversarial ties.
- Peak memory and runtime beat `top_k` on the target sizes.
- The result is tested for float32 and float64.
- If it does not win, keep the result as a documented rejected experiment.

### Suggested thread request

> Implement Task A4 from `docs/jax_spo_performance_improvement_handoff.md`: benchmark an exact cutoff/radix-selection alternative to sorted `top_k`. Preserve deterministic ties and exact truncation statistics. Keep it standalone first and ask before modifying `kernels.py`.

---

## Task A5: Packed-Key Layout Study

### Type

Representation research.

### Goal

Measure whether a different packed X/Z key layout reduces sort/search cost or
row memory.

### Current limitation

The representation stores separately padded X and Z halves:

```text
2 * ceil(N / 32) uint32 words
```

A tightly packed `2N`-bit key needs:

```text
ceil(2N / 32) uint32 words
```

This saves one 32-bit word per row for some system sizes, for example systems
just above a 32-qubit boundary or systems at 16 qubits and below.

At exactly 64 qubits it does not save bytes.

### Candidate experiments

- current multiword uint32 layout,
- tightly packed symplectic bits,
- uint64 key words where supported,
- structure-of-arrays versus row-major keys,
- scalar or hash-assisted primary sort keys.

### Risks

- Commutation and product kernels currently rely on separate X/Z halves.
- Fewer sort keys can still lose if 64-bit operations are slower.
- A hash fingerprint adds resident memory and must handle collisions exactly.

### Likely files

- benchmark-only prototype first
- no production files until results exist

### `kernels.py`

Yes for any production layout change. Ask before implementation.

### Acceptance criteria

- Report bytes per row by system size.
- Benchmark population-count, XOR/product, lexsort, and partner lookup.
- Preserve exact key equality.
- Do not migrate production storage without an end-to-end win.

### Suggested thread request

> Implement Task A5 from `docs/jax_spo_performance_improvement_handoff.md`: run a packed-key layout benchmark covering memory, commutation, Pauli product, lexsort, and lookup. Keep it benchmark-only first; any production change to `kernels.py` requires my approval.

---

## Task A6: Fixed-Capacity GPU Hash-Table Engine

### Type

Complete exact sparse-engine redesign.

### Goal

Replace lexsorted sparse-table reconstruction with direct hash lookup, update,
and insertion.

### Proposed state

```text
fixed-capacity key table
primal coefficient table
gradient coefficient table
occupancy / tombstone metadata
live count
```

### Proposed rotation step

1. compute partner key `Q = P xor generator`,
2. assign one owner for each pair,
3. look up the partner in expected constant time,
4. rotate existing pairs once,
5. insert missing partners with atomic coordination,
6. mark rows removed by threshold,
7. compact or rebuild only when load or tombstones require it,
8. enforce the global cap with an exact selection primitive.

### Why

This removes the requirement to lexsort the full state after every gate. It
trades extra resident table capacity for much lower temporary memory and less
global data movement.

### Important design questions

- open addressing scheme and load factor,
- exact multiword-key equality,
- atomic insertion ownership,
- tombstone cleanup,
- deterministic behavior,
- fixed capacity and overflow reporting,
- interaction with exact top-K selection,
- forward/backward alignment,
- Pallas versus CUDA/FFI implementation.

### Recommended stages

1. CPU reference hash-table step.
2. Single-device correctness prototype.
3. Standalone lookup/insert benchmark.
4. Forward integration.
5. Backward integration.
6. Large GPU benchmark.

Do not implement multi-GPU routing in the same first task.

### Likely files

- new algorithm/backend modules
- custom-kernel integration files
- new tests and examples
- documentation

### `kernels.py`

Maybe or yes, depending on integration. Ask before changing it.

### Acceptance criteria

- Exact parity on small circuits.
- Duplicate partner insertions cannot create duplicate live keys.
- Clear overflow behavior.
- Exact or explicitly selected cap policy.
- Peak memory and runtime beat gather-first search/update on target workloads.
- The current algorithms remain available as references.

### Suggested thread request

> Implement Task A6 from `docs/jax_spo_performance_improvement_handoff.md`: design the first fixed-capacity GPU hash-table SPO engine stage. Start with a small exact reference and primitive benchmark, keep current algorithms intact, and ask before modifying `kernels.py`.

---

## Task X1: Error-Budgeted Tiny-Angle Pruning

### Type

Approximate. Changes the simulated circuit.

### Goal

Skip rotations whose effect is below an explicit error budget.

### Evidence

The checked-in TFI diagnostic found:

```text
tiny nonzero Rz values:
  unique rows ~= 6600

same Rz values set exactly to zero:
  unique rows ~= 230
```

The measured energy and sector masses were effectively unchanged in that case.

### Error bound idea

For a Pauli rotation, skipping the rotation changes the anti-commuting operator
component by an L2 amount bounded by:

```text
2 * abs(sin(theta / 2)) * ||O_anti||_2
```

Use a bound like this to charge an explicit error ledger. Do not use an
undocumented angle tolerance.

### Important gradient issue

At exact zero, the parameter derivative may still be nonzero. Approximate
pruning must define whether:

- the parameter is constrained/frozen,
- the approximate objective uses a zero derivative,
- or an alternative consistent gradient is computed.

Do not silently return an inconsistent gradient.

### Likely files

- optional circuit preprocessing or runner policy
- error-accounting utilities
- dedicated example and tests
- documentation of changed semantics

### `kernels.py`

No for circuit-level pruning.

### Acceptance criteria

- Pruning is opt-in.
- The error budget is reported.
- Energy and gradient error are measured against the exact path.
- Exact zero fast paths remain separate from approximate pruning.

### Suggested thread request

> Implement Task X1 from `docs/jax_spo_performance_improvement_handoff.md`: prototype opt-in, error-budgeted tiny-angle pruning. Keep exact zero handling separate, define consistent gradient semantics, and report accuracy versus support/runtime reduction.

---

## Task X2: Approximate Cap Selection Or Adaptive Threshold

### Type

Approximate. Changes which terms survive the cap.

### Goal

Avoid exact global top-K when a controlled approximate selection is acceptable.

### Candidate designs

- `jax.lax.approx_max_k`,
- log-magnitude histogram cutoff,
- adaptive `trunc_val` chosen to target capacity,
- local threshold followed by a small boundary correction,
- approximate local compaction with periodic exact checkpoints.

### Required reporting

- retained top-K recall,
- operator L1/L2 difference,
- expectation and gradient error,
- support size,
- runtime and memory.

### Risks

- Optimizer behavior may be sensitive to discontinuous selection.
- Approximation may be platform-specific.
- `K` is often a large fraction of `2M`; approximate top-K may not be faster in
  that regime.

### Likely files

- standalone benchmark first
- optional new algorithm name if results are good
- accuracy tests and example

### `kernels.py`

Maybe. Ask before integration.

### Acceptance criteria

- Exact algorithms remain unchanged.
- Approximation is opt-in and named clearly.
- Accuracy/runtime curves cover real workloads.
- No default change without user approval.

### Suggested thread request

> Implement Task X2 from `docs/jax_spo_performance_improvement_handoff.md`: benchmark approximate cap selection and adaptive-threshold policies. Keep them opt-in, report support/energy/gradient error, and ask before changing `kernels.py`.

---

## Task X3: Commuting-Layer Or Small-Block Fusion

### Type

Exact before truncation, but usually changes truncation timing. Treat as an
approximate algorithm unless per-gate truncation is reproduced.

### Goal

Reduce the number of global merge/compaction events by applying several
commuting rotations as one structured block.

### Motivation

TFI and AFH ansätze contain layers of translated commuting rotations with a
shared angle. Exact unitary conjugations commute inside such a layer, but the
current per-gate truncation makes order observable.

### Candidate stages

1. Combine adjacent identical generators by adding angles.
2. Fuse a very small block while retaining an exact sparse map internally.
3. Apply truncation only at block boundaries and measure the difference.
4. Explore subgroup/coset or local tensor representations only if small-block
   results are promising.

### Risks

- Intermediate support can grow exponentially.
- Layer-boundary truncation changes current results.
- A large fusion may only move the same merge cost elsewhere.

### Likely files

- circuit preprocessing or a new experimental algorithm
- new correctness/accuracy tests
- TFI/AFH benchmark

### `kernels.py`

Maybe. Ask before changing it.

### Acceptance criteria

- Exact and changed-truncation modes are clearly distinguished.
- Gate ordering and error differences are documented.
- The benchmark includes support growth inside the block.

### Suggested thread request

> Implement Task X3 from `docs/jax_spo_performance_improvement_handoff.md`: prototype small commuting-rotation block fusion, clearly separate exact unitary behavior from changed truncation timing, and benchmark support/runtime/error before wider integration.

---

## Task N1: MPO / Operator Tensor-Train Backend

### Type

New approximate backend design.

### Goal

Represent the evolved operator by tensor-network bond dimension instead of an
explicit list of Pauli strings.

### Best target

One-dimensional local circuits where operator-space entanglement remains
moderate even when explicit Pauli support is large.

### Proposed first stage

- Implement a small standalone MPO reference for 1D Pauli operators.
- Apply local one- and two-qubit gates.
- Compress with SVD and a controlled discarded-weight tolerance.
- Compute the same product-basis expectation values.
- Compare against exact small SPD cases.

### Later questions

- adjoint gradients,
- checkpointing or recomputation,
- OSE interpretation,
- 2D snake ordering or PEPO alternatives,
- conversion between sparse Pauli and MPO forms.

### Risks

- Bond dimension can grow rapidly.
- Two-dimensional workloads may be poor.
- The approximation error differs from coefficient truncation.

### Likely files

- a new backend package
- new examples and tests
- documentation

### `kernels.py`

No. Keep it separate from the current JAX sparse backend.

### Acceptance criteria

- Small 1D exact parity before compression.
- Controlled discarded-weight reporting.
- Runtime/memory curve versus bond dimension.
- Energy and gradient comparison where supported.

### Suggested thread request

> Implement Task N1 from `docs/jax_spo_performance_improvement_handoff.md`: design a standalone 1D MPO/operator tensor-train prototype backend with controlled SVD compression and small-case parity against SPD. Do not modify the current JAX sparse kernels.

---

## Task N2: Stochastic Pauli Propagation

### Type

New stochastic backend design.

### Goal

Sample Pauli propagation paths instead of storing and merging all branches.

### Basic idea

For an anti-commuting rotation, sample the two branches associated with the
cosine and sine contributions using importance weights. Propagate a population
of Pauli trajectories and estimate observables statistically.

### Candidate designs

- independent importance-sampled paths,
- particle population with resampling,
- common random numbers across nearby optimizer evaluations,
- deterministic large-coefficient head plus stochastic tail.

### Required analysis

- estimator bias,
- variance growth with circuit depth,
- sign or weight instability,
- gradient estimator consistency,
- optimizer robustness to noise.

### Likely files

- separate experimental backend/package
- sampling examples and statistical tests
- no changes to exact JAX algorithms initially

### `kernels.py`

No.

### Acceptance criteria

- Known small cases have confidence intervals covering exact results.
- Bias and variance are reported separately.
- Memory scales with sample population rather than exact support.
- A hybrid deterministic/stochastic option is evaluated if pure sampling has
  high variance.

### Suggested thread request

> Implement Task N2 from `docs/jax_spo_performance_improvement_handoff.md`: build a standalone stochastic Pauli-propagation prototype, measure bias and variance against exact SPD, and evaluate a deterministic-head/stochastic-tail variant.

---

## Task D1: Multi-GPU Hash-Sharded SPO

### Type

Distributed architectural change.

### Goal

Distribute one very large SPO or SPGO across several devices by row ownership.

### Dependency

Stabilize the single-device ownership and compaction design first, preferably
after Task A6 or a simpler fixed-capacity equivalent.

### Recommended ownership

Start with hash sharding:

```text
owner = hash(xz_row) mod num_shards
```

Equal keys then route to the same owner, making duplicate resolution local.

### Proposed step

1. compute local commutation and partners,
2. update locally owned partners,
3. compute destination owner for missing rows,
4. exchange insertion rows,
5. merge or insert on the owner shard,
6. apply local threshold,
7. use global scalar reductions for diagnostics,
8. perform global cap selection only when required.

### Capacity representation

```text
xz_shards[num_shards, capacity_per_shard, words]
c_shards[num_shards, capacity_per_shard]
grad_c_shards[num_shards, capacity_per_shard]
live_counts[num_shards]
```

### Development stages

1. fake shards on CPU,
2. sharded representation on one device,
3. multiple fake CPU devices,
4. small real multi-GPU test,
5. DGX benchmark.

### Risks

- all-to-all insertion traffic,
- shard imbalance,
- local overflow,
- exact global top-K cost,
- hash collision handling,
- static shape constraints.

### Likely files

- new distributed algorithm/backend modules
- sharding configuration and tests
- new examples and handoff updates

### `kernels.py`

Maybe. Ask before changing it.

### Acceptance criteria

- Fake-shard parity with the single-device exact path.
- Duplicate rows from different shards merge correctly.
- Explicit overflow behavior.
- Communication volume and load balance are reported.
- Current single-device algorithms remain available.

### Suggested thread request

> Implement Task D1 from `docs/jax_spo_performance_improvement_handoff.md`: begin the fake-shard CPU stage of a fixed-capacity, hash-owned multi-device SPO design. Preserve exact small-case parity and ask before modifying `kernels.py`.

---

## Task D2: Embarrassingly Parallel Workload Decomposition

### Type

Distributed orchestration experiment.

### Goal

Use multiple devices without sharding one SPO when the scientific workload has
independent evaluations.

### Candidate parallel work

- independent observables or Hamiltonian terms,
- independent optimizer seeds,
- hyperparameter sweeps,
- independent target operators,
- selected line-search or population-based optimizer evaluations.

### Why

This is much simpler than distributed sparse-table mutation. It may provide a
useful multi-GPU path for workloads that do not require one SPO to exceed one
device.

### Limits

- Many examples already exploit translation symmetry and evolve one local
  representative observable.
- L-BFGS evaluations are mostly sequential.
- This does not solve single-SPO memory limits.

### Likely files

- example/orchestration code
- no core algorithm changes initially

### `kernels.py`

No.

### Acceptance criteria

- The decomposed result matches the serial aggregation.
- Device utilization and communication overhead are reported.
- The use case is documented clearly so it is not confused with row sharding.

### Suggested thread request

> Implement Task D2 from `docs/jax_spo_performance_improvement_handoff.md`: evaluate embarrassingly parallel decomposition across independent SPD observables or runs before attempting row-sharded execution. Keep core algorithms unchanged.

---

## Recommended Implementation Order

### Phase 1: Establish facts

1. Task P0: profiling and telemetry.
2. Task P1: reproducible environment metadata.

### Phase 2: Exact low-risk wins

1. Task E1: expose the donated algorithm.
2. Task E2: remove dtype host materialization.
3. Task E6: skip unnecessary full-size `top_k`.
4. Task E7: gather selected `M` rows before lexsort.
5. Task E8: exact zero-angle fast paths.
6. Task E9: same-support OSE initialization, after `kernels.py` approval.
7. Tasks E3, E4, E5, and E10 as supporting workflow improvements.

### Phase 3: Stable-shape execution

1. Task A1: explicit capacity and live count.
2. Task A2: compiled rotation blocks.

### Phase 4: Choose one larger sparse-engine direction

- Task A3 if insertion deltas are small.
- Task A4 if exact cap selection remains dominant.
- Task A6 if sorting remains the fundamental limit.
- Task A5 only if key width is a measured bottleneck.

### Phase 5: Changed-semantics and new-backend research

- Task X1: tiny-angle pruning.
- Task X2: approximate selection.
- Task X3: commuting-layer fusion.
- Task N1: MPO backend.
- Task N2: stochastic backend.

### Phase 6: Distribution

- Task D2 for easy independent parallelism.
- Task D1 for a single SPO that exceeds one device.

## First Recommended New Algorithm Task

If only one new implementation task is selected, start with Task E7:

```text
2M scalar scores
-> select M indices
-> gather M rows
-> lexsort M rows
-> return M rows inside a donated full-step JIT
```

It directly attacks the proven `2M` bottleneck. It can remain within the
algorithm modules in its first form. It preserves the current mathematical
model and provides a clean comparison before a hash-table or new-backend
rewrite.

## Handoff Assumptions

- JAX on GPU remains the main production path.
- Memory pressure is driven mainly by merge intermediates and workflow buffer
  lifetimes, not by the stored SPO alone.
- Exact semantics should remain the default unless the user chooses an
  approximate task explicitly.
- Existing dirty-worktree files belong to the user and must be preserved.
- Existing algorithms should remain available as references while experiments
  are evaluated.
