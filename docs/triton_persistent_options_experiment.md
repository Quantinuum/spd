# Persistent-storage option evaluation

## Baseline and isolation

Verified user commit **350828acd8428e133a7cb180201dffd01ff9fb59**
(`persitent-storage-base`): persistent module, tests and report, based on
anticommuting-index commit `63f1786`. Those committed files are unchanged.
Experiments are uncommitted on `experiment/triton-persistent-options`; default
backend dispatch and diagnostic/backward production paths are unchanged.

Each listed area was evaluated separately, in priority order, against persistent
storage. Experiments use one H100 80GB HBM3, float64, the original 11×11 OBC XX+Z
circuit, 341 original-order/wrapped-angle gates per step, 23 steps, strict cutoff
2^-18, and a nonbinding cap. Every timed step warms from its own input. Metadata
construction/maintenance, capacity changes, compaction and final materialization
are included in propagation time. Observations and sampling are outside it.
GPU trials run sequentially. These are bounded design experiments, not an
exhaustive search over layouts, index algorithms or all possible combinations.

`experiment_kernels.py` and `persistent_options.py` provide explicit variant
selection; shared address/update code makes comparisons and adjoint probes easier.
A no-feature `control` variant checks the common experimental harness against the
unchanged committed baseline. Original trial source hashes are saved in each
`option.json`; repeat trials validate the final experiment code. No variant
changes full-key equality or numerical truncation semantics.

## Results and recommendation

**Select persistent storage plus the maintained bitset inverted index as the
next integration candidate.** Keep the simple packed-bit scan available as a
lower-memory alternative. Do not add local specialization or fingerprints to
the selected design on the present evidence.

All figures below are uninstrumented seconds for the full 23-step trajectory;
memory is peak allocated decimal GB. Ranges show independent repeat trials,
not confidence intervals.

| Option | Total (s) | Peak GB | Assessment |
|---|---:|---:|---|
| Committed persistent baseline, fresh runs | 9.520–9.618 | 50.91 | Preserved baseline; earlier runs were 9.399 s |
| No-feature experimental control | 9.230–9.240 | 50.91 | Matched control for small option effects |
| Local Z/XX specialization | 9.357–9.367 | 50.91 | No advantage over matched control |
| Maintained bitset inverted index | **5.779–5.823** | **60.70** | Clear winner; about 1.59× versus matched control |
| Fresh packed-bit active-row scan | 6.379 | 52.00 | Lower-memory alternative, about 10% slower than bitsets |
| Word-oriented keys | 11.749 | 50.91 | Reject this tested layout |
| Cached fingerprints | 9.214–9.237 | 54.88 | No established gain versus matched control |
| Compact at 3% dead rows | 10.981 | 51.27 | Too frequent in this workload |
| Compact at 25% dead rows | 9.461 | 60.42 | No clear benefit; more memory |
| Compact at 50% dead rows | 9.689 | 60.14 | No benefit; allocator retried once |
| Growth factor 1.1 | 9.643 | 51.97 | No demonstrated benefit |
| Growth factor 1.5 | 9.547 | 55.32 | No demonstrated benefit; more memory |
| Additional reusable buffers | 9.528 | 72.64 | No speed benefit; substantially more live memory |
| Device-resident counts in eight-gate batches | 11.315 | 54.85 | Current conservative launch strategy loses |

The committed kernel and experimental scaffold have a measurable timing
difference, and the fresh committed baseline varies from its earlier 9.399 s.
Therefore tiny gains relative to that earlier number are not attributed to an
option: the matched no-feature scaffold is the stricter comparison. This is why
the apparent standalone fingerprint improvement is not accepted as a win.

The maintained index averages **5.801 s**, with final steps **1.2868–1.2874 s**.
It is approximately 1.64–1.66× faster than the fresh committed baseline, or 1.59×
versus the matched scaffold. Its additional peak allocation is **9.79 GB**.
That tradeoff matters when adding adjoints or targeting smaller GPUs; the small
backward probes do not establish full backward memory use or speedup.

Three targeted interaction checks did not improve the recommendation:

| Combination | Total (s) | Peak GB |
|---|---:|---:|
| Inverted index + local specialization | 5.810 | 60.70 |
| Inverted index + fingerprints | 6.001 | 64.67 |
| Inverted index + local specialization + fingerprints | 6.036 | 64.67 |

The specialization combination is within standalone run variation, so prefer
the simpler standalone index. Fingerprint combinations cost more time and
memory in these trials. No other combinations were tested.

All **22 full-trajectory trials** completed all 23 steps and matched every term
count. Across them, maximum expectation difference is **3.33e-16** and maximum
norm difference **2.22e-16**. **221 distinct tests passed**: the initial 205-test
suite passed in 22.81 s, and 16 additional missing-partner adjoint checks passed
in 2.02 s. Together these comprise 76 option/adjoint/precision checks plus the
existing 145 persistent/forward invariant tests.

## Winner profile

The independent inverted-index final-step profile takes **1.307 s**:

| Device stage | Calls | Seconds |
|---|---:|---:|
| Active-row lookup/update | 254 | 0.4591 |
| Bitset query and row-list construction | 340 | 0.3361 |
| Bitset construction/incremental maintenance | 222 | 0.2202 |
| Hash construction/incremental insertion | 222 | 0.0983 |
| Compaction, including final materialization | 5 | 0.0902 |
| Direct device-to-device copies | 14 | 0.0367 |

There are fewer update launches because gates with no selected rows can skip
that kernel after the first full pass. Additional metadata copies, fills and
initial counting are present in the raw trace; table rows are exclusive GPU
stage times, not an exhaustive wall-time decomposition. CPU count readback
waits overlap these kernels and must not be added again.

The original persistent update took about 2.065 s at the final step; selective
execution reduces it to 0.459 s, while paying for selection and index maintenance.
The benefit is primarily at large support. Early steps favor the simpler scan
path; choosing a crossover for automatic dispatch is future integration work,
not an implemented hybrid in these results.

## What each trial changes

1. **Local Z/XX:** derive parity and phase from the one/two participating sites.
   Site indices are device scalars, avoiding hundreds of compiled gate-position
   specializations. Other generators use the generic kernel.
2. **Active-row selection:** compare two methods with the same active-row update
   kernel. `select` scans the relevant packed bits at each gate. `inverted`
   maintains X/Z bitplanes, updates appended rows, rebuilds after compaction, and
   queries those bitplanes to produce a row list. Both include row-list creation
   and its count readback. The first gate processes all rows so initially small
   commuting coefficients are truncated at the correct point; subsequent gates
   use the fixed-cutoff invariant. Dead keys remain indexed, independent of
   whether their primal coefficient is zero.
3. **Word-oriented layout (`soa`):** store keys by packed word rather than by row;
   adapt insertion, exact lookup and compaction. Materialize normal row-oriented
   arrays at the public boundary. This tests one concrete layout, not tiled or
   other possible layouts.
4. **Fingerprints:** cache each stored key's 32-bit hash and check it before loading
   candidate keys. A matching fingerprint still requires every key word to match.
   Metadata follows growth, append and compaction. It does not derive an XOR
   partner's hash from the cached original hash.
5. **Physical cleanup / growth:** independently sweep dead-row thresholds 3%,
   25%, 50% (baseline 10%), then reserve growth factors 1.1 and 1.5 (baseline
   1.25). Logical truncation remains immediate after every gate.
6. **Buffers:** reuse a table when capacity is unchanged and pool one old key/
   coefficient workspace for subsequent growth/compaction. The committed version
   already reuses its working arrays and counters across gates; this tests
   additional reuse at storage-maintenance boundaries.
7. **Device counts:** batch eight gates between host count checks, maintaining
   used/live counts on-device. Preflight capacity checks stop the batch before
   unsafe mutation; binding caps stop it immediately after the relevant gate so
   the host can apply top-k before continuing. Physical compaction is deferred to
   batch boundaries. Conservative launch grids and extra insertion kernels are
   included in the measured cost.

## CUDA graph feasibility probe

`probe_persistent_capture.py` separately compares identical kernels with/without
capture on closed Pauli support: 64 or 4,096 rows, 32 rotations, cutoff zero.
Every XOR partner already exists, so keys and row count stay fixed. Both methods
match the ordinary forward path's coefficients. Ten warm trials per method:

| Rows | Uncaptured median | Captured median | Capture setup |
|---|---:|---:|---:|
| 64 | 1.466 ms | 0.131 ms | 75.5 ms |
| 4,096 | 1.047 ms | 0.260 ms | 33.8 ms |

This establishes a dispatch benefit on fixed support only. Setup requires reuse
for amortization. It is **not** a captured run of the growing 11×11 workload and
must not be used to predict its speedup. Dynamic growth, compaction, pointer
changes and graph reuse still need a suitable execution design. The full-scale
device-count trial tests a prerequisite independently and does not capture it.

## Semantics and backward-representative checks

All full-scale completed runs are compared against every reference count, norm
and expectation; full 221.9M-term coefficient equality is not claimed. The
`comparison.json` records per-run maximum scalar differences and allocation
retries as well as time and peak memory.

The option suite includes multi-gate coefficient comparisons with the committed
forward implementation, binding caps, float32/float64, mixed local/generic
rotations and word-boundary packing. Backward-representative probes exercise
Z/XX gates on closed two-qubit support and gradient-only missing-partner growth,
comparing primal coefficients, adjoints,
meaningful-row counts and angle gradients with the existing diagnostic backward
kernel. They include a zero-primal/nonzero-adjoint row and run on the indexing,
layout, fingerprint and shortlisted combination variants.

These probes verify that row/index choices can carry adjoints and preserve pair
ownership. They do **not** implement a full persistent backward trajectory,
backward compaction/top-k, diagnostic accounting, or demonstrate backward speedup.
The production persistent storage's definition of live rows must be extended to
include nonzero adjoints before full backward integration. Fixed-cutoff active
selection also needs a fresh full pass when entering a new cutoff regime.

## Reproduction

Run sequentially from the repository root:

```sh
python -m pytest -q tests/test_triton_persistent_options.py tests/test_triton_persistent.py tests/test_triton_backend.py tests/test_triton_invariants.py
python benchmarks/benchmark_persistent_options.py --variant inverted --output-dir /tmp/spd-inverted
python benchmarks/benchmark_persistent_options.py --variant baseline --dead-fraction .25 --output-dir /tmp/spd-compact25
python benchmarks/benchmark_persistent_options.py --variant baseline --growth-factor 1.1 --output-dir /tmp/spd-growth110
python benchmarks/probe_persistent_capture.py --output /tmp/spd-capture.json
python benchmarks/summarize_persistent_options.py benchmarks/results/persistent_options_20260910
```

Other isolated variants: `local`, `select`, `soa`, `fingerprint`, `buffers`,
`device`; `baseline` runs committed code, while `control` uses the experimental
kernel scaffold without an optimization. Results, source hashes, logs, memory
counters and tests are under `benchmarks/results/persistent_options_20260910/`.
