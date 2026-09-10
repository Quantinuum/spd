# Anticommuting-only forward index experiment

Branch: `experiment/triton-anticommuting-index`, based on
`c9a2b004d2598f1741c04619dfa30865e86a9aad`. Changes are uncommitted for review.

## Implementation

The state-only `conjugate_pauli_rotation` now calls a separate
`build_anticommuting_index` kernel. It computes symplectic commutation parity,
hashes the anticommuting rows, and inserts their original row IDs. A predicated
PTX integer CAS instruction suppresses atomic memory traffic from inactive
lanes; Triton's `atomic_cas` API has no mask argument. Probing still compares
complete keys in the unchanged rotation kernel. No coefficient atomics are used.

The kernel handles arbitrary packed Pauli generators, including Y and gates
spanning multiple words. This first experiment does not specialize on Z/XX
locations, shrink the table, change the output layout, or introduce persistent
storage. The table still has at least 2N slots and output capacity remains 2N.
The full index and its callers in backward/diagnostics/algebra are unchanged.

A queried partner of P is P XOR G. Its commutation parity with G equals that
of P, so every partner requested by `rotate` remains in the sparse index.
Coefficient arithmetic, disabled FP fusion, strict combined-coefficient cutoff,
gate order, wrapped angles, input immutability and size-cap behavior remain as
before. The index retains original row IDs, so coefficient addressing is unchanged.

## Measurement protocol

Same H100 80GB HBM3, PyTorch 2.8.0+cu128 and Triton 3.4.0 as the preceding
[profile investigation](monoprop_gpu_profile_20260910.md). Each process runs alone
on the GPU. The original 11×11, float64, 23-step XX+Z benchmark warms each step
from its own input and times the second propagation with synchronization.
Expectations, norms, copies and allocator counters are outside the timer.
Expandable allocator segments remain enabled by the backend default.

The first candidate run uses `investigate_triton_allocator.py` directly. The
comparison runner `benchmark_anticommuting_index.py` records source hashes and
hardware/software provenance. Its `--index full` control substitutes the original
full-index kernel and discards the extra gate argument; the rest of the production
path is identical. That control adds a small Python wrapper around each index
launch. It is compared with the earlier unwrapped baseline to detect meaningful
timing differences. Profiled timings are reported separately from these runs.

## Uninstrumented results

| Run | Total propagation (s) | Final step (s) |
|---|---:|---:|
| Earlier full-index baseline | 73.503657 | 18.649192 |
| Fresh full-index control | 73.481375 | 18.663887 |
| Anticommuting index, run 1 | 27.086327 | 6.809514 |
| Anticommuting index, run 2 | 27.076855 | 6.816342 |

The two candidate totals average **27.081591 s**, a **2.714× speedup** over
the earlier baseline (63.2% less propagation time), or **2.713×** versus the
fresh full-index control. The control is within 0.04% of the earlier baseline. These are two independently
launched trajectories, with one timed warm sample per step in each; they do not
constitute a broad confidence interval or a survey of workloads.

For context, the saved MonoProp totals are 58.769949 s at 32 threads and
51.819652 s at 48 threads. The candidate is about 1.91× faster than the latter
by elapsed propagation time, but MonoProp uses different truncation semantics
and produces different term counts. This is an approximate performance comparison,
not an equivalent-accuracy speedup.

Peak allocated memory remains **51,658,223,104 bytes (48.11 GiB)** and peak
reserved **70,701,285,376 bytes (65.85 GiB)**, with no allocation retries or OOMs.
This optimization saves GPU work, not buffer capacity.

## Profile attribution

Independent warmed CPU/CUDA profiling passes, seconds per step:

| Step | Original index build | Sparse index build | Original rotate | Sparse rotate |
|---|---:|---:|---:|---:|
| 3 | 0.002351 | 0.000543 | 0.001179 | 0.001004 |
| 14 | 0.393511 | 0.041409 | 0.257488 | 0.245099 |
| 23 | 12.679052 | 1.007484 | 5.745365 | 5.559089 |

Final-step index construction is **12.59× faster**. The unchanged rotation
kernel is about 3.2% faster, consistent with reduced occupancy of the lookup
table; this profile does not independently isolate the reason for that smaller
gain. Initialization remains 0.222 s and count D→H copies total 0.000475 s.
Final profiled wall time is 6.817 s, close to the uninstrumented 6.810–6.816 s.

The remaining rotation kernel accounts for **81.5%** of profiled final-step wall
time, versus 14.8% for index construction. It includes partner lookup, arithmetic,
thresholding, compaction and output writes; 81.5% must not be attributed entirely
to commuting-row copies. Early-step profiler overhead remains substantial.
The profiled trajectory also matches all 23 counts; maximum expectation and norm
differences versus the fresh full control are 3.33e-16 and 1.11e-16.

## Validation and limits

The broad Triton suite passed **372 tests in 152.16 s**, covering forward,
backward, diagnostics, algebra, Clifford, runner and JAX conformance paths,
as well as the first six sparse-index membership cases. Three additional cases
exercise no active lanes, sparse active lanes across block boundaries and all
active lanes; the complete nine-case index suite passed in 2.04 s. Together these runs cover
375 distinct tests (six membership cases appear in both runs).

Existing independent NumPy and dense-oracle tests cover coefficient values,
strict threshold boundaries, cancellation, multiword packing, collisions and
input preservation. New tests check table membership directly against Pauli-letter
commutation rules, including a partial final block. Both candidate trajectories match every baseline term count, ending at
221,899,620 terms. Maximum expectation and norm differences are each 1.11e-16.
The fresh full-index control also matches all counts, with expectation differences
up to 3.33e-16 and norm differences up to 1.11e-16. These full-trajectory scalar
checks do not establish exact full-coefficient equivalence for the 221.9M-term state.

Reported speedups apply to this workload. Dense anticommuting workloads and other
GPU architectures have not been performance-characterized. PTX is NVIDIA-specific,
as is this backend. Multi-GPU execution has not been tested: distributed partner
communication remains separate work, and this change does not define ownership
or repartition rows by gate.

## Next priority

Keep this as the first experiment and persistent storage/incremental indexing as
the second priority. With rotation/output now dominating, persistence could reduce remaining output
rewrites as well as repeated index construction. Profile those internal costs
before predicting its benefit. It must preserve immediate logical deletion at
each cutoff and the immutable-input contract. Local gate specialization, smaller
tables, layout changes and inverted indexing remain independent candidates; this
experiment does not rule them out.

## Reproduction

Run sequentially from the repository root:

```sh
python -m pytest -q tests/test_triton_*.py
python -u benchmarks/benchmark_anticommuting_index.py --index anti --output-dir /tmp/spd-anti
python -u benchmarks/benchmark_anticommuting_index.py --index full --output-dir /tmp/spd-full
python -u benchmarks/profile_triton_forward.py --output-dir /tmp/spd-anti-profile
```

Artifacts are under `benchmarks/results/anticommuting_index_20260910/`.
The earlier synchronized baseline and full-index traces are under
`benchmarks/results/forward_profile_20260910/`.
