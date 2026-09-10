# Persistent forward storage experiment

The anticommuting-index change was committed as `63f1786` on
`experiment/triton-anticommuting-index`, containing exactly the five requested
files. This second experiment is on `experiment/triton-persistent-storage`,
branched from that commit. The new prototype is opt-in and uncommitted for review;
the default state-only forward path still uses the committed sparse index.

## Results

Same H100 80GB HBM3, float64, 11×11 OBC XX+Z, 341 gates per timestep, 23 steps,
cutoff 2^-18, original pytket gate order and wrapped angles. Each step warms from
its own input and times a second synchronized propagation. Initial private-state
copying, index construction, growth, periodic compaction and final materialization
are **inside** the propagation timer. Observations are outside it. GPU processes
run sequentially; allocator defaults are unchanged.

| Run | Total propagation (s) | Final step (s) |
|---|---:|---:|
| Original full-index control | 73.481375 | 18.663887 |
| Committed anticommuting index, run 1 | 27.086327 | 6.809514 |
| Committed anticommuting index, run 2 | 27.076855 | 6.816342 |
| Fresh anticommuting-index control | 27.083108 | 6.805292 |
| Persistent storage, run 1 | 9.398818 | 2.311488 |
| Persistent storage, run 2 | 9.398502 | 2.310850 |

The persistent runs average **9.398660 s**, about **2.88× faster** than the
committed anticommuting-index version and **7.82× faster** than the original
full-index control. The fresh anticommuting control took 27.083108 s,
confirming the 2.88× comparison. These are two independently launched trajectories with one
warmed timed sample per step, not a confidence interval across machines or
workloads. Saved MonoProp-48 takes 51.819652 s, but has different truncation
semantics; that comparison is not an equivalent-accuracy speedup.

All 23 counts match the committed sparse-index trajectory, ending at
221,899,620 terms. Maximum expectation and norm differences across the two runs
are both 2.22e-16. Full-scale validation compares counts and scalar summaries,
not every coefficient of the 221.9M-term result.

Peak allocated memory is **50,914,279,936 bytes**, versus 51,658,223,104 for the
sparse-index version. Peak reserved is **70,428,655,616 bytes**, versus
70,701,285,376. Neither persistent run has an allocation retry or OOM. Final
returned tensor storage is **8,875,984,800 bytes**, versus 17,751,969,600: output
is materialized at its actual live size. Peak workspace remains much larger than
that final state because growth and compaction temporarily hold old/new buffers.

## Profile attribution

The independent final-step profile takes 2.327 s (uninstrumented repeats:
2.311 s). Exclusive device totals:

| Stage | Calls | Time (s) |
|---|---:|---:|
| Pair update / lookup | 341 | 2.065120 |
| Full builds plus incremental insertion | 222 | 0.098242 |
| Periodic and final compaction | 5 | 0.089575 |
| Device-to-device copies for initial ownership and growth | 14 | 0.036816 |
| Table/counter initialization | 357 | 0.006847 |
| Per-gate two-counter D→H copies | 341 | 0.000470 |

The copy column is direct CUDA D→D transfer time; the separate compaction column
also reads/writes state buffers. Initial live-count reduction and other small
kernels add about 0.001 s. Host `cudaMemcpyAsync` appears to take 2.272 s because
the pageable counter readback waits for preceding GPU work; that time overlaps
the kernel totals and must not be added as independent communication cost.

Pair update/lookup now accounts for about **89%** of final-step wall time. It
still scans all used rows, computes generic generator parity/phase and hashes
anticommuting partners. Local Z/XX specialization or an inverted index to visit
only affected rows are plausible next targets; this profile does not separate
those internal costs sufficiently to predict their benefit. No additional
optimization or compaction-policy tuning is included in these measurements.

The profiled trajectory matches all counts, with expectation and norm differences
versus the persistent repeat no larger than 3.33e-16 and 1.11e-16.

## Storage and semantic invariants

`spd/triton_backend/persistent.py` provides `evolve_step_persistent`. It is a
separate experimental entry point; no default dispatch or shared backward or
diagnostic kernel is changed.

- Copy the input once into privately owned storage. Stable row IDs and one hash
  table survive across gates **within a timestep**. Materialize compact arrays
  on return. Persistence across public timestep calls is not implemented.
- Keep dead keys in the index, but set their coefficients to zero immediately
  at the original strict cutoff after every gate. No discarded coefficient is
  retained for future numerical use. A zeroed key may receive a newly generated
  contribution later, which is legitimate reactivation, not resurrection of its
  old value.
- Select one owner per XOR pair using stable row indices, including when the
  smaller-index row is dead. That owner reads both old coefficients and updates
  both outputs. Ownership must not depend on concurrently modified coefficients.
  Nonowners do not load/store those coefficients. No coefficient atomics are used.
- Append previously absent keys in the update kernel. Insert them into the table
  in a subsequent kernel, after all current-gate partner lookups have completed.
  Existing keys, including commuting keys, are not rewritten by gate updates.
  The update kernel still scans all allocated logical slots and writes commuting
  coefficients; this is not an inverted-index traversal of affected terms only.
- Hash entries retain exact original row IDs and lookups compare every packed
  word. Dead keys do not break probe chains. Physical compaction removes them,
  renumbers live rows and rebuilds the index.
- Compact when dead rows exceed both 16 slots and 10% of used slots, or when a
  size cap binds. Reserve enough row/table capacity before a gate for worst-case
  doubling, so overflow cannot leave an externally visible partial update.
- Preserve the original floating-point expression order for each paired output,
  FP fusion disabled, strict combined-output truncation, generator order, and
  immutable public input. Top-k tie order remains unspecified as in the baseline.

At the final timestep, each timed persistent call performs **5 full index builds**
(initial plus rebuilds), **5 compactions including final materialization**, and
**6 buffer growths**. It also incrementally inserts newly appended rows. Thus this
is a maintained index with occasional rebuilds, not a literally constant table.
The committed approach builds a gate-specific index 341 times per timestep.

## Validation

**145 tests passed in 4.18 s**: 29 new persistent-storage tests plus the existing
116 forward and invariant tests. New coverage includes:

- Multi-gate coefficient comparisons, including 3-, 33- and 121-qubit packing,
  zero/nonzero cutoffs and binding caps.
- Exact hash membership, live counts, unique keys and coefficient comparisons
  after each of 24 gates, under both frequent and infrequent compaction.
- Deleted-key reactivation, adversarial collision chains, empty input and
  empty operation sequences, and adjacent representable cutoff values.
- Independent dense-matrix checks for every two-qubit Pauli generator in both
  float32 and float64, with per-gate truncation.

The new module is not used by backward, diagnostic or public runner paths;
those paths are not claimed to gain this speedup. Multi-GPU distribution is
untested. Local mutation and exact pair ownership are useful prerequisites, but
distributed partner routing and snapshot ownership require separate work.
Performance with other GPUs, dense anticommuting workloads, binding caps or
alternative compaction policies has not been characterized.

## Reproduction and artifacts

Run sequentially from the repository root:

```sh
python -m pytest -q tests/test_triton_persistent.py tests/test_triton_backend.py tests/test_triton_invariants.py
python -u benchmarks/benchmark_persistent_storage.py --output-dir /tmp/spd-persistent
python -u benchmarks/benchmark_persistent_storage.py --mode anti --output-dir /tmp/spd-anti-control
python -u benchmarks/benchmark_persistent_storage.py --profile --output-dir /tmp/spd-persistent-profile
```

Raw pickles, logs, allocator counters, source hashes, per-call storage statistics,
comparison JSON and profile traces are in
`benchmarks/results/persistent_storage_20260910/`. Earlier baselines and traces
remain in `benchmarks/results/anticommuting_index_20260910/` and
`benchmarks/results/forward_profile_20260910/`.
