# Stage 4: final persistent-storage acceptance

Status: complete, validated, and approved by the user.
Validated branch: `integrate/persistent-storage`, initially clean at stage 3
commit `b7c63f8`. No merge or push was performed during stage 4.

The generic persistent integration covers state-only forward sequences, public
forward diagnostics/history, coefficient adjoints, angle gradients, and ordinary
and noise-analysis backward execution. SPO/SPGO own the storage. Functional APIs
preserve inputs; explicit Triton in-place forward/backward calls retain storage
across calls. NumPy/JAX reject requested in-place execution. The consuming
variational wrapper remains the separate
[ownership-transfer TODO](triton_variational_pipeline_todo.md).

## Correctness acceptance

- Final full repository suite, including the allocator-lifetime fix:
  **1,203 passed, 1 expected failure in 251.32 s**.
- The initial full suite also passed before the fix (1,203 passed, 1 expected
  failure in 258.29 s). Focused validation after the fix passed all **120**
  persistent forward/backward/diagnostic/ownership tests in **7.08 s**.
- All final small TFI/AFH benchmarks pass key/coefficient, adjoint, angle-gradient
  and diagnostic comparisons against the retained original per-gate kernels.
- Large same-terminal backward comparisons match keys, coefficients and adjoints
  exactly. Maximum angle errors: TFI 1.39e-16, AFH 8.33e-17. Per-gate discarded
  counts match exactly; maximum L1/L2 diagnostic errors are TFI
  2.78e-17 / 1.08e-19 and AFH 1.42e-14 / 1.39e-17.
- The final 23-step TFI state-only and diagnostic runs match stage 1's term counts
  at every step and energies/norms to roundoff, ending at 221,899,620 terms.
  This large forward check compares counts/scalars, not all 222 million rows.
- `git diff --check` passed.

The expected failure is the existing strict JAX stack cutoff-equality diagnostic
case in `tests/test_backend_semantic_contract.py`. It is not a Triton failure.
The full suite covers mixed rotations and all supported exact Cliffords, wide
keys, binding caps, cutoff equality, gradient-only growth, shared-buffer safety,
serialization, independent NumPy/JAX results, dense finite differences, terminal
basis/OSE/L2 losses, analysis, public runners and example optimization paths.

```sh
python -m pytest tests -q
python -m pytest tests/test_triton_persistent.py tests/test_triton_persistent_diagnostics.py tests/test_triton_persistent_backward.py tests/test_triton_storage_ownership.py -q
```

## Reserved-memory investigation and fix

The trace reproduced the previously reported TFI discrepancy on the same H100:

| Implementation / controlled change | Peak allocated GB | Peak reserved GB |
|---|---:|---:|
| Committed stage 1 (`3abb844`) | 50.915 | 70.429 |
| Committed stage 3 (`b7c63f8`) | 50.915 | 79.530 |
| Stage 3 with original table-replacement lifetime restored | 50.915 | 70.429 |

The cause is the lifetime of the old index during non-final compaction. Stage 3
explicitly cleared `self.table` before `_rebuild()` allocated its replacement.
Stage 1 kept the old table alive until the replacement allocation completed.
A disposable checkout changing **only that lifetime** reproduced stage 1's lower
reservation. Allocation traces place the added segment mappings in subsequent
row-buffer growth/compaction allocations: changing when the old index becomes
reusable changes allocator placement across the growing workload.

The final fix lets `_rebuild()` replace the old table directly. Final
materialization still releases its index. It changes neither mathematical
storage contents nor cutoff/cap behavior and adds no allocator flushing to the
execution loop.

Before the fix, the completed run had 8.877 GB of live tensors and 70.654 GB of
inactive cached blocks within its 79.530 GB reservation. Releasing unused cache
**after measurement** lowered reservation to 8.915 GB, with those live tensors
unchanged. The largest inactive block was 49.312 GB. Both original and fixed
traces had zero allocation retries and OOMs. The allocator reported zero inactive
split bytes; that counter alone did not describe the reservation difference.

This demonstrates a cache-placement effect, not 79.5 GB of simultaneously live
tensors. It does not establish identical headroom for arbitrary workloads or
allocations outside PyTorch. Nor does this allocation order minimize reservation
universally: capped AFH diagnostics reserved 9.110 GB in the final run, versus
7.957 GB in the earlier stage 2 sample, with unchanged allocated peak. No
workload-specific allocator tuning or optional storage experiment was added.

Reproduce in separate processes with
`benchmarks/trace_persistent_allocator.py --repo CHECKOUT --output /tmp/TRACE`.
The default traces TFI 11x11 for 23 steps. It records allocation events, step
metrics and cache release afterward. Trace runtimes are **not** used in the
performance tables below. Use `--no-history` for timing and add `--diagnostics`
for the public diagnostic path. Tiny disposable inputs compile the update and
compaction specializations first; there is no full-workload warm-up.

## Final forward and diagnostic performance

H100 80GB, Torch 2.8.0+cu128, Triton 3.4.0, float64. One untraced measured pass per path,
separate processes, no concurrent GPU benchmarks. Timings include copying,
execution and final compaction; observations are outside the timed intervals.
These single samples are sanity checks, not statistical timing bounds.

| Workload | State-only | Diagnostics | Overhead | Main diagnostics (historical) |
|---|---:|---:|---:|---:|
| TFI 11×11, 23 steps | 10.246 s | 10.831 s | +5.7% | 74.713 s |
| AFH 6×6×6, uncapped | 0.887 s | 1.108 s | +24.9% | 2.505 s |
| AFH 6×6×6, exact 20M cap | 14.822 s | 16.928 s | +14.2% | 29.564 s |

TFI uses h=3.044382, dt=.04, cutoff 2^-18 and a nonbinding cap. AFH uses the
seed-0 two-layer local-Hamiltonian workload, with cutoff 3e-4 uncapped or 1e-5
capped. AFH final counts remain 2,265,180 / 20,000,000. Its exact-cap diagnostic
measurement uses the shared operation loop, bypassing public cap rounding;
TFI diagnostics uses public `spd.evolve` with a nonbinding rounded cap.

**Main diagnostics** reuses the actual-main measurements recorded in stage 2,
commit `c9a2b004d2598f1741c04619dfa30865e86a9aad`; main was not rerun here. Current
measurements are from the final implementation with the allocation-lifetime fix.
The smaller overhead percentages do not prove that diagnostics became cheaper:
state-only timing also varies. In particular, the uncapped AFH sample is slower
than the earlier 0.689-second repeat. The sizable cases retain the demonstrated
advantage over historical main diagnostics, but no no-regression timing guarantee
is claimed.

| Workload | State allocated GB | Diagnostic allocated GB | State reserved GB | Diagnostic reserved GB |
|---|---:|---:|---:|---:|
| TFI 11×11 | 50.915 | 50.915 | 70.429 | 63.529 |
| AFH uncapped | 1.095 | 1.095 | 1.623 | 1.623 |
| AFH exact 20M cap | 7.341 | 7.552 | 8.544 | 9.110 |

All memory figures are peaks in decimal GB. TFI's extra preprepared gate metadata
in the tracing/timing harness adds approximately 0.00035 GB to allocated peaks
compared with the earlier harness. That is separate from the large reservation
change. The AFH allocated peaks match the earlier results.

The stage 2 main-comparison limitation remains: exact top-k boundary ties can
select different keys, leading to different later histories for capped AFH.
This is the existing unspecified tie policy, not exact trajectory equality.
Final acceptance does not erase that limitation or weaken strict comparisons
for unambiguous caps. See the stage 2 report for the original tie investigation.

## Final backward performance

Both implementations receive the same terminal SPGO, initialized from basis
expectation plus 0.13 * OSE(alpha=2). Forward construction and initialization are
excluded from these backward intervals. The reference is the original per-gate
backward kernel retained in the current checkout, with current wrappers—not an
actual-main process. Both paths use the shared diagnostic loop and exact caps.

| Workload | Original per-gate backward | Persistent backward | Original allocated MB | Persistent allocated MB | Original reserved MB | Persistent reserved MB |
|---|---:|---:|---:|---:|---:|---:|
| TFI 11×11, **10 steps** | 0.809 s | 0.507 s | 247.970 | 279.972 | 274.727 | 318.767 |
| AFH 6×6×6, uncapped | 1.942 s | 1.128 s | 845.970 | 968.319 | 1033.896 | 1012.924 |

Decimal MB. Terminal supports are 994,285 / 2,265,180 terms; final supports are
89,945 / 384,943. Persistent backward remains faster on these cases and uses
about 13–14% more peak allocated memory. These measurements do **not** cover
backpropagation of 23-step TFI or exact-20M AFH; binding-cap and gradient-only
backward behavior are covered by the correctness suite.

The final small forward/backward benchmarks also passed. Their allocation peaks
match stage 3, and coefficients/adjoints match exactly. As before, the very small
TFI backward case has no speedup (9.841 ms reference, 10.558 ms persistent).
Full small-case metrics and error summaries are in the result artifact.

## Artifacts and completion boundary

Durable metrics: [persistent_integration_stage4.json](../benchmarks/results/persistent_integration_stage4.json).
Reproduction scripts:

- `benchmarks/trace_persistent_allocator.py`: TFI trace and untraced timing.
- `benchmarks/benchmark_persistent_large.py`: large exact-cap forward/diagnostic
  and same-terminal backward comparisons (`--case afh|tfi --phase state|diagnostics|backward --output FILE`).
- Existing `benchmark_persistent_integration.py --diagnostics` and
  `benchmark_persistent_backward.py`: small reference comparisons.

Raw logs, snapshots, measurements and disposable checkouts are in
`/tmp/spd-stage4`. Previous session temporary files had been removed; this stage
reconstructed the allocator baseline from committed stage 1 code and preserved
new summary results in the repository.

All four integration stages are now implemented, validated, and approved.
Remaining work is outside this round: the dedicated
consuming variational wrapper, any tighter workspace-memory target, and the
explicitly excluded optional arithmetic/index/layout/padding/multi-GPU experiments.
No merge into main is part of this acceptance step.
