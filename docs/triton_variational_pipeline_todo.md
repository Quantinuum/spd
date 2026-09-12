# TODO: variational pipeline with ownership transfer

Status: deferred follow-up, outside the current persistent-storage integration.
The feasibility probe succeeded. The intended implementation is a small
higher-level energy-and-gradient wrapper around existing forward, initialization
and backward operations, supported by an explicit storage-ownership transfer.

## Proposed work

- Define a consuming SPO-to-SPGO transfer: reuse primal buffers/index and make
  the source SPO unavailable for further use; specify treatment of exposed or
  shared tensor views. Keep existing non-consuming APIs unchanged.
- Wrap forward in-place evolution, energy extraction, adjoint initialization,
  ownership transfer, and backward in-place evolution in one higher-level call.
- Retain compatible capacity at the transition where beneficial; allocate the
  adjoint channel without copying primal keys or coefficients.
- Validate energy/gradients and source lifetime, and measure complete-pipeline
  allocated/reserved peaks against the existing functional and in-place paths.

This should reuse the current kernels. Further workspace optimization is a
separate question if a peak close to the compact SPGO size is required.

## Feasibility results

No repository implementation or public API was changed for this probe.
Raw scripts/results remain in `/tmp/spd-stage3-transfer-probe`. From that
directory, run `python driver.py`, then `python compare.py`. This document
preserves the findings even if the temporary artifacts are later removed.

The controlled script evolves an exclusively owned SPO, records the energy,
initializes the adjoints using the existing public initializer, transfers its
existing storage/index to the SPGO, and deletes the SPO. It checks collection
with a weak reference, verifies unchanged primal buffer pointers at transfer,
and restores exclusive ownership only after eliminating the source object.
It then runs backward in place. This uses private internals and assumes no
external tensor aliases; it is an experiment, not a proposed public API.

The retained-capacity variant allocates the adjoint channel to match existing
capacity; only adjoints are copied into that channel. The compact variant first
compacts the forward SPO and can directly adopt the initialized adjoint array.
Neither copies primal keys/coefficients at handoff. Both show zero _Storage
working-copy constructions in backward, while ordinary capacity growth and
compaction inside backward can still allocate/copy buffers.

Workloads: TFI 11x11, 10 steps, cutoff 2^-18; AFH 6x6x6, cutoff 3e-4, no binding
cap. Basis expectation + 0.13 * OSE(alpha=2). Tiny compilation only and one
measured pipeline per variant. Phase metrics include forward, initialization,
and backward; host comparison is excluded. No GPU benchmarks overlap.

Compare against `/tmp/spd-stage3-inplace-memory` (same inputs and loss).

| Case | Previous in-place full peak MB | Transfer full peak MB | Previous backward peak MB | Transfer backward peak MB |
|---|---:|---:|---:|---:|
| TFI 11x11, 10 steps | 327.790 | 218.428 | 327.790 | 218.428 |
| AFH 6x6x6 uncapped | 1207.420 | 1094.670 | 1207.420 | 798.293 |

All numbers are peak allocated decimal MB. Transfer reserved peaks are 297.796
MB / 1497.367 MB. Functional-default full peaks from the earlier probe were
279.622 / 1094.671 MB, respectively; AFH forward still dominates the full peak.

The transfer + compaction variant has full allocated peaks 231.895 / 1094.670 MB,
and backward peaks 231.895 / 805.223 MB. In this experiment retaining capacity
at transfer gives a smaller backward peak than compacting and then growing again.

Compact terminal logical SPGO sizes are only 47.726 / 163.093 MB. Physical SPGO
buffers immediately after retained-capacity transfer, including the index, are
127.877 / 482.462 MB. These distinctions matter: removing an independently
preserved SPO does NOT yet establish a full-pipeline peak near compact SPGO size.
Growth, compaction, indexing, loss arithmetic workspace, and the forward peak
remain to be managed in a dedicated pipeline.

Both variants match original functional keys and coefficients exactly; adjoints,
angles, energies and diagnostic norms match to roundoff; all per-gate discarded
counts match exactly. `summary.json` has errors, phase peaks, ownership assertions
and copy counts. JSON/NPZ files retain the raw outputs and arrays.
