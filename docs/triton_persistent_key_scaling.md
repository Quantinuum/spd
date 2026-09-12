# Persistent key-width investigation (2026-09-11)

This investigation stays on `experiment/triton-persistent-options`; production
storage kernels and default dispatch are unchanged.

## Matched physical benchmark

A fresh 11×11 run uses the generalization harness, two timed repeats per step,
and the same persistent baseline as the completed 21×21 run (one timed repeat).
Compare the first 19 steps, not 23 steps at 11×11 against 19 at 21×21.

| Quantity | 11×11 | 21×21 |
|---|---:|---:|
| Physical qubits | 121 | 441 |
| Packed uint32 words per key | 8 | 28 |
| Rotations per step | 341 | 1,281 |
| Final strings after step 19 | 76,647,324 | 76,647,367 |
| Steps 1–19 evolution time | 2.837 s | 20.591 s |
| Step 19 evolution time | 0.875 s | 6.577 s |

Overall time rises 7.257×, while measured input live-row/rotation work rises
3.755×. Time per unit of this logical work rises **1.932×**, against **3.5×**
key width. Over steps 12–19, whose input supports exceed one million, the
normalized increase is 1.993×; at step 19 it is 2.002×. This demonstrates a
substantial system-size-associated cost, but does not isolate hashing or prove
a linear scaling law. Gate sequence, key width and slightly different support
trajectories still change together. Work includes commuting rows and is measured
in warm passes, as documented in the generalization report.

## What the code does

`persistent._insert` hashes every packed word before inserting a row.
`persistent._update` traverses words for parity/phase, recomputes the hash of the
XOR partner over all words, and compares the full key at each occupied probe.
With bounded average probe count, build and lookup contain O(W) work per key;
lookup includes O(W × probes) comparisons. Persisting the table reduces how
often it is built. It does not make variable-length keys constant-cost.

Exact full-key equality remains necessary in this representation. A cached hash
can accelerate addressing and reject collisions, but cannot replace equality
without changing the exact semantics. The previously tested `fingerprint` mode
still recomputes the query hash; it is a collision-filter experiment, not a
cached partner-hash experiment.

## Fixed-support probe

`benchmarks/benchmark_persistent_key_width.py` embeds identical paired operators
in wider keys using zero identity padding. It holds row count, one-site X gate,
100% anti-commuting fraction, 100% partner-hit fraction and table capacity/load
fixed. All keys are unique by construction; every lookup is checked against its
known partner. Generic/local updates match exactly, preserve squared norm, and
produce no support changes. No production kernel is modified.

Each kernel has three warm calls and 15 CUDA-event samples. Table clearing and
coefficient resets are outside the measured interval. The probe separates hash
insertion, generic updates, local arithmetic, query-hash calculation and exact
lookup. The ideal cached-query lookup uses a precomputed hash and retains full
key comparison; cache creation/maintenance is deliberately excluded. It measures
an opportunity, not an end-to-end optimized algorithm. Timings at one million
and 8.39 million rows also show that the answer depends on support size.

At **8,388,608 rows**, a nearby-width sweep gives:

| Words | Insert (ms) | Generic update (ms) | Exact lookup (ms) | Ideal cached-query lookup (ms) |
|---|---:|---:|---:|---:|
| 28 | 0.871 | 1.557 | 1.193 | 1.144 |
| 30 | 0.911 | 1.766 | 1.343 | 1.314 |
| 32 | 1.197 | 5.283 | 3.903 | 2.807 |
| 34 | 1.035 | 2.206 | 1.541 | 1.374 |
| 36 | 0.992 | 2.311 | 1.782 | 1.496 |
| 64 | 2.220 | 10.585 | 7.697 | 5.466 |
| 66 | 1.616 | 4.473 | 3.064 | 2.306 |

The 32→34 and 64→66 discontinuities contradict a simple smooth per-word cost
model in this implementation. Wider padded keys are substantially faster in
these probes. Padding changes row stride, hash inputs/distribution and compiled
kernels; this test does **not** isolate which hardware/compiler effect causes
the discontinuity. The 28/32/64 trend reproduced in the preceding width sweep.

The ideal cache saves only about 4% of lookup time at W=28, versus roughly
28–29% at W=32/64 in this sweep. Recomputing that cache from scratch every gate
would pay most or all of the saving back. A useful implementation would need a
maintainable/composable hash, plus exact equality and full collision/invariant
tests. This has not been implemented. Padding is a simpler immediate hypothesis
to evaluate on a physical circuit.

## Physical AFH padding trial

The real 8×8×8 two-layer AFH circuit uses the same seeded scale-1 parameters,
cutoff 1e-5 and 20M cap. The experimental runner's `--extra-padding-qubits 32`
embeds it in 544 rather than 512 packed qubits: 34 instead of 32 uint32 words.
The additional qubits carry identity and no gates act on them. This is a storage
experiment, not a change to the physical Hamiltonian or ansatz.

A padded run followed by a fresh unpadded control, each with a full warm pass
and one timed pass on the same H100, gives:

| Simplest persistent baseline | Unpadded W=32 | Identity-padded W=34 |
|---|---:|---:|
| Evolution | 77.900 s | **47.295 s** |
| Peak allocated decimal GB | 14.908 | 15.753 |
| Final strings | 20,000,000 | 20,000,000 |

Padding reduces elapsed time **39.29%**, or increases throughput **1.647×**,
for 0.845 GB additional peak allocation (5.67%). Each stored key grows
from 128 to 136 bytes, a persistent 6.25% increase in key-buffer memory,
including compaction temporaries. The full-key hash table stores int32 row IDs;
its per-entry size and capacity do not increase in this matched run. Coefficient
arrays are unchanged in size. This is not merely construction-time overhead. No row-selection option, inverted
index or local-arithmetic option is enabled. The fresh unpadded control also
agrees closely with the previous 77.645 s baseline.

Measured warm input live/slot work, gate counts, cap-ending rotations,
compactions, rebuilds and growths all match. Final energy differs by 2.08e-17
and squared norm by 1.27e-9. No full 20M snapshots were collected, so this is
not a claim of identical large truncated states; the established cap-tie
validation limits still apply. The default AFH full-coefficient check gives identical canonical keys and
bit-identical coefficients for all 1,701 final terms after removing identity
padding, recorded in `padding_coefficient_validation.json`.

This is direct end-to-end evidence of a useful padding effect in a realistic
workload, beyond the synthetic probe. It is still a single timed padded sample
at 20M; there is no new 50M or full-backward padding measurement. Padding changes
stride, hash distribution and compiled code together. No hardware-counter or
fixed-hash/variable-stride ablation establishes the causal mechanism yet.

## Recommendation

Retain persistent storage as the integration foundation. The concern about key
width is valid, but raw 11×11/21×21 totals are not a hash-scaling measurement.
Do not rewrite the hash solely on those totals or assume that all observed
width cost is unavoidable. The padding trial provides a larger, simpler
measured opportunity than the ideal cached-query probe at W=28.

Carry the padding candidate into the persistent-storage layout work, keeping
physical qubit count separate from internal storage capacity. Validate diagnostic
and backward behavior and performance before making it a default. Do not assume
that adding 32 identity qubits is universally optimal across widths or GPUs.
A composable cached hash remains a possible later experiment, not an implemented
or measured full-circuit optimization.

## Artifacts

Compact records are in `benchmarks/results/persistent_key_scaling_20260911/`:
matched run/comparison JSONs and fixed-support kernel samples with source hashes.
The fixed-support probe is diagnostic, not a replacement for physical workloads.

## Reproduction

```sh
python benchmarks/benchmark_persistent_key_width.py --rows 8388608 --words 28 30 32 34 36 64 66
python benchmarks/benchmark_persistent_generalization.py --workload afh --size 8 --layers 2 --random-scale 1 --cutoff .00001 --cap 20000000 --variant baseline --extra-padding-qubits 32 --output /tmp/afh20m-pad32.json
python benchmarks/benchmark_persistent_generalization.py --workload afh --size 8 --layers 2 --random-scale 1 --cutoff .00001 --cap 20000000 --variant baseline --output /tmp/afh20m-pad0.json
python benchmarks/check_persistent_identity_padding.py
```
