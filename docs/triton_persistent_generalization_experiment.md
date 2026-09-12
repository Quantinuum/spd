# Persistent-storage options across system sizes and models

Experiment branch: `experiment/triton-persistent-options`. These are opt-in
forward experiments; production dispatch and the committed persistent baseline
are unchanged. This extends the [11×11 option scan](triton_persistent_options_experiment.md).

## Questions and method

The original 11×11 XX+Z result cannot by itself establish a universal storage
choice. This scan separates larger packed keys from a different physical model
and from a different angle/support regime.

- **21×21 OBC XX+Z:** same central Z observable, h=3.044382, dt=0.04,
  strict cutoff 2^-18, double precision, original pytket order and angles.
  441 physical qubits pad to 448: 28 uint32 key words rather than 8.
  There are 1,281 rotations per timestep. Compare complete timesteps 1–19,
  with a nonbinding 1-billion-term cap. Stop before the next timestep after
  crossing the 65-million-row safety boundary; do not alter truncation to fit.
- **8×8×8 AFH:** use the actual Hamiltonian and circuit builders in
  `examples/gradient/run_afh_gs_3d.py`, including its `full_H=False` local
  nine-term Hamiltonian, seed 0, double precision, cutoff 0.001, and zero-state
  expectation after Néel preparation. Keys contain 32 uint32 words.
  Two layers have 10,240 rotations and 256 X preparation gates; three layers
  have 15,360 rotations and the same preparation. Measurements/barriers are
  skipped as in SPD. The benchmark evaluates energy at initialization rather
  than running the optimizer or backward pass.
- **Harder AFH:** increase the existing uniform initialization scale from 0.1
  to 1.0 with the same seeded vector. This multiplies every input parameter by
  ten; it is not a change to normal sampling. At cutoff 0.001 this still peaks
  at only 383,462 terms. An intermediate cutoff-0.0001 scan reaches five million
  terms. The main large-support sweep uses cutoff **0.00001** and caps of
  **20 and 50 million**, keeping the circuit and angles fixed across caps.

All reported timings run in isolated processes, sequentially on the same H100
80GB. One interrupted low-support trial is explicitly excluded because its last
timed pass overlapped the start of the harder pilot warm-up.
Each timestep/evaluation first runs a warm pass from its own input, then a
synchronized timed pass. The 5-million intermediate scan uses two timed passes;
the main 20/50-million scan uses one. Timing includes initial
storage creation, per-gate truncation/top-k, all index/buffer maintenance,
physical compaction, materialization and the original Clifford dispatch.
`seconds` measures evolution; `energy_seconds` additionally includes the final
expectation reduction. The expectation reduction is warmed as well in the scan;
the initial pilot's first reduction included compilation and is not used for
energy timing comparisons.

Work counters are collected only in the warm pass. `input_live_sum` counts
live terms presented to each rotation, including commuting terms; this is a
logical workload measure, not executed anti-commuting updates. They describe
the warm pass; under binding-cap ties, timed repeats can in principle select
different tied strings. Each repeat therefore records its own count, energy and
norm. Throughput based on these counters is a warm-work proxy in that case. Device-batched
mode does not expose these per-gate counters; its throughput field is null.
Peak allocated and reserved bytes are recorded separately. Initial circuit
construction is outside timing. Source and circuit hashes are recorded per run.

## Naming across the tables

Every option below uses persistent storage. The extra index in `inverted` is
an auxiliary qubit-to-row bitset index, separate from the persistent full-key
hash table used to find Pauli partners.

| Optimization family | 21×21 code name | Generalized AFH code name | What changes |
|---|---|---|---|
| Persistent baseline | `baseline` | `baseline` | Keep rows and the full-key hash table between gates |
| Experimental control | `control` | `control` | Same algorithm through the experimental scaffold |
| Local arithmetic | `local` | `local_all` | Compute parity/phase from occupied generator sites |
| Fresh active-row scan | `select` | `select_all` | Scan relevant packed-key bits each gate to select anti-commuting rows |
| Maintained inverted index | `inverted` | `inverted_all` | Select rows using auxiliary per-qubit X/Z bitsets |
| Cached fingerprints | `fingerprint` | No all-axis rename needed | Cache a key fingerprint |
| Transposed key layout | `soa` | No all-axis rename needed | Change physical key layout |

`_all` extends the optimized path from Z/XX to all single- and two-site axes;
it does not mean a different storage algorithm or arbitrary-weight specialization.
`+` combines independent optimizations, such as selection plus local arithmetic.
The 21×21 Z/XX paths cover its entire gate set. AFH additionally needs YY/ZZ,
so the generalized variants cover its entire gate set. These are comparable
families, but not identical kernel binaries. The 21×21 table also includes
fingerprint/layout experiments; the main AFH table concentrates on the promising
families and their combinations. They are deliberately not identical sweeps.

## Axis coverage

The retained original modes (`local`, `select`, `inverted`) optimize only single
Z and two-site XX; YY/ZZ execute correctly through the generic fallback. Those
fast paths cover 4,096 of the AFH circuit's 10,240 rotations (40%). Their AFH
measurements are therefore explicitly labelled **restricted prototype**.

New opt-in modes remove that restriction for all one- and two-site Pauli axes:

| Mode | Local arithmetic | Active-row selection |
|---|---|---|
| `local_all` | All X/Y/Z and all nine two-site axis pairs | Full row pass |
| `select_all` | Generic packed-key phase | Packed-key scan for every local axis |
| `inverted_all` | Generic packed-key phase | Maintained bitsets for every local axis |
| `select_all+local_all` | All local axes | Packed-key scan |
| `inverted_all+local_all` | All local axes | Maintained bitsets |

Each occupied site stores its word, bit, and generator X/Z flags. Selection
computes symplectic parity from those flags; local arithmetic computes the same
integer phase formula as the full packed-key kernel, omitting identity sites
whose contributions cancel. This includes YY, ZZ, mixed XY/XZ/YZ/etc., and all
single-site rotations. Higher-weight Pauli strings retain the generic fallback.
All 10,240 AFH rotations can use these new local paths. The existing bitplane
layout and exact full-key hash equality are unchanged.

The original X preparation gates remain in their parsed execution order and use
exact Clifford dispatch. They are not approximated by floating-point pi
rotations or omitted from the energy computation.

## Results

The completed runs support persistent storage as the foundation, but do **not**
support maintained bitsets as a universal default. Growing XX+Z support favors
maintained indexing; heavily capped AFH favors a fresh active-row scan.
All memory figures below are peak allocated decimal GB.

### Larger XX+Z keys: 21×21

These are the original Z/XX-specialized modes, which cover every rotation in
this circuit. The new all-axis modes were not separately timed on this case.
All 19 timestep counts match the same-size baseline; the final state contains
**76,647,367 strings**. This was a planned stopping boundary, not a measured OOM.

| Option | Total evolution, steps 1–19 (s) | Last step (s) | Peak GB |
|---|---:|---:|---:|
| Committed persistent baseline | 20.591 | 6.577 | 50.692 |
| Experimental control | 20.966 | 6.601 | 50.692 |
| Local arithmetic (`local`) | 15.492 | 4.780 | 50.692 |
| Maintained inverted index (`inverted`) | **6.605** | **1.790** | 60.920 |
| Fresh active-row scan (`select`) | 9.989 | 2.890 | 51.045 |
| Cached fingerprints | 20.978 | 6.598 | 52.086 |
| Transposed key layout | 31.831 | 10.342 | 50.692 |

Maintained indexing is 3.12× faster overall and 3.53× faster over steps 12–19,
whose input supports all exceed one million. Local arithmetic now has a clear
benefit with wider keys. Fingerprints and transposed layout still do not help.
Maximum energy difference across these comparisons is 3.33e-16.

### AFH: 20 million strings, all axes optimized

The following runs use two layers, scale 1, cutoff 1e-5, and the same complete
AFH circuit. Every run ends at the 20-million cap.

| Option | Evolution (s) | Peak GB |
|---|---:|---:|
| Committed persistent baseline | 77.645 | 14.908 |
| Experimental control | 78.479 | 14.920 |
| `local_all` | 62.194 | 14.925 |
| `select_all` | 59.588 | 15.023 |
| `select_all+local_all` | **59.270** | 15.023 |
| `inverted_all` | 107.743 | 18.155 |
| `inverted_all+local_all` | 107.347 | 18.155 |

The scan plus local arithmetic is 23.7% lower in time than the baseline.
Its 0.5% advantage over the plain all-axis scan is too small to establish an
additional benefit from combining the two with these single timed samples.
It was retained as the representative scan contender at 50 million.

The restricted Z/XX prototypes took 67.961 s (scan), 69.673 s (local),
67.796 s (scan plus local), and 117.918 s (maintained index). Generalizing axis
coverage improves AFH performance, but does not reverse the scan/index ranking.

### AFH scaling to 50 million strings

| Option | 20M time (s) | 50M time (s) | 50M peak GB | 50M logical billion rows/rotation/s |
|---|---:|---:|---:|---:|
| Persistent baseline | 77.645 | 195.515 | 35.887 | 1.718 |
| All-axis scan + local | **59.270** | **147.441** | **36.121** | **2.278** |
| All-axis maintained index + local | 107.347 | 258.995 | 43.088 | 1.297 |

The scan saves **48.07 s (24.6%)** at 50 million, or **1.326× throughput**,
for approximately 0.234 GB additional peak allocation. The maintained index is
32.5% slower than baseline and uses 7.20 GB more. None of these runs had an
allocator retry. Reserved memory at 50M was 42.308 / 42.534 / 50.042 GB,
respectively. All three finish with exactly 50 million strings.

At 20M the warm trajectory presents 136.110 billion live rows to rotations,
with 6,670 rotations ending at the cap and 2,231 physical compactions.
At 50M these become 335.934 billion rows, 6,657 cap-ending rotations and
2,269 compactions. The frequently rebuilt maintained index plausibly explains
its reversal relative to the growing XX+Z case. At the end of the original sweep this was an inference from
work/maintenance counters and end-to-end timings. The September 11 follow-up
profile below measures the bitset construction cost directly. The earlier dead-row compaction sweep does not eliminate
physical compaction forced by top-k caps.

The user requested stopping at 50M. The started 100M baseline was cancelled,
its timing was marked invalid, and GPU allocations were released. There are
no completed 100M results or measured AFH maximum-capacity claims. The original queued
additional repeats, profiles and buffer trials were not run; the September 11
follow-up separately profiles two contenders at 20M.

### Smaller controls

Default AFH has only 1,701 final / 2,034 peak live terms with two layers, and
8,736 final / 9,223 peak live terms with three. Simply increasing the cap does
not increase work. The default two-layer energy agrees with the existing JAX
initial-energy result to 8.88e-16, using the identical saved parameter vector.
These small supports are not the optimization target.

At five million with scale 1 and cutoff 1e-4, the restricted prototypes took
approximately 19.57 s baseline, 17.33 s fresh scan, 17.78 s local, 26.60 s
maintained index, 19.79 s fingerprints, 19.94 s transposed layout and 24.31 s
device batching (two timed passes each). This independently shows the AFH
ranking reversal, but does not substitute for all-axis large-support results.

## September 11: direct AFH profile evidence

Two sequential H100 runs repeat the 20M, scale-1, cutoff-1e-5 AFH circuit with
`inverted_all+local_all` and `select_all+local_all`. Each has a warm evaluation,
one uninstrumented evaluation and one separate CPU/CUDA profiler replay. The
kernel/storage source hashes match the previously validated 458-test snapshot.
These are new bounded measurements; no cap above 20M was used in this follow-up.

| Measured time (s) | Maintained bitsets + local | Fresh scan + local |
|---|---:|---:|
| Uninstrumented evolution | 106.886 | 58.947 |
| Instrumented evolution | 107.439 | 59.713 |
| All GPU kernels, summed | 99.070 | 51.182 |
| Auxiliary bitset construction (`make_planes`) | **50.774** | **0** |
| Active-row selection (`select_rows`) | **0.677** | **3.583** |
| Hash insertion (`insert_variant`) | 6.076 | 6.067 |
| Pauli updates (`update_variant`) | 1.756 | 1.757 |
| Non-top-k compaction kernel | 0.091 | 0.090 |
| Other kernels, including top-k and gathers/copies | 39.696 | 39.685 |

All recorded kernels use one GPU stream. Summed kernel durations omit host work,
synchronization gaps and memcpy events, so they should not be equated with wall
time. Both replays contain 2,224 `aten::topk` calls. The bitset replay contains
2,344 `make_planes` launches; **2,224** have grid `[625000, 32, 1]`, covering all
20M rows and all 32 key words. These full-cap builds take **50.672 s**, over
99.8% of bitset construction time. The code calls `_rebuild()` after physical
compaction; the inverted variant additionally calls `_planes()` to reconstruct
row-position-dependent bitsets. The matching top-k/full-cap-build counts and
measured kernel times directly support cap-driven reconstruction as the dominant
cost in this run.

Bitsets save **2.906 s** in selection but cost **50.774 s** to construct: a net
**47.868 s** penalty. The measured total GPU-kernel gap is **47.888 s**, and the
instrumented wall-time gap is **47.726 s**. Thus bitset maintenance minus its
selection savings accounts for essentially the entire measured slowdown
relative to the fresh scan in this 20M AFH case. Hash insertion, Pauli updates
and the remaining kernel groups take almost identical times.

This establishes the bottleneck in the current implementation and this capped
AFH regime. It does not prove that AFH generally disfavors inverted indexing,
or that every bitset implementation must have this cost. No same-model
cap-frequency ablation or new 21×21 phase profile was performed. The 50M
end-to-end ranking remains from the earlier sweep, not a new phase profile.

Both replays finish at 20M rows, with energy difference 1.39e-17 and squared-norm
difference approximately 3.09e-12 between them. No new full coefficient
snapshots were collected. The previously documented cap-tie validation limits
still apply; these profiles do not establish deterministic state equality.

Artifacts are in `benchmarks/results/persistent_generalization_20260911_profile/`:
original run JSONs, Chrome traces, per-kernel summaries and `comparison.json`.
`benchmarks/summarize_persistent_profile.py` reproduces the kernel aggregation.

## Design implication

Prioritize integrating the generic persistent engine across ordinary forward,
diagnostic forward and backward. Keep both row-selection options available on
the experiment branch; they need not be part of the first main integration.
The evidence favors fresh all-axis scanning for capped AFH and maintained
indexing for growing XX+Z. A universal maintained-index default would regress
a realistic workload. A future policy can use observed selection and maintenance
costs, but no automatic switching policy was implemented or validated here.
Local arithmetic is useful with wide keys; whether to combine it with an
already efficient scan needs repeat measurements before treating a sub-percent
advantage as a design decision. Full diagnostic forward and backward integration
should follow this distinction, with backward performance assessed separately.

## Validation and limits

Every timestep is compared with the same-size persistent baseline, rather than
assuming the 21×21 trajectory equals 11×11. AFH saves full packed keys and
coefficients for order-independent comparisons through the 20-million cap.
The 50-million cases validate counts, norms and energies; they do not claim
full coefficient equality. Term-cap ties may select
different equal-magnitude strings when row ordering changes; report any such
differences rather than describing scalar agreement as full coefficient equality.
Full saved coefficients match exactly for every completed default AFH control
and every five-million variant after matching packed keys. The intermediate
two-million control differs by one missing and one extra key, with maximum
common-coefficient difference 8.56e-8 and energy difference 6.07e-9. The
respective unmatched magnitudes are 3.65e-4 and 3.47e-4. Binding-cap trajectories
are therefore not universally bit-identical even for the control scaffold.
Equal-magnitude top-k ties have unspecified ordering under the existing backend
contract; this is a plausible source of trajectory differences, but no
first-divergence trace was collected to establish their cause.

At 20 million, the selected `select_all+local_all` contender matches all
baseline keys and coefficients **exactly**, as does `local_all`. Some other
variants exchange three keys out of 20 million, with maximum shared-coefficient
difference 1.73e-5 and unmatched magnitudes at most 9.84e-5. This includes the
control scaffold. Maximum energy and squared-norm differences across the 20M
runs are approximately 5e-17 and 1.27e-9, respectively. These observed support
differences remain a validation limit rather than proof of a particular cause.

At 50 million the maximum energy difference is 2.78e-17; the maximum
squared-norm difference is approximately 6.20e-10. This scalar agreement is
useful but is not a substitute for the absent full 50M snapshots.

Raw results, circuit/source hashes and full comparison records are under
`benchmarks/results/persistent_generalization_20260910/`, with the main AFH
sweep in `cutoff1e5/`. `comparison.json` contains order-independent coefficient
checks; `stop_at_50m.json` records the cancellation. The validated source hashes
were rechecked against all completed all-axis large-support runs.

The expanded suite passed **458 tests in 224.84 s**, including all twelve
single-/two-site axis patterns and explicit active-set membership. Its validated
source hashes are recorded in `all_axes_validation.json`. The checks cover mixed XX/YY/ZZ/Z
rotations, word boundaries, binding caps,
immutable inputs, and backward-representative primal/adjoint/angle-gradient
updates including gradient-only missing-partner growth. Full backward timing
and integration remain outside this experiment.

## Reproduction

```sh
# Main large-support driver expects completed 20M and 50M baseline pilots:
python benchmarks/benchmark_persistent_generalization.py --workload afh --size 8 --layers 2 --random-scale 1 --cutoff .00001 --cap 20000000 --variant baseline --save-state --output benchmarks/results/persistent_generalization_20260910/cutoff1e5/afh2_baseline_20000000_scan.json
python benchmarks/benchmark_persistent_generalization.py --workload afh --size 8 --layers 2 --random-scale 1 --cutoff .00001 --cap 50000000 --variant baseline --output benchmarks/results/persistent_generalization_20260910/cutoff1e5/afh2_baseline_50000000_scan.json
python benchmarks/run_persistent_large_support.py
python benchmarks/summarize_persistent_generalization.py benchmarks/results/persistent_generalization_20260910
python benchmarks/summarize_persistent_generalization.py benchmarks/results/persistent_generalization_20260910/cutoff1e5
python -m pytest -q tests/test_triton_persistent_all_axes.py tests/test_triton_persistent_wide_options.py tests/test_triton_persistent_options.py tests/test_triton_persistent.py tests/test_triton_backend.py tests/test_triton_invariants.py
```

## Follow-up benchmark priorities

1. Full forward diagnostics and backward on real energy/gradient workloads:
   validate histories, truncation diagnostics, coefficient adjoints, angle
   gradients, memory and elapsed time against existing paths. Forward timings
   alone do not establish backward speedups.
2. Within one model, vary cap/cutoff to obtain growing, weakly capped and heavily
   capped support. This separates maintenance frequency from the XX+Z/AFH model
   difference. Keep all additional runs at or below the requested 50M limit.
3. At fixed key width and support above one million, control anti-commuting
   fraction, partner hit rate and new/dead row turnover. This tests when selection
   savings can pay for its overhead; use synthetic cases as diagnostic tools,
   not substitutes for physical circuits.
4. Add higher-weight Pauli generators and interleaved Clifford gates, plus
   several initialization seeds and repeated timing in varied order. These
   exercise generic fallbacks, key changes and robustness beyond one trajectory.

These are proposed future tests, not completed measurements.

## Artifact retention

Full state snapshots and the September 11 full traces were removed after retaining
the compact results. See [artifact retention](triton_persistent_artifact_retention.md)
for the manifest and regeneration requirements. References to full-state checks
and traces above describe measurements completed before cleanup.

## Key-width follow-up

The [key-width investigation](triton_persistent_key_scaling.md) separates the
11×11/21×21 workload effects and reports a new, simpler identity-padding
candidate on 20M AFH. It does not change the historical option timings above.
