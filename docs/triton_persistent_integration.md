# Generic persistent-storage integration

## Verified starting point

Worktree: `/teamspace/studios/this_studio/spd-persistent-main`.
Branch: `integrate/persistent-storage`; initially clean at `4a84b0f`, the
cherry-pick of `350828a`. The persistent module and tests match that source commit.
The user reports baseline testing complete; it was not repeated as a separate
baseline stage. No applicable AGENTS.md files were found.

## Review and commit stages

Pause after each stage for user review and commit instruction. No merge or push.

1. **Standalone forward sequence (approved).** Route `evolve_step`
   through basic persistent storage, support mixed exact Cliffords, preserve the
   per-gate forward implementation as an independent reference. Check ownership,
   gate order, cutoff/top-k, wide keys, empty/zero rows and input immutability.
2. **Public forward diagnostics/history.** Extend pair updates with optional
   discarded-count/L1/L2 reductions. Retain one private storage session across
   the public runner's operation loop. Preserve progress, history aggregation,
   save-strings behavior, requested/effective caps, and public return types.
   Observation access must not reconstruct storage or expose mutable snapshots.
3. **Backward and analysis.** Extend storage with adjoints and support defined by
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
