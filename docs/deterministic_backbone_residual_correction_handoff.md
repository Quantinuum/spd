# Deterministic-Backbone Residual-Correction R-SPD Handoff

Status: Phase 0.5, the deterministic Phase 1 feasibility study, the minimum
NumPy Phase 2 stochastic prototype, and the decisive Phase 3 benchmarks are
complete.  The evidence supports starting the time-blocked Phase 4 estimator.
This handoff uses the terminology in `R-SPD/main.tex` as updated on
18 August 2026.

## Objective

Investigate an unbiased estimator that retains a stable deterministic top-
`K_d` SPD calculation as a control-variate backbone and randomizes only its
discarded error.  The motivation is the sharp sampled-SPO norm inflation seen
in direct R-SPD when the coefficient spectrum becomes flat.  The new method
must remain NumPy-only and forward-only for its first prototype.  Do not modify
`spd/numpy_backend/kernels.py` without asking the user first.

This is not a request to enlarge the Pauli-string budget of direct R-SPD, add a
small deterministic cutoff to unbiased mode, or replace pivotal truncation by
a locally inferior sampling design.

## Canonical terminology

Use these terms consistently in new code and documentation:

- **R-SPD:** Randomized Sparse Pauli Dynamics.
- **Pauli-string budget `K`:** maximum persistent sampled-SPO support after a
  randomized truncation.  Do not call it a population size.
- **Run** or **realization:** one complete stochastic execution at fixed `K`.
- **Estimator sample:** the final scalar returned by one run.
- **Sampled SPO:** the sparse random operator evolved within a run.
- **Randomized truncation:** the water-filled pivotal `K`-truncation.
- **Trajectory/Pauli path:** reserve for a method following one Pauli string at
  a time.  A finite-`K` R-SPD run is not a trajectory.

The initial prototype used FP-SPP/population terminology.  Compatibility
aliases and historical `trajectory_*.pkl` filenames may remain when required
to read or resume existing data, but new interfaces and human-facing messages
must use the terms above.

## Why direct R-SPD is being reconsidered

Direct R-SPD propagates and randomly truncates the entire sampled SPO.  Once
inverse-probability reweighting inflates its norm, the next randomized
truncation acts on that already random, inflated SPO.  The deterministic heavy
core then collapses and terminal estimates become rare-event dominated.

Representative diagnostics from the 11x11 XX+Z benchmark were:

| budget/run set | step | mean sampled-SPO norm squared | median heavy count |
|---|---:|---:|---:|
| `K=1,000` | 6 | 1.4 | 176 |
| `K=1,000` | 7 | 17 | 65 |
| `K=1,000` | 8 | 1,695 | 7 |
| `K=10,000` | 7 | 1.5 | 914 |
| `K=10,000` | 8 | 58 | 176 |
| `K=10,000` | 9 | 17,576 | 15 |

Increasing `K` tenfold delayed the transition by only about one Trotter step.
After exact support reaches the cap, a branching gate produces candidate
support proportional to `K`; in a flat-spectrum regime the local relative
noise depends mainly on the candidate-support ratio and spectrum, not absolute
`K`.  Exact support grows exponentially, so increasing `K` shifts the
crossover only logarithmically.

The numerical-zero hypothesis was also checked.  At the `K=10,000` variance
transition, the smallest candidate coefficients were already physical-scale:

| step | smallest observed candidate | median per-record minimum |
|---:|---:|---:|
| 7 | `5.5e-6` | `1.8e-4` |
| 8 | `5.5e-4` | `1.4e-3` |
| 9 | `5.2e-3` | `1.8e-2` |

A `1e-10` cutoff therefore cannot explain or cure the transition.  Exact zeros
are already removed.  Unbiased mode must continue to use zero deterministic
magnitude threshold.

## Exact deterministic-backbone identity

Let `T_l` be exact linear propagation through gate `l`.  Starting from the
exact observable `d_0 = c_0`, define a deterministic top-`K_d` backbone:

```text
u_(l+1) = T_l d_l
d_(l+1) = TopK_Kd(u_(l+1))
r_(l+1) = u_(l+1) - d_(l+1)
```

Here `r_(l+1)` is the complete signed or complex residual discarded by the
backbone at that gate.  The definition gives

```text
d_(l+1) = T_l d_l - r_(l+1).
```

Unrolling this recursion yields the exact identity

```text
c_L = d_L + sum_j T_(L-1) ... T_(j+1) r_(j+1).
```

This identity is exact even though top-`K` is nonlinear: the residual is
defined after applying top-`K` to the deterministic backbone candidate.

## Proposed first correction recurrence

Maintain a separate sampled correction SPO `z_hat_l`, initialized to zero.  At
each gate:

```text
u             = propagate_and_merge(d, gate)
d_next        = deterministic_top_k(u, K_d)
r             = u - d_next

z_candidate   = propagate_and_merge(z_hat, gate) + r
merge equal keys in z_candidate
z_hat_next     = pivotal_truncate(z_candidate, K_c, rng)

d             = d_next
z_hat         = z_hat_next
```

Randomized truncation acts as the identity when correction support does not
exceed `K_c`.  Only exact zeros may be removed in unbiased mode.

Inductively,

```text
E[z_hat_l] = c_l - d_l,
```

because

```text
E[z_hat_(l+1)]
  = T_l E[z_hat_l] + r_(l+1)
  = T_l(c_l - d_l) + (T_l d_l - d_(l+1))
  = c_(l+1) - d_(l+1).
```

The final scalar estimator is

```text
mu_hat = overlap(d_L) + overlap(z_hat_L),
```

and is unbiased for the exact zero-threshold expectation.

## Difference from the existing deterministic heavy core

The pivotal heavy core is local to the current random candidate.  The
randomized output replaces the whole propagated SPO, so stochastic norm
inflation changes later candidates and can erase the heavy core.

In the residual-correction design, the top-`K_d` backbone is never random and
never receives the sampled correction.  Randomness estimates only the additive
error of that fixed deterministic approximation.  This separation is the
control variate.  Merely reserving more deterministic entries inside the
existing pivotal truncation is not equivalent and would generally worsen tail
variance at fixed `K`.

## Phase 0: specification before implementation

1. Fix circuit orientation and confirm the identity using the repository's
   Heisenberg reverse-gate convention.
2. Define `K_d` (deterministic backbone budget), `K_c` (sampled-correction
   budget), one correction run, and an independent-run ensemble.
3. Specify whether deterministic top-`K_d` breaks equal-magnitude ties by
   canonical Pauli key to make the residual stream reproducible.
4. Specify diagnostics and checkpoint schema before running the hard benchmark.
5. Keep direct R-SPD and residual-corrected R-SPD as separate APIs; do not add a
   mode flag that makes unbiasedness ambiguous.

## Phase 1: deterministic residual feasibility study

Before randomizing the correction, instrument a deterministic top-`K_d`
backbone and record at every actual top-`K` truncation:

- candidate, retained, and residual support;
- residual `l1` norm and squared `l2` norm;
- discarded fraction of candidate squared norm;
- cancellation when residual injections are combined after propagation;
- residual statistics grouped by gate and Trotter step;
- deterministic expectation and its difference from the best available
  reference.

For small circuits, propagate each residual exactly through the remaining
gates and compute its final scalar contribution.  Measure

```text
sum_j |Delta mu_j| / |sum_j Delta mu_j|.
```

A very large ratio signals a residual sign/cancellation problem: the small
deterministic bias would then be the difference of large contributions and may
still be expensive to estimate stochastically.

Go/no-go question: is the deterministic correction operator substantially
smaller and more compressible than the full propagated observable?

### Phase 0.5 and Phase 1 findings

The research scripts and checkpointed records are in
`results/test_residual_correction/`.  They use the NumPy backend, double
precision, and numerical-zero tolerance `2^-30`.  Numerical-zero removals are
reported once per timestep with their count, `l1` norm, squared `l2` norm, and
largest removed magnitude.

The 11x11 deterministic `K_d=1,000` backbone was first instrumented through
step 10.  It reproduced the existing deterministic expectations exactly.  At
step 10, 61 of 341 gates discarded terms, the largest candidate support was
1,316, and the accumulated local discarded `l2` norm for that timestep was
0.0799.  The sum of all discarded squared norms through step 10 agreed with
`1 - ||d||_2^2` to `9e-16`.

An exact 2x2 XX+Z audit with `K_d=8` completed eight steps.  Backbone plus
correction agreed with an independently propagated full operator to
`3.1e-16`.  Separately propagating every signed residual to the final scalar
gave

```text
sum_j |Delta mu_j| / |sum_j Delta mu_j| = 1.44.
```

This does not show a severe residual-sign cancellation problem on the small
audit.

A 3x3 XX+Z study with `K_d=100` completed eight steps.  The maximum operator
`l2` difference between backbone plus correction and the independent full
reference was `1.5e-9`, consistent with the declared numerical-zero policy.
At step 8 the exact correction had support 64,525, squared norm 0.0517, and
effective support 1,173.  Its predicted pivotal relative MSE was:

| correction budget | predicted MSE / correction norm squared |
|---:|---:|
| 100 | 10.7 |
| 1,000 | 0.462 |
| 10,000 | 0.00157 |

For the 11x11 `K_d=1,000` case, the exact correction after step 2 already had
support 16,904 but squared norm only `2.81e-9`.  Its predicted pivotal relative
MSE was 15.1, 0.655, and `1.01e-4` at correction budgets 100, 1,000, and
10,000 respectively.  During step 3, the guarded study stopped before an
XX gate whose estimated transient correction support was 50,316; the current
correction support was 25,158.  This is a scalability result, not an
algorithmic failure: exact correction propagation is already too expensive
for the hard benchmark.

The Phase 1 answer is mixed.  The correction is much smaller in norm at
shallow depth, and its absolute predicted randomized-truncation error is lower
than truncating the full observable.  It is not smaller in support and is less
compressible relative to its own norm.  The evidence supports a minimal
stochastic prototype with early norm-inflation stopping, starting at
`K_c=1,000` and `K_c=10,000`; it does not support a large production ensemble
yet.

## Phase 2: minimum NumPy prototype

Implement only:

- NumPy backend;
- forward Heisenberg propagation;
- expectation estimation;
- one deterministic backbone;
- one independently seeded sampled correction;
- an independent-run ensemble returning every correction estimate, mean,
  sample variance, standard error, and seeds.

Reuse existing exact NumPy propagation and pivotal truncation.  Do not modify
`kernels.py` without permission.  The correction candidate can contain up to
the propagated correction support plus the deterministic residual support, so
the old direct-R-SPD `2K` transient-capacity argument does not automatically
apply.  Resolve capacity and merging explicitly rather than silently dropping
entries.

Possible execution layouts:

1. **Correctness-first independent runs:** recompute the deterministic
   backbone in each run.  Simple and independently checkpointable, but wastes
   deterministic work.
2. **Lockstep ensemble:** propagate one backbone and `R` independent correction
   SPOs gate by gate.  Efficient but requires ensemble-level checkpointing.
3. **Residual tape:** compute the backbone once and persist every residual for
   later independent runs.  Appendable, but the tape may be large.

Start with option 1 for correctness.  Select option 2 or 3 only after residual
sizes have been measured.

### Phase 2 implementation

The correctness-first implementation is in `spd/residual_corrected.py` and is
exported through three APIs:

- `evolve_residual_corrected(...)`;
- `run_residual_corrected_spd(...)`;
- `run_residual_corrected_ensemble(...)`.

It recomputes the deterministic backbone independently in every ensemble run,
uses canonical packed-Pauli keys to break equal-magnitude top-`K_d` ties, and
derives each child seed from `(master_seed, run_index)`.  The initial
observable is split into a top-`K_d` backbone and residual before the first
gate, so both persistent support budgets are enforced from initialization.

Exact NumPy propagation uses no transient top-`K` cap.  The backbone is split
only after propagation and merging; the correction is propagated, merged with
the complete signed residual, and then passed once through pivotal
`K_c`-truncation.  Maximum observed backbone and correction candidate supports
are retained in diagnostics.  `kernels.py` was not changed.

The default `numerical_zero_tolerance=1e-12` uses
`np.isclose(value, 0, rtol=0, atol=tolerance)` and emits one aggregate warning
per run or ensemble if non-exact values are removed.  Removed count, `l1`
norm, squared `l2` norm, and maximum magnitude are stored in diagnostics.  With
a positive tolerance, unbiasedness is stated only up to that declared
numerical-zero policy.  Setting the tolerance to zero preserves all nonzero
coefficients and the formal unbiasedness guarantee.

Focused tests cover telescoping, exact agreement when `K_c` covers the
correction, empirical unbiasedness, multiple residual injections, merging and
cancellation, deterministic ties, stable seeds, initial support above both
budgets, empty correction behavior, repeated `R_Z(pi/4)`, tiny nonzero
coefficients, and numerical-zero warnings.  The full repository test suite
passed after the Phase 2 implementation.  Phase 3 adds a separate tested
results collector without changing the estimator.

## Required tests

1. Exact backbone-plus-residual telescoping identity without correction
   truncation.
2. Exact agreement when `K_c` covers every correction candidate.
3. Empirical end-to-end unbiasedness on dense-solvable small circuits.
4. Residual injection at multiple gates.
5. Merging and cancellation between propagated correction and new residual.
6. Deterministic top-`K_d` tie reproducibility.
7. Stable independent run seeds and worker-count reproducibility.
8. Initial observable support above either budget.
9. Empty residual/correction behavior.
10. Repeated `R_Z(pi/4)` stress case.
11. No deterministic cutoff, clipping, normalization, or second top-`K` on the
    sampled correction.
12. Diagnostics agree with independently recomputed support and norms.

## Phase 3: decisive benchmarks

Compare at both fixed peak memory and fixed total work:

- deterministic top-`K_d` SPD;
- direct pivotal R-SPD;
- deterministic backbone plus correction R-SPD;
- the best available high-support deterministic reference.

Initial matrix:

```text
K_d = 1,000
K_c = 100, 1,000, 10,000
R   = enough independent correction runs for a stable variance estimate
```

Report by Trotter step:

- deterministic backbone bias;
- correction ensemble mean and standard error;
- corrected estimate and RMSE;
- sampled-correction norm squared;
- correction heavy-core count and fraction;
- accumulated predicted randomized-truncation MSE;
- terminal nonzero-hit fraction and estimate quantiles;
- runtime, peak memory, and variance-times-work.

Success requires more than matching the mean at shallow depth.  The correction
norm must remain controlled materially beyond the direct R-SPD transition, and
the corrected estimator must beat direct R-SPD at comparable resources.

### Initial Phase 3 evidence

The appendable 11x11 runs use `K_d=1,000`, numerical-zero tolerance `1e-12`,
and master seed `20260818`.  The high-support deterministic reference at step 8
is `0.8007972471`.

| correction budget | runs | corrected mean | sample variance | standard error | terminal hit fraction | median correction norm squared |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 16 | 0.84143 | 0.10448 | 0.08081 | 0.125 | 1,113 |
| 10,000 | 8 | 0.79926 | 0.003849 | 0.02193 | 0.500 | 60.6 |

Increasing `K_c` tenfold therefore helps substantially.  It does not yet show
a different asymptotic regime.  At step 8, the median heavy count among actual
correction truncations is 2 for `K_c=1,000` and 33 for `K_c=10,000`, only
`0.20%` and `0.33%` of their budgets.  The corresponding direct R-SPD
`K=10,000`, `R=8` sample variance is `0.002006`, smaller than the corrected
estimator's `0.003849` at similar persistent support and runtime.  The median
sampled norm squared is also similar: 58.7 for direct R-SPD and 60.6 for the
sampled correction.

The low-budget comparison is more favorable but not fixed-memory: at `R=16`,
direct `K=1,000` had variance `0.3515`, while corrected
`K_d=K_c=1,000` had variance `0.1045`.  Their mean runtimes were approximately
105 and 110 seconds per run, but the corrected method retains up to 2,000
strings across its two components.  At the larger comparison, direct
`K=10,000` and corrected `(K_d,K_c)=(1,000,10,000)` took approximately 695 and
796 seconds per run, and variance-times-runtime favored direct R-SPD by about
2.2 times.  The existing evidence therefore does not yet meet the Phase 3
success condition at comparable resources.

At `K_c=1,000`, extending the corrected run to step 12 gave zero terminal
correction overlap in all 16 runs while the median correction norm squared was
`4.55e14`.  The zero sample variance is false convergence.

These observations support two conclusions.  First, a deterministic backbone
can protect the scalar estimate after the sampled correction begins to fail.
Second, increasing only `K_c` appears to move the repeated-truncation crossover
rather than remove it.  The next controlled comparison swaps the same total
persistent support between the two components:

```text
(K_d, K_c) = (1,000, 10,000)  # existing
(K_d, K_c) = (10,000, 1,000)  # deterministic-cancellation test
```

If the second allocation is materially better, deterministic cancellation
before residual injection is important and larger backbones deserve further
study.  Test `(10,000, 10,000)` and step 9 only after this allocation pilot;
do not spend runs at step 12.  A hard-benchmark `K_c=100` ensemble is also not
useful now because `K_c=1,000` is already beyond its stable regime at step 8.

Two `(K_d,K_c)=(10,000,1,000)` pilot runs were completed.  The larger backbone
estimate was `0.8014008744`, reducing the absolute deterministic bias from
`0.01317` to `0.000604`.  Both sampled corrections nevertheless had zero
terminal overlap, with correction norm squared 617 and 1,406.  Their step-8
median heavy count was 1.  The resulting zero sample variance is not a variance
estimate; it is another rare-event warning.  At fixed total persistent support
of 11,000, allocating 10,000 strings to the correction produced a much healthier
sampled correction than allocating them to the backbone.  Larger `K_d` improves
the deterministic approximation but does not by itself cure unbiased-correction
degeneracy.

The final `(K_d,K_c)=(10,000,10,000)` step-8 pilot completed eight runs:

| runs | corrected mean | sample variance | standard error | terminal hit fraction | median correction norm squared | mean seconds/run |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.80978 | 0.002231 | 0.01670 | 0.375 | 55.6 | 1,258 |

Increasing `K_d` substantially reduces the sampled-correction norm at shallow
depth, but the benefit is erased by repeated truncation.  Comparing
`K_d=1,000` with `K_d=10,000` at fixed `K_c=10,000`, the median correction norm
squared changes from `5.90e-7` to `3.01e-9` at step 3, from `0.00980` to
`0.00490` at step 6, and only from `60.6` to `55.6` at step 8.  The step-8
heavy count is 20, or `0.20%` of `K_c`.

Direct `K=10,000`, `R=8` had variance `0.002006` and took approximately 695
seconds/run.  The corrected `(10,000,10,000)` estimator therefore has slightly
higher observed variance, about 1.8 times the runtime, and twice the persistent
support.  Small-ensemble variance uncertainty does not change the diagnostic
conclusion: the correction has already reached the same norm-inflated regime.

Phase 3 does not show a useful large-`K` scaling crossover.  A crossover must
exist when `K` approaches the exact effective support, but the accessible data
are consistent with moving the divergence boundary by roughly one Trotter step
per decade of correction budget.  Use that logarithmic-delay behavior as the
working hypothesis.  Do not run deeper persistent-correction ensembles unless
new evidence contradicts it; proceed to Phase 4.

`results/get_residual_corrected_2d_obc_xx_z_statistics.py` creates
`phase3_ensemble.csv`, `phase3_steps.csv`, and `phase3_statistics.pkl` from any
completed residual-corrected result directory.  It reports estimator tails,
terminal hit fraction, correction norm growth, heavy-core statistics, RMSE,
runtime, and variance-times-runtime.

## Phase 4 if one persistent correction still degenerates

Use the equivalent scalar decomposition

```text
mu = mu_backbone + sum_j Delta mu_j
```

and estimate residual injections from different time blocks independently.
This prevents early degenerate correction histories from being merged into
every later residual.  Use pilot second moments to allocate independent samples
across time blocks.  This is a second algorithm and should not complicate the
first prototype.

## What to retain from direct R-SPD

Direct R-SPD remains valuable as:

- the validated conditionally unbiased baseline;
- a shallow-circuit method while the sampled SPO retains a substantial heavy
  core;
- a compressibility and rare-event diagnostic;
- the stress baseline for any proposed variance reduction;
- a documented negative result showing that local optimality does not prevent
  repeated-truncation norm inflation.

Do not spend large ensembles trying to average through a clearly degenerate
regime.  The user plans to stop the active `K=10,000` run after 32 completed
runs.  Preserve those result files and use them to document the transition.

Recommended operational diagnostics for direct R-SPD are sampled-SPO norm
inflation, deterministic-heavy-core fraction, accumulated predicted MSE,
terminal nonzero-hit fraction, quantiles, and ordinary sample variance.  A
zero sample variance before any rare terminal hit is not evidence of
convergence.

## Open questions for the next session

1. Is the correction operator actually more compressible than the full SPO?
2. Do residual injections from different gates cancel before or only at the
   final scalar?
3. What is the best reproducible top-`K_d` tie rule in the existing NumPy path?
4. Can residuals be streamed or checkpointed without materializing an
   `O(number_of_gates * K_d)` tape?
5. Does one persistent correction reproduce direct R-SPD norm inflation, only
   at a smaller scale?
6. If time-stratified residual estimation is needed, what pilot statistic gives
   a stable allocation?
7. What comparison reference is sufficiently close to zero-threshold exact
   propagation for unbiasedness claims at the hard benchmark scale?

Begin with a separate Phase 4 design pass.  Derive the unbiased time-blocked
scalar estimator, independent seed hierarchy, block diagnostics, and pilot
allocation rule before implementation.  Keep the existing persistent-
correction API intact as the Phase 2/3 reference implementation.

## Phase 4 initial implementation and pilot

The minimum NumPy research prototype is in `spd/time_blocked_residual.py`.  It
is intentionally not exported as top-level production API and keeps the
existing direct and persistent-correction APIs unchanged.  A deterministic
pass stores only top-`K_d` backbones at block boundaries.  Each block run
regenerates its local residual injections, then propagates only that block's
sampled correction to the terminal scalar.  The implementation uses the
repository's reverse-Heisenberg operation stream and has exact small-circuit
tests for the ordering and blockwise telescoping identity.

The appendable pilot wrapper is
`examples/benchmark_time_blocked_residual_2d_obc_xx_z.py`.  Its seed hierarchy
is `(master_seed, stage, block_index, correction_budget, run_index)`.  It
writes one deterministic `backbone.pkl`, one atomic pickle per block run,
`pilot_blocks.csv`, and `pilot_summary.pkl`.  After consolidation, the full
repository suite passes with 583 tests.  `kernels.py` was not changed.

The first 11x11 step-8 pilot used eight one-Trotter-step blocks,
`K_d=K_c=1,000`, four runs per block, master seed `20260819`, and the default
numerical-zero tolerance `1e-12`.  Block 0 is the last physical Trotter step
because the block order is reverse Heisenberg.

| block | injected local l2 norm squared | terminal hit fraction | mean correction | maximum sampled norm squared | maximum norm-inflation ratio | minimum heavy fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 1 | `2.29e-9` | 0 | 0 | 149 | `6.52e10` | 0 |
| 2 | `3.15e-7` | 0 | 0 | 1,479 | `4.70e9` | 0 |
| 3 | `6.74e-6` | 0 | 0 | 98.2 | `1.46e7` | 0 |
| 4 | `5.62e-5` | 0 | 0 | 9.71 | `1.73e5` | 0 |
| 5 | `2.70e-4` | 0.25 | `0.00948` | 1.67 | `6.16e3` | 0.001 |
| 6 | `8.66e-4` | 0 | 0 | 0.249 | 288 | 0.003 |
| 7 | `2.06e-3` | 1 | `-0.00861` | 0.0338 | 16.4 | 0.035 |

The backbone estimate was `0.8139695955`.  The nominal corrected pilot value
was `0.8148344526` with an ordinary plug-in standard error of `0.00962`, but
this is not a valid convergence claim: blocks 1--4 and 6 had nonempty residual
injections, zero terminal hits, and zero sample variance.  The pilot summary
therefore marks false convergence and intentionally withholds a production
run allocation.  Block 0 is not marked because it is genuinely empty.

The time blocking works as an isolation mechanism: failures in early reverse
blocks no longer erase the observable contribution from block 7.  It does not
remove the long repeated-truncation tail that an early residual block must
still traverse.  Do not add `K_c=1,000` production runs to the zero-hit blocks.
The next resource decision is whether to spend a targeted `K_c=10,000` budget
pilot on the borderline blocks 4--6, or to change their stratification before
further sampling.  The Phase 4 pilot itself does not resolve that choice.

### Targeted Phase 4 `K_c=10,000` budget pilot

A staged two-run pilot was subsequently completed at `K_d=1,000` and
`K_c=10,000`.  Blocks 5 and 6 were tested first.  Their diagnostics showed a
clear budget crossover, so block 4 was then added.  The existing deterministic
boundary checkpoint was reused; blocks 0--3 and 7 were not rerun.

| block | runs | terminal hit fraction | mean correction | sample standard error | maximum norm-inflation ratio | minimum heavy fraction | mean seconds/run |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 0.5 | `-0.00648` | `0.00648` | `7.49e3` | 0.0023 | 290 |
| 5 | 2 | 1 | `-0.00577` | `0.00108` | 204 | 0.0124 | 225 |
| 6 | 2 | 1 | `-0.00286` | `0.000430` | 9.84 | 0.105 | 138 |

At `K_c=1,000`, blocks 5 and 6 had hit fractions 0.25 and 0, maximum
norm-inflation ratios `6.16e3` and 288, and minimum heavy fractions 0.001 and
0.003.  At `K_c=10,000`, both blocks hit in every run and their heavy fractions
rose to 0.0124 and 0.105.  Neither triggered the early degeneracy alarm.  This
is a real correction-budget crossover for blocks 5 and 6, although two runs
are not enough for production variance estimates.

Block 4 improved from a maximum norm-inflation ratio `1.73e5` and zero heavy
fraction to `7.49e3` and 0.0023.  One of its two runs still had zero terminal
overlap.  It should therefore be treated as partially recovered but still
rare-event dominated.  Do not use its two-run sample variance for production
allocation.

The targeted pilot is stored in
`results/time_blocked_residual_2d_obc_xx_z_n11_Kd1000_Kc10000_steps8_seed20260819/`.
The remaining narrow empirical question is the early-block boundary.  Blocks
1--3 had `K_c=1,000` norm-inflation ratios from `1.46e7` to `6.52e10`; the
Phase 3 logarithmic-delay evidence makes a full `K_c=10,000` ensemble there
difficult to justify.  A single diagnostic run of block 3, followed by block 2
only if the heavy core recovers, would be the smallest useful continuation if
a later study needs a more precise crossover curve.

### Consolidation decision

The direct, persistent-correction, and time-blocked experiments now support
the same practical conclusion.  Independent blocks recover the late residual
contributions, but they cannot shorten the long randomized propagation tail
of an early injection.  The unresolved block-3 point is therefore not expected
to change the algorithmic conclusion and is not an active benchmark request.

Further tuning of repeated local unbiased hard-`K` truncation is deprioritized.
Future work should change what is sampled or represented---for example a
terminal-scalar conditional estimator, a global history sampler, a different
operator representation, or an explicit controlled-bias method.  Phase 4 is
retained as a small research prototype and diagnostic, not promoted to the
top-level `spd` API.
