# Time-Blocked Residual Correction

This is the Phase 4 NumPy prototype. It is separate from direct R-SPD and
from the persistent residual-correction estimator in
`spd/residual_corrected.py`.

## Exact identity and ordering

Suppose the circuit stores gates `g_0, ..., g_(L-1)`. The repository's
Heisenberg loop calls `apply_forward` on them in reverse order. Define

```text
h_l = g_(L-1-l)
```

and let `T_l` be the exact linear map for `h_l`. Split the initial observable
and every deterministic backbone candidate as

```text
c_0 = d_0 + r_0
T_l d_l = d_(l+1) + r_(l+1).
```

Then

```text
c_L = d_L + sum_(j=0)^L Q_j r_j,
Q_j = T_(L-1) ... T_j,
Q_L = identity.
```

Partition the injection indices `0, ..., L` into blocks `I_b` and define

```text
Delta_mu_b = expectation(sum_(j in I_b) Q_j r_j).
```

This gives the exact scalar identity

```text
mu = mu_backbone + sum_b Delta_mu_b.
```

Within one block run, pivotal truncation is applied after every propagation
and residual merge. Its conditional expectation is the identity. Therefore
the block estimate is unbiased for `Delta_mu_b`, and independently seeded
block sample means give an unbiased sum. This formal statement requires
`numerical_zero_tolerance=0`. The default `1e-12` result is exact only up to
the declared numerical-zero policy.

## Block assignment

Blocks use the reverse-Heisenberg operation stream. A gate at reverse index
`l` and its new residual `r_(l+1)` belong to the same block. The initial
residual `r_0` belongs to block 0. Thus every residual injection is assigned
once.

The generic API takes explicit block sizes because `CircuitIR` does not carry
Trotter-step labels. For the XX+Z pilot, one block is one Trotter step. Block 0
is the last physical Trotter step because propagation is reversed. One step
per block is the initial compromise: one gate per block creates thousands of
strata, while several steps per block lengthen the shared noisy history.

## Streaming and memory

`prepare_time_blocked_backbone` makes one deterministic pass and stores the
top-`K_d` backbone only at block boundaries. It does not store residuals. A
block run reloads one boundary backbone, regenerates residuals inside that
block, and propagates only that block's sampled correction through the
remaining gates.

The stored deterministic state is `O(B K_d)`. Blocks run sequentially, so the
live sampled state is `O(K_c)`, not `O(B K_c)`. This favors a small first
prototype over a lockstep implementation.

## Seeds and allocation

Pilot seeds are derived from

```text
(master_seed, stage, block_index, correction_budget, run_index).
```

They do not depend on the requested number of runs or worker count. Stage 0 is
the pilot stream; production should use a different stage.

The first pilot uses one common correction budget. For block sample `Y_b`, it
records the uncentered pilot second moment

```text
m2_b = mean(Y_b ** 2).
```

If every block has a healthy pilot, a later production study could allocate
runs under a total-work constraint using the continuous rule

```text
n_b proportional to sqrt(m2_b / c_b)
```

with a nonzero minimum in every block. The present pilot never reached that
stage: zero-hit blocks made their second moments unusable for allocation. The
research prototype therefore records the pilot statistics but does not expose
a production allocator. Choosing different budgets requires pilots at two or
more budget candidates and comparison of second moment times work, terminal
hits, and heavy-core health.

The pilot wrapper does not emit a suggested production allocation.

## Diagnostics and degeneracy

Each block run records residual support and norms, correction candidate and
sampled support, sampled norm squared, pivotal heavy count, predicted MSE,
terminal scalar hit, and imaginary leakage. Block summaries include second
moment, variance, standard error, scalar quantiles, and terminal hit fraction.

A zero terminal hit fraction together with zero sample variance is explicitly
marked `false_convergence` when the block had a nonempty residual injection.
The per-gate records also report sampled norm squared divided by cumulative
injected residual squared norm. The first randomized truncation where this
ratio reaches `1e6` and the heavy fraction is at most `1%` is recorded as a
heuristic degeneracy-warning gate. This is an operational alarm, not a proof
of bias or divergence. It locates a transition before the terminal scalar and
should be inspected before adding production runs.

## Checkpoint format and pilot

`examples/benchmark_time_blocked_residual_2d_obc_xx_z.py` writes:

```text
backbone.pkl
pilot/block_XXXX/run_XXXXXX.pkl
pilot_blocks.csv
pilot_summary.pkl
```

Every pickle contains immutable metadata. The deterministic checkpoint stores
only boundary backbones. Each completed block run is replaced atomically and
can be extended by increasing `--pilot-runs`.

The smallest useful hard pilot is step 8 on 11x11 with eight one-step blocks,
`K_d=1000`, `K_c=1000`, and four pilot runs per block. Four samples are not a
production variance estimate. They are enough for a first second-moment,
terminal-hit, and norm-inflation screen. Do not allocate production runs to a
zero-hit block until its correction budget or sampling design has been
reconsidered.
