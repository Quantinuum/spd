# TFI Symmetry-Breaking Findings

## Main Observation

Optimized one-dimensional TFI symmetry-breaking runs can have nearly identical
energy but very different operator stabilizer entropy (OSE):

```text
run       energy density          OSE
seed0    -1.3400264103795254    4.4959199766
seed1118 -1.3400264092759370    1.7571262274
```

This is not evidence of a degenerate exact ground state. The optimized ansatz
states are approximate, separated minima, while the energy is a lossy scalar
objective.

## State Diagnostics

For `L=13` and `g=1.1`, dense statevector checks reproduce the saved energies.
The two states have:

```text
fidelity                         = 0.9787898751
global-phase-adjusted distance  = 0.1460267133
Var(H), seed0                   = 9.9211e-02
Var(H), seed1118                = 9.9174e-02
```

Both put about `0.9524` of their spectral weight on the exact ground state,
but their excited-state phases differ. Their half-cut entanglement is also
similar: about `0.805` bits for seed0 and `0.795` bits for seed1118. The OSE
difference is therefore mainly an operator-space effect rather than a large
state-entanglement difference.

The active-parameter Hessians have no eigenvalues below `1e-3`, and the direct
line between the two optima rises from about `-1.34` to `-0.59` in energy
density. They look like distinct near-optimal minima, not a local flat valley.

## Visible and Hidden Pauli Sectors

In the `+` basis:

```text
<+|P|+> = 1  when P contains only I and X
         = 0  when P contains Y or Z.
```

Energy therefore sees only the `I/X` sector. The same-energy runs distribute
their squared coefficient mass very differently:

```text
run       I/X l2 mass (fraction)       Y/Z l2 mass (fraction)
seed0     0.2763945988 (0.125065)      1.9336054012 (0.874935)
seed1118  1.2910411939 (0.584182)      0.9189588061 (0.415818)
```

This hidden-sector difference explains much of the OSE gap.

## Hidden-Sector Regularization

A targeted terminal penalty is:

```text
C_hidden = sum_s w(h(s)) |c_s|^2
```

where `h(s)` can count `Y/Z` letters or total Pauli weight. Implemented weight
families are:

```text
power: w(h) = h^p                              for h > cutoff
exp:   w(h) = exp(gamma * (h - cutoff)) - 1   for h > cutoff
```

The terminal coefficient gradient is `2 w(s) c_s`, so the experiment can add
this SPGO to the ordinary basis-expectation SPGO and reuse
`spd.backpropagate(...)` without changing backend kernels.

Weak regularization did not change the basin:

```text
p=0, lambda_hidden=0.01:
energy      = -1.3400230721
OSE         =  4.4913894525
hidden frac =  0.874632
```

At the seed0 initial point, the hidden and energy gradients had cosine about
`0.99998`, so a small penalty mostly rescaled the original descent direction.

Stronger or size-weighted penalties found low-hidden solutions with a small
energy cost:

```text
setting                  energy          OSE       hidden fraction
p=0, lambda=0.1        -1.3369474044    1.366845      0.335699
p=2, lambda=0.01       -1.3379043416    1.420430      0.342875
```

Treat this regularizer as a basin-selection tool rather than the final physics
objective. A useful protocol is to start with stronger regularization, reduce
`lambda_hidden` in separate fixed-objective runs, and finish with an optional
energy-only optimization. If the hidden mass returns, report the energy versus
hidden-mass Pareto curve.

## Depolarizing-Noise Probe

A finite-difference probe applied two-qubit depolarizing noise after each
nonzero `ZZPhase` gate. Across four nearly equal-energy seeds:

```text
run       OSE       I/X l2 mass    dE/dp
seed0     4.495920   0.27639460    15.931976
seed43    4.495948   0.27639109    15.932056
seed86    3.657084   0.40839987    18.266425
seed1118  1.757126   1.29104119    22.050942
```

For these four samples:

```text
corr(OSE, dE/dp)       = -0.9968
corr(OSE, I/X l2 mass) = -0.9832
```

The sample is too small for a general conclusion, but it supports the visible-
sector interpretation. Future checks should use
`backpropagate_noise_analysis` instead of a finite-difference probability
sweep.

## Tiny `Rz` Tails

One alpha-0.5 initfile run retained optimized `Rz` parameters below `2e-6`.
Those small nonzero angles expanded the raw SPO from 256 to 8192 rows, mostly
through coefficients near the truncation threshold. Setting the `Rz`
parameters exactly to zero reduced the support without materially changing
energy, sector mass, or OSE.

Small parameters that are physically intended to be zero should therefore be
canonicalized before interpreting SPO row count or memory use.

## Next Questions

- Does a low-hidden solution remain low-hidden as `lambda_hidden` approaches
  zero?
- What is the energy versus hidden-mass/OSE Pareto curve?
- Do `Y/Z` count, total Pauli weight, and exponential weights select different
  minima?
- Would matching additional physical correlators remove the underconstraint
  more naturally than a Pauli-sector penalty?
- Is there a free-fermion correlation-matrix explanation for the different
  operator-space distributions?
- How sensitive are the results to truncation and `max_num_str`?

Keep `kernels.py` unchanged unless hidden-sector loss becomes a supported
backend feature.
