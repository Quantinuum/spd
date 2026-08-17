# Free-Fermion Noise-Sensitivity TODO

## Question

Test whether operator stabilizer entropy (OSE) predicts depolarizing-noise
sensitivity for random one-dimensional free-fermion circuits.

## Previous Experiment

The first experiment used:

- an open chain with 12 qubits;
- six layers of onsite `Z` and nearest-neighbor `XX`/`YY` rotations;
- five random-angle scales with 50 seeds per scale;
- finite differences over several small depolarizing probabilities.

For the 250 samples, the preliminary correlations were:

```text
corr(dE/dp, OSE)                            = -0.156
corr(dE/dp, visible l2 mass)                = -0.144
corr(dE/dp, visible Pauli-weighted l2 mass) = -0.144
corr(dE/dp, clean energy)                   = -0.971
corr(clean energy, OSE)                     =  0.186
```

These fixed-size results do not show a strong relationship between OSE and
noise sensitivity. The strong clean-energy correlation may dominate the raw
slope and should be controlled for in a future analysis.

## Updated Method

Use `backpropagate_noise_analysis` instead of a finite-difference noise sweep:

1. Evolve the clean observable once.
2. Initialize the terminal SPGO for the chosen expectation value.
3. Call `backpropagate_noise_analysis`.
4. Sum the operation-aligned two-qubit susceptibilities when all two-qubit
   channels share the same probability.
5. Use one finite-difference sample only as an end-to-end sanity check.

## Open Question

A hopping block is represented by separate `YY` and `XX` Pauli rotations.
Decide whether the intended physical model applies depolarizing noise after
each rotation or once after the combined hopping block before implementing a
new sweep.

## Scope of Future Work

If the experiment is resumed, keep generated parameters and results outside
the repository. Add a small reproducible example and focused tests only after
the channel-placement convention is fixed.
