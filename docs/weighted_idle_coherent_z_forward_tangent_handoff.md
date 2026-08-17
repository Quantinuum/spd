# Weighted Idle Coherent-Z Susceptibility

## Status

Coherent idle-Z susceptibility is not implemented. The current
`backpropagate_noise_analysis` result contains only operation-aligned one- and
two-qubit depolarizing susceptibilities.

The previous reverse-mode `idle_coherent_z` estimate was removed because it
could silently return zero when the correct derivative was nonzero.
Depolarizing channels do not have this problem: they rescale existing Pauli
coefficients and do not create new Pauli support.

## Why Restricted Reverse Mode Fails

At a zero-angle coherent insertion with generator `K`, an anticommuting primal
term `P` creates a tangent partner `Q = K P`. The coefficient of `Q` is zero at
the insertion point, but the loss derivative with respect to that coefficient
can be nonzero. An SPGO whose adjoint is restricted to the current primal
support can therefore omit the required direction.

For basis-0 expectation, the formal terminal derivative is nonzero for every
string containing only `I` and `Z`. Explicitly storing all `2**system_size`
diagonal strings is not practical.

A three-qubit counterexample is:

```text
initial state: |000>
observable:    IZI

RX(0.4) on q1
RX(0.2) on q0
idle RZ(epsilon) on q1 after the q0 gate
RY(0.7) on q1
```

The dense derivative at `epsilon=0` is:

```text
-sin(0.4) * sin(0.7) = -0.250870183...
```

The restricted reverse-SPGO calculation returns zero because the tangent
support is absent. TFI circuits can have exact symmetry-protected zeros, but
that special case does not validate the general method.

## Recommended Result: One Weighted Total

Compute one scalar:

```text
chi_idle = sum_g weights[g] * chi_idle[g]
```

rather than retaining one tangent operator per gate. `weights=None` can mean
all weights are one. Hardware-derived weights can represent quantities such as
frequency times idle duration.

Use a forward tangent with two sparse operators:

```text
ideal_spo
tangent_spo
```

Initialize `tangent_spo` to zero. During the reverse traversal used for
Heisenberg evolution, construct the idle-Z source after each physical gate and
before conjugating through that gate:

```text
K_g = weights[g] * sum(Z_q for q not active in gate g)
source_g = (i / 2) [K_g, ideal_spo]

tangent_spo = tangent_spo + source_g
ideal_spo   = apply_ideal_forward(ideal_spo, gate_g)
tangent_spo = apply_ideal_forward(tangent_spo, gate_g)
```

Skipped operations add no source. Because the source always uses the ideal
SPO, coherent-error cross terms are excluded: this is the derivative with
respect to one common error scale, not a finite-angle simulation.

At the end, basis-0 expectation uses:

```python
coherent_total = final_tangent_spo.get_expectation_value(basis="0")
```

Only diagonal strings present in the sparse tangent are inspected.

## Suggested Implementation

Add backend primitives for both NumPy and JAX:

```python
get_pauli_rotation_tangent(spo, xzk) -> SparsePauliOp
```

An optional fused idle helper may scan the primal SPO once:

```python
get_idle_z_tangent(spo, active_qubits, system_size) -> SparsePauliOp
```

Add an internal runner before changing the public API:

```python
_evolve_with_weighted_idle_tangent(
    spo,
    circuit_ir,
    weights,
    trunc_val,
    max_num_str,
    backend,
) -> (final_spo, final_tangent_spo, info)
```

If this becomes stable, a higher-level `analyze_noise` function can own both
the forward tangent and the ordinary reverse pass. The existing
`backpropagate_noise_analysis` does not receive the original observable and
therefore does not have enough information to construct this forward tangent.

## Experimental Alternative: Support Probe

A finite-angle probe can be used only to discover terminal support:

1. Evolve the ideal observable.
2. Evolve a second circuit with small idle-Z probe rotations.
3. Form the union of ideal and probe terminal supports.
4. Use ideal coefficients, including explicit zeros on probe-only support.
5. Initialize the terminal SPGO on this enlarged support.
6. Backpropagate through the ideal circuit.

The perturbed coefficients must not be used as the derivative. A heuristic
threshold separation is:

```text
theta_probe**2 < support_probe_trunc_val < abs(theta_probe)
```

Coefficient scales and cancellation mean this is not a correctness guarantee.
At least two probe angles should produce stable support and gradients. Keep
this method explicitly experimental and implement it only after forward mode
is available as a reference.

## Complexity and Truncation

For one weighted total, forward mode stores one primal and one tangent SPO:

```text
memory = O(size(primal SPO) + size(tangent SPO))
```

There is no explicit `O(num_gates)` tangent-memory multiplier, although tangent
support can grow differently from primal support. Tangent truncation and
maximum-size loss must be reported separately from primal truncation.

## Required Validation

Both backends should test:

1. local tangent kernels against dense finite differences;
2. the nonzero three-qubit counterexample;
3. all-one and nonuniform positive and negative weights;
4. weighted totals against sums of separate dense susceptibilities;
5. exact TFI symmetry zeros;
6. rotations, Cliffords, skipped operations, and padded storage;
7. unchanged primal and ordinary reverse-mode results;
8. independent primal and tangent truncation information;
9. support-probe agreement and probe-angle stability, if the experimental
   method is implemented.

## Deferred Scope

This document is a proposal. It does not imply that forward tangents,
finite-angle support probing, weighted coherent totals, or per-gate coherent
maps are currently available.
