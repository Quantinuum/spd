# TODO: averaged local propagation for HVA-TMERA

Status: planned; not implemented. This is a follow-up to the working
[full-Hamiltonian HVA-TMERA implementation](hva_tmera.md).

## Goal

Compute the finite-ring energy density and all shared-parameter gradients by
ascending a bounded local operator through the MERA. Match the current
full-Hamiltonian result without constructing an N-qubit observable or enumerating
all spatial tensor placements in the new execution path.

Keep the current default W/U parameterization, parameter ordering, angle units,
ancilla preparation, optional physical bottom layer, and finite top termination.
The proposed symmetric U/top experiments are not part of this implementation.
Parameters remain shared spatially within each layer, and independent between
scales, W/U types, and brickwork rounds.

Initial acceptance target: NumPy, chi=2 and chi=4, exact forward and backward
calculations. Add JAX/Triton and truncation after the reference is validated.
Preserve the existing uncommitted work; this TODO does not authorize replacing
or discarding it.

## Why averaging is necessary

Sharing tensors within a layer does not guarantee an exactly translation-invariant
state. Selecting one two- or four-site physical cell and summing shared gate
parameter gradients does not recover the omitted Hamiltonian terms.

For a homogeneous binary layer, use both inequivalent placements of the local
operator. After mapping their outputs into the same ordered coarse-site basis,

```text
h_next = (A_left(h) + A_right(h)) / 2.
```

Merge at each layer before proceeding to the next. Do not retain a branching
tree of all causal cones. The average represents all translated local terms;
it is not an assumption that their individual expectation values agree.

Use the identity

```text
V_layer^dagger [ (1/M) sum_j translate_j(h) ] V_layer
    = (1/(M/2)) sum_k translate_k(h_next)
```

as the normalization and correctness contract for an ordinary binary layer on
M fine sites. Derive and test the appropriate embeddings for the dimension-changing
bottom layers; handle small periodic rings explicitly rather than applying a
bulk formula where its sites overlap.

For the TFI model, start at the physical lattice with
`h = -Z_0 Z_1 - g X_0`, padded with identities as needed. Its spatial average is
`H/N`. Carry that normalization through every layer. Return `e = <H>/N` and
`grad_e = grad(<H>)/N`; multiply by N only if a caller explicitly requests total
energy and its gradient.

## Expected scaling and scope

At fixed chi and brickwork depth, the local operator width and number of branch
maps per layer are bounded independently of N. There are O(log N) layers, so
energy and reverse-mode gradient evaluation can cost `O(C(chi, rounds) log N)`.
This is a per-evaluation statement, not a bound on optimizer iterations.

A standard binary MERA generally needs three coarse sites for a closed local
operator description. For chi=4 these use six encoded qubits, allowing at most
4096 Pauli coefficients between layers. Temporary branch support can be wider
and may dominate memory and runtime. Measure it rather than treating six qubits
as the maximum size of the entire calculation.

Forward live operator memory can be bounded independently of N; retaining
records for the backward pass normally adds O(log N) storage. Sparse support
can become dense within this bounded space. Performance improvements over the
current implementation must be benchmarked, particularly on small systems.

Avoid hidden O(N) work: the new path must not construct the full circuit, generate
all `binary_mera_layers` placements, or allocate Pauli strings of length N.

## Code organization

Use a small MERA evaluator above the existing circuit runner. Do not introduce
a general tensor-network framework or a general branching circuit IR.

| Responsibility | Proposed home |
| --- | --- |
| W/U gate definitions, site dimensions, parameter layout, local branch construction | `spd/ansatz/hva_mera.py`, with small shared helpers if needed |
| Layer-by-layer averaging, forward records, backward orchestration, energy-density evaluation | New `spd/mera.py` |
| Execution of each local gate/channel sequence | Existing `CircuitIR`, `evolve`, and `backpropagate` |
| Sparse coefficient alignment, weighted merging, explicit coefficient-gradient construction | Backend-neutral operations exposed through `BackendAdapter` |
| Optimizer selection and reporting | Existing HVA-TMERA example, after equivalence is established |

A branch description needs only its local circuit, input embedding, output
reindexing/identity padding, averaging weight, and gate-to-parameter metadata.
A layer groups its branches and describes input/output site widths. Near the top,
use a bounded explicit ring calculation rather than forcing the bulk branch
representation to cover exceptional geometry.

## Phase 1: local geometry and exact forward reference

- [ ] Factor parameter counts/slices and tensor gate construction so full-circuit
  and local builders share one definition. Keep existing circuits and parameter
  metadata unchanged; avoid enumerating the full lattice to count parameters.
- [ ] Build both local placements for a binary layer using local qubit indices.
  Preserve retained inputs, ancillas, preparation rotations, and gate order.
- [ ] Define input embedding and output alignment explicitly, including the
  transpose operations required by the later backward pass. Distinguish
  reindexing/identity padding from the physical ancilla contraction.
- [ ] Implement exact weighted SPO merging on aligned Pauli keys. Audit current
  arithmetic: some NumPy addition paths use `np.isclose`, so ordinary `+` is not
  automatically a zero-cutoff exact merge. Do not silently remove small terms.
- [ ] Implement NumPy averaged ascent, merging the two results at each scale.
- [ ] Support actual bottom-layer dimensions and the optional chi=4 physical
  bottom layer. Do not assume all layers have the same site width.
- [ ] When the ring is too small for a generic local cone, assemble the spatial
  average on that small ring and finish with the existing top circuit/state.
  Derive the switch condition from geometry so periodic sites are never duplicated.
- [ ] Verify each branch against a dense local isometry calculation, and verify
  the normalized layer identity above against explicit translated sums.
- [ ] Compare complete energy density with the full-Hamiltonian result for
  N=4,8,16; chi=2,4; multiple seeds; one/two rounds; and both valid bottom options.

Acceptance: exact finite-ring energy agreement to double-precision tolerance,
with no term truncation or binding term cap, and the original ansatz unchanged.

## Phase 2: exact shared-parameter gradients

- [ ] Add a narrow way to run a local circuit backward with an arbitrary output
  coefficient gradient and that circuit's own forward record/checkpoints.
  `init_gradient_spo` currently initializes terminal losses; a branch needs an
  incoming gradient from the rest of the MERA computation.
- [ ] Give each branch ownership of its output coefficients, channel checkpoints,
  and execution metadata. Do not rely on a merged SPO inheriting one branch's
  checkpoints through ordinary arithmetic.
- [ ] Reverse the weighted merge: send half of the incoming coefficient gradient
  into each ordinary binary branch, undoing output alignment first.
- [ ] Backpropagate through each branch and sum its contributions to the shared
  input-operator gradient and the layer's parameter gradient. Reverse the input
  embedding as well, including identity-padding selection where applicable.
- [ ] Preserve the common input coefficients once. An SPGO stores both coefficients
  and adjoints; adding two input SPGOs naively doubles the primal coefficients.
  Sum adjoints explicitly and attach them to the saved common input.
- [ ] Preserve zero-primal/nonzero-adjoint terms and dependencies through exact
  cancellations. Align sparse keys without assuming adjoint support equals the
  nonzero forward support.
- [ ] Accumulate all W/U occurrences through the existing parameter-index and
  angle-factor conventions. Do not add a spatial multiplicity factor on top of
  the factors already represented by averaging.
- [ ] Differentiate the actual finite top circuit and all bottom-layer parameters.
- [ ] Test merge and embedding adjoints, including cancellations and zero angles.
- [ ] Compare every parameter gradient with the full-Hamiltonian gradient divided
  by N, and check finite differences of the new energy-density evaluator.

Acceptance: energy and all gradients match the reference for the Phase 1 cases,
including top parameters, with no truncation. Keep the full-Hamiltonian path as
an independent regression reference.

## Phase 3: backend integration and approximation

- [ ] Add equivalent local execution and merge/adjoint operations for JAX and
  Triton, reusing existing rotation and channel kernels where possible.
- [ ] Keep sparse merges, coefficient-gradient alignment, and branch state on the
  selected device. Avoid transferring complete intermediate operators to the host.
- [ ] Validate backend agreement before enabling approximation.
- [ ] Specify truncation points and record the corresponding backward decisions.
  Prefer merging contributions before any additional layer-level truncation;
  measure whether branch-internal truncation is also needed for temporary memory.
- [ ] Compare approximation error against the exact energy and gradient.
  Averaging and truncation do not generally commute, so the new approximate
  result need not match the old truncated full-Hamiltonian result.
- [ ] Report per-layer and per-branch term counts, active widths, truncation,
  runtime, and peak memory. Distinguish measurements from error estimates.

Acceptance: backend agreement at zero cutoff, documented approximation behavior,
and no loss of gradient support through sparse merging.

## Phase 4: optimization example and scaling measurements

- [ ] Add an explicit evaluator choice to the example, retaining the full result
  as a comparison mode. Normalize optimizer costs and gradients consistently.
- [ ] Compare short optimization runs using the same initial parameters and
  energy-density tolerances.
- [ ] Measure increasing powers of two in N at fixed chi and rounds. Report
  construction time separately from repeated energy/gradient evaluation time.
- [ ] Confirm bounded local support and no O(N) allocation/setup in the local path.
- [ ] Measure crossover size and temporary branch memory before changing defaults.
- [ ] Document supported cases, settings, normalization, and benchmark results.

Acceptance: correctness and measured efficiency justify selecting the local
method by default; the full method remains available for validation.

## Suggested work split and handoff

One focused coding session is a reasonable unit for Phase 1. A session may be
able to finish Phases 1 and 2 together, but exact sparse-gradient handling is
substantial enough that it should remain a separate acceptance milestone.
Do not make all backends, truncation, and optimizer integration prerequisites
for reviewing the exact NumPy implementation.

Suggested sequence of reviewable changes:

1. Shared layer descriptions and exact NumPy energy density.
2. Explicit branch gradients and exact NumPy energy/gradient agreement.
3. JAX/Triton support and truncation validation.
4. Optimizer integration and scaling measurements.

For the next session: read this TODO and `hva_tmera.md`, inspect the existing
uncommitted changes, and implement Phase 1 first. Update the checkboxes with
actual validation results. Record any geometry or gradient limitation rather
than weakening equivalence checks to make them pass.

## References

- [Evenbly and Vidal, Algorithms for entanglement renormalization](https://arxiv.org/pdf/0707.1454):
  Section II.C distinguishes binary three-site from ternary two-site operator
  support; Section II.D explains the translation-invariance terminology;
  Section III.E derives averaged ascending/descending evaluation. Its displayed
  three-branch formulas and chi^8 cost concern the ternary scheme; derive the
  two-placement binary maps for this implementation rather than copying those
  formulas unchanged.
- Existing regression references: `tests/test_hva_mera_bond_dimension.py`,
  `tests/test_hva_mera_channels.py`, `tests/test_hva_mera_example.py`, and
  `tests/test_hva_mera_triton.py`.
