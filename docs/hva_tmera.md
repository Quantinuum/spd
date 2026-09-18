# Open-chain HVA brickwork binary MERA

## Scope

This feature implements a paper-inspired HVA analogue of standard binary MERA
in finite periodic 1D systems. It is not an exact reproduction of the ansatz or
benchmarks in arXiv:2108.13401. The primary target is `chi=4`; `chi=2` and
`chi=8` retain structural coverage.

The implementation is deliberately small. It uses fixed integer qubit indices,
explicit W/U placement, and `VariationalCircuit` metadata. It is not a general
tensor-network framework.

## Final tensor ansatz

One four-qubit brickwork round on ordered positions `[0,1,2,3]` is:

1. `ZZPhase(p0)` on `(0,1)` and `ZZPhase(p1)` on `(2,3)`;
2. full-register barrier;
3. independent `Rx(p2),...,Rx(p5)` on positions `0,...,3`;
4. full-register barrier;
5. `ZZPhase(p6)` on `(1,2)`;
6. full-register barrier;
7. independent `Rx(p7),...,Rx(p10)` on positions `0,...,3`;
8. full-register barrier.

A two-qubit physical tensor uses three parameters per round:
`ZZ(0,1), X(0), X(1)`.

Parameter sharing is:

- independent between gate positions within a tensor;
- shared across every spatial tensor instance of the same scale and type (W or U);
- independent across scales, tensor types (W or U), and brickwork rounds.

Every variable rotation depends on one optimizer parameter with scalar factor
one. Fixed `Ry(0.5)` preparation rotations use parameter index `-1`.

## Isometries

An isometry is a unitary on retained inputs plus fresh zero inputs. Fresh inputs
receive fixed `Ry(0.5)` before the brickwork unitary:

```text
W |psi> = U_brickwork (|+...+>_ancillas tensor |psi>_retained).
```

`HVATensor.outputs`, `.retained`, and `.ancillas` make those roles explicit.
Because the preparation followed by a unitary preserves inner products, every
W satisfies `W dagger W = I` on its retained space.

## How the code is organized

The implementation separates **where tensors go** from **which gates they
contain**. This keeps the MERA indexing independent of the HVA brickwork.

`HVATensor` describes one placement. `outputs` is its ordered qubit chain. For
a W isometry, `retained` names inputs that already exist and `ancillas` names
fresh zero inputs. For a U disentangler, `retained` and `ancillas` are empty.

`BinaryMERALayer(scale=s, ...)` groups every W and U acting at child scale `s`:

```text
coarse scale s+1       [ parent 0 ]             [ parent 1 ]
                            | W0                      | W1
                            v                         v
fine scale s           [ child 0 ][ child 1 ]--U--[ child 2 ][ child 3 ]
                            ^                                      |
                            +----------- periodic U ---------------+

preparation order:                    W sublayer  ->  U sublayer
backward observable propagation:      U sublayer  ->  W sublayer
```

For `L=16, chi=4`, preparation descends through the scales as follows. Each
line shows the site representation after the W expansion; the U tensors then
act across adjacent sites.

```text
scale 4 top:              (14,15)
                              |
                         W at scale 3
                              v
scale 3 sites:        (6,7)  (14,15)                 [top has no U]
                              |
                         W then U at scale 2
                              v
scale 2 sites:    (2,3) (6,7) (10,11) (14,15)
                              |
                         W then U at scale 1
                              v
scale 1 sites:   (0,1) (2,3) ... (12,13) (14,15)
                              |
                    optional W then U at scale 0
                              v
physical sites:       0  1  2  3  ...  12 13 14 15
```

The main call path is:

```text
tfi_binary_mera
  -> _build_network
       -> _selected_layers
            -> binary_mera_layers
                 -> binary_mera_sites
       -> _Builder.prepare_zero_inputs
       -> _Builder.add_tensors           [all W, then all U, at each scale]
       -> _Builder.build                 [VariationalCircuit metadata]

tfi_binary_mera_channels
  -> _build_network                      [same gates and parameter layout]
  -> parse_pytket_circuit
  -> insert CreateZero before each group of new qubits is prepared
```

`binary_mera_parameter_shape` walks the same selected layers without emitting
gates. It counts one parameter block per scale, tensor type (W or U), and brickwork
round. Spatial tensor instances within that block reuse the same indices.

## Binary placement

At scale `s`, site `j` owns the rightmost
`min(log2(chi), 2**s)` fixed indices in its physical block of length `2**s`.
Preparation proceeds coarse to fine. Each layer applies all W tensors, then all
U tensors. The top layer has two child sites, one W, and no U.

For `L=16, chi=4`, the four-qubit part is:

| Scale | W outputs | U outputs |
| --- | --- | --- |
| 3 | `(6,7,14,15)` | none |
| 2 | `(2,3,6,7)`, `(10,11,14,15)` | `(6,7,10,11)`, `(14,15,2,3)` |
| 1 | `(0,1,2,3)`, `(4,5,6,7)`, `(8,9,10,11)`, `(12,13,14,15)` | `(2,3,4,5)`, `(6,7,8,9)`, `(10,11,12,13)`, `(14,15,0,1)` |

The optional physical layer has square W blocks
`(0,1),(2,3),...,(14,15)` followed by U blocks
`(1,2),(3,4),...,(15,0)`. It creates no qubits. Set
`physical_bottom=False` to omit both physical W and U layers.

The physical layer remains an explicit modeling choice. The production default
includes it because its dimension map is valid (`chi=4 -> 2 x 2`) and it
improved the one-round finite-budget trial. Deeper finite-budget results were
mixed, so this is not a claim that it always optimizes better.

## Translation symmetry and the energy objective

Sharing W and U tensors within a layer does **not** make the finite MERA state
invariant under translation by two physical sites (`chi=2`) or four physical
sites (`chi=4`). A translation that preserves the bottom layer can change which
sites share a parent in the next layer. Coarser layers and the top input still
distinguish those positions. `2 * log2(chi)` counts the qubits in a pair of
full-dimension MERA sites; it is not a translation period of the whole state.

For `h_i = -Z_i Z_(i+1) - g X_i`, the energy density and its gradient are

```text
e = (1/N) sum_i <h_i>
grad e = (1/N) sum_i grad <h_i>.
```

Replacing these sums with `i=0,...,p-1`, normalized by `p`, requires those
expectations and derivatives to repeat every `p` sites. Summing gate derivatives
for shared parameters does not supply the omitted Hamiltonian terms.

The proposed local-cell initialization was checked at `N=8, g=1.1`, one
brickwork round, physical bottom layer included, and parameters drawn using
`np.random.default_rng(7).normal(0, 0.15, shape)`:

| chi | Cell size | Full energy density | First-cell energy density | Maximum absolute gradient difference |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | -0.997539061761 | -0.993832017579 | 0.051116945802 |
| 4 | 4 | -0.809768853062 | -0.844523125240 | 0.811649782743 |

Both gradients in the last column are normalized per physical site. Dense
state derivatives and untruncated NumPy SPD backpropagation agree on this
mismatch. Tests also cover two rounds and omission of the chi=4 physical layer.
The example therefore keeps the full Hamiltonian and returns total energy and
its gradient; divide both by `N` to obtain energy density and its gradient.

An exact local evaluation must average **all** cells, including their
right-edge bonds. This can reduce peak memory if cells are evaluated separately,
but repeats circuit propagation and may increase runtime. A smaller exact
calculation would require an implementation that averages the different causal
cones at each scale, rather than selecting one physical cell.

## API

```python
import numpy as np
from spd.ansatz import binary_mera_parameter_shape, tfi_binary_mera

shape = binary_mera_parameter_shape(
    16, brickwork_rounds=2, chi=4, physical_bottom=False,
)
params = np.random.default_rng(7).normal(0, 0.05, shape)
ansatz = tfi_binary_mera(
    params, 16, brickwork_rounds=2, chi=4, physical_bottom=False,
)
```

Run the channel-backed optimizer from the repository root. The positional
arguments are system size, bond dimension, and L-BFGS iteration limit:

```bash
python examples/gradient/run_1d_tfi_hva_tmera.py 16 4 100 --rounds 2 --backend triton
```

Use `--no-physical-bottom` for the corresponding upper-layer-only run:

```bash
python examples/gradient/run_1d_tfi_hva_tmera.py 16 4 100 --rounds 2 --backend triton --no-physical-bottom
```

The script reports the exact finite-ring ground energy

```text
E0(N,g) = -sum_(m=0)^(N-1) sqrt(1 + g^2 - 2g cos((2m+1) pi/N)),
```

obtained from the antiperiodic momenta in the even Jordan-Wigner parity sector.
This is an `O(N)` analytic reference rather than exact diagonalization; see
P. Pfeuty, *Annals of Physics* **57**, 79-90 (1970),
<https://doi.org/10.1016/0003-4916(70)90270-8>.

The flat parameter layout is coarse to fine, W before U at each scale, then
brickwork round and gate position. At `L=16, chi=4`, each round has 55
parameters without the physical layer and 61 with it.

Other public helpers are:

- `binary_mera_sites` and `binary_mera_layers` for fixed-index placement;
- `tfi_hva_tensor_layer` for spatially shared W or U tensors;
- `binary_mera_qubit_initializations` for the circuit positions and indices of new qubits;
- `binary_mera_causal_cone` for conservative structural support;
- `tfi_binary_mera_channels` for direct `CircuitIR` with `CreateZero`.

`binary_mera_qubit_initializations` returns `(command_offset, qubit_indices)`
pairs, including the top input and the ancillas introduced by each layer. The
offset is where their preparation starts in the reference circuit's command
list, including barriers. These are new **qubits**, not necessarily whole MERA
sites: one coarse site can contain several qubits. A square W adds no qubits.
This helper was previously called `binary_mera_creation_boundaries`.

## Channel integration

This HVA feature branch is based on `main` after the static quantum-channel
implementation was merged.

`tfi_binary_mera_channels` constructs the merged `CircuitIR`, inserts
`CreateZero(i)` before qubit `i` is first prepared, and preserves original integer
indices. Channel execution, active-index tracking, contraction, checkpoints,
and backpropagation remain owned by the merged infrastructure.

The adapter has been exercised directly from this worktree with NumPy, JAX on
CPU, and native Triton on GPU. All agree with the independent reference on the
checked applications, preserve the expected active-qubit counts, and discard
zero terms for the final open-chain ansatz. Backend and runner changes in this
worktree are covered by their own tests.

## Validation

The compact machine-readable record is
[`hva_tmera_validation.json`](hva_tmera_validation.json).

Current checks cover:

- four-qubit gate order, barriers, position-dependent angles, and spatial sharing;
- fixed index placement and parameter counts for `chi=2,4,8`;
- unitary blocks and expanding/square isometry conditions;
- independent dense states, analytic tangents, and finite differences on small systems;
- channel metadata ordering and angle units;
- local-cell versus full energy-density gradients, checked with dense derivatives
  and untruncated NumPy SPD;
- native Triton energy/gradient agreement at `N=8` and `N=16`;
- zero discarded forward/backward terms for the final open-chain ansatz.

Run the CPU feature suite with JAX explicitly on CPU so its initialization
does not reserve GPU memory:

```bash
JAX_PLATFORMS=cpu python -m pytest \
  tests/test_hva_mera.py \
  tests/test_hva_mera_bond_dimension.py \
  tests/test_hva_mera_channels.py \
  tests/test_hva_mera_example.py \
  --ignore=tests/test_hva_mera_triton.py
```

Native Triton GPU validation is separate; JAX can stay on CPU:

```bash
JAX_PLATFORMS=cpu python -m pytest tests/test_hva_mera_triton.py
```

For `L=16, chi=4, g=1.1`, seed 7, native Triton SPD, cutoff zero, and 100
L-BFGS-B iterations:

| Rounds | Physical layer | Parameters | Total energy | Gradient inf-norm |
| ---: | --- | ---: | ---: | ---: |
| 1 | omitted | 55 | -21.1023917565 | 0.07298 |
| 1 | included | 61 | -21.3553215950 | 0.02650 |
| 2 | omitted | 110 | -21.4324774952 | 0.11564 |
| 2 | included | 122 | -21.3589965910 | 0.05106 |
| 3 | omitted | 165 | -21.3481876297 | 0.06265 |
| 3 | included | 183 | -21.3570920319 | 0.08146 |

All six optimizers reached the iteration limit; these compare useful finite
budgets rather than converged minima. Initial states are paired between the
physical-layer variants at each round count. Different round counts use
different seeded vectors.

## Planned averaged local evaluator

The phased implementation plan is in
[`hva_tmera_averaged_ascending_todo.md`](hva_tmera_averaged_ascending_todo.md).
It covers exact local averaging, shared-parameter gradients, backend integration,
and scaling measurements. This is a planned follow-up; the current example
continues to use the full Hamiltonian.

## Remaining work

No additional ansatz or channel integration code is currently required. A
full repository test run before committing is prudent, especially if main moves
again. Useful follow-up experiments are multiple seeds with converged
optimization, larger `L`, and a controlled decision on the physical layer.
Those are numerical studies rather than blockers for committing this feature.
