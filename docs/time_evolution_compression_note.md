# Time-Evolution Compression Note

## Outcome
An exploratory time-evolution compression workflow now lives under
[`examples/time_evolution/`](../examples/time_evolution/).

The workflow uses SPD's existing forward and backward public runners to:
- precompute cached target evolved operators for representative single-site
  `X` and `Z`
- optimize a shallower variational circuit against those cached targets using
  the local operator L2 cost
- reload saved optimization results and recompute the channel-wise cost split
  together with a direct-Trotter baseline comparison

This work stays at the example layer and does not introduce new `spd/` public
APIs.

## Implemented Scripts

### Shared helper
- [`examples/time_evolution/common.py`](../examples/time_evolution/common.py)
  centralizes:
  - benchmark constants
  - target/optimization artifact paths
  - 2D TFI circuit builders
  - metadata helpers
  - translationally invariant gradient compression helpers

### Constant-TFI compression example
- [`precompute_target_tfi_2d.py`](../examples/time_evolution/precompute_target_tfi_2d.py)
- [`optimize_tfi_2d_compression.py`](../examples/time_evolution/optimize_tfi_2d_compression.py)

Current setup:
- system size `4 x 4`
- target circuit: first-order Trotter, total time `0.3`, `100` steps
- variational circuit: first-order Trotter, `5` layers

### Linear-ramp compression example
- [`precompute_target_tfi_ramp_2d.py`](../examples/time_evolution/precompute_target_tfi_ramp_2d.py)
- [`optimize_tfi_ramp_compression.py`](../examples/time_evolution/optimize_tfi_ramp_compression.py)

Current setup:
- system size `14 x 14`
- target circuit: linear ramp in `g` from `0.0` to `3.2`
- target circuit style: second-order Trotter, total time `0.06`, `6` layers
- variational circuit: first-order Trotter, `2` layers

### Result inspection
- [`check_result.py`](../examples/time_evolution/check_result.py)

This script reloads:
- target metadata
- cached target `X` / `Z` SPOs
- saved optimization result

and then prints:
- recomputed total cost
- `X` and `Z` channel costs
- direct-Trotter baseline cost
- parameter vectors for both the baseline and the optimized circuit

## Important Implementation Notes
- The compression cost is the SPD-friendly local operator cost:
  `||X - X_target||^2 + ||Z - Z_target||^2`
  using one representative site per channel under translational invariance.
- Raw SPD gradients are reduced back to shared layer parameters in the example
  code by summing all sitewise gate contributions belonging to the same shared
  parameter, then converting from SPD's internal angle convention back to
  pytket's parameterization by a factor of `pi`.
- The constant and ramp examples now carry separate lattice-size settings in
  `common.py`, so changing the ramp benchmark size no longer changes the
  original constant-TFI example.
- The example scripts suppress SPD's internal per-gate progress printing during
  normal forward/backward calls so the console output stays readable.

## Validation
- The constant-TFI example was run through target precomputation and
  optimization, and its saved result can be checked with `check_result.py`.
- The linear-ramp example was run through target precomputation and
  optimization, and its saved result can also be checked with
  `check_result.py --scenario ramp`.
- `check_result.py` currently reports both:
  - recomputed saved-result cost
  - direct-Trotter baseline cost

## Status
- [x] constant-TFI compression example added
- [x] linear-ramp compression example added
- [x] cached target metadata and target SPO artifacts added
- [x] saved-result inspection script added
- [x] direct-Trotter baseline comparison added
- [x] constant and ramp benchmark sizes separated

## Follow-Up
Likely next steps for this example area:
- try alternate local cost functions in `check_result.py`
- raise optimization iteration limits for heavier ramp runs when needed
- extend the ansatz beyond full translational invariance, for example to a
  `2 x 2` unit cell pattern
