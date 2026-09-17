# Static channel benchmark design

The primary benchmark should measure a complete expectation-and-gradient
iteration on a layered preparation circuit. Use separate microbenchmarks to
explain where time is spent. No performance claims are established by the
correctness tests.

## Workloads

| Workload | Construction | What it measures |
| --- | --- | --- |
| Layered preparation | Fixed 64-index IR, staged creation growing 8→16→32→64, with rotations and entanglers between stages; reverse evolution contracts 64→32→16→8 | Grouped contractions, shrinking packed width, and reconstruction across boundaries |
| Repeated reset | Fixed-width entangling circuit with distinct-index reset groups between unitary segments | Row merging without width reduction and repeated checkpoints |
| Discard | Entangle active inputs, discard subsets, and measure the remaining compact outputs | Identity insertion and its transpose without contraction checkpoints |
| Controlled contraction | Generate known packed rows with specified I/Z and X/Y fractions on contracted columns and known duplicate images | Isolated packing, merging, and gradient lookup costs |

Use a small dense-solvable version of each circuit as a correctness check, then
scale the same construction. Include shared rotation parameters across channel
boundaries. Keep random seeds, angles, observables, and fixed index mappings
identical across backends. Include widths 31, 32, 33, 64, and 65 in kernel tests.

For controlled contractions, construct survivor and duplicate fractions explicitly:
fully random Pauli axes can kill nearly every row when many columns contract,
producing a misleadingly easy benchmark. Include cancellation and zero-valued
coordinates as well as nonzero merged images. Sweep 1, 4, and 16 contracted
columns and input term counts around 10^3, 10^4, 10^5, and 10^6, subject to memory.
These are starting points, not promised feasible circuit sizes.

## Execution matrix

Compare Triton and JAX on the same GPU, in separate processes. Use NumPy as a
small-case reference and CPU timing baseline. Report single and double precision
separately. Channel execution uses JAX's compact functional rotation path, so
changing the ordinary JAX rotation algorithm does not produce a distinct channel
implementation to benchmark.

Start with exact execution (cutoff zero, nonbinding term cap) at manageable
sizes. Add approximate cases with identical cutoffs and effective caps. Record
actual input/output row counts, zero-coordinate counts, active widths, and
expectation/gradient error: equal caps alone do not guarantee equal work or
support, especially with ties.

For each workload run three checkpoint configurations:

1. All snapshots fit in device memory.
2. Device budget zero, enough CPU RAM to avoid disk spill.
3. Both budgets zero, forcing disk spill.

Also measure the default budgets (4 GiB CPU, 1 GiB per device) on a case large
enough to trigger real eviction. Record checkpoint counts, retained bytes, and
bytes written to disk; report the disk type and directory. Peak process/device
memory is distinct from retained-checkpoint bytes and includes live states,
compiler/allocator reservations, and temporary arrays.

## Timing and reporting

Report cold first-evaluation time separately from warmed steady-state time.
Warm up the complete forward/backward evaluation: widths, row counts, and column
maps can each create JAX/Triton specializations. An optimizer-like sequence of
changing angles is a useful second run, since it exposes recompilation when
support sizes change. Do not assume a single fixed-angle warmup covers it.

Use wall-clock timing with synchronization before and after each measured
region (`jax.block_until_ready` for JAX arrays; CUDA synchronization for Triton).
Measure forward, gradient initialization, backward, and total iteration time.
Keep transfers and disk I/O inside the total iteration timing. Use fresh tapes
for every iteration because backward consumes them; close tapes in
expectation-only runs.

Start with 3 warmups and 10 measured iterations, reporting the median and range;
increase repeats for noisy or very short cases. Exclude random input generation
and initial observable upload from steady-state timings and report setup
separately. Record GPU, driver, library versions, precision, allocator settings,
seed, cutoff, requested/effective cap, and checkpoint budgets.

The first decision is whether channel transforms, unitary reconstruction, or
checkpoint transfers dominate total time. Optimize duplicate merging or fuse
GPU kernels only when the isolated timings explain a material end-to-end cost.
