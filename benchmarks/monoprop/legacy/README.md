# Historical 11x11 comparison

Archived drivers and results from the older XX+Z, center-site Z benchmark:
11x11, h=3.044382, dt=.04, 23 steps, cutoff 2^-18. These are different inputs
from the published tilted-field Ising ZZ benchmark in the parent directory.
Do not combine their timings or term counts with the new comparison.

The scripts retain their original per-step full-input warm-up policy. They now
import their shared circuit builder from `examples/` and write here by default.
The MonoProp driver is retained for reproducibility; it was not run for the new
A100 comparison. The old 32-/48-thread data are historical CPU measurements.
See `docs/monoprop_gpu_performance_handoff.md` for their original provenance.
