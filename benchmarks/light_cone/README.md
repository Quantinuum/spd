# Pruning benchmarks

## Public TFI forward/backward benchmark

`benchmark_barrier_api.py` compares no pruning, whole-circuit pruning and barrier
refresh using one public `evolve` call and one public `backpropagate` call.
It builds the circuit from `run_tfi_gs.py 2 8 1000 --linear-system-size 10` at the
default initial parameters: four variational layers, 20 barriers, 1,200 rotations,
g=3.1, seed=0, float64, cutoff=1e-8. The requested cap 10,485,760 is rounded to
16,777,216 by the existing API; it does not bind in these probes. This is not a
1,000-iteration optimization run.

Recorded NVIDIA L4 results, medians of three fresh processes per mode:

| Mode | Forward + backward | Retained rotations |
|---|---:|---:|
| Unpruned | 0.837 s | 1,200 |
| Whole-circuit cone | 0.586 s | 801 |
| Barrier refresh | 0.219 s | 248 |

All modes produced 1,154 bitwise-identical final terms and coefficients.
Initial-parameter gradients agreed within 4.4e-16. A separate probe with 3× the
initial angles produced 26,848 identical forward terms, with gradient differences
below 1.4e-10. These measurements do not establish performance or cap behavior
at arbitrary later optimizer parameters.

Timing includes public diagnostics, forward planning, CUDA synchronization,
gradient initialization and shared-parameter gradient aggregation. All modes
use `in_place=True` for forward and `progress=False`. Circuit construction,
common tiny warmup, energy readback and file output are excluded. Dumps happen
**after both timed stages** because `to_host()` compacts storage. `support_s`
measures support extraction only, not total planning.

From the repository root, use fresh output names and run modes sequentially:

```sh
python benchmarks/light_cone/benchmark_barrier_api.py --mode baseline --output /tmp/pruning-base.json --dump /tmp/pruning-base
python benchmarks/light_cone/benchmark_barrier_api.py --mode whole --output /tmp/pruning-whole.json --dump /tmp/pruning-whole
python benchmarks/light_cone/benchmark_barrier_api.py --mode barrier --output /tmp/pruning-barrier.json --dump /tmp/pruning-barrier
OPENBLAS_NUM_THREADS=1 python benchmarks/light_cone/compare_final_arrays.py /tmp/pruning-base /tmp/pruning-barrier --output /tmp/pruning-coefficients.json
```

Repeat with unique JSON names for timing statistics. Use `--scale 3` for the
larger-angle probe. The JSON files include energies, parameter gradients, support
scan counts and GPU memory, enabling numerical comparisons as well as timings.
Generated JSON, coefficient arrays and logs are local artifacts, not source files.

## Other measured workloads

With the GPU support reduction, the original diagnostics-free MonoProp loop
measured 17.472 s unpruned versus 8.046 s with per-step pruning at 12×12, and
63.936 s versus 11.540 s at 18×18 (28 steps, L4, float64, cutoff=1e-6, no cap).
Complete final coefficient maps matched in both cases. These loop timings are
not directly comparable to the diagnostics-enabled public TFI runner above.
The fixed observable indices [20,21] change distance to the boundary as lattice
size changes, so this is not a controlled system-size scaling experiment.

A single whole-circuit cone at 12×12 removed mostly cheap early gates and improved
public-runner time by only about 1.9%. Refreshing from actual intermediate support
can therefore matter more than making a single plan cheaper. Pruning reduces gate
work; it does not eliminate packed-key memory or per-gate system-size dependence.

Full implementation validation: **1,370 passed, 1 existing expected failure**.
Correctness regressions live in `tests/test_pruning.py` and
`tests/test_barrier_pruning.py`; benchmark timings are not CI thresholds.
