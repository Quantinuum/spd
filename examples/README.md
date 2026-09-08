# Examples

`run_randomized.py` demonstrates the NumPy R-SPD API. It runs independent
fixed-budget realizations and prints the ensemble mean, sample variance-derived
standard error, reproducible child seeds, and accumulated randomized-truncation count.

`run_residual_corrected.py` demonstrates the separate residual-corrected
R-SPD API. It combines a deterministic top-`K_d` backbone with an independently
sampled pivotal top-`K_c` correction and reports both estimator components.

`benchmark_residual_corrected_2d_obc_xx_z.py` runs the appendable 2D OBC XX+Z
benchmark. For the initial 11x11, eight-step, `K_d=1,000`, `K_c=1,000` test:

```bash
python examples/benchmark_residual_corrected_2d_obc_xx_z.py \
    --n 11 --num-steps 8 \
    --backbone-budget 1000 --correction-budget 1000 \
    --runs 4 --workers 1
```

After inspecting four runs, increase the requested ensemble without rerunning
them:

```bash
python examples/benchmark_residual_corrected_2d_obc_xx_z.py \
    --n 11 --num-steps 8 \
    --backbone-budget 1000 --correction-budget 1000 \
    --runs 8 --workers 2
```

The second command starts only run indices 4 through 7. Each completed run is
stored as `run_XXXXXX.pkl`; `statistics.csv` reports prefix ensembles with
`R=1,2,4,...`. Seeds depend only on the master seed and run index, so changing
`--workers` does not change existing or future samples. A realization runs its
full eight-step circuit in one call; if interrupted before it is saved, only
that realization must restart.

For Phase 3 tail and per-step diagnostics, collect an existing result directory
against the Figure 7 deterministic reference:

```bash
python results/get_residual_corrected_2d_obc_xx_z_statistics.py \
    --input-dir results/residual_corrected_2d_obc_xx_z_n11_Kd1000_Kc10000_steps8_seed20260818 \
    --reference ../paper/fig_7_throughput_benchmark/jax_search_update_merge_donate_benchmark_data_dt_0.04_total_t_0.92_threshold_log_18.pkl
```

This writes `phase3_ensemble.csv`, `phase3_steps.csv`, and
`phase3_statistics.pkl` alongside the completed runs.  The complementary
fixed-total-support pilot uses:

```bash
python examples/benchmark_residual_corrected_2d_obc_xx_z.py \
    --n 11 --num-steps 8 \
    --backbone-budget 10000 --correction-budget 1000 \
    --runs 2 --workers 1
```

The two completed runs had zero terminal correction overlap and large
correction norms, so this allocation should not be extended.  The final
large-budget pilot used:

```bash
python examples/benchmark_residual_corrected_2d_obc_xx_z.py \
    --n 11 --num-steps 8 \
    --backbone-budget 10000 --correction-budget 10000 \
    --runs 8 --workers 1
```

Its correction norm at step 8 was nearly the same as the existing
`K_d=1,000`, `K_c=10,000` result despite the larger backbone.  Phase 3 is
therefore complete; do not extend the persistent-correction benchmark to step
9 before testing the time-blocked Phase 4 estimator.

`benchmark_time_blocked_residual_2d_obc_xx_z.py` runs the separate Phase 4
pilot. It uses one reverse-Heisenberg Trotter step per block and stores one
deterministic boundary-backbone checkpoint plus one file per block run. The
smallest useful 11x11 pilot is:

```bash
python examples/benchmark_time_blocked_residual_2d_obc_xx_z.py \
    --n 11 --num-steps 8 \
    --backbone-budget 1000 --correction-budget 1000 \
    --pilot-runs 4
```

Increasing `--pilot-runs` appends only missing `(block, run)` pairs. The pilot
writes `pilot_blocks.csv` with second moments, hit fractions, norm and
heavy-core diagnostics. It withholds any production allocation when a nonempty
block has zero-hit false convergence. See
`docs/time_blocked_residual_correction.md` for the identity, seed hierarchy,
and checkpoint schema.

Use `--blocks 5,6` for a targeted budget-scaling pilot. Block indices follow
the reverse-Heisenberg order and can be added later without changing existing
seeds or checkpoint files.

`benchmark_rspd_2d_obc_xx_z_stepwise.py` is the harder R-SPD counterpart to the
Figure 7 stepwise 2D OBC XX+Z throughput benchmark.
Each run proceeds through all timesteps independently and is saved in its own
pickle. Progress is checkpointed after every timestep, so an interrupted run
resumes from its last completed step:

```bash
python examples/benchmark_rspd_2d_obc_xx_z_stepwise.py \
    --n 11 --num-steps 8 \
    --pauli-budget 100000 \
    --runs 8 \
    --workers 4 \
    --output-dir results/fpspp_n11_K100000_steps8
```

`--runs R` means “ensure run indices `0` through `R-1` exist.” Re-running the
command with `--runs 16` skips the first 8 completed runs and adds indices 8
through 15. Alternatively, `--additional-runs 8` appends eight indices after
the largest completed or in-progress index. `--run-index I` executes one stable
run index and is useful with an external scheduler. Seeds depend only on the
master seed, run index, and timestep, so changing the requested ensemble size or
worker count does not change existing runs.

Every completed `trajectory_XXXXXX.pkl` contains one R-SPD run's expectation, support,
norm, runtime, throughput work, randomized-truncation count, predicted
randomized-truncation MSE, raw/merged child counts, and full truncation
diagnostics at every timestep.
The historical filename is retained so existing result directories stay
resumable. The larger sampled SPO is retained only in `.progress` while a run
is incomplete.

For a quick environment check before a larger run, use:

```bash
python examples/benchmark_rspd_2d_obc_xx_z_stepwise.py \
    --n 3 --num-steps 1 --pauli-budget 100 --runs 2 \
    --output-dir results/fpspp_smoke
```

Each worker can hold a full `K`-string sampled SPO, so increase `--workers` only
when memory allows. Each worker finishes one whole run before starting
another.

After any number of runs have completed, aggregate them and compare
them with the deterministic Figure 7 values using the separate collector:

```bash
python results/get_rspd_2d_obc_xx_z_statistics.py \
    --input-dir results/fpspp_n11_K100000_steps8 \
    --reference ../paper/fig_7_throughput_benchmark/jax_search_update_merge_donate_benchmark_data_dt_0.04_total_t_0.92_threshold_log_18.pkl
```

The collector writes `statistics.pkl` and `statistics.csv`. By default it
reports prefix ensembles of size `1, 2, 4, ..., R`, including the full set of
currently completed runs. Use `--run-counts 4,8,16,32` to pick specific sample
counts. Re-run it after adding runs; it never modifies
the individual samples.

This directory mixes a few polished demos with exploratory scripts that are still useful during development.

`variational_tfi.py` demonstrates `spd.VariationalCircuit` with the stable 1D
TFI generator from `spd.ansatz`. It runs forward and backward propagation, then
sums the rotation-gate gradients into the shared layer parameters.

## Dependencies

Most scripts in this directory use the `pytket` frontend and therefore require the optional `pytket` dependency to be installed.

## Recommended Starting Points

- [`run_simple_circuit_1.py`](run_simple_circuit_1.py): smallest in-code `pytket` circuit example using `spd.create_spo(...)`, `evolve(...)`, `get_expectation_value(...)`, and truncation info
- [`gradient/run_tfi_gs_1d.py`](gradient/run_tfi_gs_1d.py): main forward + backward workflow with `create_spo(...)`, `evolve(...)`, `init_gradient_spo(...)`, and `backpropagate(...)`
- [`run_with_backend_adapter.py`](run_with_backend_adapter.py): small example with a reusable configured backend
- [`tfi_noise_susceptibility.py`](tfi_noise_susceptibility.py): operation-aligned one- and two-qubit depolarizing susceptibilities for a short TFI Trotter circuit
- [`run_simple_circuit_2.py`](run_simple_circuit_2.py): runs a stored sample circuit from [`simple_test_circuit.pkl`](simple_test_circuit.pkl)
- [`functionality/translate_spo.py`](functionality/translate_spo.py): demonstrates cyclic physical-site translation on a sparse Pauli operator
- [`gen_simple_test_circuit.py`](gen_simple_test_circuit.py): regenerates the sample pickled `pytket` circuit used by `run_simple_circuit_2.py`

## Advanced / Exploratory Scripts

These are still intentionally kept because they are useful for inspection, comparison, or ad hoc investigation:

- [`run_forward_backward.py`](run_forward_backward.py): compact public-API forward/backward example
- [`benchmark_jax_memory_donation.py`](benchmark_jax_memory_donation.py): large-SPO JAX memory benchmark for forward/backward paths and experimental buffer-donation wrappers
- [`open_qasm/compare_frontends.py`](open_qasm/compare_frontends.py): advanced comparison example for the built-in OpenQASM frontend and the `pytket` import path

## Gradient / TFI Work

[`gradient/`](gradient/) contains the larger TFI and AFH workflows. The scripts there are still research-oriented, but [`gradient/run_tfi_gs_1d.py`](gradient/run_tfi_gs_1d.py) is also one of the main end-to-end examples for the current SPD workflow.

The gradient scripts use positional arguments for model size and iteration count, plus shared optional flags:

```bash
python examples/gradient/run_tfi_gs_1d.py 6 3.1 + 100 --method lbfgs
python examples/gradient/run_tfi_gs_1d.py 6 3.1 + 100 --init-params-path previous/final_params.txt
python examples/gradient/run_tfi_gs_1d.py 6 3.1 + 100 --lambda-ose 0.1
python examples/gradient/run_tfi_gs_1d.py 6 3.1 + 100 --system-size 15
python examples/gradient/run_tfi_gs_2d.py 6 100 --linear-system-size 12
python examples/gradient/run_tfi_gs_3d.py 6 100 --linear-system-size 6
python examples/gradient/run_tfi_gs_2d.py 6 100 --algorithm search_update_merge
```

If `--init-params-path` is omitted, parameters are initialized randomly. If it is provided, the script initializes from that file. `lambda_ose` is constant within one training run and is stored in `metadata.json`. To decrease it, start a new run from the previous `final_params.txt` with a smaller `--lambda-ose`.

The JAX algorithm defaults to `stack_sort_merge`. The gradient scripts also accept `--algorithm search_update_merge` for large-run experiments. Each run prints a simple JAX storage estimate before the first evaluation and records `elapsed_s` in `evals.csv` and `history.csv`.

## OpenQASM

[`open_qasm/run_openqasm_file.py`](open_qasm/run_openqasm_file.py) is available if you want the built-in OpenQASM frontend, but it is not the main example path in this repo.

## Stored Artifacts

Several example scripts depend on checked-in data files:

- [`simple_test_circuit.pkl`](simple_test_circuit.pkl)
- `.pkl` files under [`gradient/`](gradient/)

For stable regression coverage, prefer the test suite in [`tests/`](../tests/) rather than relying on scripts in this directory.
