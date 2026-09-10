# Examples

This directory mixes a few polished demos with exploratory scripts that are still useful during development.

`variational_tfi.py` demonstrates `spd.VariationalCircuit` with the stable 1D
TFI generator from `spd.ansatz`. It runs forward and backward propagation, then
sums the rotation-gate gradients into the shared layer parameters.

## Dependencies

Most scripts in this directory use the `pytket` frontend and therefore require the optional `pytket` dependency to be installed.

## Recommended Starting Points

- [`run_simple_circuit_1.py`](run_simple_circuit_1.py): smallest in-code `pytket` circuit example using `spd.create_spo(...)`, `evolve(...)`, `get_expectation_value(...)`, and truncation info
- [`gradient/run_tfi_gs.py`](gradient/run_tfi_gs.py): 1D/2D/3D TFI optimization with `VariationalCircuit`
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

[`gradient/`](gradient/) contains the larger TFI and AFH workflows. The unified
TFI runner uses the stable ansatz generators from `spd.ansatz` and reduces gate
gradients with `VariationalCircuit.parameter_gradients(...)`.

The first positional argument selects the spatial dimension. Periodic lattice
dimensions must be even. If `--linear-system-size` is omitted, the runner uses
`number_of_parameters + 2`.

```bash
python examples/gradient/run_tfi_gs.py 1 6 100 --method lbfgs --g 3.1
python examples/gradient/run_tfi_gs.py 1 6 100 --basis 0
python examples/gradient/run_tfi_gs.py 2 6 100 --linear-system-size 12
python examples/gradient/run_tfi_gs.py 3 6 100 --linear-system-size 6
python examples/gradient/run_tfi_gs.py 2 6 100 --algorithm search_update_merge
python examples/gradient/run_tfi_gs.py 1 6 100 --init-params-path previous/final_params.txt
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

## Stepwise cuPauliProp benchmark

[`benchmark_2d_obc_xx_z_cupauliprop_stepwise.py`](benchmark_2d_obc_xx_z_cupauliprop_stepwise.py)
is the cuPauliProp counterpart of
[`benchmark_2d_obc_xx_z_stepwise.py`](benchmark_2d_obc_xx_z_stepwise.py).
It uses the same numerical defaults: 11×11 OBC, dt=.04, total_t=.92,
h=3.044382, cutoff=2^-18, double precision and 23 steps. It shares the original
circuit builder to preserve gate ordering and angle conventions. No top-k cap
is applied; the original default cap of 1e9 does not bind on this problem.

Install `cuquantum-python-cu12==26.6.0` and `cupy-cuda12x==14.2.0` in the
benchmark environment, alongside SPD and pytket. From the repository root:

```sh
python examples/benchmark_2d_obc_xx_z_cupauliprop_stepwise.py --output-dir /tmp/cupauliprop
# In this Studio, the external packages live in a separate target directory:
PYTHONPATH=/tmp/spd-m5-packages:$PWD \
  python examples/benchmark_2d_obc_xx_z_cupauliprop_stepwise.py --allocator async --output-dir /tmp/cupauliprop-async
```

The script defaults to `--allocator async`, which completed all 23 steps on
the H100. `--allocator default` selects the usual CuPy caching pool, which ran
out of memory during step 23. Async was about 20% slower over the shared 22
steps; the full-run file consistently uses async for every step. Plot the two
policies as separate series. `--memory-limit` controls cuPauliProp's scratch
budget (default `80%`); it is not a hard bound on total device memory.
All numerical parameters remain unchanged. Both settings are saved in metadata.

The output filename is
`cupauliprop_benchmark_data_dt_0.04_total_t_0.92_threshold_log_18.pkl`.
Its six fields exactly match the original plotting format:

- `num_paulis`, `times`, `norms`: one entry per completed step; norms are square roots, not squared norms.
- `all_results`: initial expectation followed by each completed step.
- `avg_num_paulis`, `avg_speeds`: begin at step 2, as in the original script.
  Throughput is boundary-average support × 341 / seconds, not an instrumented count of intermediate rows.

Each step warms once from the same input, then times a synchronized second pass.
Observations and checkpoint writes are outside that timer. The pickle is updated
after every completed step, so an interrupted run remains usable. Versions,
configuration and timing policy live in a separate `.metadata.json` file without
changing the plotting dictionary. `--num-steps` supports bounded runs.

[Generated default-parameter runs](../benchmarks/results/cupauliprop_default/)
and [`compare_stepwise_results.py`](compare_stepwise_results.py) provide numeric
checks against the original Triton results.

For the separate Triton retention issue, see the
[allocator investigation and measured launch setting](../benchmarks/ALLOCATOR_RETENTION.md).
