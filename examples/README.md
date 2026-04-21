# Examples

This directory mixes a few polished demos with exploratory scripts that are still useful during development.

## Dependencies

Most scripts in this directory use the `pytket` frontend and therefore require the optional `pytket` dependency to be installed. The OpenQASM examples under [`open_qasm/`](open_qasm/) are the main exception.

## Recommended Starting Points

- [`run_simple_circuit_1.py`](run_simple_circuit_1.py): smallest in-code `pytket` circuit example using `create_initial_spo(...)`, `evolve(...)`, and `get_expectation_value(...)`
- [`run_simple_circuit_2.py`](run_simple_circuit_2.py): runs a stored sample circuit from [`simple_test_circuit.pkl`](simple_test_circuit.pkl)
- [`functionality/translate_spo.py`](functionality/translate_spo.py): demonstrates cyclic physical-site translation on a sparse Pauli operator
- [`open_qasm/run_openqasm_file.py`](open_qasm/run_openqasm_file.py): built-in OpenQASM parse-plus-execute example with the NumPy backend
- [`gen_simple_test_circuit.py`](gen_simple_test_circuit.py): regenerates the sample pickled `pytket` circuit used by `run_simple_circuit_2.py`

## Advanced / Exploratory Scripts

These are still intentionally kept because they are useful for inspection, comparison, or ad hoc investigation:

- [`benchmark/benchmark_2d_obc_xx_z_stepwise.py`](benchmark/benchmark_2d_obc_xx_z_stepwise.py): advanced benchmark example that reproduces the legacy SPD stepwise OBC workload using SPD internals
- [`run_forward_backward.py`](run_forward_backward.py): older forward/backward exploration with backend internals
- [`run_hard.py`](run_hard.py): heavier large-circuit run on a stored pickled circuit
- [`estimate_tfi_y_prop.py`](estimate_tfi_y_prop.py): TFI-specific exploratory script
- [`open_qasm/compare_frontends.py`](open_qasm/compare_frontends.py): advanced comparison example for the built-in OpenQASM frontend and the `pytket` import path
- [`open_qasm/test_l2_error.py`](open_qasm/test_l2_error.py): investigation script for a specific OpenQASM workflow

## Gradient / TFI Work

[`gradient/`](gradient/) is intentionally an exploratory research area for now. It contains larger TFI workflows, helpers, and saved results, and is not being treated as a polished example surface yet.

## Stored Artifacts

Several example scripts depend on checked-in data files:

- [`simple_test_circuit.pkl`](simple_test_circuit.pkl)
- [`6x6_periodic_2steps_pauliexpbox.pkl`](6x6_periodic_2steps_pauliexpbox.pkl)
- `.pkl` files under [`open_qasm/`](open_qasm/)
- `.pkl` files under [`gradient/`](gradient/)

For stable regression coverage, prefer the test suite in [`tests/`](../tests/) rather than relying on scripts in this directory.
