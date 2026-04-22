# Examples

This directory mixes a few polished demos with exploratory scripts that are still useful during development.

## Dependencies

Most scripts in this directory use the `pytket` frontend and therefore require the optional `pytket` dependency to be installed.

## Recommended Starting Points

- [`run_simple_circuit_1.py`](run_simple_circuit_1.py): smallest in-code `pytket` circuit example using `spd.create_spo(...)`, `evolve(...)`, and `get_expectation_value(...)`
- [`gradient/run_tfi_gs_1d.py`](gradient/run_tfi_gs_1d.py): main forward + backward workflow with `create_spo(...)`, `evolve(...)`, `init_gradient_spo(...)`, and `backpropagate(...)`
- [`run_with_backend_adapter.py`](run_with_backend_adapter.py): small example with a reusable configured backend
- [`run_simple_circuit_2.py`](run_simple_circuit_2.py): runs a stored sample circuit from [`simple_test_circuit.pkl`](simple_test_circuit.pkl)
- [`functionality/translate_spo.py`](functionality/translate_spo.py): demonstrates cyclic physical-site translation on a sparse Pauli operator
- [`gen_simple_test_circuit.py`](gen_simple_test_circuit.py): regenerates the sample pickled `pytket` circuit used by `run_simple_circuit_2.py`

## Advanced / Exploratory Scripts

These are still intentionally kept because they are useful for inspection, comparison, or ad hoc investigation:

- [`run_forward_backward.py`](run_forward_backward.py): compact public-API forward/backward example
- [`open_qasm/compare_frontends.py`](open_qasm/compare_frontends.py): advanced comparison example for the built-in OpenQASM frontend and the `pytket` import path

## Gradient / TFI Work

[`gradient/`](gradient/) contains the larger TFI and AFH workflows. The scripts there are still research-oriented, but [`gradient/run_tfi_gs_1d.py`](gradient/run_tfi_gs_1d.py) is also one of the main end-to-end examples for the current SPD workflow.

## OpenQASM

[`open_qasm/run_openqasm_file.py`](open_qasm/run_openqasm_file.py) is available if you want the built-in OpenQASM frontend, but it is not the main example path in this repo.

## Stored Artifacts

Several example scripts depend on checked-in data files:

- [`simple_test_circuit.pkl`](simple_test_circuit.pkl)
- `.pkl` files under [`gradient/`](gradient/)

For stable regression coverage, prefer the test suite in [`tests/`](../tests/) rather than relying on scripts in this directory.
