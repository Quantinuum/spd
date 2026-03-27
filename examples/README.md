# Examples

This directory contains exploratory scripts, small demos, and larger gradient-oriented experiments.

## Good Starting Points

- [`run_simple_circuit_1.py`](run_simple_circuit_1.py): small forward example built directly in code
- [`run_forward_backward.py`](run_forward_backward.py): older forward/backward exploration script

## Other Scripts

- [`run_simple_circuit_2.py`](run_simple_circuit_2.py): runs a pickled example circuit
- [`run_hard.py`](run_hard.py): heavier example intended for larger runs
- [`estimate_tfi_y_prop.py`](estimate_tfi_y_prop.py): specific TFI-related experiment
- [`gen_simple_test_circuit.py`](gen_simple_test_circuit.py): generates a stored example circuit

## Gradient / TFI Subdirectory

[`gradient/`](gradient/) contains larger experiments and utilities around transverse-field Ising model workflows.

Main files:

- [`gradient/tfi_setup.py`](gradient/tfi_setup.py): circuit and Hamiltonian construction helpers
- [`gradient/test_tfi_gradient.py`](gradient/test_tfi_gradient.py): experiment-style gradient verification script
- [`gradient/run_tfi_gs_1d.py`](gradient/run_tfi_gs_1d.py): 1D workflow
- [`gradient/run_tfi_gs_2d.py`](gradient/run_tfi_gs_2d.py): 2D workflow

## Data / Generated Artifacts

These files are stored artifacts rather than hand-authored examples:

- [`simple_test_circuit.pkl`](simple_test_circuit.pkl)
- [`6x6_periodic_2steps_pauliexpbox.pkl`](6x6_periodic_2steps_pauliexpbox.pkl)
- gradient `.pkl` files under [`examples/gradient/`](gradient/)

For stable regression coverage, prefer the `tests/` directory rather than relying on these scripts directly.
