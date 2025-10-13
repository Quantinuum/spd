# spd

A Python package for sparse pauli dynamics simulation.
Currnetly supports only running `pytket` circuit.
Current implementation with `jax` can be run directly on GPU. 

## Features
- Sparse Pauli dynamics simulation
- [TODO] qiskit circuit
- [TODO] open qasm circuit
- [TODO] Trotterization time evolution of Hamiltonian dynamics
- [TODO] Add different backend: numpy, pytorch


### Installing
```
pip install -e . 
```


### Usage
```
import spd
spd.run_pytket_circuit(circ, measure_qubits_list=[0, 1], trunc_val=1e-3)  # < Z0Z1 >
```

