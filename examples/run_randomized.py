"""Small Randomized Sparse Pauli Dynamics (R-SPD) example."""

import numpy as np

import spd
from spd.circuit_ir import CircuitIR, PauliRotation


observable = spd.create_spo({"XX": 1.0}, precision="double")
circuit = CircuitIR(
    system_size=2,
    operations=(
        PauliRotation("RZ", "ZI", 0.37),
        PauliRotation("RZ", "IZ", -0.61),
    ),
)

result = spd.run_ensemble(
    observable,
    circuit,
    pauli_budget=2,
    runs=1_000,
    master_seed=7,
    basis="X",
)

exact = np.cos(0.37) * np.cos(-0.61)
print("exact expectation:", exact)
print("R-SPD mean:", result.mean)
print("standard error:", result.standard_error)
print("individual estimates:", result.estimates[:10])
print("child seeds:", result.seeds[:3])
print(
    "total randomized truncations:",
    result.diagnostics.total_randomized_truncations,
)
