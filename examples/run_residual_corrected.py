"""Run a small residual-corrected R-SPD ensemble."""

import numpy as np

import spd
from spd.circuit_ir import CircuitIR, PauliRotation


def main():
    depth = 6
    angle = 0.3
    observable = spd.create_spo({"X": 1.0}, precision="double")
    circuit = CircuitIR(
        1,
        tuple(PauliRotation("RZ", "Z", angle) for _ in range(depth)),
    )

    result = spd.run_residual_corrected_ensemble(
        observable,
        circuit,
        backbone_budget=1,
        correction_budget=1,
        runs=2_000,
        master_seed=7,
        basis="X",
    )

    print("exact expectation:", float(np.cos(depth * angle)))
    print("backbone expectation:", result.backbone_estimate)
    print("correction mean:", result.correction_mean)
    print("corrected mean:", result.mean)
    print("standard error:", result.standard_error)
    print("first five seeds:", result.seeds[:5])


if __name__ == "__main__":
    main()
