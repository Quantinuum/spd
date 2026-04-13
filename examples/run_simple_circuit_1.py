"""Minimal in-code `pytket` example for forward expectation evaluation."""

from pytket.circuit import Circuit

import spd

if __name__ == "__main__":
    circ = Circuit(3)
    circ.Rz(0.5, 0)
    circ.Rx(0.5, 1)
    circ.ZZPhase(0.25, 0, 2)
    circ.measure_all()
    measure_qubits = [1]
    trunc_val = 3e-5

    exp_val, final_spo = spd.run_pytket_circuit(
        circ,
        measure_qubits,
        trunc_val,
        int(1e6),
        backend_name="numpy",
    )

    print("trunc_val:", trunc_val)
    print("expectation value:", exp_val)
    print("final SPO size:", final_spo.get_size())
