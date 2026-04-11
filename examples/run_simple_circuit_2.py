"""Run the stored sample `pytket` circuit with the public SPD API."""

import pickle
from pathlib import Path

import spd

if __name__ == "__main__":
    file_path = Path(__file__).resolve().parent / "simple_test_circuit.pkl"
    with file_path.open("rb") as handle:
        circ = pickle.load(handle)

    m_qubits = [0, 1]
    trunc_val = 3e-5

    exp_val, final_spo = spd.run_pytket_circuit(
        circ,
        m_qubits,
        trunc_val,
        int(1e6),
        backend_name="numpy",
    )

    print("circuit file:", file_path.name)
    print("trunc_val:", trunc_val)
    print("expectation value:", exp_val)
    print("final SPO size:", final_spo.get_size())
