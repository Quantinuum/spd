"""Run the stored sample `pytket` circuit with the public SPD API."""

import pickle
from pathlib import Path

import spd

if __name__ == "__main__":
    file_path = Path(__file__).resolve().parent / "simple_test_circuit.pkl"
    with file_path.open("rb") as handle:
        circ = pickle.load(handle)

    trunc_val = 3e-5
    backend = spd.BackendAdapter.from_name("numpy", packbit=32)
    initial_spo = backend.create_initial_spo({"ZZ": 1.0})

    final_spo = spd.evolve(
        initial_spo,
        circ,
        trunc_val,
        int(1e6),
    )
    exp_val = final_spo.get_expectation_value()

    print("circuit file:", file_path.name)
    print("trunc_val:", trunc_val)
    print("expectation value:", exp_val)
    print("final SPO size:", final_spo.get_size())
