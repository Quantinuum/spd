"""Run the stored sample `pytket` circuit with the public SPD API."""

import pickle
from pathlib import Path

import spd

if __name__ == "__main__":
    file_path = Path(__file__).resolve().parent / "simple_test_circuit.pkl"
    with file_path.open("rb") as handle:
        circ = pickle.load(handle)

    trunc_val = 3e-5
    initial_spo = spd.create_spo({"ZZ" + "I" * (circ.n_qubits - 2): 1.0},
                                 backend_name="numpy",
                                 )

    final_spo, info = spd.evolve(
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
    print(f"RUN INFO: sum-l1-norm-err: {info['sum_truncated_l1_norm']}, sum-l2-norm-err: {info['sum_truncated_l2_norm']}, total-l2-norm-err: {info['total_truncated_l2_norm']}")
