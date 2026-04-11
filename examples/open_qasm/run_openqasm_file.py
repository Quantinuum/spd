"""Minimal built-in OpenQASM frontend example."""

from pathlib import Path

import spd


if __name__ == "__main__":
    qasm_path = Path(__file__).resolve().parent / "spd_periodic_trunc5e-4_70steps_time05.qasm"
    measure_qubits = [0]
    trunc_val = 5e-4
    max_num_str = int(1e6)

    exp_val, final_spo = spd.run_openqasm_file(
        str(qasm_path),
        measure_qubits,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        backend_name="numpy",
    )

    print("OpenQASM file:", qasm_path.name)
    print("Expectation value:", exp_val)
    print("Final SPO size:", final_spo.get_size())
