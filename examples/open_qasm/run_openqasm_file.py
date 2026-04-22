"""Minimal built-in OpenQASM frontend example."""

from pathlib import Path

import spd
from spd.openqasm_frontend import parse_openqasm_file


if __name__ == "__main__":
    qasm_path = Path(__file__).resolve().parent / "spd_periodic_trunc5e-4_70steps_time05.qasm"
    trunc_val = 5e-4
    max_num_str = int(1e6)
    system_size, operations = parse_openqasm_file(str(qasm_path))
    initial_spo = spd.create_spo({"Z" + "I" * (system_size - 1): 1.0})

    final_spo = spd.evolve(initial_spo, operations, trunc_val=trunc_val, max_num_str=max_num_str)
    exp_val = final_spo.get_expectation_value()

    print("OpenQASM file:", qasm_path.name)
    print("Expectation value:", exp_val)
    print("Final SPO size:", final_spo.get_size())
