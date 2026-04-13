"""Generate the stored sample circuit used by `run_simple_circuit_2.py`."""

import pickle
from pathlib import Path

from pytket import Circuit
import numpy as np

np.random.seed(0)


def gen_simple_test_circuit() -> Circuit:
    circ = Circuit(64, 64)

    for i in range(64):
        circ.Rx(np.random.rand(), i)

    for i in range(64):
        q1 = np.random.randint(0, 63)
        q2 = (q1 + np.random.randint(1, 63)) % 64
        circ.ZZPhase(np.random.rand(), q1, q2)

    for i in range(63):
        circ.H(i)

    for i in range(64):
        q1 = np.random.randint(0, 63)
        q2 = (q1 + np.random.randint(1, 63)) % 64
        circ.ZZPhase(np.random.rand(), q1, q2)

    for i in range(64):
        circ.Rx(np.random.rand(), i)

    for i in range(64):
        q1 = np.random.randint(0, 63)
        q2 = (q1 + np.random.randint(1, 63)) % 64
        circ.ZZPhase(np.random.rand(), q1, q2)
        # circ.ZZPhase(np.random.rand(), i, (i + 1) % 64)


    for i in range(64):
        circ.Rx(np.random.rand(), i)

    for i in range(64):
        circ.Measure(i, i)

    return circ


if __name__ == "__main__":
    circ = gen_simple_test_circuit()
    output_path = Path(__file__).resolve().parent / "simple_test_circuit.pkl"
    with output_path.open("wb") as handle:
        pickle.dump(circ, handle)
    print(f"Circuit saved to {output_path}")
