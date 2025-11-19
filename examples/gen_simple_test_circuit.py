import pickle
import pytket
from pytket import Circuit
import numpy as np
np.random.seed(0)

def gen_simple_test_circuit() -> Circuit:
    circ = Circuit(64, 64)

    for i in range(64):
        circ.Rx(np.random.rand(), i)

    # for i in range(64):
    #     q1 = np.random.randint(0, 63)
    #     q2 = (q1 + np.random.randint(1, 63)) % 64
    #     circ.ZZPhase(np.random.rand(), q1, q2)

    # for i in range(63):
    #     circ.H(i)

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
    with open("simple_test_circuit.pkl", "wb") as f:
        pickle.dump(circ, f)
    print("Circuit saved to simple_test_circuit.pkl")
