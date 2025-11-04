import pytket
from pytket.circuit import Circuit
import time
import sys
sys.path.append('../')
import spd
import pickle
import numpy as np

if __name__ == "__main__":
    # Example usage
    circ = Circuit(3)
    circ.Rz(0.5, 0)
    circ.Rx(1.0, 1)
    circ.ZZPhase(0.25, 0, 2)
    circ.measure_all()

    file_path = 'phasecraft_hard.pkl'

    circ = pickle.load(open(file_path, 'rb'))

    measurement_dict = {}
    for x in range(6):
        for y in range(6):
            i = y * 6 + x
            measurement_dict[(i,)] = np.power(-1, (x+y)) / 72
            measurement_dict[(i+36,)] = -np.power(-1, (x+y)) / 72

    print(measurement_dict)
    # m_qubits = [0,]

    for trunc_val in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
    # for trunc_val in [3e-6, 1e-6, 3e-7, 1e-7]:
        t0 = time.time()
        print("\n Truncation Value:", trunc_val)

        exp_val = spd.run_pytket_circuit(circ, measurement_dict,
                                         trunc_val)
        print("\n Expectation Value:", exp_val)
        print("Time:", time.time() - t0, "s")
