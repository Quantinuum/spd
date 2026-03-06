import pytket
from pytket.circuit import Circuit
import sys
sys.path.append('../')
import spd

if __name__ == "__main__":
    import pickle
    file_path = 'simple_test_circuit.pkl'
    circ = pickle.load(open(file_path, 'rb'))

    m_qubits = [0, 1]
    for trunc_val in [3e-5]:
        print("\n Truncation Value:", trunc_val)

        exp_val, spo = spd.run_pytket_circuit(circ, m_qubits, trunc_val, backend_name='jax')
        print("\n Expectation Value:", exp_val)
