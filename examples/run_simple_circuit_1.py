import pytket
from pytket.circuit import Circuit
import sys
sys.path.append('../')
import spd

if __name__ == "__main__":
    # Example usage
    circ = Circuit(3)
    circ.Rz(0.5, 0)
    circ.Rx(0.5, 1)
    circ.ZZPhase(0.25, 0, 2)
    circ.measure_all()

    m_qubits = [1,]
    for trunc_val in [3e-5]:
        print("\n Truncation Value:", trunc_val)

        # exp_val = spd.run_pytket_circuit(circ, m_qubits, trunc_val, int(1e6), backend_name='jax')
        exp_val = spd.run_pytket_circuit(circ, m_qubits, trunc_val, int(1e6), backend_name='numpy')
        print("\n Expectation Value:", exp_val)
