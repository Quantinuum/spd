import pickle
import pytket
from pytket import Circuit
import numpy as np
np.random.seed(0)
import spd

def gen_TFI_ansatz_circuit(theta1: float,
                           theta2: float,
                           theta3: float,
                           theta4: float,
                           system_size: int = 12
                           ) -> Circuit:
    circ = Circuit(system_size, system_size)

    for i in range(system_size):
        circ.Rx(theta1, i)

    for i in range(system_size):
        circ.ZZPhase(theta2, i, (i + 1) % system_size)

    for i in range(system_size):
        circ.Rx(theta3, i)

    for i in range(system_size):
        circ.ZZPhase(theta4, i, (i + 1) % system_size)

    for i in range(system_size):
        circ.Measure(i, i)

    return circ

def gen_Hamiltonian_dict(system_size):
    ham_dict = {}
    for i in range(system_size):
        pauli_str = ['I'] * system_size
        pauli_str[i] = 'Z'
        pauli_str[(i + 1) % system_size] = 'Z'
        ham_dict[''.join(pauli_str)] = -1.0

        pauli_str = ['I'] * system_size
        pauli_str[i] = 'X'
        ham_dict[''.join(pauli_str)] = -1.0

    return ham_dict

if __name__ == "__main__":
    system_size = 12

    random_thetas = np.random.rand(4)
    circ = gen_TFI_ansatz_circuit(random_thetas[0],
                                  random_thetas[1],
                                  random_thetas[2],
                                  random_thetas[3],
                                  system_size,
                                  )

    ham_dict = gen_Hamiltonian_dict(system_size)

    trunc_val = 3e-5
    print("\n Truncation Value:", trunc_val)

    exp_val = spd.run_pytket_circuit(circ, ham_dict, trunc_val, backend_name='numpy')
    print("\n Expectation Value:", exp_val)

    # Compute the finite difference gradient
    gradients = []

    for i in range(4):
        new_random_thetas = random_thetas.copy()
        new_random_thetas[i] += 1e-4
        new_circ = gen_TFI_ansatz_circuit(new_random_thetas[0],
                                          new_random_thetas[1],
                                          new_random_thetas[2],
                                          new_random_thetas[3],
                                          system_size,
                                          )
        new_exp_val = spd.run_pytket_circuit(new_circ, ham_dict, trunc_val, backend_name='numpy')
        print("new_exp_val = ", new_exp_val)
        gradient = (new_exp_val - exp_val) / 1e-4
        print(f" Gradient wrt theta[{i}]:", gradient)
        gradients.append(gradient)

    print("\n Finite Difference Gradients:", gradients)


