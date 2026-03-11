import pytket
from pytket import Circuit

def gen_1d_TFI_ansatz_circuit(thetas,
                              system_size: int = 12
                              ) -> Circuit:
    circ = Circuit(system_size, system_size)

    assert len(thetas) % 3 == 0, "The length of thetas should be a multiple of 3."
    depth = len(thetas) // 3

    for d in range(depth):
        for i in range(system_size):
            circ.ZZPhase(thetas[3*d], i, (i + 1) % system_size)

        circ.add_barrier(list(range(system_size)), ) # add a barrier on all qubits and bits

        for i in range(system_size):
            circ.Rx(thetas[3*d + 1], i)

        circ.add_barrier(list(range(system_size)), ) # add a barrier on all qubits and bits

        for i in range(system_size):
            circ.Rz(thetas[3*d + 2], i)

        circ.add_barrier(list(range(system_size)), ) # add a barrier on all qubits and bits

    for i in range(system_size):
        circ.Measure(i, i)

    return circ

def gen_2d_TFI_ansatz_circuit(thetas,
                              system_size_x: int = 4,
                              system_size_y: int = 4,
                              ) -> Circuit:
    system_size = system_size_x * system_size_y
    circ = Circuit(system_size, system_size)

    depth = len(thetas) // 3

    for d in range(depth):
        for i in range(system_size):
            circ.Rz(thetas[3*d], i)

        circ.add_barrier(list(range(system_size)), )

        for i in range(system_size):
            circ.Rx(thetas[3*d+1], i)

        circ.add_barrier(list(range(system_size)), )

        for x in range(system_size_x):
            for y in range(system_size_y):
                i = x * system_size_y + y
                j = ((x + 1) % system_size_x) * system_size_y + y
                circ.ZZPhase(thetas[3*d+2], i, j)

        for x in range(system_size_x):
            for y in range(system_size_y):
                i = x * system_size_y + y
                j = x * system_size_y + (y + 1) % system_size_y
                circ.ZZPhase(thetas[3*d+2], i, j)

        circ.add_barrier(list(range(system_size)), )
        # # alternative ansatz

        # for i in range(system_size):
        #     circ.Rx(thetas[2*d], i)

        # circ.add_barrier(list(range(system_size)), )

        # for x in range(system_size_x):
        #     for y in range(system_size_y):
        #         i = x * system_size_y + y
        #         j = ((x + 1) % system_size_x) * system_size_y + y
        #         circ.ZZPhase(thetas[2*d+1], i, j)

        # for x in range(system_size_x):
        #     for y in range(system_size_y):
        #         i = x * system_size_y + y
        #         j = x * system_size_y + (y + 1) % system_size_y
        #         circ.ZZPhase(thetas[2*d+1], i, j)

        # circ.add_barrier(list(range(system_size)), )

    for i in range(system_size):
        circ.Measure(i, i)

    return circ

def gen_1d_Hamiltonian_dict(system_size, g):
    ham_dict = {}
    for i in range(system_size):
        pauli_str = ['I'] * system_size
        pauli_str[i] = 'Z'
        pauli_str[(i + 1) % system_size] = 'Z'
        ham_dict[''.join(pauli_str)] = -1.0

        pauli_str = ['I'] * system_size
        pauli_str[i] = 'X'
        ham_dict[''.join(pauli_str)] = -g

    return ham_dict

def gen_2d_Hamiltonian_dict(system_size_x, system_size_y, g):
    system_size = system_size_x * system_size_y
    ham_dict = {}
    # single term
    pauli_str = ['I'] * system_size
    pauli_str[0] = 'Z'
    pauli_str[1] = 'Z'
    ham_dict[''.join(pauli_str)] = -1.0

    pauli_str = ['I'] * system_size
    pauli_str[0] = 'Z'
    pauli_str[system_size_y] = 'Z'
    ham_dict[''.join(pauli_str)] = -1.0

    pauli_str = ['I'] * system_size
    pauli_str[0] = 'X'
    ham_dict[''.join(pauli_str)] = -g
    return ham_dict

    # translational invariant 2D TFI Hamiltonian
    for x in range(system_size_x):
        for y in range(system_size_y):
            i = x * system_size_y + y

            pauli_str = ['I'] * system_size
            pauli_str[i] = 'X'
            ham_dict[''.join(pauli_str)] = -g

            j = ((x + 1) % system_size_x) * system_size_y + y
            pauli_str = ['I'] * system_size
            pauli_str[i] = 'Z'
            pauli_str[j] = 'Z'
            ham_dict[''.join(pauli_str)] = -1.0

            j = x * system_size_y + (y + 1) % system_size_y
            pauli_str = ['I'] * system_size
            pauli_str[i] = 'Z'
            pauli_str[j] = 'Z'
            ham_dict[''.join(pauli_str)] = -1.0

    return ham_dict

