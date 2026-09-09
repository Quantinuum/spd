from pytket import Circuit


def gen_1d_TFI_ansatz_circuit(thetas, system_size=12, basis="+"):
    """Legacy helper retained for odd-size TFI time-evolution experiments."""
    circuit = Circuit(system_size, system_size)
    if len(thetas) % 2:
        raise ValueError("The number of parameters must be even.")
    if basis not in ("+", "0"):
        raise ValueError("basis must be '+' or '0'.")

    for layer in range(len(thetas) // 2):
        first = 2 * layer
        if basis == "0":
            for qubit in range(system_size):
                circuit.Rx(thetas[first], qubit)
            circuit.add_barrier(list(range(system_size)))

        for qubit in range(system_size):
            circuit.ZZPhase(
                thetas[first if basis == "+" else first + 1],
                qubit,
                (qubit + 1) % system_size,
            )
        circuit.add_barrier(list(range(system_size)))

        if basis == "+":
            for qubit in range(system_size):
                circuit.Rx(thetas[first + 1], qubit)
            circuit.add_barrier(list(range(system_size)))

    circuit.measure_all()
    return circuit


def gen_1d_TFI_symm_breaking_ansatz_circuit(thetas,
                                            system_size: int = 12,
                                            ) -> Circuit:
    circ = Circuit(system_size, system_size)

    assert len(thetas) % 3 == 0, "The length of thetas should be a multiple of 2."
    depth = len(thetas) // 3

    # basis == '+'
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

def gen_1d_Hamiltonian_dict(system_size, g, full=True):
    if not full:
        ham_dict = {}
        pauli_str = ['I'] * system_size
        pauli_str[0] = 'Z'
        pauli_str[1] = 'Z'
        ham_dict[''.join(pauli_str)] = -1.0

        pauli_str = ['I'] * system_size
        pauli_str[0] = 'X'
        ham_dict[''.join(pauli_str)] = -g
    else:
        ## translational invariant 1D TFI Hamiltonian
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
    pauli_str[system_size_x] = 'Z'
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

def gen_3d_Hamiltonian_dict(system_size_x, system_size_y, system_size_z, g):
    system_size = system_size_x * system_size_y * system_size_z
    ham_dict = {}
    # single term
    pauli_str = ['I'] * system_size
    pauli_str[0] = 'Z'
    pauli_str[1] = 'Z'
    ham_dict[''.join(pauli_str)] = -1.0

    pauli_str = ['I'] * system_size
    pauli_str[0] = 'Z'
    pauli_str[system_size_x] = 'Z'
    ham_dict[''.join(pauli_str)] = -1.0

    pauli_str = ['I'] * system_size
    pauli_str[0] = 'Z'
    pauli_str[system_size_x * system_size_y] = 'Z'
    ham_dict[''.join(pauli_str)] = -1.0

    pauli_str = ['I'] * system_size
    pauli_str[0] = 'X'
    ham_dict[''.join(pauli_str)] = -g
    return ham_dict
