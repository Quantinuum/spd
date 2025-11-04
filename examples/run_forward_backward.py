import pytket
from pytket.circuit import Circuit
import sys
sys.path.append('../')
import spd

import spd.jax_backend as backend
from spd.run_circuit import _ROT_DISPATCH, _CLIFFORD_FUNC_DISPATCH, parse_pauli_theta
import time
import jax.numpy as jnp
import psutil

def run_pytket_circuit_forward(circ, measure_qubits_list, trunc_val, packbit=32, loggin=True):
    # [TODO] Separate the backend.run part from the parse circuit part.

    from pytket.passes import DecomposeBoxes, AutoRebase
    from pytket.circuit import OpType

    total_start_time = time.time()
    system_size = circ.n_qubits
    system_size = packbit * ((system_size + packbit - 1) // packbit)  # pad to multiple of packbit
    print("SYSTEM SIZE (PADDED):", system_size)
    commands = circ.get_commands()
    total_num_gate = len(commands)

    if packbit == 8:
        pauli_str_to_uint = backend.pauli_str_to_uint8
    elif packbit == 32:
        pauli_str_to_uint = backend.pauli_str_to_uint32
    elif packbit == 64:
        raise NotImplementedError("64-bit has some bug in the code. It cannot generate correct result")
        pauli_str_to_uint = backend.pauli_str_to_uint64
    else:
        raise ValueError("packbit must be 8 or 32")


    measure_Zs = ''.join(['Z' if i in measure_qubits_list else 'I' for i in range(system_size)])
    xz_array = pauli_str_to_uint(measure_Zs).reshape([1, -1])
    c_array = jnp.ones((1,), dtype=jnp.complex64)
    print("The initial xz_array: ", xz_array)
    print("The intial c_array: ", c_array)

    merge_pauli = backend.merge_and_pad
    max_num_string = 0

    for command_idx, command in enumerate(commands[::-1]):
        t0 = time.time()

        if command.op.type in _ROT_DISPATCH:
            P, theta = parse_pauli_theta(command, system_size)
            xzk = pauli_str_to_uint(P)
            # Parsing the rotation: u = exp(-i * theta * P)

            xz_array_1, c_array_1, xz_array_2, c_array_2 = backend.conjugated_pauli_batched_uint_(xz_array, c_array, xzk, theta)
            xz_array, c_array, num_string = merge_pauli(xz_array_1, c_array_1,
                                                        xz_array_2, c_array_2,
                                                        trunc_val=trunc_val,
                                                        )
            max_num_string = max(max_num_string, num_string)
        elif command.op.type in [OpType.H, OpType.S, OpType.Sdg]:
            func = _CLIFFORD_FUNC_DISPATCH[command.op.type]
            qubit = command.args[0].index[0]
            xz_array, c_array = func(xz_array, c_array, qubit)
        elif command.op.type in [OpType.CX, OpType.CZ]:
            func = _CLIFFORD_FUNC_DISPATCH[command.op.type]
            control_qubit = command.args[0].index[0]
            target_qubit = command.args[1].index[0]
            xz_array, c_array = func(xz_array, c_array, control_qubit, target_qubit)
        elif command.op.type in [OpType.Measure, OpType.Barrier]:
            # print("Skipping measurement/barrier")
            continue
        else:
            raise ValueError(f"Unsupported gate type: {command.op.type}")

        t1 = time.time()

        weight_left = jnp.linalg.norm(c_array) ** 2

        current_row_size = len(c_array)
        process = psutil.Process()
        print(command.op.type,
              weight_left,
              process.memory_info().rss / 1e6, "MB", f"Size: {current_row_size}",
              "Progress: {:.2f}%".format(100 * (command_idx+1) / total_num_gate),
              "gates:", (command_idx+1), "/", total_num_gate - (command_idx+1),
              "M rows/s:", current_row_size / (t1 - t0) / 1e6,
              "Time:", "{:.2f}".format(t1 - t0),
              "Total Time:", "{:.2f}s".format(time.time() - total_start_time),
              "--------",
              end='\r')

    exp_val = backend.get_expectation_value(xz_array, c_array)
    return exp_val, xz_array, c_array


def run_pytket_circuit_backward(circ, xz_array, c_array, trunc_val, packbit=32, loggin=True):
    # [TODO] Separate the backend.run part from the parse circuit part.

    from pytket.passes import DecomposeBoxes, AutoRebase
    from pytket.circuit import OpType

    total_start_time = time.time()
    system_size = circ.n_qubits
    system_size = packbit * ((system_size + packbit - 1) // packbit)  # pad to multiple of packbit
    print("SYSTEM SIZE (PADDED):", system_size)
    commands = circ.get_commands()
    total_num_gate = len(commands)

    if packbit == 8:
        pauli_str_to_uint = backend.pauli_str_to_uint8
    elif packbit == 32:
        pauli_str_to_uint = backend.pauli_str_to_uint32
    elif packbit == 64:
        raise NotImplementedError("64-bit has some bug in the code. It cannot generate correct result")
        pauli_str_to_uint = backend.pauli_str_to_uint64
    else:
        raise ValueError("packbit must be 8 or 32")


    merge_pauli = backend.merge_and_pad
    max_num_string = 0

    for command_idx, command in enumerate(commands):
        t0 = time.time()

        if command.op.type in _ROT_DISPATCH:
            P, theta = parse_pauli_theta(command, system_size)
            xzk = pauli_str_to_uint(P)
            # Parsing the rotation: u = exp(-i * theta * P)

            xz_array_1, c_array_1, xz_array_2, c_array_2 = backend.conjugated_pauli_batched_uint_(xz_array, c_array, xzk, -theta)
            xz_array, c_array, num_string = merge_pauli(xz_array_1, c_array_1,
                                                        xz_array_2, c_array_2,
                                                        trunc_val=trunc_val,
                                                        )
            max_num_string = max(max_num_string, num_string)
        elif command.op.type in [OpType.H, OpType.S, OpType.Sdg]:
            func = _CLIFFORD_FUNC_DISPATCH[command.op.type]
            qubit = command.args[0].index[0]
            xz_array, c_array = func(xz_array, c_array, qubit)
        elif command.op.type in [OpType.CX, OpType.CZ]:
            func = _CLIFFORD_FUNC_DISPATCH[command.op.type]
            control_qubit = command.args[0].index[0]
            target_qubit = command.args[1].index[0]
            xz_array, c_array = func(xz_array, c_array, control_qubit, target_qubit)
        elif command.op.type in [OpType.Measure, OpType.Barrier]:
            # print("Skipping measurement/barrier")
            continue
        else:
            raise ValueError(f"Unsupported gate type: {command.op.type}")

        t1 = time.time()

        weight_left = jnp.linalg.norm(c_array) ** 2

        current_row_size = len(c_array)
        process = psutil.Process()
        print(command.op.type,
              weight_left,
              process.memory_info().rss / 1e6, "MB", f"Size: {current_row_size}",
              "Progress: {:.2f}%".format(100 * (command_idx+1) / total_num_gate),
              "gates:", (command_idx+1), "/", total_num_gate - (command_idx+1),
              "M rows/s:", current_row_size / (t1 - t0) / 1e6,
              "Time:", "{:.2f}".format(t1 - t0),
              "Total Time:", "{:.2f}s".format(time.time() - total_start_time),
              "--------",
              end='\r')

    exp_val = backend.get_expectation_value(xz_array, c_array)
    return exp_val, xz_array, c_array


if __name__ == "__main__":
    import pickle
    file_path = 'simple_test_circuit.pkl'
    circ = pickle.load(open(file_path, 'rb'))

    from pytket.passes import DecomposeBoxes, AutoRebase
    from pytket.circuit import OpType
    AutoRebase({# OpType.CX, OpType.CY, OpType.CZ,
                OpType.ZZPhase, OpType.YYPhase, OpType.XXPhase,
                OpType.Rx, OpType.Ry, OpType.Rz,
                }).apply(circ)



    m_qubits = [0, 1]
    # for trunc_val in [1e-1, 5e-2, 2.5e-2, 1e-2,]:
    # for trunc_val in [3e-3, 1e-3, 3e-4, 1e-4]:
    for trunc_val in [3e-5, 1e-5]:
        print("================================ ")
        print("\n Truncation Value:", trunc_val)
        print("================================ ")

        exp_val, xz_array, c_array = run_pytket_circuit_forward(circ, m_qubits, trunc_val)
        print("\n Expectation Value:", exp_val)

        exp_val, xz_array, c_array = run_pytket_circuit_backward(circ, xz_array, c_array, trunc_val)

        print("\n")
        print(xz_array)
        print(c_array)

