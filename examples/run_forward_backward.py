import pytket
from pytket.circuit import Circuit
import sys
from pathlib import Path
sys.path.append('../')
import spd

import spd.jax_backend as backend
from spd.pytket_frontend import parse_pauli_theta
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
        pauli_str_to_uint = backend.utils.pauli_str_to_uint8
    elif packbit == 32:
        pauli_str_to_uint = backend.utils.pauli_str_to_uint32
    elif packbit == 64:
        raise NotImplementedError("64-bit has some bug in the code. It cannot generate correct result")
        pauli_str_to_uint = backend.utils.pauli_str_to_uint64
    else:
        raise ValueError("packbit must be 8 or 32")


    measure_Zs = ''.join(['Z' if i in measure_qubits_list else 'I' for i in range(system_size)])
    xz_array = pauli_str_to_uint(measure_Zs).reshape([1, -1])
    c_array = jnp.ones((1,), dtype=backend.utils.get_real_dtype())
    print("The initial xz_array: ", xz_array)
    print("The intial c_array: ", c_array)

    from pytket.circuit import OpType
    rotation_ops = {
        OpType.Rz, OpType.Rx, OpType.Ry, OpType.ZZPhase, OpType.XXPhase, OpType.YYPhase, OpType.PauliExpBox
    }
    clifford_dispatch = {
        OpType.H: backend.conjugated_pauli_batched_uint32_H,
        OpType.S: backend.conjugated_pauli_batched_uint32_S,
        OpType.Sdg: backend.conjugated_pauli_batched_uint32_Sdg,
        OpType.CX: backend.conjugated_pauli_batched_uint32_CX,
        OpType.CY: backend.conjugated_pauli_batched_uint32_CY,
        OpType.CZ: backend.conjugated_pauli_batched_uint32_CZ,
        OpType.X: backend.conjugated_pauli_batched_uint32_X,
        OpType.Y: backend.conjugated_pauli_batched_uint32_Y,
        OpType.Z: backend.conjugated_pauli_batched_uint32_Z,
    }
    spo = backend.SparsePauliOp(xz_array, c_array)
    max_num_string = 0

    for command_idx, command in enumerate(commands[::-1]):
        t0 = time.time()

        if command.op.type in rotation_ops:
            P, theta = parse_pauli_theta(command, system_size)
            xzk = pauli_str_to_uint(P)
            # Parsing the rotation: u = exp(-i * theta * P)
            spo, num_string = backend.conjugated_pauli_forward(
                spo, xzk, theta, trunc_val, max_num_str=1 << 30
            )
            max_num_string = max(max_num_string, num_string)
        elif command.op.type in [OpType.H, OpType.S, OpType.Sdg, OpType.X, OpType.Y, OpType.Z]:
            func = clifford_dispatch[command.op.type]
            qubit = command.args[0].index[0]
            spo = func(spo, qubit)
        elif command.op.type in [OpType.CX, OpType.CY, OpType.CZ]:
            func = clifford_dispatch[command.op.type]
            control_qubit = command.args[0].index[0]
            target_qubit = command.args[1].index[0]
            spo = func(spo, control_qubit, target_qubit)
        elif command.op.type in [OpType.Measure, OpType.Barrier]:
            # print("Skipping measurement/barrier")
            continue
        else:
            raise ValueError(f"Unsupported gate type: {command.op.type}")

        t1 = time.time()

        c_array = spo.c_array
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

    exp_val = spo.get_expectation_value()
    return exp_val, spo.xz_array, spo.c_array


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
        pauli_str_to_uint = backend.utils.pauli_str_to_uint8
    elif packbit == 32:
        pauli_str_to_uint = backend.utils.pauli_str_to_uint32
    elif packbit == 64:
        raise NotImplementedError("64-bit has some bug in the code. It cannot generate correct result")
        pauli_str_to_uint = backend.utils.pauli_str_to_uint64
    else:
        raise ValueError("packbit must be 8 or 32")


    from pytket.circuit import OpType
    rotation_ops = {
        OpType.Rz, OpType.Rx, OpType.Ry, OpType.ZZPhase, OpType.XXPhase, OpType.YYPhase, OpType.PauliExpBox
    }
    clifford_dispatch = {
        OpType.H: backend.conjugated_pauli_batched_uint32_H,
        OpType.S: backend.conjugated_pauli_batched_uint32_S,
        OpType.Sdg: backend.conjugated_pauli_batched_uint32_Sdg,
        OpType.CX: backend.conjugated_pauli_batched_uint32_CX,
        OpType.CY: backend.conjugated_pauli_batched_uint32_CY,
        OpType.CZ: backend.conjugated_pauli_batched_uint32_CZ,
        OpType.X: backend.conjugated_pauli_batched_uint32_X,
        OpType.Y: backend.conjugated_pauli_batched_uint32_Y,
        OpType.Z: backend.conjugated_pauli_batched_uint32_Z,
    }
    spo = backend.SparsePauliOp(xz_array, c_array)
    max_num_string = 0

    for command_idx, command in enumerate(commands):
        t0 = time.time()

        if command.op.type in rotation_ops:
            P, theta = parse_pauli_theta(command, system_size)
            xzk = pauli_str_to_uint(P)
            # Parsing the rotation: u = exp(-i * theta * P)
            spo, num_string = backend.conjugated_pauli_forward(
                spo, xzk, -theta, trunc_val, max_num_str=1 << 30
            )
            max_num_string = max(max_num_string, num_string)
        elif command.op.type in [OpType.H, OpType.S, OpType.Sdg, OpType.X, OpType.Y, OpType.Z]:
            func = clifford_dispatch[command.op.type]
            qubit = command.args[0].index[0]
            spo = func(spo, qubit)
        elif command.op.type in [OpType.CX, OpType.CY, OpType.CZ]:
            func = clifford_dispatch[command.op.type]
            control_qubit = command.args[0].index[0]
            target_qubit = command.args[1].index[0]
            spo = func(spo, control_qubit, target_qubit)
        elif command.op.type in [OpType.Measure, OpType.Barrier]:
            # print("Skipping measurement/barrier")
            continue
        else:
            raise ValueError(f"Unsupported gate type: {command.op.type}")

        t1 = time.time()

        c_array = spo.c_array
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

    exp_val = spo.get_expectation_value()
    return exp_val, spo.xz_array, spo.c_array


if __name__ == "__main__":
    import pickle
    file_path = Path(__file__).resolve().parent / "simple_test_circuit.pkl"
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
