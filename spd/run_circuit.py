import jax
import jax.numpy as jnp
import numpy as np
import psutil
import time
import sys
from pytket import Circuit
from pytket.circuit import OpType

from . import jax_backend as backend

# BACKEDN related functions:
# pauli_str_to_uint8, uint32, ...
# checktype
# merge_pauli
# conjugated_pauli_batched_uint_
# X, Y, Z, CX, CZ, H, S, Sdg

## Maybe we do `import spd.jax as backend`
## backend.pauli_str_to_uint...
## backend.conjugated_pauli_batched_uint...

# [TODO]: Merging the conj, merge into a single function
# [TODO]: make sure c_array always float32

def run_pytket_circuit(circ, measure_qubits_list, trunc_val, packbit=32, loggin=True):
    # [TODO] Separate the backend.run part from the parse circuit part.

    from pytket.passes import DecomposeBoxes, AutoRebase
    from pytket.circuit import OpType

    AutoRebase({# OpType.CX, OpType.CY, OpType.CZ,
                OpType.CX, OpType.CZ,
                OpType.ZZPhase, OpType.YYPhase, OpType.XXPhase,
                OpType.Rx, OpType.Ry, OpType.Rz,
                OpType.H, OpType.S, OpType.Sdg,
                }).apply(circ)

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
              end='\n')

    exp_val = backend.get_expectation_value(xz_array, c_array)

    if loggin:
        # We log trunc_val, final number of terms, max number string, final weight, total time, exp_val
        with open("spd_log.txt", "a") as f:
            f.write(f"{trunc_val}, {len(c_array)}, {max_num_string}, {jnp.linalg.norm(c_array) ** 2}, {time.time() - total_start_time}, {exp_val}\n")
    else:
        pass

    return exp_val

def parse_pauli_theta(command, system_size):
    P = ['I'] * system_size
    theta = command.op.params[0] * np.pi # / 2
    op_type = command.op.type
    _ROT_DISPATCH[op_type](command, P)
    return ''.join(P), theta

# global dispatch table
_ROT_DISPATCH = {
        OpType.Rz: lambda cmd, P: _single_pauli_rot(cmd, P, "Z"),
        OpType.Rx: lambda cmd, P: _single_pauli_rot(cmd, P, "X"),
        OpType.Ry: lambda cmd, P: _single_pauli_rot(cmd, P, "Y"),
        OpType.ZZPhase: lambda cmd, P: _two_pauli_rot(cmd, P, "Z"),
        OpType.XXPhase: lambda cmd, P: _two_pauli_rot(cmd, P, "X"),
        OpType.YYPhase: lambda cmd, P: _two_pauli_rot(cmd, P, "Y"),
        }

def _single_pauli_rot(command, P, axis):
    qubit = command.args[0].index[0]
    P[qubit] = axis

def _two_pauli_rot(command, P, axis):
    qubit1 = command.args[0].index[0]
    qubit2 = command.args[1].index[0]
    P[qubit1] = axis
    P[qubit2] = axis

_CLIFFORD_FUNC_DISPATCH = {
        OpType.H: backend.conjugated_pauli_batched_uint32_H,
        OpType.S: backend.conjugated_pauli_batched_uint32_S,
        OpType.Sdg: backend.conjugated_pauli_batched_uint32_Sdg,
        OpType.CX: backend.conjugated_pauli_batched_uint32_CX,
        OpType.CZ: backend.conjugated_pauli_batched_uint32_CZ,
        OpType.X: backend.conjugated_pauli_batched_uint32_X,
        OpType.Y: backend.conjugated_pauli_batched_uint32_Y,
        OpType.Z: backend.conjugated_pauli_batched_uint32_Z,
        }
