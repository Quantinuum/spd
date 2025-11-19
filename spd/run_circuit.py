import math, psutil, time, sys
from pytket.circuit import OpType


# [TODO]: Merging the conj, merge into a single function
# [TODO]: make sure c_array always float32


def set_backend(backend_name, _PACKBIT=32):
    # [TODO] Make this more elegant, not using globals
    global backend, utils, _ROT_DISPATCH, _CLIFFORD_FUNC_DISPATCH
    if backend_name == 'numpy':
        from . import numpy_backend as backend
        from .numpy_backend import utils
    elif backend_name == 'jax':
        from . import jax_backend as backend
        from .jax_backend import utils
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")

    utils.set_packbit(_PACKBIT)

    # global dispatch table
    _ROT_DISPATCH = {
            OpType.Rz: lambda cmd, P: _single_pauli_rot(cmd, P, "Z"),
            OpType.Rx: lambda cmd, P: _single_pauli_rot(cmd, P, "X"),
            OpType.Ry: lambda cmd, P: _single_pauli_rot(cmd, P, "Y"),
            OpType.ZZPhase: lambda cmd, P: _two_pauli_rot(cmd, P, "Z"),
            OpType.XXPhase: lambda cmd, P: _two_pauli_rot(cmd, P, "X"),
            OpType.YYPhase: lambda cmd, P: _two_pauli_rot(cmd, P, "Y"),
            OpType.PauliExpBox: _pauli_exp_box,
            }

    _CLIFFORD_FUNC_DISPATCH = {
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



def run_pytket_circuit(circ,
                       measure_qubits_data,
                       trunc_val,
                       backend_name='numpy',
                       rebase=False,
                       log_filename=None,
                       save_strings=False,
                       ):
    _PACKBIT = 32
    set_backend(backend_name, _PACKBIT)


    # [TODO] Separate the backend.run part from the parse circuit part.
    from pytket.circuit import OpType

    if rebase:
        # from pytket.passes import DecomposeBoxes
        from pytket.passes import AutoRebase
        AutoRebase({OpType.CX, OpType.CY, OpType.CZ,
                    OpType.ZZPhase, OpType.YYPhase, OpType.XXPhase,
                    OpType.Rx, OpType.Ry, OpType.Rz,
                    OpType.H, OpType.S, OpType.Sdg,
                    }).apply(circ)

    total_start_time = time.time()
    original_system_size = circ.n_qubits
    padded_system_size = _PACKBIT * ((original_system_size + _PACKBIT - 1) // _PACKBIT)  # pad to multiple of _PACKBIT
    print("SYSTEM SIZE (PADDED):", padded_system_size)

    commands = circ.get_commands()
    total_num_gate = len(commands)

    if type(measure_qubits_data) is dict:
        key = next(iter(measure_qubits_data))
        if type(key) == tuple:
            sparse_pauli_op = backend.create_measurement_op(measure_qubits_data, padded_system_size,)
        elif type(key) == str:
            sparse_pauli_op = backend.create_op(measure_qubits_data)
        else:
            raise ValueError("measure_qubits_data dict key must be tuple or str")
    elif type(measure_qubits_data) is list:
        m_dict = {tuple(measure_qubits_data): 1.0}
        sparse_pauli_op = backend.create_measurement_op(m_dict, padded_system_size, _PACKBIT)

    max_num_string = 0
    initial_weight = backend.get_norm_square(sparse_pauli_op)

    for command_idx, command in enumerate(commands[::-1]):
        t0 = time.time()

        if command.op.type in _ROT_DISPATCH:
            P, theta = parse_pauli_theta(command, padded_system_size)
            xzk = utils.pauli_str_to_uint(P)
            # Parsing the rotation: u = exp(-i * theta * P)

            if backend_name == 'numpy':
                sparse_pauli_op, num_string = backend.conjugated_pauli_batched_uint_(sparse_pauli_op, xzk, theta, trunc_val=trunc_val)
            else:
                sparse_pauli_op_1, sparse_pauli_op_2 = backend.conjugated_pauli_batched_uint_(sparse_pauli_op, xzk, theta)
                sparse_pauli_op, num_string = backend.merge_and_pad(sparse_pauli_op_1,
                                                                    sparse_pauli_op_2,
                                                                    trunc_val=trunc_val,
                                                                    )

            max_num_string = max(max_num_string, num_string)
        elif command.op.type in [OpType.H, OpType.S, OpType.Sdg, OpType.X, OpType.Y, OpType.Z]:
            func = _CLIFFORD_FUNC_DISPATCH[command.op.type]
            qubit = command.args[0].index[0]
            sparse_pauli_op = func(sparse_pauli_op, qubit)
        elif command.op.type in [OpType.CX, OpType.CY, OpType.CZ]:
            func = _CLIFFORD_FUNC_DISPATCH[command.op.type]
            control_qubit = command.args[0].index[0]
            target_qubit = command.args[1].index[0]
            sparse_pauli_op = func(sparse_pauli_op, control_qubit, target_qubit)
        elif command.op.type in [OpType.Measure, OpType.Barrier]:
            # print("Skipping measurement/barrier")
            continue
        else:
            raise ValueError(f"Unsupported gate type: {command.op.type}")

        t1 = time.time()

        current_weight = backend.get_norm_square(sparse_pauli_op)
        weight_left = current_weight / initial_weight

        current_row_size = backend.get_size(sparse_pauli_op)

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

    exp_val = backend.get_expectation_value(sparse_pauli_op)

    if log_filename is not None:
        # We log trunc_val, final number of terms, max number string, final weight, total time, exp_val
        assert type(log_filename) == str
        with open(log_filename, "a") as f:
            f.write(f"{trunc_val}, {current_row_size}, {max_num_string}, {weight_left}, {time.time() - total_start_time}, {exp_val}\n")
    else:
        pass

    if save_strings:
        import pickle
        pickle.dump(sparse_pauli_op, open(f'strings_{trunc_val}.pickle','wb'))

    return exp_val

def parse_pauli_theta(command, padded_system_size):
    P = ['I'] * padded_system_size
    op_type = command.op.type
    P, theta = _ROT_DISPATCH[op_type](command, P)
    return ''.join(P), theta

def _single_pauli_rot(command, P, axis):
    qubit = command.args[0].index[0]
    P[qubit] = axis
    theta = command.op.params[0] * math.pi # / 2
    return P, theta

def _two_pauli_rot(command, P, axis):
    qubit1 = command.args[0].index[0]
    qubit2 = command.args[1].index[0]
    P[qubit1] = axis
    P[qubit2] = axis
    theta = command.op.params[0] * math.pi # / 2
    return P, theta

def _pauli_exp_box(command, P):
    # (Pdb) cmd.op.get_paulis()
    # [Pauli.X, Pauli.X, Pauli.X]
    # (Pdb) cmd.op.get_phase()
    # 0.0318309886183791
    n_qubits = command.op.n_qubits
    q_indices = [command.args[i].index[0] for i in range(n_qubits)]
    paulis = command.op.get_paulis()
    for q_idx, pauli in zip(q_indices, paulis):
        P[q_idx] = str(pauli)[-1]

    theta = command.op.get_phase() * math.pi # / 2
    return P, theta

