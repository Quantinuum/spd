"""Parse `pytket` circuits into the internal SPD execution IR.

This module is the current frontend boundary. Its job is to translate pytket's
gate representation and parameter conventions into backend-agnostic IR objects,
including conversion to SPD's rotation-angle convention.
"""

import math

from pytket.circuit import OpType

from .circuit_ir import (
    PauliRotation,
    SingleQubitClifford,
    SkippedOperation,
    TwoQubitClifford,
)


PYTKET_REBASE_GATES = {
    OpType.CX,
    OpType.CY,
    OpType.CZ,
    OpType.ZZPhase,
    OpType.YYPhase,
    OpType.XXPhase,
    OpType.Rx,
    OpType.Ry,
    OpType.Rz,
    OpType.H,
    OpType.S,
    OpType.Sdg,
}


_SINGLE_QUBIT_CLIFFORDS = {OpType.H, OpType.S, OpType.Sdg, OpType.X, OpType.Y, OpType.Z}
_TWO_QUBIT_CLIFFORDS = {OpType.CX, OpType.CY, OpType.CZ}
_SKIPPED_GATES = {OpType.Measure, OpType.Barrier}


def maybe_rebase_pytket_circuit(circ):
    """Rebase a pytket circuit onto the gate subset currently supported by SPD."""
    from pytket.passes import AutoRebase

    AutoRebase(PYTKET_REBASE_GATES).apply(circ)


def parse_pytket_circuit(circ, padded_system_size):
    """Lower a pytket circuit into the internal SPD execution IR."""
    operations = []
    for command in circ.get_commands():
        op_type = command.op.type
        gate_name = str(op_type)

        if op_type in _ROTATION_DISPATCH:
            pauli, theta = parse_pauli_theta(command, padded_system_size)
            operations.append(PauliRotation(gate_name=gate_name, pauli=pauli, theta=theta))
        elif op_type in _SINGLE_QUBIT_CLIFFORDS:
            operations.append(
                SingleQubitClifford(gate_name=gate_name, qubit=command.args[0].index[0])
            )
        elif op_type in _TWO_QUBIT_CLIFFORDS:
            operations.append(
                TwoQubitClifford(
                    gate_name=gate_name,
                    control_qubit=command.args[0].index[0],
                    target_qubit=command.args[1].index[0],
                )
            )
        elif op_type in _SKIPPED_GATES:
            operations.append(SkippedOperation(gate_name=gate_name))
        else:
            raise ValueError(f"Unsupported gate type: {command.op.type}")

    return operations


def parse_pauli_theta(command, padded_system_size):
    """Extract the Pauli string and SPD theta for a pytket rotation command."""
    pauli = ["I"] * padded_system_size
    op_type = command.op.type
    pauli, theta = _ROTATION_DISPATCH[op_type](command, pauli)
    return "".join(pauli), theta


def _single_pauli_rot(command, pauli, axis):
    qubit = command.args[0].index[0]
    pauli[qubit] = axis
    # pytket uses exp(-i * param * pi * P / 2), while SPD stores rotations as
    # exp(-i * theta * P / 2). Therefore theta = param * pi here.
    theta = command.op.params[0] * math.pi
    return pauli, theta


def _two_pauli_rot(command, pauli, axis):
    qubit1 = command.args[0].index[0]
    qubit2 = command.args[1].index[0]
    pauli[qubit1] = axis
    pauli[qubit2] = axis
    # pytket uses exp(-i * param * pi * P / 2), while SPD stores rotations as
    # exp(-i * theta * P / 2). Therefore theta = param * pi here.
    theta = command.op.params[0] * math.pi
    return pauli, theta


def _pauli_exp_box(command, pauli):
    n_qubits = command.op.n_qubits
    q_indices = [command.args[i].index[0] for i in range(n_qubits)]
    for q_idx, pauli_term in zip(q_indices, command.op.get_paulis()):
        pauli[q_idx] = str(pauli_term)[-1]

    # PauliExpBox follows the same exp(-i * phase * pi * P / 2) convention.
    theta = command.op.get_phase() * math.pi
    return pauli, theta


_ROTATION_DISPATCH = {
    OpType.Rz: lambda cmd, pauli: _single_pauli_rot(cmd, pauli, "Z"),
    OpType.Rx: lambda cmd, pauli: _single_pauli_rot(cmd, pauli, "X"),
    OpType.Ry: lambda cmd, pauli: _single_pauli_rot(cmd, pauli, "Y"),
    OpType.ZZPhase: lambda cmd, pauli: _two_pauli_rot(cmd, pauli, "Z"),
    OpType.XXPhase: lambda cmd, pauli: _two_pauli_rot(cmd, pauli, "X"),
    OpType.YYPhase: lambda cmd, pauli: _two_pauli_rot(cmd, pauli, "Y"),
    OpType.PauliExpBox: _pauli_exp_box,
}
