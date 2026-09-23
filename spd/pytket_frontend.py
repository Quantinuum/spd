"""Translate between `pytket` circuits and the SPD execution IR.

This module translates pytket's gate representation and parameter conventions
into backend-agnostic IR objects and exports the supported IR subset back to
pytket.
"""

import math

from pytket.circuit import OpType

from .circuit_ir import (
    CircuitIR, CreateZero, ResetZero, Discard,
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

_BARRIER_NAMES = {"barrier", "optype.barrier"}
_MEASURE_NAMES = {"measure", "measurement", "optype.measure"}
_SINGLE_QUBIT_CLIFFORD_NAMES = {
    "h": "H", "s": "S", "sdg": "Sdg", "x": "X", "y": "Y", "z": "Z",
}
_TWO_QUBIT_CLIFFORD_NAMES = {"cx": "CX", "cy": "CY", "cz": "CZ"}


def maybe_rebase_pytket_circuit(circ):
    """Rebase a pytket circuit onto the gate subset currently supported by SPD."""
    from pytket.passes import AutoRebase

    AutoRebase(PYTKET_REBASE_GATES).apply(circ)


def parse_pytket_circuit(circ):
    """Lower a pytket circuit into logical-width SPD execution IR."""
    from pytket.circuit import Qubit
    if circ.qubits != [Qubit(i) for i in range(circ.n_qubits)]:
        raise ValueError("SPD requires canonical integer qubits q[0], ..., q[n-1].")
    if any(a != b for a, b in circ.implicit_qubit_permutation().items()):
        raise ValueError("Implicit qubit permutations must be materialized before SPD import.")
    operations = [CreateZero(q.index[0]) for q in sorted(circ.created_qubits)]
    for command in circ.get_commands():
        op_type = command.op.type
        gate_name = str(op_type)

        if op_type in _ROTATION_DISPATCH:
            pauli, theta = parse_pauli_theta(command, circ.n_qubits)
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
        elif op_type == OpType.Reset:
            operations.append(ResetZero(command.qubits[0].index[0]))
        elif op_type == OpType.Measure and (circ.created_qubits or circ.discarded_qubits
                                            or any(c.op.type == OpType.Reset for c in circ.get_commands())):
            raise ValueError("Measurements are unsupported in channel circuits.")
        elif op_type in _SKIPPED_GATES:
            operations.append(SkippedOperation(gate_name=gate_name))
        else:
            raise ValueError(f"Unsupported gate type: {command.op.type}")

    operations.extend(Discard(q.index[0]) for q in sorted(circ.discarded_qubits))
    return CircuitIR(system_size=circ.n_qubits, operations=tuple(operations))


def export_pytket_circuit(circuit_ir):
    """Export the supported :class:`CircuitIR` subset to a pytket circuit.

    Unitary operations, resets, barriers, and terminal full-register
    measurements are supported. A measurement block must contain one marker per
    qubit and is exported as ``q[i] -> c[i]``. ``CreateZero`` and ``Discard``
    cannot retain their IR positions in pytket, so they are rejected.

    No ordering barriers are added. Callers that require a fixed total command
    order should include barriers in the input IR.
    """
    from pytket.circuit import Circuit

    if not isinstance(circuit_ir, CircuitIR):
        raise TypeError("circuit_ir must be a CircuitIR.")

    operations = circuit_ir.operations
    measurement_start = len(operations)
    while (measurement_start > 0
           and _operation_name(operations[measurement_start - 1]) in _MEASURE_NAMES):
        measurement_start -= 1
    measurements = operations[measurement_start:]
    if any(_operation_name(op) in _MEASURE_NAMES for op in operations[:measurement_start]):
        raise ValueError("Measurements must form one terminal full-register block.")
    if measurements and len(measurements) != circuit_ir.system_size:
        raise ValueError(
            "A terminal measurement block must contain one measurement per qubit."
        )

    circuit = Circuit(circuit_ir.system_size,
                      circuit_ir.system_size if measurements else 0)
    for operation in operations[:measurement_start]:
        if isinstance(operation, PauliRotation):
            _export_pauli_rotation(circuit, operation)
        elif isinstance(operation, SingleQubitClifford):
            gate = _SINGLE_QUBIT_CLIFFORD_NAMES.get(_operation_name(operation))
            if gate is None:
                raise ValueError(f"Unsupported single-qubit Clifford: {operation.gate_name}")
            getattr(circuit, gate)(operation.qubit)
        elif isinstance(operation, TwoQubitClifford):
            gate = _TWO_QUBIT_CLIFFORD_NAMES.get(_operation_name(operation))
            if gate is None:
                raise ValueError(f"Unsupported two-qubit Clifford: {operation.gate_name}")
            getattr(circuit, gate)(operation.control_qubit, operation.target_qubit)
        elif isinstance(operation, ResetZero):
            circuit.Reset(operation.qubit)
        elif isinstance(operation, (CreateZero, Discard)):
            raise ValueError(f"{type(operation).__name__} export is unsupported.")
        elif isinstance(operation, SkippedOperation):
            if _operation_name(operation) not in _BARRIER_NAMES:
                raise ValueError(f"Unsupported skipped operation: {operation.gate_name}")
            circuit.add_barrier(list(range(circuit_ir.system_size)))
        else:
            raise TypeError(f"Unsupported circuit operation: {type(operation)!r}")

    if measurements:
        for qubit in range(circuit_ir.system_size):
            circuit.Measure(qubit, qubit)
    return circuit


def _operation_name(operation):
    """Return a normalized gate name without an optional ``OpType.`` prefix."""
    name = operation.gate_name.lower()
    return name.removeprefix("optype.")


def _export_pauli_rotation(circuit, operation):
    """Append one SPD-convention Pauli rotation to a pytket circuit."""
    from pytket.circuit import PauliExpBox
    from pytket.pauli import Pauli

    parameter = operation.theta / math.pi
    support = [(qubit, pauli) for qubit, pauli in enumerate(operation.pauli)
               if pauli != "I"]
    if len(support) == 1:
        qubit, pauli = support[0]
        getattr(circuit, {"X": "Rx", "Y": "Ry", "Z": "Rz"}[pauli])(
            parameter, qubit
        )
        return
    if len(support) == 2 and support[0][1] == support[1][1]:
        pauli = support[0][1]
        getattr(circuit, {"X": "XXPhase", "Y": "YYPhase", "Z": "ZZPhase"}[pauli])(
            parameter, support[0][0], support[1][0]
        )
        return

    # Keeping only the non-identity support gives the same unitary. For an
    # identity rotation, retain one identity qubit so pytket can carry the
    # global phase as a PauliExpBox command and the parser can recover it.
    if support:
        qubits = [qubit for qubit, _ in support]
        paulis = [getattr(Pauli, pauli) for _, pauli in support]
    else:
        qubits = [0]
        paulis = [Pauli.I]
    circuit.add_pauliexpbox(PauliExpBox(paulis, parameter), qubits)


def parse_pauli_theta(command, system_size):
    """Extract the Pauli string and SPD theta for a pytket rotation command."""
    pauli = ["I"] * system_size
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
