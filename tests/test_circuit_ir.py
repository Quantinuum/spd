from dataclasses import FrozenInstanceError

import pytest

import spd
from spd.circuit_ir import (
    CircuitIR,
    CreateZero,
    Discard,
    PauliRotation,
    ResetZero,
    SingleQubitClifford,
    SkippedOperation,
    TwoQubitClifford,
    get_operation_qubits,
)


def test_circuit_ir_is_immutable_and_normalizes_operations_to_tuple():
    circuit_ir = CircuitIR(
        system_size=3,
        operations=[SingleQubitClifford("OpType.H", 0)],
    )

    assert isinstance(circuit_ir.operations, tuple)
    assert spd.CircuitIR is CircuitIR
    with pytest.raises(FrozenInstanceError):
        circuit_ir.system_size = 4


@pytest.mark.parametrize("system_size", [0, -1])
def test_circuit_ir_rejects_nonpositive_system_size(system_size):
    with pytest.raises(ValueError, match="at least 1"):
        CircuitIR(system_size=system_size, operations=())


@pytest.mark.parametrize(
    "operation",
    [
        SingleQubitClifford("OpType.H", 2),
        TwoQubitClifford("OpType.CX", 0, 2),
        PauliRotation("RX", "IIZ", 0.1),
    ],
)
def test_circuit_ir_rejects_operations_outside_system(operation):
    with pytest.raises(ValueError, match="outside"):
        CircuitIR(system_size=2, operations=(operation,))


def test_get_operation_qubits_covers_ir_operation_types():
    assert get_operation_qubits(PauliRotation("RZZ", "IZZI", 0.1)) == (1, 2)
    assert get_operation_qubits(SingleQubitClifford("OpType.H", 1)) == (1,)
    assert get_operation_qubits(TwoQubitClifford("OpType.CX", 1, 3)) == (1, 3)
    assert get_operation_qubits(SkippedOperation("barrier")) == ()


def test_circuit_statistics_count_operation_kinds_and_gate_arities():
    circuit_ir = CircuitIR(4, (
        SingleQubitClifford("H", 0),
        PauliRotation("Rz", "IIZI", 0.1),
        TwoQubitClifford("CX", 0, 1),
        PauliRotation("RXX", "IIXX", 0.2),
        PauliRotation("RXYZ", "IXYZ", 0.3),
        PauliRotation("Phase", "IIII", 0.4),
        ResetZero(0),
        SkippedOperation("barrier"),
        Discard(3),
    ))

    statistics = circuit_ir.statistics()

    assert statistics["total_operations"] == 9
    assert statistics["gate_operations"] == 6
    assert statistics["single_qubit_gates"] == 2
    assert statistics["two_qubit_gates"] == 2
    assert statistics["multi_qubit_gates"] == 1
    assert statistics["zero_qubit_gates"] == 1
    assert statistics["two_qubit_depth"] == 1
    assert statistics["reset_zero"] == 1
    assert statistics["discard"] == 1
    assert statistics["skipped_operations"] == 1
    assert statistics["operation_types"]["PauliRotation"] == 4
    assert statistics["gate_names"] == {
        "CX": 1, "H": 1, "Phase": 1, "RXX": 1, "RXYZ": 1, "Rz": 1,
    }


def test_two_qubit_depth_accounts_for_dependencies_through_other_operations():
    circuit_ir = CircuitIR(4, (
        TwoQubitClifford("CX", 0, 1),
        TwoQubitClifford("CX", 2, 3),
        PauliRotation("RXYZ", "IXYZ", 0.3),
        TwoQubitClifford("CX", 0, 1),
    ))

    assert circuit_ir.statistics()["two_qubit_depth"] == 2


def test_print_statistics(capsys):
    CircuitIR(2, (CreateZero(1), ResetZero(1))).print_statistics()

    output = capsys.readouterr().out
    assert "CircuitIR statistics:" in output
    assert "Two-qubit gates: 0" in output
    assert "CreateZero: 1" in output
    assert "ResetZero: 1" in output
