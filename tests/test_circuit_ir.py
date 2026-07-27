from dataclasses import FrozenInstanceError

import pytest

import spd
from spd.circuit_ir import (
    CircuitIR,
    PauliRotation,
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
