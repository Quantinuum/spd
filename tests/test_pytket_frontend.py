import numpy as np
from pytket.circuit import Circuit, PauliExpBox
from pytket.pauli import Pauli

from spd.circuit_ir import PauliRotation, SingleQubitClifford, SkippedOperation, TwoQubitClifford
from spd.pytket_frontend import parse_pytket_circuit


def test_parse_pytket_circuit_emits_backend_agnostic_ir():
    circ = Circuit(3, 1)
    circ.Rz(0.25, 0)
    circ.CX(0, 1)
    circ.H(2)
    circ.add_barrier([0, 1, 2])
    circ.Measure(0, 0)

    circuit_ir = parse_pytket_circuit(circ, padded_system_size=32)
    operations = circuit_ir.operations

    assert circuit_ir.system_size == 3
    assert isinstance(operations[0], PauliRotation)
    assert operations[0].pauli[:3] == "ZII"
    assert np.isclose(operations[0].theta, 0.25 * np.pi)

    assert isinstance(operations[1], SingleQubitClifford)
    assert operations[1].gate_name == "OpType.H"
    assert operations[1].qubit == 2

    assert isinstance(operations[2], TwoQubitClifford)
    assert operations[2].gate_name == "OpType.CX"
    assert operations[2].control_qubit == 0
    assert operations[2].target_qubit == 1

    assert isinstance(operations[3], SkippedOperation)
    assert isinstance(operations[4], SkippedOperation)


def test_parse_pytket_circuit_handles_pauli_exp_box():
    circ = Circuit(3)
    box = PauliExpBox([Pauli.X, Pauli.Y, Pauli.Z], 0.125)
    circ.add_pauliexpbox(box, [0, 1, 2])

    circuit_ir = parse_pytket_circuit(circ, padded_system_size=32)
    operations = circuit_ir.operations

    assert circuit_ir.system_size == 3
    assert len(operations) == 1
    assert isinstance(operations[0], PauliRotation)
    assert operations[0].gate_name == "OpType.PauliExpBox"
    assert operations[0].pauli[:3] == "XYZ"
    assert np.isclose(operations[0].theta, 0.125 * np.pi)
