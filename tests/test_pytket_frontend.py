import numpy as np
import pytest
from pytket.circuit import Circuit, PauliExpBox
from pytket.pauli import Pauli

from spd.circuit_ir import (
    CircuitIR, CreateZero, Discard, PauliRotation, ResetZero,
    SingleQubitClifford, SkippedOperation, TwoQubitClifford,
)
from spd.pytket_frontend import export_pytket_circuit, parse_pytket_circuit


def test_parse_pytket_circuit_emits_backend_agnostic_ir():
    circ = Circuit(3, 1)
    circ.Rz(0.25, 0)
    circ.CX(0, 1)
    circ.H(2)
    circ.add_barrier([0, 1, 2])
    circ.Measure(0, 0)

    circuit_ir = parse_pytket_circuit(circ)
    operations = circuit_ir.operations

    assert circuit_ir.system_size == 3
    assert isinstance(operations[0], PauliRotation)
    assert operations[0].pauli == "ZII"
    assert len(operations[0].pauli) == circuit_ir.system_size
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

    circuit_ir = parse_pytket_circuit(circ)
    operations = circuit_ir.operations

    assert circuit_ir.system_size == 3
    assert len(operations) == 1
    assert isinstance(operations[0], PauliRotation)
    assert operations[0].gate_name == "OpType.PauliExpBox"
    assert operations[0].pauli == "XYZ"
    assert np.isclose(operations[0].theta, 0.125 * np.pi)


def _assert_round_trip_matches(expected, actual):
    assert actual.system_size == expected.system_size
    assert len(actual.operations) == len(expected.operations)
    for wanted, got in zip(expected.operations, actual.operations):
        assert type(got) is type(wanted)
        if isinstance(wanted, PauliRotation):
            assert got.pauli == wanted.pauli
            angle_difference = (
                (got.theta - wanted.theta + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
            )
            assert angle_difference == pytest.approx(0.)
        elif isinstance(wanted, SkippedOperation):
            assert got.gate_name.lower().endswith(wanted.gate_name.lower().split(".")[-1])
        else:
            wanted_values = vars(wanted).copy()
            got_values = vars(got).copy()
            wanted_values.pop("gate_name", None)
            got_values.pop("gate_name", None)
            assert got_values == wanted_values


def test_export_pytket_unitary_round_trip_with_explicit_ordering_barriers():
    barrier = SkippedOperation("barrier")
    circuit_ir = CircuitIR(3, (
        PauliRotation("custom-x", "XII", .17), barrier,
        PauliRotation("custom-yy", "IYY", -.23), barrier,
        PauliRotation("custom-xyz", "XYZ", .31), barrier,
        PauliRotation("global", "III", -.11), barrier,
        SingleQubitClifford("H", 2), barrier,
        SingleQubitClifford("OpType.Sdg", 1), barrier,
        TwoQubitClifford("CX", 0, 2), barrier,
        TwoQubitClifford("OpType.CY", 2, 1), barrier,
        TwoQubitClifford("CZ", 1, 0),
    ))

    exported = export_pytket_circuit(circuit_ir)

    assert sum(command.op.type.name == "Barrier" for command in exported.get_commands()) == 8
    _assert_round_trip_matches(circuit_ir, parse_pytket_circuit(exported))


def test_export_pytket_does_not_add_ordering_barriers():
    circuit_ir = CircuitIR(2, (
        SingleQubitClifford("H", 0),
        SingleQubitClifford("X", 1),
    ))

    exported = export_pytket_circuit(circuit_ir)

    assert all(command.op.type.name != "Barrier" for command in exported.get_commands())


@pytest.mark.parametrize("name", ["H", "S", "Sdg", "X", "Y", "Z"])
def test_export_pytket_all_single_qubit_cliffords(name):
    circuit_ir = CircuitIR(1, (SingleQubitClifford(name, 0),))
    _assert_round_trip_matches(
        circuit_ir, parse_pytket_circuit(export_pytket_circuit(circuit_ir))
    )


@pytest.mark.parametrize("name", ["CX", "CY", "CZ"])
def test_export_pytket_all_two_qubit_cliffords(name):
    circuit_ir = CircuitIR(2, (TwoQubitClifford(name, 0, 1),))
    _assert_round_trip_matches(
        circuit_ir, parse_pytket_circuit(export_pytket_circuit(circuit_ir))
    )


@pytest.mark.parametrize("pauli", ["XI", "YI", "ZI", "XX", "YY", "ZZ"])
def test_export_pytket_native_pauli_rotations(pauli):
    circuit_ir = CircuitIR(2, (PauliRotation("descriptive-name", pauli, -.27),))
    _assert_round_trip_matches(
        circuit_ir, parse_pytket_circuit(export_pytket_circuit(circuit_ir))
    )


def test_export_pytket_reset_round_trip():
    barrier = SkippedOperation("OpType.Barrier")
    circuit_ir = CircuitIR(2, (
        PauliRotation("Ry", "YI", .2), barrier,
        ResetZero(0), barrier,
        PauliRotation("YX", "YX", -.4),
    ))

    _assert_round_trip_matches(
        circuit_ir,
        parse_pytket_circuit(export_pytket_circuit(circuit_ir)),
    )


def test_export_pytket_terminal_full_register_measurement_round_trip():
    circuit_ir = CircuitIR(2, (
        SingleQubitClifford("H", 0),
        SkippedOperation("barrier"),
        SkippedOperation("measure"),
        SkippedOperation("OpType.Measure"),
    ))

    exported = export_pytket_circuit(circuit_ir)

    assert exported.n_bits == 2
    measurements = [command for command in exported.get_commands()
                    if command.op.type.name == "Measure"]
    assert [
        [unit.index[0] for unit in command.args] for command in measurements
    ] == [[0, 0], [1, 1]]
    _assert_round_trip_matches(circuit_ir, parse_pytket_circuit(exported))


@pytest.mark.parametrize("operations, message", [
    ((SkippedOperation("measure"),), "one measurement per qubit"),
    ((SkippedOperation("measure"), SingleQubitClifford("H", 0)), "terminal"),
    ((SkippedOperation("unknown"),), "Unsupported skipped operation"),
    ((CreateZero(0),), "CreateZero export is unsupported"),
    ((Discard(0),), "Discard export is unsupported"),
])
def test_export_pytket_rejects_unsupported_ir(operations, message):
    with pytest.raises(ValueError, match=message):
        export_pytket_circuit(CircuitIR(2, operations))


def test_export_pytket_rejects_unknown_clifford_name():
    with pytest.raises(ValueError, match="Unsupported single-qubit Clifford"):
        export_pytket_circuit(CircuitIR(1, (SingleQubitClifford("V", 0),)))
