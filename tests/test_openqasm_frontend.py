import numpy as np

from spd.circuit_ir import PauliRotation, SingleQubitClifford, SkippedOperation, TwoQubitClifford
from spd.openqasm_frontend import parse_openqasm_file, parse_openqasm_str


def test_parse_openqasm_str_emits_backend_agnostic_ir():
    source = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[1];
    rx(pi/2) q[0];
    rzz(-pi/4) q[0], q[1];
    h q[1];
    cx q[0], q[1];
    barrier q;
    measure q[0] -> c[0];
    """

    system_size, operations = parse_openqasm_str(source, padded_system_size=32)

    assert system_size == 2
    assert isinstance(operations[0], PauliRotation)
    assert operations[0].gate_name == "OpenQASM.rx"
    assert operations[0].pauli[:2] == "XI"
    assert np.isclose(operations[0].theta, np.pi / 2)

    assert isinstance(operations[1], PauliRotation)
    assert operations[1].gate_name == "OpenQASM.rzz"
    assert operations[1].pauli[:2] == "ZZ"
    assert np.isclose(operations[1].theta, -np.pi / 4)

    assert isinstance(operations[2], SingleQubitClifford)
    assert operations[2].gate_name == "OpType.H"
    assert operations[2].qubit == 1

    assert isinstance(operations[3], TwoQubitClifford)
    assert operations[3].gate_name == "OpType.CX"
    assert operations[3].control_qubit == 0
    assert operations[3].target_qubit == 1

    assert isinstance(operations[4], SkippedOperation)
    assert isinstance(operations[5], SkippedOperation)


def test_parse_openqasm_file_handles_sample_circuit():
    path = "tests/fixtures/open_qasm/periodic_small_8q.qasm"

    system_size, operations = parse_openqasm_file(path, padded_system_size=32)

    assert system_size == 8
    assert len(operations) == 36
    assert isinstance(operations[0], PauliRotation)
    assert operations[0].pauli[0] == "X"
    assert np.isclose(operations[0].theta, -0.41367656401114183)
    assert isinstance(operations[8], PauliRotation)
    assert operations[8].pauli[0] == "Z"
    assert operations[8].pauli[3] == "Z"
