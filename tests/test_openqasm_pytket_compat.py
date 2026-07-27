from collections import Counter

import numpy as np
import pytest

import spd
from spd.circuit_ir import PauliRotation, SingleQubitClifford, SkippedOperation, TwoQubitClifford
from spd.openqasm_frontend import parse_openqasm_file, parse_openqasm_str
from spd.pytket_frontend import parse_pytket_circuit
from tests.helpers import make_initial_spo, to_term_dict

pytest.importorskip("pytket.qasm", reason="pytket extra not installed")

BACKENDS = {
    "numpy": spd.numpy_backend,
    "jax": spd.jax_backend,
}


def _normalize_operation(operation):
    if isinstance(operation, PauliRotation):
        return ("rotation", operation.pauli, _normalize_theta(operation.theta))
    if isinstance(operation, SingleQubitClifford):
        return ("single_clifford", _normalize_gate_name(operation.gate_name), operation.qubit)
    if isinstance(operation, TwoQubitClifford):
        return (
            "two_clifford",
            _normalize_gate_name(operation.gate_name),
            operation.control_qubit,
            operation.target_qubit,
        )
    if isinstance(operation, SkippedOperation):
        return ("skipped", _normalize_gate_name(operation.gate_name))
    raise TypeError(f"Unsupported operation type: {type(operation)!r}")


def _normalize_gate_name(gate_name):
    gate_name = gate_name.lower()
    if gate_name.startswith("optype."):
        return gate_name.split(".", 1)[1]
    if gate_name.startswith("openqasm."):
        return gate_name.split(".", 1)[1]
    return gate_name


def _normalize_theta(theta):
    period = 4.0 * np.pi
    theta_mod = float(theta) % period
    if np.isclose(theta_mod, period, atol=1e-12):
        return 0.0
    return round(theta_mod, 12)


def _assert_semantic_ir_match(native_ops, pytket_ops):
    native_counter = Counter(_normalize_operation(operation) for operation in native_ops)
    pytket_counter = Counter(_normalize_operation(operation) for operation in pytket_ops)
    assert native_counter == pytket_counter


def test_openqasm_string_matches_pytket_lowered_ir():
    from pytket.qasm import circuit_from_qasm_str

    source = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[1];
    rz(pi/7) q[0];
    rx(-3*pi/8) q[1];
    rzz(pi/5) q[0], q[2];
    h q[2];
    cx q[0], q[1];
    barrier q;
    measure q[0] -> c[0];
    """

    native_ir = parse_openqasm_str(source, padded_system_size=32)
    circ = circuit_from_qasm_str(source)
    pytket_ir = parse_pytket_circuit(circ, padded_system_size=32)

    assert native_ir.system_size == pytket_ir.system_size == circ.n_qubits == 3
    _assert_semantic_ir_match(native_ir.operations, pytket_ir.operations)


def test_openqasm_sample_matches_pytket_lowered_ir():
    from pytket.qasm import circuit_from_qasm

    path = "tests/fixtures/open_qasm/periodic_small_8q.qasm"

    native_ir = parse_openqasm_file(path, padded_system_size=32)
    circ = circuit_from_qasm(path)
    pytket_ir = parse_pytket_circuit(circ, padded_system_size=32)

    assert native_ir.system_size == pytket_ir.system_size == circ.n_qubits == 8
    assert len(native_ir.operations) == 36
    _assert_semantic_ir_match(native_ir.operations, pytket_ir.operations)


def test_openqasm_ir_matches_pytket_forward_execution(backend_name):
    from pytket.qasm import circuit_from_qasm_str

    source = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    rx(pi/7) q[0];
    rz(-pi/6) q[1];
    rzz(2*pi/9) q[0], q[2];
    h q[2];
    cx q[2], q[1];
    """

    pytket_circ = circuit_from_qasm_str(source)
    native_ir = parse_openqasm_str(source, padded_system_size=32)
    initial_spo = make_initial_spo(backend_name, [1], 3)

    native_final_spo, _ = spd.evolve(initial_spo, native_ir, trunc_val=1e-12, max_num_str=1000)
    pytket_final_spo, _ = spd.evolve(initial_spo, pytket_circ, trunc_val=1e-12, max_num_str=1000)

    native_exp_val = native_final_spo.get_expectation_value()
    pytket_exp_val = pytket_final_spo.get_expectation_value()
    assert np.isclose(float(np.asarray(native_exp_val)), float(np.asarray(pytket_exp_val)), atol=1e-6)

    module = BACKENDS[backend_name]
    native_terms = to_term_dict(backend_name, module, native_final_spo, n_qubits=3)
    pytket_terms = to_term_dict(backend_name, module, pytket_final_spo, n_qubits=3)
    assert native_terms.keys() == pytket_terms.keys()
    for term in native_terms:
        assert np.isclose(native_terms[term], pytket_terms[term], atol=1e-6)


@pytest.mark.parametrize("backend_name", ["numpy"], ids=["numpy"])
def test_openqasm_file_ir_matches_pytket_forward_execution_on_sample(backend_name):
    from pytket.qasm import circuit_from_qasm

    path = "tests/fixtures/open_qasm/periodic_small_8q.qasm"
    pytket_circ = circuit_from_qasm(path)
    measurement = list(range(8))
    native_ir = parse_openqasm_file(path, padded_system_size=32)
    initial_spo = make_initial_spo(backend_name, measurement, 8)

    native_final_spo, _ = spd.evolve(initial_spo, native_ir, trunc_val=1e-4, max_num_str=100000)
    pytket_final_spo, _ = spd.evolve(initial_spo, pytket_circ, trunc_val=1e-4, max_num_str=100000)

    native_exp_val = native_final_spo.get_expectation_value()
    pytket_exp_val = pytket_final_spo.get_expectation_value()
    assert np.isclose(float(np.asarray(native_exp_val)), float(np.asarray(pytket_exp_val)), atol=1e-6)
    assert native_final_spo.get_size() == pytket_final_spo.get_size()

    module = BACKENDS[backend_name]
    native_terms = to_term_dict(backend_name, module, native_final_spo, n_qubits=8)
    pytket_terms = to_term_dict(backend_name, module, pytket_final_spo, n_qubits=8)
    assert native_terms.keys() == pytket_terms.keys()
    for term in native_terms:
        assert np.isclose(native_terms[term], pytket_terms[term], atol=1e-6)
