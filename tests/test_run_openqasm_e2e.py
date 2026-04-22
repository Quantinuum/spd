import numpy as np

import spd
from spd.openqasm_frontend import parse_openqasm_str
from tests.helpers import assert_info_consistent, make_initial_spo


def test_openqasm_ir_single_qubit_rx_z_expectation(backend_name):
    source = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    rx(pi/2) q[0];
    """
    _, operations = parse_openqasm_str(source, padded_system_size=32)
    initial_spo = make_initial_spo(backend_name, [0], 1)
    final_spo, info = spd.evolve(initial_spo, operations, trunc_val=1e-12, max_num_str=1000)
    exp_val = final_spo.get_expectation_value()

    assert np.isclose(exp_val, 0.0, atol=1e-6)
    assert final_spo.get_size() >= 1
    assert_info_consistent(info, expected_steps=1)


def test_openqasm_ir_backward_single_parameter_gradient(backend_name):
    source = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    rx(pi/4) q[0];
    """
    _, operations = parse_openqasm_str(source, padded_system_size=32)
    initial_spo = make_initial_spo(backend_name, [0], 1)
    final_spo, _ = spd.evolve(initial_spo, operations, trunc_val=1e-12, max_num_str=1000)
    initial_spgo = spd.init_gradient_spo(final_spo)
    backward_final_spo, grads, info = spd.backpropagate(
        initial_spgo,
        operations,
        trunc_val=1e-12,
        max_num_str=1000,
    )

    assert np.isclose(final_spo.get_expectation_value(), np.cos(np.pi / 4), atol=1e-6)
    assert len(grads) == 1
    assert np.isclose(grads[0], -np.sin(np.pi / 4), atol=1e-6)
    assert backward_final_spo.get_size() >= 1
    assert_info_consistent(info, expected_steps=1)
