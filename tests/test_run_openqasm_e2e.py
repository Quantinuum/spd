import numpy as np

import spd


def test_run_openqasm_str_single_qubit_rx_z_expectation(backend_name):
    source = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    rx(pi/2) q[0];
    """

    exp_val, final_spo = spd.run_openqasm_str(
        source,
        [0],
        trunc_val=1e-12,
        max_num_str=1000,
        backend_name=backend_name,
    )

    assert np.isclose(exp_val, 0.0, atol=1e-6)
    assert final_spo.get_size() >= 1


def test_run_openqasm_str_backward_single_parameter_gradient(backend_name):
    source = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    rx(pi/4) q[0];
    """

    exp_val, final_spo = spd.run_openqasm_str(
        source,
        [0],
        trunc_val=1e-12,
        max_num_str=1000,
        backend_name=backend_name,
    )
    grads, backward_final_spo = spd.run_openqasm_str_backward(
        source,
        final_spo,
        trunc_val=1e-12,
        max_num_str=1000,
        backend_name=backend_name,
    )

    assert np.isclose(exp_val, np.cos(np.pi / 4), atol=1e-6)
    assert len(grads) == 1
    assert np.isclose(grads[0], -np.sin(np.pi / 4), atol=1e-6)
    assert backward_final_spo.get_size() >= 1
