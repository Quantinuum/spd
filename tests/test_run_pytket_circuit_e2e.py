import numpy as np
import pytest
from pytket.circuit import Circuit

import spd


def test_run_pytket_circuit_single_qubit_ry_z_expectation(backend_name):
    circ = Circuit(1)
    circ.Ry(0.25, 0)

    exp_val, final_spo = spd.run_pytket_circuit(
        circ,
        [0],
        1e-12,
        backend_name=backend_name,
    )

    assert np.isclose(float(np.asarray(exp_val)), np.cos(np.pi / 4), atol=1e-6)
    assert final_spo.get_size() == 2


def test_run_pytket_circuit_bell_state_observables(backend_name):
    circ = Circuit(2)
    circ.H(0)
    circ.CX(0, 1)

    exp_zz, _ = spd.run_pytket_circuit(
        circ,
        [0, 1],
        1e-12,
        backend_name=backend_name,
    )
    exp_xx, _ = spd.run_pytket_circuit(
        circ,
        {"XX": 1.0},
        1e-12,
        backend_name=backend_name,
    )

    assert np.isclose(float(np.asarray(exp_zz)), 1.0, atol=1e-6)
    assert np.isclose(float(np.asarray(exp_xx)), 1.0, atol=1e-6)


def test_run_pytket_circuit_backward_single_parameter_gradient(backend_name):
    circ = Circuit(1)
    circ.Ry(0.25, 0)

    exp_val, final_spo = spd.run_pytket_circuit(
        circ,
        [0],
        1e-12,
        backend_name=backend_name,
    )
    grads, backward_final_spo = spd.run_pytket_circuit_backward(
        circ,
        final_spo,
        1e-12,
        backend_name=backend_name,
    )

    theta = np.pi / 4
    assert np.isclose(float(np.asarray(exp_val)), np.cos(theta), atol=1e-6)
    assert len(grads) == 1
    assert np.isclose(float(np.asarray(grads[0])), -np.sin(theta), atol=1e-6)
    assert backward_final_spo.get_size() == 1
