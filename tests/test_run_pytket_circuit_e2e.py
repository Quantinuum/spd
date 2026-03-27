import numpy as np
import pytest
from pytket.circuit import Circuit

import spd

MAX_NUM_STR = 1_000_000


def test_run_pytket_circuit_single_qubit_ry_z_expectation(backend_name):
    circ = Circuit(1)
    circ.Ry(0.25, 0)

    exp_val, final_spo = spd.run_pytket_circuit(
        circ,
        [0],
        1e-12,
        MAX_NUM_STR,
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
        MAX_NUM_STR,
        backend_name=backend_name,
    )
    exp_xx, _ = spd.run_pytket_circuit(
        circ,
        {"XX": 1.0},
        1e-12,
        MAX_NUM_STR,
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
        MAX_NUM_STR,
        backend_name=backend_name,
    )
    grads, backward_final_spo = spd.run_pytket_circuit_backward(
        circ,
        final_spo,
        1e-12,
        MAX_NUM_STR,
        backend_name=backend_name,
    )

    theta = np.pi / 4
    assert np.isclose(float(np.asarray(exp_val)), np.cos(theta), atol=1e-6)
    assert len(grads) == 1
    assert np.isclose(float(np.asarray(grads[0])), -np.sin(theta), atol=1e-6)
    assert backward_final_spo.get_size() == 1


def test_jax_run_pytket_circuit_max_num_str_caps_size_and_rounds_to_pow2():
    circ = Circuit(2)
    circ.Ry(0.25, 0)
    circ.Ry(0.25, 1)

    _, spo_no_cap = spd.run_pytket_circuit(
        circ,
        [0, 1],
        1e-12,
        MAX_NUM_STR,
        backend_name="jax",
    )
    _, spo_cap_2 = spd.run_pytket_circuit(
        circ,
        [0, 1],
        1e-12,
        2,
        backend_name="jax",
    )
    _, spo_cap_3 = spd.run_pytket_circuit(
        circ,
        [0, 1],
        1e-12,
        3,
        backend_name="jax",
    )

    assert spo_no_cap.get_size() == 4
    assert spo_cap_2.get_size() == 2
    assert spo_cap_3.get_size() == 4


def test_jax_run_pytket_circuit_backward_respects_max_num_str():
    circ = Circuit(2)
    circ.Ry(0.25, 0)
    circ.Ry(0.25, 1)

    _, final_spo = spd.run_pytket_circuit(
        circ,
        [0, 1],
        1e-12,
        MAX_NUM_STR,
        backend_name="jax",
    )
    grads, backward_final_spo = spd.run_pytket_circuit_backward(
        circ,
        final_spo,
        1e-12,
        2,
        backend_name="jax",
    )

    assert len(grads) == 2
    assert backward_final_spo.get_size() <= 2


def test_numpy_run_pytket_circuit_respects_max_num_str():
    circ = Circuit(2)
    circ.Ry(0.25, 0)
    circ.Ry(0.25, 1)

    _, spo = spd.run_pytket_circuit(
        circ,
        [0, 1],
        1e-12,
        2,
        backend_name="numpy",
    )

    assert spo.get_size() <= 2


def test_numpy_run_pytket_circuit_backward_respects_max_num_str():
    circ = Circuit(2)
    circ.Ry(0.25, 0)
    circ.Ry(0.25, 1)

    _, final_spo = spd.run_pytket_circuit(
        circ,
        [0, 1],
        1e-12,
        MAX_NUM_STR,
        backend_name="numpy",
    )
    grads, backward_final_spo = spd.run_pytket_circuit_backward(
        circ,
        final_spo,
        1e-12,
        2,
        backend_name="numpy",
    )

    assert len(grads) == 2
    assert backward_final_spo.get_size() <= 2
