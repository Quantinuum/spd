import numpy as np
import pytest
from pytket.circuit import Circuit

import spd

MAX_NUM_STR = 1_000_000
BACKENDS = {
    "numpy": spd.numpy_backend,
    "jax": spd.jax_backend,
}


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


def test_run_pytket_backward_from_spgo_matches_wrapper(backend_name):
    circ = Circuit(1)
    circ.Ry(0.25, 0)

    _, final_spo = spd.run_pytket_circuit(
        circ,
        [0],
        1e-12,
        MAX_NUM_STR,
        backend_name=backend_name,
    )
    initial_spgo = spd.init_gradient_spo(
        final_spo,
        basis="0",
        backend_name=backend_name,
    )

    direct_grads, direct_final_spgo = spd.run_pytket_backward_from_spgo(
        circ,
        initial_spgo,
        1e-12,
        MAX_NUM_STR,
        backend_name=backend_name,
    )
    wrapped_grads, wrapped_final_spgo = spd.run_pytket_circuit_backward(
        circ,
        final_spo,
        1e-12,
        MAX_NUM_STR,
        backend_name=backend_name,
    )

    assert np.allclose(np.asarray(direct_grads), np.asarray(wrapped_grads), atol=1e-6)
    assert direct_final_spgo.get_size() == wrapped_final_spgo.get_size()


def test_run_pytket_circuit_backward_l2_difference_matches_finite_difference(backend_name):
    circ = Circuit(1)
    circ.Ry(0.25, 0)
    target_spo = BACKENDS[backend_name].create_op({"Z": 0.4, "X": -0.2})

    _, final_spo = spd.run_pytket_circuit(
        circ,
        [0],
        1e-12,
        MAX_NUM_STR,
        backend_name=backend_name,
    )
    grads, backward_final_spgo = spd.run_pytket_circuit_backward(
        circ,
        final_spo,
        1e-12,
        MAX_NUM_STR,
        backend_name=backend_name,
        loss_type="l2_difference",
        target_spo=target_spo,
    )

    def l2_loss(param):
        shifted = Circuit(1)
        shifted.Ry(param, 0)
        _, shifted_spo = spd.run_pytket_circuit(
            shifted,
            [0],
            1e-12,
            MAX_NUM_STR,
            backend_name=backend_name,
        )
        diff = shifted_spo - target_spo
        return float(np.asarray(diff.get_norm_square()))

    eps = 1e-5
    finite_difference_grad = (l2_loss(0.25 + eps) - l2_loss(0.25 - eps)) / (2 * eps)

    assert len(grads) == 1
    assert np.isclose(float(np.asarray(grads[0])), finite_difference_grad / np.pi, atol=5e-4)
    assert backward_final_spgo.get_size() >= 1


def test_run_pytket_circuit_backward_basis_expectation_plus_ose_matches_direct_runner(backend_name):
    circ = Circuit(1)
    circ.Ry(0.25, 0)
    lambda_ose = 0.2
    alpha = 1.0

    _, final_spo = spd.run_pytket_circuit(
        circ,
        [0],
        1e-12,
        MAX_NUM_STR,
        backend_name=backend_name,
    )
    grads, _ = spd.run_pytket_circuit_backward(
        circ,
        final_spo,
        1e-12,
        MAX_NUM_STR,
        backend_name=backend_name,
        loss_type="basis_expectation",
        lambda_ose=lambda_ose,
        alpha=alpha,
    )
    initial_spgo = spd.init_gradient_spo(
        final_spo,
        loss_type="basis_expectation",
        lambda_ose=lambda_ose,
        alpha=alpha,
        backend_name=backend_name,
    )
    direct_grads, direct_final_spgo = spd.run_pytket_backward_from_spgo(
        circ,
        initial_spgo,
        1e-12,
        MAX_NUM_STR,
        backend_name=backend_name,
    )

    assert len(grads) == 1
    assert np.allclose(np.asarray(grads), np.asarray(direct_grads), atol=1e-6)
    assert direct_final_spgo.get_size() == 1


def test_init_gradient_spo_rejects_backend_mismatched_spo():
    numpy_spo = spd.numpy_backend.create_op({"Z": 1.0})

    with pytest.raises(TypeError, match="final_spo must be a jax SparsePauliOp"):
        spd.init_gradient_spo(
            numpy_spo,
            backend_name="jax",
        )


def test_init_gradient_spo_rejects_backend_mismatched_target_spo():
    numpy_spo = spd.numpy_backend.create_op({"Z": 1.0})
    jax_target = spd.jax_backend.create_op({"Z": 0.5})

    with pytest.raises(TypeError, match="target_spo must be a numpy SparsePauliOp"):
        spd.init_gradient_spo(
            numpy_spo,
            backend_name="numpy",
            loss_type="l2_difference",
            target_spo=jax_target,
        )


def test_run_pytket_backward_from_spgo_rejects_backend_mismatched_spgo():
    circ = Circuit(1)
    circ.Ry(0.25, 0)

    _, numpy_final_spo = spd.run_pytket_circuit(
        circ,
        [0],
        1e-12,
        MAX_NUM_STR,
        backend_name="numpy",
    )
    numpy_spgo = spd.init_gradient_spo(
        numpy_final_spo,
        backend_name="numpy",
    )

    with pytest.raises(TypeError, match="initial_spgo must be a jax SparsePauliGradientOp"):
        spd.run_pytket_backward_from_spgo(
            circ,
            numpy_spgo,
            1e-12,
            MAX_NUM_STR,
            backend_name="jax",
        )


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
