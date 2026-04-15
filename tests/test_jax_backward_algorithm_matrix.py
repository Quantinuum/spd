import numpy as np
import pytest
from pytket.circuit import Circuit

import spd
from spd import jax_backend, numpy_backend
from tests.helpers import to_grad_term_dict


JAX_BACKWARD_ALGORITHMS = ["stack_sort_merge", "search_update_merge"]
RUNNER_TRUNC_VAL = 1e-6


def _configure_rotation_backends():
    for module in (jax_backend, numpy_backend):
        module.utils.set_packbit(32)
    jax_backend.set_precision("single")


def _assert_grad_term_dicts_close(actual, expected, atol=1e-6):
    actual = {
        key: value
        for key, value in actual.items()
        if not (np.isclose(value[0], 0.0, atol=atol) and np.isclose(value[1], 0.0, atol=atol))
    }
    expected = {
        key: value
        for key, value in expected.items()
        if not (np.isclose(value[0], 0.0, atol=atol) and np.isclose(value[1], 0.0, atol=atol))
    }
    assert set(actual.keys()) == set(expected.keys())
    for key in actual:
        assert actual[key][0] == pytest.approx(expected[key][0], abs=atol)
        assert actual[key][1] == pytest.approx(expected[key][1], abs=atol)


@pytest.mark.parametrize("jax_algorithm", JAX_BACKWARD_ALGORITHMS, indirect=True)
def test_jax_backward_algorithms_match_numpy_on_direct_kernel_basis_expectation(jax_algorithm):
    _configure_rotation_backends()

    spo_jax = jax_backend.create_op({"XIII": 1.0, "IIII": 0.25})
    spo_numpy = numpy_backend.create_op({"XIII": 1.0, "IIII": 0.25})
    spgo_jax = jax_backend.init_gradient_spo(spo_jax, loss_type="basis_expectation", basis="Z")
    spgo_numpy = numpy_backend.init_gradient_spo(spo_numpy, loss_type="basis_expectation", basis="Z")
    sigma_jax = np.asarray(jax_backend.utils.pauli_str_to_uint32("ZIII"))
    sigma_numpy = np.asarray(numpy_backend.utils.pauli_str_to_uint32("ZIII"))

    spgo_jax_out, _, grad_jax = jax_backend.conjugate_pauli_rot_backward(
        spgo_jax, sigma_jax, np.pi / 3, trunc_val=1e-12, max_num_str=1000
    )
    spgo_numpy_out, _, grad_numpy = numpy_backend.conjugate_pauli_rot_backward(
        spgo_numpy, sigma_numpy, np.pi / 3, trunc_val=1e-12, max_num_str=1000
    )

    actual_terms = to_grad_term_dict("jax", jax_backend, spgo_jax_out, n_qubits=4)
    expected_terms = to_grad_term_dict("numpy", numpy_backend, spgo_numpy_out, n_qubits=4)
    _assert_grad_term_dicts_close(actual_terms, expected_terms)
    assert grad_jax == pytest.approx(grad_numpy, abs=1e-6)


@pytest.mark.parametrize("jax_algorithm", JAX_BACKWARD_ALGORITHMS, indirect=True)
def test_jax_backward_algorithms_match_numpy_on_runner_basis_expectation(jax_algorithm):
    _configure_rotation_backends()

    circ = Circuit(1)
    circ.Ry(0.25, 0)

    _, final_spo_jax = spd.run_pytket_circuit(
        circ, [0], RUNNER_TRUNC_VAL, 1000, backend_name="jax"
    )
    _, final_spo_numpy = spd.run_pytket_circuit(
        circ, [0], RUNNER_TRUNC_VAL, 1000, backend_name="numpy"
    )

    grads_jax, spgo_jax = spd.run_pytket_circuit_backward(
        circ, final_spo_jax, RUNNER_TRUNC_VAL, 1000, backend_name="jax"
    )
    grads_numpy, spgo_numpy = spd.run_pytket_circuit_backward(
        circ, final_spo_numpy, RUNNER_TRUNC_VAL, 1000, backend_name="numpy"
    )

    actual_terms = to_grad_term_dict("jax", jax_backend, spgo_jax, n_qubits=1)
    expected_terms = to_grad_term_dict("numpy", numpy_backend, spgo_numpy, n_qubits=1)
    _assert_grad_term_dicts_close(actual_terms, expected_terms)
    assert np.asarray(grads_jax) == pytest.approx(np.asarray(grads_numpy), abs=1e-6)


@pytest.mark.parametrize("jax_algorithm", JAX_BACKWARD_ALGORITHMS, indirect=True)
def test_jax_backward_algorithms_match_numpy_on_runner_l2_difference(jax_algorithm):
    _configure_rotation_backends()

    circ = Circuit(1)
    circ.Ry(0.25, 0)
    target_jax = jax_backend.create_op({"Z": 0.4, "X": -0.2})
    target_numpy = numpy_backend.create_op({"Z": 0.4, "X": -0.2})

    _, final_spo_jax = spd.run_pytket_circuit(
        circ, [0], RUNNER_TRUNC_VAL, 1000, backend_name="jax"
    )
    _, final_spo_numpy = spd.run_pytket_circuit(
        circ, [0], RUNNER_TRUNC_VAL, 1000, backend_name="numpy"
    )

    grads_jax, spgo_jax = spd.run_pytket_circuit_backward(
        circ,
        final_spo_jax,
        RUNNER_TRUNC_VAL,
        1000,
        backend_name="jax",
        loss_type="l2_difference",
        target_spo=target_jax,
    )
    grads_numpy, spgo_numpy = spd.run_pytket_circuit_backward(
        circ,
        final_spo_numpy,
        RUNNER_TRUNC_VAL,
        1000,
        backend_name="numpy",
        loss_type="l2_difference",
        target_spo=target_numpy,
    )

    actual_terms = to_grad_term_dict("jax", jax_backend, spgo_jax, n_qubits=1)
    expected_terms = to_grad_term_dict("numpy", numpy_backend, spgo_numpy, n_qubits=1)
    _assert_grad_term_dicts_close(actual_terms, expected_terms)
    assert np.asarray(grads_jax) == pytest.approx(np.asarray(grads_numpy), abs=1e-6)
