import numpy as np
import pytest

from spd import jax_backend, numpy_backend
from tests.helpers import to_grad_term_dict


def _configure_rotation_backends():
    for module in (jax_backend, numpy_backend):
        module.utils.set_packbit(32)
    jax_backend.set_precision("single")


def test_jax_search_update_merge_backward_matches_numpy_reference():
    _configure_rotation_backends()
    previous_algorithm = jax_backend.get_algorithm()
    jax_backend.set_algorithm("search_update_merge")

    try:
        spo_jax = jax_backend.create_op({"XIII": 1.0, "IIII": 0.25})
        spo_numpy = numpy_backend.create_op({"XIII": 1.0, "IIII": 0.25})
        spgo_jax = jax_backend.init_gradient_spo(
            spo_jax,
            loss_type="basis_expectation",
            basis="Z",
        )
        spgo_numpy = numpy_backend.init_gradient_spo(
            spo_numpy,
            loss_type="basis_expectation",
            basis="Z",
        )
        sigma_jax = np.asarray(jax_backend.utils.pauli_str_to_uint32("ZIII"))
        sigma_numpy = np.asarray(numpy_backend.utils.pauli_str_to_uint32("ZIII"))

        spgo_jax_out, _, grad_jax = jax_backend.conjugate_pauli_rot_backward(
            spgo_jax,
            sigma_jax,
            np.pi / 3,
            trunc_val=1e-12,
            max_num_str=1000,
        )
        spgo_numpy_out, _, grad_numpy = numpy_backend.conjugate_pauli_rot_backward(
            spgo_numpy,
            sigma_numpy,
            np.pi / 3,
            trunc_val=1e-12,
            max_num_str=1000,
        )

        actual_terms = to_grad_term_dict("jax", jax_backend, spgo_jax_out, n_qubits=4)
        expected_terms = to_grad_term_dict("numpy", numpy_backend, spgo_numpy_out, n_qubits=4)
        assert set(actual_terms.keys()) == set(expected_terms.keys())
        for key in actual_terms:
            assert actual_terms[key][0] == pytest.approx(expected_terms[key][0], abs=1e-6)
            assert actual_terms[key][1] == pytest.approx(expected_terms[key][1], abs=1e-6)
        assert grad_jax == pytest.approx(grad_numpy, abs=1e-6)
    finally:
        jax_backend.set_algorithm(previous_algorithm)
