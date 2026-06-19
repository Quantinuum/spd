import numpy as np
import pytest

from spd import jax_backend
from tests.helpers import assert_step_info_close, to_grad_term_dict, to_term_dict


def _configure_jax():
    jax_backend.utils.set_packbit(32)
    jax_backend.set_precision("single")


def test_search_update_merge_donate_forward_matches_search_update_merge_at_cap():
    _configure_jax()
    previous_algorithm = jax_backend.get_algorithm()
    sigma = np.asarray(jax_backend.utils.pauli_str_to_uint32("ZIII"))

    try:
        jax_backend.set_algorithm("search_update_merge")
        spo_search = jax_backend.create_op({"XIII": 1.0, "ZIII": 0.5})
        expected_spo, expected_count, expected_info = jax_backend.conjugate_pauli_rot_forward(
            spo_search,
            sigma,
            np.pi / 3,
            trunc_val=1e-12,
            max_num_str=2,
        )

        jax_backend.set_algorithm("search_update_merge_donate")
        spo_donate = jax_backend.create_op({"XIII": 1.0, "ZIII": 0.5})
        actual_spo, actual_count, actual_info = jax_backend.conjugate_pauli_rot_forward(
            spo_donate,
            sigma,
            np.pi / 3,
            trunc_val=1e-12,
            max_num_str=2,
        )

        assert actual_count == expected_count
        assert actual_spo.get_size() == expected_spo.get_size()
        assert to_term_dict("jax", jax_backend, actual_spo, n_qubits=4) == pytest.approx(
            to_term_dict("jax", jax_backend, expected_spo, n_qubits=4),
            abs=1e-6,
        )
        assert_step_info_close(actual_info, expected_info)
    finally:
        jax_backend.set_algorithm(previous_algorithm)


def test_search_update_merge_donate_forward_matches_search_update_merge_when_shrinking():
    _configure_jax()
    previous_algorithm = jax_backend.get_algorithm()
    sigma = np.asarray(jax_backend.utils.pauli_str_to_uint32("ZIII"))

    try:
        jax_backend.set_algorithm("search_update_merge")
        spo_search = jax_backend.create_op({"XIII": 1.0, "ZIII": 0.5})
        expected_spo, expected_count, expected_info = jax_backend.conjugate_pauli_rot_forward(
            spo_search,
            sigma,
            np.pi / 4,
            trunc_val=0.8,
            max_num_str=2,
        )

        jax_backend.set_algorithm("search_update_merge_donate")
        spo_donate = jax_backend.create_op({"XIII": 1.0, "ZIII": 0.5})
        actual_spo, actual_count, actual_info = jax_backend.conjugate_pauli_rot_forward(
            spo_donate,
            sigma,
            np.pi / 4,
            trunc_val=0.8,
            max_num_str=2,
        )

        assert actual_count == expected_count
        assert actual_spo.get_size() == expected_spo.get_size()
        assert to_term_dict("jax", jax_backend, actual_spo, n_qubits=4) == pytest.approx(
            to_term_dict("jax", jax_backend, expected_spo, n_qubits=4),
            abs=1e-6,
        )
        assert_step_info_close(actual_info, expected_info)
    finally:
        jax_backend.set_algorithm(previous_algorithm)


def test_search_update_merge_donate_backward_matches_search_update_merge_at_cap():
    _configure_jax()
    previous_algorithm = jax_backend.get_algorithm()
    sigma = np.asarray(jax_backend.utils.pauli_str_to_uint32("ZIII"))

    try:
        jax_backend.set_algorithm("search_update_merge")
        spo_search = jax_backend.create_op({"XIII": 1.0, "IIII": 0.25})
        spgo_search = jax_backend.init_gradient_spo(
            spo_search,
            loss_type="basis_expectation",
            basis="Z",
        )
        expected_spgo, expected_count, expected_grad, expected_info = (
            jax_backend.conjugate_pauli_rot_backward(
                spgo_search,
                sigma,
                np.pi / 3,
                trunc_val=1e-12,
                max_num_str=2,
            )
        )

        jax_backend.set_algorithm("search_update_merge_donate")
        spo_donate = jax_backend.create_op({"XIII": 1.0, "IIII": 0.25})
        spgo_donate = jax_backend.init_gradient_spo(
            spo_donate,
            loss_type="basis_expectation",
            basis="Z",
        )
        actual_spgo, actual_count, actual_grad, actual_info = jax_backend.conjugate_pauli_rot_backward(
            spgo_donate,
            sigma,
            np.pi / 3,
            trunc_val=1e-12,
            max_num_str=2,
        )

        assert actual_count == expected_count
        assert actual_spgo.get_size() == expected_spgo.get_size()
        assert actual_grad == pytest.approx(expected_grad, abs=1e-6)
        assert to_grad_term_dict("jax", jax_backend, actual_spgo, n_qubits=4) == pytest.approx(
            to_grad_term_dict("jax", jax_backend, expected_spgo, n_qubits=4),
            abs=1e-6,
        )
        assert_step_info_close(actual_info, expected_info)
    finally:
        jax_backend.set_algorithm(previous_algorithm)


def test_search_update_merge_donate_falls_back_while_growing():
    _configure_jax()
    previous_algorithm = jax_backend.get_algorithm()
    sigma = np.asarray(jax_backend.utils.pauli_str_to_uint32("ZIII"))

    try:
        jax_backend.set_algorithm("search_update_merge")
        spo_search = jax_backend.create_op({"XIII": 1.0})
        expected_spo, expected_count, expected_info = jax_backend.conjugate_pauli_rot_forward(
            spo_search,
            sigma,
            np.pi / 3,
            trunc_val=1e-12,
            max_num_str=1000,
        )

        jax_backend.set_algorithm("search_update_merge_donate")
        spo_donate = jax_backend.create_op({"XIII": 1.0})
        actual_spo, actual_count, actual_info = jax_backend.conjugate_pauli_rot_forward(
            spo_donate,
            sigma,
            np.pi / 3,
            trunc_val=1e-12,
            max_num_str=1000,
        )

        assert actual_count == expected_count
        assert actual_spo.get_size() == expected_spo.get_size()
        assert to_term_dict("jax", jax_backend, actual_spo, n_qubits=4) == pytest.approx(
            to_term_dict("jax", jax_backend, expected_spo, n_qubits=4),
            abs=1e-6,
        )
        assert_step_info_close(actual_info, expected_info)
    finally:
        jax_backend.set_algorithm(previous_algorithm)
