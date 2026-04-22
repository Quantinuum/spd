import numpy as np
import pytest

from spd import jax_backend, numpy_backend
from spd.jax_backend.kernels import DEFAULT_ALGORITHM
from tests.helpers import to_term_dict


def _configure_rotation_backends():
    for module in (jax_backend, numpy_backend):
        module.utils.set_packbit(32)
    jax_backend.set_precision("single")


def test_jax_set_algorithm_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported JAX algorithm"):
        jax_backend.set_algorithm("not_a_real_algorithm")


def test_jax_default_algorithm_is_stack_sort_merge():
    assert DEFAULT_ALGORITHM == "stack_sort_merge"


def test_jax_stack_sort_merge_selection_preserves_current_rotation_behavior():
    _configure_rotation_backends()
    previous_algorithm = jax_backend.get_algorithm()
    jax_backend.set_algorithm("stack_sort_merge")

    try:
        spo_jax = jax_backend.create_op({"XIII": 1.0})
        spo_numpy = numpy_backend.create_op({"XIII": 1.0})
        sigma_jax = np.asarray(jax_backend.utils.pauli_str_to_uint32("ZIII"))
        sigma_numpy = np.asarray(numpy_backend.utils.pauli_str_to_uint32("ZIII"))

        spo_jax_out, _ = jax_backend.conjugate_pauli_rot_forward(
            spo_jax,
            sigma_jax,
            np.pi / 3,
            trunc_val=1e-12,
            max_num_str=1000,
        )
        spo_numpy_out, _ = numpy_backend.conjugate_pauli_rot_forward(
            spo_numpy,
            sigma_numpy,
            np.pi / 3,
            trunc_val=1e-12,
            max_num_str=1000,
        )

        assert to_term_dict("jax", jax_backend, spo_jax_out, n_qubits=4) == pytest.approx(
            to_term_dict("numpy", numpy_backend, spo_numpy_out, n_qubits=4),
            abs=1e-6,
        )
    finally:
        jax_backend.set_algorithm(previous_algorithm)


def test_jax_search_update_merge_forward_matches_numpy_reference():
    _configure_rotation_backends()
    previous_algorithm = jax_backend.get_algorithm()
    jax_backend.set_algorithm("search_update_merge")

    try:
        spo_jax = jax_backend.create_op({"XIII": 1.0, "ZIII": 0.5})
        spo_numpy = numpy_backend.create_op({"XIII": 1.0, "ZIII": 0.5})
        sigma_jax = np.asarray(jax_backend.utils.pauli_str_to_uint32("ZIII"))
        sigma_numpy = np.asarray(numpy_backend.utils.pauli_str_to_uint32("ZIII"))

        spo_jax_out, _ = jax_backend.conjugate_pauli_rot_forward(
            spo_jax,
            sigma_jax,
            np.pi / 3,
            trunc_val=1e-12,
            max_num_str=1000,
        )
        spo_numpy_out, _ = numpy_backend.conjugate_pauli_rot_forward(
            spo_numpy,
            sigma_numpy,
            np.pi / 3,
            trunc_val=1e-12,
            max_num_str=1000,
        )

        actual_terms = to_term_dict("jax", jax_backend, spo_jax_out, n_qubits=4)
        expected_terms = to_term_dict("numpy", numpy_backend, spo_numpy_out, n_qubits=4)
        assert actual_terms == pytest.approx(expected_terms, abs=1e-6)
    finally:
        jax_backend.set_algorithm(previous_algorithm)
