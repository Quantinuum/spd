import numpy as np

import spd
from tests.helpers import assert_step_info_close


BACKENDS = {
    "numpy": spd.numpy_backend,
    "jax": spd.jax_backend,
}


def _configure_backend(module):
    module.utils.set_packbit(32)
    module.set_precision("single")
    if module is spd.jax_backend:
        module.set_algorithm("search_update_merge")


def test_forward_rotation_threshold_info_matches_expected(backend_name):
    module = BACKENDS[backend_name]
    _configure_backend(module)

    spo = module.create_op({"XIII": 1.0})
    sigma = np.asarray(module.utils.pauli_str_to_uint32("ZIII"))

    spo_out, num_string, step_info = module.conjugate_pauli_rot_forward(
        spo,
        sigma,
        np.pi / 4,
        trunc_val=0.8,
        max_num_str=1000,
    )

    assert num_string == 0
    if backend_name == "numpy":
        assert_step_info_close(
            step_info,
            {
                "num_str_truncated": 2,
                "truncated_l1_norm": np.sqrt(2),
                "truncated_l2_norm": 1.0,
            },
        )
    else:
        assert_step_info_close(
            step_info,
            {
                "num_str_truncated": 1,
                "truncated_l1_norm": np.sqrt(0.5),
                "truncated_l2_norm": np.sqrt(0.5),
            },
        )


def test_forward_rotation_max_num_str_info_matches_expected(backend_name):
    module = BACKENDS[backend_name]
    _configure_backend(module)

    spo = module.create_op({"XIII": 1.0})
    sigma = np.asarray(module.utils.pauli_str_to_uint32("ZIII"))

    spo_out, num_string, step_info = module.conjugate_pauli_rot_forward(
        spo,
        sigma,
        np.pi / 4,
        trunc_val=1e-12,
        max_num_str=1,
    )

    assert num_string == 1
    assert spo_out.get_size() == 1
    assert_step_info_close(
        step_info,
        {
            "num_str_truncated": 1,
            "truncated_l1_norm": np.sqrt(0.5),
            "truncated_l2_norm": np.sqrt(0.5),
        },
    )


def test_backward_rotation_threshold_info_matches_expected(backend_name):
    module = BACKENDS[backend_name]
    _configure_backend(module)

    spo = module.create_op({"XIII": 1.0})
    spgo = module.init_gradient_spo(spo, loss_type="basis_expectation", basis="0")
    sigma = np.asarray(module.utils.pauli_str_to_uint32("ZIII"))

    spgo_out, num_string, _, step_info = module.conjugate_pauli_rot_backward(
        spgo,
        sigma,
        np.pi / 4,
        trunc_val=0.8,
        max_num_str=1000,
    )

    assert num_string == 0
    if backend_name == "numpy":
        assert_step_info_close(
            step_info,
            {
                "num_str_truncated": 2,
                "truncated_l1_norm": np.sqrt(2),
                "truncated_l2_norm": 1.0,
            },
        )
    else:
        assert_step_info_close(
            step_info,
            {
                "num_str_truncated": 1,
                "truncated_l1_norm": np.sqrt(0.5),
                "truncated_l2_norm": np.sqrt(0.5),
            },
        )


def test_backward_rotation_max_num_str_info_matches_expected(backend_name):
    module = BACKENDS[backend_name]
    _configure_backend(module)

    spo = module.create_op({"XIII": 1.0})
    spgo = module.init_gradient_spo(spo, loss_type="basis_expectation", basis="0")
    sigma = np.asarray(module.utils.pauli_str_to_uint32("ZIII"))

    spgo_out, num_string, _, step_info = module.conjugate_pauli_rot_backward(
        spgo,
        sigma,
        np.pi / 4,
        trunc_val=1e-12,
        max_num_str=1,
    )

    assert num_string == 1
    assert spgo_out.get_size() == 1
    assert_step_info_close(
        step_info,
        {
            "num_str_truncated": 1,
            "truncated_l1_norm": np.sqrt(0.5),
            "truncated_l2_norm": np.sqrt(0.5),
        },
    )
