import numpy as np

from spd import jax_backend, numpy_backend
from tests.helpers import to_grad_term_dict, to_term_dict


BACKENDS = {
    "numpy": numpy_backend,
    "jax": jax_backend,
}


def _set_packbit():
    for module in BACKENDS.values():
        module.utils.set_packbit(32)
    jax_backend.set_precision("single")


def _build_ops(builder, *args, **kwargs):
    _set_packbit()
    return {
        backend_name: builder(module, *args, **kwargs)
        for backend_name, module in BACKENDS.items()
    }


def _assert_term_dicts_close(actual, expected, atol=1e-6):
    assert set(actual.keys()) == set(expected.keys())
    for key in actual:
        assert np.isclose(actual[key], expected[key], atol=atol)


def _assert_grad_term_dicts_close(actual, expected, atol=1e-6):
    assert set(actual.keys()) == set(expected.keys())
    for key in actual:
        assert np.isclose(actual[key][0], expected[key][0], atol=atol)
        assert np.isclose(actual[key][1], expected[key][1], atol=atol)


def test_create_op_semantics_match_between_backends():
    pauli_dict = {
        "XIZY": 1.5,
        "ZZII": -0.25,
        "IIII": 2.0,
    }

    ops = _build_ops(lambda module, payload: module.create_op(payload), pauli_dict)
    terms_by_backend = {
        backend_name: to_term_dict(backend_name, BACKENDS[backend_name], spo, n_qubits=4)
        for backend_name, spo in ops.items()
    }

    _assert_term_dicts_close(terms_by_backend["numpy"], terms_by_backend["jax"])


def test_create_measurement_op_semantics_match_between_backends():
    measurement_dict = {
        (0, 2): 1.0,
        (1,): -0.5,
    }

    ops = _build_ops(
        lambda module, payload, padded_system_size: module.create_measurement_op(
            payload, padded_system_size
        ),
        measurement_dict,
        32,
    )
    terms_by_backend = {
        backend_name: to_term_dict(backend_name, BACKENDS[backend_name], spo, n_qubits=4)
        for backend_name, spo in ops.items()
    }

    _assert_term_dicts_close(terms_by_backend["numpy"], terms_by_backend["jax"])


def test_size_norm_and_expectation_match_between_backends():
    pauli_dict = {
        "IIII": 2.0,
        "ZIII": -0.5,
        "XIII": 0.75,
        "IIIX": 1.25,
    }

    ops = _build_ops(lambda module, payload: module.create_op(payload), pauli_dict)

    sizes = {backend_name: ops[backend_name].get_size() for backend_name in BACKENDS}
    norm_squares = {
        backend_name: float(np.asarray(ops[backend_name].get_norm_square()))
        for backend_name in BACKENDS
    }
    exp_z = {
        backend_name: float(np.asarray(ops[backend_name].get_expectation_value(basis="Z")))
        for backend_name in BACKENDS
    }
    exp_x = {
        backend_name: float(np.asarray(ops[backend_name].get_expectation_value(basis="X")))
        for backend_name in BACKENDS
    }

    assert sizes["numpy"] == sizes["jax"]
    assert np.isclose(norm_squares["numpy"], norm_squares["jax"], atol=1e-6)
    assert np.isclose(exp_z["numpy"], exp_z["jax"], atol=1e-6)
    assert np.isclose(exp_x["numpy"], exp_x["jax"], atol=1e-6)


def test_operator_stabilizer_entropy_matches_between_backends():
    pauli_dict = {
        "IIII": 2.0,
        "ZIII": -0.5,
        "XIII": 0.75,
        "IIIX": 1.25,
    }

    ops = _build_ops(lambda module, payload: module.create_op(payload), pauli_dict)

    ose_alpha_1 = {
        backend_name: float(np.asarray(ops[backend_name].get_operator_stabilizer_entropy(alpha=1.0)))
        for backend_name in BACKENDS
    }
    ose_alpha_2 = {
        backend_name: float(np.asarray(ops[backend_name].get_operator_stabilizer_entropy(alpha=2.0)))
        for backend_name in BACKENDS
    }

    assert np.isclose(ose_alpha_1["numpy"], ose_alpha_1["jax"], atol=1e-6)
    assert np.isclose(ose_alpha_2["numpy"], ose_alpha_2["jax"], atol=1e-6)


def test_sparse_pauli_arithmetic_matches_between_backends():
    ops_a = _build_ops(lambda module: module.create_op({"IIII": 1.0, "ZIII": -0.5, "XIII": 0.25}))
    ops_b = _build_ops(lambda module: module.create_op({"ZIII": 0.5, "IIIX": 1.25}))

    sum_terms = {
        backend_name: to_term_dict(backend_name, BACKENDS[backend_name], ops_a[backend_name] + ops_b[backend_name], n_qubits=4)
        for backend_name in BACKENDS
    }
    diff_terms = {
        backend_name: to_term_dict(backend_name, BACKENDS[backend_name], ops_a[backend_name] - ops_b[backend_name], n_qubits=4)
        for backend_name in BACKENDS
    }
    scaled_terms = {
        backend_name: to_term_dict(backend_name, BACKENDS[backend_name], 2.0 * ops_a[backend_name], n_qubits=4)
        for backend_name in BACKENDS
    }

    _assert_term_dicts_close(sum_terms["numpy"], sum_terms["jax"])
    _assert_term_dicts_close(diff_terms["numpy"], diff_terms["jax"])
    _assert_term_dicts_close(scaled_terms["numpy"], scaled_terms["jax"])


def test_init_gradient_spo_basis_expectation_semantics_match_between_backends():
    pauli_dict = {
        "IIII": 2.0,
        "ZIII": -0.5,
        "XIII": 0.75,
        "YIIX": -1.25,
    }
    ops = _build_ops(lambda module, payload: module.create_op(payload), pauli_dict)
    grads = {
        backend_name: BACKENDS[backend_name].init_gradient_spo(
            spo,
            loss_type="basis_expectation",
            basis="Z",
        )
        for backend_name, spo in ops.items()
    }

    grad_terms = {
        backend_name: to_grad_term_dict(backend_name, BACKENDS[backend_name], spgo, n_qubits=4)
        for backend_name, spgo in grads.items()
    }

    assert grad_terms["numpy"] == grad_terms["jax"]
    assert np.isclose(
        float(np.asarray(grads["numpy"].get_norm_square())),
        float(np.asarray(grads["jax"].get_norm_square())),
        atol=1e-6,
    )
    assert np.isclose(
        float(np.asarray(grads["numpy"].get_operator_stabilizer_entropy(alpha=1.0))),
        float(np.asarray(grads["jax"].get_operator_stabilizer_entropy(alpha=1.0))),
        atol=1e-6,
    )
    assert np.isclose(
        float(np.asarray(grads["numpy"].get_operator_stabilizer_entropy(alpha=2.0))),
        float(np.asarray(grads["jax"].get_operator_stabilizer_entropy(alpha=2.0))),
        atol=1e-6,
    )


def test_sparse_pauli_gradient_op_arithmetic_matches_between_backends():
    ops = _build_ops(
        lambda module: module.init_gradient_spo(
            module.create_op({"IIII": 1.0, "ZIII": -0.5, "XIII": 0.25}),
            loss_type="basis_expectation",
            basis="Z",
        )
    )
    ose_ops = _build_ops(
        lambda module: module.init_gradient_from_ose(
            module.create_op({"IIII": 1.0, "ZIII": -0.5, "XIII": 0.25}),
            alpha=1.0,
        )
    )

    sum_terms = {
        backend_name: to_grad_term_dict(
            backend_name,
            BACKENDS[backend_name],
            ops[backend_name] + 0.25 * ose_ops[backend_name],
            n_qubits=4,
        )
        for backend_name in BACKENDS
    }

    _assert_grad_term_dicts_close(sum_terms["numpy"], sum_terms["jax"])


def test_init_gradient_spo_composition_matches_between_backends():
    ops = _build_ops(lambda module: module.create_op({"IIII": 1.0, "ZIII": -0.5, "XIII": 0.25}))
    target_ops = _build_ops(lambda module: module.create_op({"IIII": 0.5, "YIII": -0.125}))

    basis_grads = {
        backend_name: BACKENDS[backend_name].init_gradient_spo(
            ops[backend_name],
            loss_type="basis_expectation",
            basis="Z",
            lambda_ose=0.5,
            alpha=2.0,
        )
        for backend_name in BACKENDS
    }
    l2_grads = {
        backend_name: BACKENDS[backend_name].init_gradient_spo(
            ops[backend_name],
            loss_type="l2_difference",
            target_spo=target_ops[backend_name],
            lambda_ose=0.0,
            alpha=1.0,
        )
        for backend_name in BACKENDS
    }

    basis_terms = {
        backend_name: to_grad_term_dict(backend_name, BACKENDS[backend_name], basis_grads[backend_name], n_qubits=4)
        for backend_name in BACKENDS
    }
    l2_terms = {
        backend_name: to_grad_term_dict(backend_name, BACKENDS[backend_name], l2_grads[backend_name], n_qubits=4)
        for backend_name in BACKENDS
    }

    _assert_grad_term_dicts_close(basis_terms["numpy"], basis_terms["jax"])
    _assert_grad_term_dicts_close(l2_terms["numpy"], l2_terms["jax"])


def test_init_gradient_spo_rejects_invalid_loss_type():
    ops = _build_ops(lambda module: module.create_op({"IIII": 1.0}))

    for backend_name, module in BACKENDS.items():
        try:
            module.init_gradient_spo(
                ops[backend_name],
                loss_type="not_a_loss",
            )
        except ValueError as exc:
            assert "Unsupported loss_type" in str(exc)
        else:
            raise AssertionError("Expected invalid loss_type to raise ValueError")


def test_init_gradient_spo_requires_target_for_l2_difference():
    ops = _build_ops(lambda module: module.create_op({"IIII": 1.0}))

    for backend_name, module in BACKENDS.items():
        try:
            module.init_gradient_spo(
                ops[backend_name],
                loss_type="l2_difference",
            )
        except ValueError as exc:
            assert "target_spo must be provided" in str(exc)
        else:
            raise AssertionError("Expected missing target_spo to raise ValueError")


def test_init_gradient_spo_rejects_invalid_basis_for_basis_expectation():
    ops = _build_ops(lambda module: module.create_op({"IIII": 1.0}))

    for backend_name, module in BACKENDS.items():
        try:
            module.init_gradient_spo(
                ops[backend_name],
                loss_type="basis_expectation",
                basis="Y",
            )
        except (ValueError, NotImplementedError) as exc:
            assert "basis" in str(exc)
        else:
            raise AssertionError("Expected invalid basis to raise an error")


def test_single_rotation_semantics_match_between_backends():
    spo_in = _build_ops(lambda module: module.create_op({"XIII": 1.0}))
    sigma_u = {
        backend_name: np.asarray(module.utils.pauli_str_to_uint32("ZIII"))
        for backend_name, module in BACKENDS.items()
    }

    outputs = {
        backend_name: BACKENDS[backend_name].conjugate_pauli_rot_forward(
            spo_in[backend_name], sigma_u[backend_name], np.pi / 3,
            trunc_val=1e-12,
            max_num_str=1000,
        )[0]
        for backend_name in BACKENDS
    }
    terms_by_backend = {
        backend_name: to_term_dict(backend_name, BACKENDS[backend_name], spo, n_qubits=4)
        for backend_name, spo in outputs.items()
    }

    _assert_term_dicts_close(terms_by_backend["numpy"], terms_by_backend["jax"])


def test_cy_clifford_semantics_match_between_backends():
    spo_in = _build_ops(lambda module: module.create_op({"IXYZ": 1.0, "ZIIX": -0.25}))

    outputs = {
        backend_name: BACKENDS[backend_name].conjugate_CY_forward(
            spo_in[backend_name], 1, 3
        )
        for backend_name in BACKENDS
    }
    terms_by_backend = {
        backend_name: to_term_dict(backend_name, BACKENDS[backend_name], spo, n_qubits=4)
        for backend_name, spo in outputs.items()
    }

    _assert_term_dicts_close(terms_by_backend["numpy"], terms_by_backend["jax"])
