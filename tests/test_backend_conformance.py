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

    sizes = {backend_name: module.get_size(ops[backend_name]) for backend_name, module in BACKENDS.items()}
    norm_squares = {
        backend_name: float(np.asarray(module.get_norm_square(ops[backend_name])))
        for backend_name, module in BACKENDS.items()
    }
    exp_z = {
        backend_name: float(np.asarray(module.get_expectation_value(ops[backend_name], basis="Z")))
        for backend_name, module in BACKENDS.items()
    }
    exp_x = {
        backend_name: float(np.asarray(module.get_expectation_value(ops[backend_name], basis="X")))
        for backend_name, module in BACKENDS.items()
    }

    assert sizes["numpy"] == sizes["jax"]
    assert np.isclose(norm_squares["numpy"], norm_squares["jax"], atol=1e-6)
    assert np.isclose(exp_z["numpy"], exp_z["jax"], atol=1e-6)
    assert np.isclose(exp_x["numpy"], exp_x["jax"], atol=1e-6)


def test_create_gradient_spo_semantics_match_between_backends():
    pauli_dict = {
        "IIII": 2.0,
        "ZIII": -0.5,
        "XIII": 0.75,
        "YIIX": -1.25,
    }
    ops = _build_ops(lambda module, payload: module.create_op(payload), pauli_dict)
    grads = {
        backend_name: BACKENDS[backend_name].create_gradient_spo(spo, basis="Z")
        for backend_name, spo in ops.items()
    }

    grad_terms = {
        backend_name: to_grad_term_dict(backend_name, BACKENDS[backend_name], spgo, n_qubits=4)
        for backend_name, spgo in grads.items()
    }

    assert grad_terms["numpy"] == grad_terms["jax"]
    assert np.isclose(
        float(np.asarray(BACKENDS["numpy"].get_norm_square(grads["numpy"]))),
        float(np.asarray(BACKENDS["jax"].get_norm_square(grads["jax"]))),
        atol=1e-6,
    )


def test_single_rotation_semantics_match_between_backends():
    spo_in = _build_ops(lambda module: module.create_op({"XIII": 1.0}))
    sigma_u = {
        backend_name: np.asarray(module.utils.pauli_str_to_uint32("ZIII"))
        for backend_name, module in BACKENDS.items()
    }

    outputs = {
        backend_name: BACKENDS[backend_name].conjugated_pauli_forward(
            spo_in[backend_name], sigma_u[backend_name], np.pi / 3, trunc_val=1e-12
        )[0]
        for backend_name in BACKENDS
    }
    terms_by_backend = {
        backend_name: to_term_dict(backend_name, BACKENDS[backend_name], spo, n_qubits=4)
        for backend_name, spo in outputs.items()
    }

    _assert_term_dicts_close(terms_by_backend["numpy"], terms_by_backend["jax"])
