import numpy as np

from tests.helpers import to_grad_term_dict, to_term_dict


def _assert_term_dict_matches(actual, expected, atol=1e-6):
    assert set(actual.keys()) == set(expected.keys())
    for key, value in expected.items():
        assert np.isclose(actual[key], value, atol=atol)


def _assert_grad_term_dict_matches(actual, expected, atol=1e-6):
    assert set(actual.keys()) == set(expected.keys())
    for key, value in expected.items():
        assert np.isclose(actual[key][0], value[0], atol=atol)
        assert np.isclose(actual[key][1], value[1], atol=atol)


def _make_spgo(backend_name, module, terms):
    if backend_name == "jax":
        xz_rows = np.asarray([module.utils.pauli_str_to_uint32(pstr) for pstr in terms])
        coeffs = np.asarray([terms[pstr][0] for pstr in terms], dtype=np.float32)
        grads = np.asarray([terms[pstr][1] for pstr in terms], dtype=np.float32)
        return module.SparsePauliGradientOp(xz_rows, coeffs, grads)

    spgo = module.SparsePauliGradientOp()
    for pstr, value_grad in terms.items():
        packed = tuple(module.utils.pauli_str_to_uint32(pstr))
        spgo[packed] = value_grad
    return spgo


def test_spo_arithmetic_exact(backend):
    backend_name, module = backend
    spo_a = module.create_op({"Z": 1.0, "X": -0.5})
    spo_b = module.create_op({"Z": -0.25, "Y": 2.0})

    sum_terms = to_term_dict(backend_name, module, spo_a + spo_b, n_qubits=1)
    diff_terms = to_term_dict(backend_name, module, spo_a - spo_b, n_qubits=1)
    scaled_terms = to_term_dict(backend_name, module, 2.0 * spo_a, n_qubits=1)
    cancelled_terms = to_term_dict(backend_name, module, spo_a + ((-1.0) * spo_a), n_qubits=1)

    _assert_term_dict_matches(sum_terms, {"Z": 0.75, "X": -0.5, "Y": 2.0})
    _assert_term_dict_matches(diff_terms, {"Z": 1.25, "X": -0.5, "Y": -2.0})
    _assert_term_dict_matches(scaled_terms, {"Z": 2.0, "X": -1.0})
    assert cancelled_terms == {}


def test_spgo_arithmetic_exact(backend):
    backend_name, module = backend
    spgo_a = _make_spgo(backend_name, module, {"Z": (1.0, 0.2), "X": (-0.5, 0.1)})
    spgo_b = _make_spgo(backend_name, module, {"Z": (-0.25, 0.3), "Y": (2.0, -1.0)})

    sum_terms = to_grad_term_dict(backend_name, module, spgo_a + spgo_b, n_qubits=1)
    scaled_terms = to_grad_term_dict(backend_name, module, 2.0 * spgo_a, n_qubits=1)
    cancelled_terms = to_grad_term_dict(backend_name, module, spgo_a + ((-1.0) * spgo_a), n_qubits=1)

    _assert_grad_term_dict_matches(sum_terms, {"Z": (0.75, 0.5), "X": (-0.5, 0.1), "Y": (2.0, -1.0)})
    _assert_grad_term_dict_matches(scaled_terms, {"Z": (2.0, 0.4), "X": (-1.0, 0.2)})
    assert cancelled_terms == {}


def test_init_gradient_from_l2_difference_exact(backend):
    backend_name, module = backend
    spo = module.create_op({"Z": 1.0, "X": 0.5})
    target_spo = module.create_op({"Z": 0.25, "Y": -1.0})

    spgo = module.init_gradient_from_l2_difference(spo, target_spo)
    grad_terms = to_grad_term_dict(backend_name, module, spgo, n_qubits=1)

    _assert_grad_term_dict_matches(grad_terms, {"Z": (1.0, 1.5), "X": (0.5, 1.0), "Y": (0.0, 2.0)})


def test_init_gradient_from_ose_exact_for_equal_weights(backend):
    backend_name, module = backend
    spo = module.create_op({"Z": 1.0, "X": 1.0})

    spgo = module.init_gradient_from_ose(spo, alpha=1.0)
    grad_terms = to_grad_term_dict(backend_name, module, spgo, n_qubits=1)

    assert np.isclose(grad_terms["Z"][0], 1.0, atol=1e-6)
    assert np.isclose(grad_terms["X"][0], 1.0, atol=1e-6)
    assert np.isclose(grad_terms["Z"][1], -2.0, atol=1e-6)
    assert np.isclose(grad_terms["X"][1], -2.0, atol=1e-6)
