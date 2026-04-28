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


def _make_jax_spo(module, terms, *, lexsorted=False):
    xz_rows = np.asarray([module.utils.pauli_str_to_uint32(pstr) for pstr in terms])
    coeffs = np.asarray([terms[pstr] for pstr in terms], dtype=np.float32)
    return module.SparsePauliOp(xz_rows, coeffs, lexsorted=lexsorted)


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


def test_jax_spo_add_handles_non_power_of_two_merged_size():
    from spd import jax_backend

    jax_backend.utils.set_packbit(32)
    spo_a = _make_jax_spo(jax_backend, {"IIII": 1.0, "IIIX": 2.0, "IIIZ": 3.0, "IIXI": 4.0})
    spo_b = _make_jax_spo(
        jax_backend,
        {
            "IXII": 5.0,
            "IXIX": 6.0,
            "IXIZ": 7.0,
            "IXXI": 8.0,
            "IZII": 9.0,
            "IZIX": 10.0,
            "IZIZ": 11.0,
            "IZXI": 12.0,
        },
    )

    summed = spo_a + spo_b

    assert summed.get_size() == 12
    _assert_term_dict_matches(
        to_term_dict("jax", jax_backend, summed, n_qubits=4),
        {
            "IIII": 1.0,
            "IIIX": 2.0,
            "IIIZ": 3.0,
            "IIXI": 4.0,
            "IXII": 5.0,
            "IXIX": 6.0,
            "IXIZ": 7.0,
            "IXXI": 8.0,
            "IZII": 9.0,
            "IZIX": 10.0,
            "IZIZ": 11.0,
            "IZXI": 12.0,
        },
    )


def test_jax_spgo_add_handles_non_power_of_two_merged_size():
    from spd import jax_backend

    jax_backend.utils.set_packbit(32)
    spgo_a = _make_spgo(
        "jax",
        jax_backend,
        {
            "IIII": (1.0, 0.1),
            "IIIX": (2.0, 0.2),
            "IIIZ": (3.0, 0.3),
            "IIXI": (4.0, 0.4),
        },
    )
    spgo_b = _make_spgo(
        "jax",
        jax_backend,
        {
            "IXII": (5.0, 0.5),
            "IXIX": (6.0, 0.6),
            "IXIZ": (7.0, 0.7),
            "IXXI": (8.0, 0.8),
            "IZII": (9.0, 0.9),
            "IZIX": (10.0, 1.0),
            "IZIZ": (11.0, 1.1),
            "IZXI": (12.0, 1.2),
        },
    )

    summed = spgo_a + spgo_b

    assert summed.get_size() == 12
    _assert_grad_term_dict_matches(
        to_grad_term_dict("jax", jax_backend, summed, n_qubits=4),
        {
            "IIII": (1.0, 0.1),
            "IIIX": (2.0, 0.2),
            "IIIZ": (3.0, 0.3),
            "IIXI": (4.0, 0.4),
            "IXII": (5.0, 0.5),
            "IXIX": (6.0, 0.6),
            "IXIZ": (7.0, 0.7),
            "IXXI": (8.0, 0.8),
            "IZII": (9.0, 0.9),
            "IZIX": (10.0, 1.0),
            "IZIZ": (11.0, 1.1),
            "IZXI": (12.0, 1.2),
        },
    )


def test_spgo_to_spo_keeps_primal_coefficients(backend):
    backend_name, module = backend
    spgo = _make_spgo(
        backend_name,
        module,
        {"Z": (1.0, 0.2), "X": (-0.5, 0.1), "Y": (2.0, -1.0)},
    )

    spo = spgo.to_spo()
    terms = to_term_dict(backend_name, module, spo, n_qubits=1)

    _assert_term_dict_matches(terms, {"Z": 1.0, "X": -0.5, "Y": 2.0})


def test_numpy_spo_dot_matches_equal_pauli_terms():
    from spd import numpy_backend

    numpy_backend.utils.set_packbit(32)
    spo_a = numpy_backend.create_op({"Z": 1.0, "X": -0.5, "Y": 2.0})
    spo_b = numpy_backend.create_op({"Z": 0.25, "X": 3.0, "I": 4.0})

    assert np.isclose(spo_a.dot(spo_b), -1.25)
    assert np.isclose(spo_a.inner_product(spo_b), -1.25)
    assert np.isclose(spo_a.inner_product(spo_a), spo_a.get_norm_square())


def test_jax_spo_lexsort_sorts_rows_and_sets_metadata():
    from spd import jax_backend

    jax_backend.utils.set_packbit(32)
    spo = _make_jax_spo(jax_backend, {"Y": 2.0, "Z": 1.0, "X": -0.5})

    sorted_spo = spo.lexsort()
    xz_rows = np.asarray(spo.xz_array)
    expected_indices = np.lexsort(xz_rows.T[::-1])

    assert sorted_spo.lexsorted
    assert np.array_equal(np.asarray(sorted_spo.xz_array), xz_rows[expected_indices])
    assert np.allclose(np.asarray(sorted_spo.c_array), np.asarray(spo.c_array)[expected_indices])


def test_jax_create_op_marks_result_lexsorted():
    from spd import jax_backend

    jax_backend.utils.set_packbit(32)
    spo = jax_backend.create_op({"Z": 1.0, "X": -0.5})

    assert spo.lexsorted


def test_jax_spo_dot_matches_numpy_for_partial_overlap():
    from spd import jax_backend

    jax_backend.utils.set_packbit(32)
    spo_a = _make_jax_spo(jax_backend, {"Y": 2.0, "Z": 1.0, "X": -0.5})
    spo_b = _make_jax_spo(jax_backend, {"I": 4.0, "X": 3.0, "Z": 0.25})

    assert np.isclose(float(np.asarray(spo_a.dot(spo_b))), -1.25, atol=1e-6)
    assert np.isclose(float(np.asarray(spo_a.dot(spo_a))), float(np.asarray(spo_a.get_norm_square())), atol=1e-6)


def test_jax_spo_dot_uses_self_lexsorted_haystack():
    from spd import jax_backend

    jax_backend.utils.set_packbit(32)
    spo_a = _make_jax_spo(jax_backend, {"Y": 2.0, "Z": 1.0, "X": -0.5}).lexsort()
    spo_b = _make_jax_spo(jax_backend, {"I": 4.0, "X": 3.0, "Z": 0.25})

    assert np.isclose(float(np.asarray(spo_a.dot(spo_b))), -1.25, atol=1e-6)


def test_jax_spo_dot_uses_other_lexsorted_haystack():
    from spd import jax_backend

    jax_backend.utils.set_packbit(32)
    spo_a = _make_jax_spo(jax_backend, {"Y": 2.0, "Z": 1.0, "X": -0.5})
    spo_b = _make_jax_spo(jax_backend, {"I": 4.0, "X": 3.0, "Z": 0.25}).lexsort()

    assert np.isclose(float(np.asarray(spo_a.dot(spo_b))), -1.25, atol=1e-6)


def test_jax_spgo_to_spo_preserves_lexsorted_metadata():
    from spd import jax_backend

    jax_backend.utils.set_packbit(32)
    spgo = _make_spgo(
        "jax",
        jax_backend,
        {"Z": (1.0, 0.2), "X": (-0.5, 0.1), "Y": (2.0, -1.0)},
    )
    spgo = jax_backend.SparsePauliGradientOp(
        spgo.xz_array,
        spgo.c_array,
        spgo.grad_c_array,
        lexsorted=True,
    )

    spo = spgo.to_spo()

    assert spo.lexsorted
    _assert_term_dict_matches(to_term_dict("jax", jax_backend, spo, n_qubits=1), {"Z": 1.0, "X": -0.5, "Y": 2.0})


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
