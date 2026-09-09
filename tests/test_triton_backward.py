"""Backward coefficients, adjoints, diagnostics and independent derivatives."""

import itertools
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA GPU")

from spd import triton_backend as gpu, numpy_backend as cpu, jax_backend
from tests.test_triton_invariants import labels_from_keys, matrix


def terms(state):
    keys, c, g = state.to_host()
    result = {tuple(k): (float(a), float(b)) for k, a, b in zip(keys, c, g)}
    assert len(result) == len(c)
    return result


def reference_state(module, data):
    if module is cpu:
        state = cpu.SparsePauliGradientOp()
        for p, pair in data.items():
            state[tuple(cpu.utils.pauli_str_to_uint(p))] = pair
        return state
    keys = np.array([module.utils.pauli_str_to_uint(p) for p in data])
    values = np.array(list(data.values()), dtype=module.utils.get_real_dtype())
    return module.SparsePauliGradientOp(keys, values[:, 0], values[:, 1])


@pytest.mark.parametrize("precision,tol", [("single", 3e-6), ("double", 3e-13)])
@pytest.mark.parametrize("qubits", [3, 33, 65, 121])
def test_backward_random_sequences_against_numpy(precision, tol, qubits):
    cpu.utils.set_packbit(32); cpu.set_precision(precision)
    rng = np.random.default_rng(847)
    labels = ["".join(rng.choice(list("IXYZ"), qubits)) for _ in range(137)]
    data = dict(zip(labels, rng.normal(size=(137, 2))))
    a = reference_state(cpu, data)
    b = gpu.create_gradient_op(data, precision=precision)
    for step in range(4):
        gate = "".join(rng.choice(list("IXYZ"), qubits))
        theta = rng.uniform(-1, 1)
        before = terms(b)
        a, size, grad, info = cpu.conjugate_pauli_rot_backward(
            a, cpu.utils.pauli_str_to_uint(gate), theta, .03, 211)
        result, count, actual_grad, actual_info = gpu.conjugate_pauli_rot_backward(
            b, gpu.utils.pauli_str_to_uint(gate), theta, .03, 211)
        actual = terms(result)
        assert count == size == len(actual)
        assert actual.keys() == a.keys()
        np.testing.assert_allclose(list(actual.values()), [a[k] for k in actual], atol=tol, rtol=tol)
        assert actual_grad == pytest.approx(grad, abs=tol, rel=tol)
        assert actual_info == pytest.approx(info, abs=tol, rel=tol)
        assert terms(b) == before
        b = result


@pytest.mark.parametrize("algorithm", ["stack_sort_merge", "search_update_merge"])
@pytest.mark.parametrize("precision,tol", [("single", 5e-6), ("double", 3e-13)])
def test_dense_support_against_jax(algorithm, precision, tol):
    previous = jax_backend.get_algorithm()
    jax_backend.utils.set_packbit(32); jax_backend.set_precision(precision)
    jax_backend.set_algorithm(algorithm)
    rng = np.random.default_rng(632)
    data = dict(zip(("".join(p) for p in itertools.product("IXYZ", repeat=2)), rng.normal(size=(16, 2))))
    try:
        a = reference_state(jax_backend, data)
        b = gpu.create_gradient_op(data, precision=precision)
        for gate, theta in [("YZ", .37), ("XI", -.23), ("ZX", .81)]:
            a, count, grad, info = jax_backend.conjugate_pauli_rot_backward(
                a, jax_backend.utils.pauli_str_to_uint(gate), theta, .01, 11)
            b, actual_count, actual_grad, actual_info = gpu.conjugate_pauli_rot_backward(
                b, gpu.utils.pauli_str_to_uint(gate), theta, .01, 11)
            expected = {tuple(k): (c, g) for k, c, g in zip(np.asarray(a.xz_array), np.asarray(a.c_array),
                                                         np.asarray(a.grad_c_array)) if c != 0 or g != 0}
            actual = terms(b)
            assert actual.keys() == expected.keys()
            assert count == actual_count
            np.testing.assert_allclose(list(actual.values()), [expected[k] for k in actual], atol=tol, rtol=tol)
            assert actual_grad == pytest.approx(float(grad), abs=tol, rel=tol)
            assert actual_info == pytest.approx(info, abs=tol, rel=tol)
    finally:
        jax_backend.set_algorithm(previous)


@pytest.mark.parametrize("precision,eps,tol", [("single", 1e-3, 1e-4), ("double", 1e-6, 1e-8)])
def test_theta_gradient_against_dense_finite_difference(precision, eps, tol):
    rng = np.random.default_rng(289)
    labels = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]
    coeff = rng.normal(size=16)
    adjoint = rng.normal(size=16)
    basis = np.array([matrix(p) for p in labels])
    observable = np.einsum("i,ijk->jk", coeff, basis)
    loss_operator = np.einsum("i,ijk->jk", adjoint, basis)
    for gate in labels[1:]:
        generator = matrix(gate)
        theta = .37
        def loss(t):
            u = np.cos(t / 2) * np.eye(4) - 1j * np.sin(t / 2) * generator
            return np.trace(loss_operator @ u.conj().T @ observable @ u).real / 4
        y = gpu.conjugate_pauli_rotation(gpu.create_op(dict(zip(labels, coeff)), precision=precision), gate, theta)
        keys, values = y.to_host()
        y_labels = labels_from_keys(keys, 2)
        g = np.array([adjoint[labels.index(p)] for p in y_labels], dtype=values.dtype)
        terminal = gpu.SparsePauliGradientOp(y.xz_array, y.c_array, torch.as_tensor(g, device='cuda'), 2)
        restored, _, grad, info = gpu.conjugate_pauli_rot_backward(terminal, gate, theta, 0.)
        assert grad == pytest.approx((loss(theta + eps) - loss(theta - eps)) / (2 * eps), abs=tol, rel=tol)
        restored_keys, restored_c, _ = restored.to_host()
        actual = dict(zip(labels_from_keys(restored_keys, 2), restored_c))
        np.testing.assert_allclose([actual[p] for p in labels], coeff, atol=tol, rtol=tol)
        assert info['num_str_truncated'] == 0


def test_gradient_only_rows_empty_states_and_exact_cutoff():
    state = gpu.create_gradient_op({'X': (0., 1.)})
    result, count, grad, _ = gpu.conjugate_pauli_rot_backward(state, 'Z', .37, 0.)
    assert count == 2 and grad == 0.
    np.testing.assert_allclose(sorted(result.to_host()[2]), sorted([np.cos(.37), np.sin(.37)]))
    result, count, _, info = gpu.conjugate_pauli_rot_backward(state, 'Z', .37, .01)
    assert count == 0 and info['num_str_truncated'] == 0
    assert gpu.conjugate_pauli_rot_backward(result, 'Z', .1, 0.)[1] == 0
    state = gpu.create_gradient_op({'X': (.5, 1.)})
    assert gpu.conjugate_pauli_rot_backward(state, 'I', 0., .5)[1] == 1
    assert gpu.conjugate_pauli_rot_forward(state.to_spo(), 'I', 0., .5)[1] == 0


@pytest.mark.parametrize("backward", [False, True])
def test_diagnostics_threshold_cap_and_zero_candidates(backward):
    state = (gpu.create_gradient_op({'I': (.8, 1.), 'X': (.6, 2.), 'Y': (.4, 3.), 'Z': (.2, 4.)})
             if backward else gpu.create_op({'I': .8, 'X': .6, 'Y': .4, 'Z': .2}))
    fn = gpu.conjugate_pauli_rot_backward if backward else gpu.conjugate_pauli_rot_forward
    result = fn(state, 'I', 0., .3, 2)
    assert result[1] == 2
    assert result[-1] == pytest.approx({'num_str_truncated': 2, 'truncated_l1_norm': .6,
                                      'truncated_l2_norm': np.sqrt(.2**2 + .4**2)})
    state = gpu.create_gradient_op({'X': (1., 1.)}) if backward else gpu.create_op({'X': 1.})
    assert fn(state, 'Z', 0., .1)[-1]['num_str_truncated'] == 0


def test_forward_diagnostics_preserve_small_lost_norm():
    # Subtracting retained norm from total would lose this discarded mass.
    _, _, info = gpu.conjugate_pauli_rot_forward(gpu.create_op({'I': 1., 'X': 1e-12}), 'I', 0., 1e-10)
    assert info['num_str_truncated'] == 1
    assert info['truncated_l2_norm'] == pytest.approx(1e-12, rel=1e-14, abs=0.)


def test_spgo_validation_and_factory_precision():
    with pytest.raises(ValueError):
        gpu.SparsePauliGradientOp(np.zeros((1, 2), np.uint32), np.array([1.]), np.array([1.], np.float32))
    with pytest.raises(ValueError):
        gpu.SparsePauliGradientOp(np.zeros((1, 2), np.uint32), np.array([1.]), np.array([1.]), 65)
    with pytest.raises(TypeError):
        gpu.conjugate_pauli_rot_backward(gpu.create_op({'X': 1.}), 'Z', .1, 0.)
    previous = gpu.utils.get_precision()
    try:
        gpu.set_precision('single')
        state = gpu.create_measurement_op({(0, 32): 1.}, 64)
        assert state.c_array.dtype == torch.float32
        assert state.get_expectation_value() == 1.
        assert gpu.create_gradient_op({'X': (1., 2.)}).grad_c_array.dtype == torch.float32
    finally:
        gpu.set_precision(previous)


@pytest.mark.parametrize("count", [127, 128, 129, 8193, "collision"])
def test_backward_compaction_and_collision_stress(count):
    from tests.test_triton_invariants import collision_observable
    cpu.utils.set_packbit(32)
    cpu.set_precision("double")
    rng = np.random.default_rng(121)
    if count == "collision":
        values = collision_observable()
    else:
        keys = np.column_stack([np.arange(count, dtype=np.uint32) << 16,
                                np.zeros(count, dtype=np.uint32)])
        values = dict(zip(labels_from_keys(keys, 16), rng.normal(size=count)))
    data = {p: (c, rng.normal()) for p, c in values.items()}
    a = reference_state(cpu, data)
    b = gpu.create_gradient_op(data, precision="double")
    gate = "YZ" * 8
    expected, n, grad, info = cpu.conjugate_pauli_rot_backward(
        a, cpu.utils.pauli_str_to_uint(gate), .43, .1, 10000)
    actual, m, actual_grad, actual_info = gpu.conjugate_pauli_rot_backward(
        b, gate, .43, .1, 10000)
    output = terms(actual)
    assert output.keys() == expected.keys()
    assert n == m == len(output)
    np.testing.assert_allclose(list(output.values()), [expected[k] for k in output], atol=1e-13)
    assert actual_grad == pytest.approx(grad, abs=1e-12)
    assert actual_info == pytest.approx(info, abs=1e-11)
