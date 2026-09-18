"""PAD storage and shared-algorithm regressions for static JAX channels."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spd
from spd import jax_backend as backend
from spd.jax_backend import channels, kernels


@pytest.fixture(autouse=True)
def configure_backend():
    spd.BackendAdapter.from_name('jax', precision='double')


@pytest.fixture(params=['stack_sort_merge', 'search_update_merge'])
def algorithm(request):
    previous = backend.get_algorithm()
    backend.set_algorithm(request.param)
    backend.set_precision('double')
    yield request.param
    backend.set_algorithm(previous)


def assert_padded(state, live):
    keys, c = np.asarray(state.xz_array), np.asarray(state.c_array)
    n = len(c)
    assert n >= 1 and n & (n - 1) == 0
    if not keys.shape[1]:
        assert n == 1
        return
    valid = keys[:, 0] != np.uint32(2**32 - 1)
    assert valid.sum() == live
    assert np.all(keys[~valid] == np.uint32(2**32 - 1))
    assert np.all(c[~valid] == 0)
    if hasattr(state, 'grad_c_array'):
        assert np.all(np.asarray(state.grad_c_array)[~valid] == 0)


def test_rotation_buckets_and_cache_reuse(algorithm):
    state = backend.create_op({'IIII': 1., 'ZIII': 2., 'IZII': 3., 'IIZI': 4., 'IIIZ': 5.})
    state = backend.reindex_spo(state, 4, range(4)).set_active_qubits(16, [2, 5, 8, 13])
    module = kernels._load_algorithm_module()
    compiled = (module.forward_stack_sort_merge_jitted if algorithm == 'stack_sort_merge'
                else module.forward_search_update_merge_top_k_jitted)
    shapes = []
    for step, (q, live) in enumerate([(0, 6), (1, 7), (2, 8), (2, 8), (3, 9), (0, 9)]):
        generator = backend.utils.pauli_str_to_uint('I' * q + 'X' + 'I' * (3-q))
        state, count, info = backend.conjugate_pauli_rot_forward(state, generator, .21, 1e-12, 100)
        state.set_active_qubits(16, [2, 5, 8, 13])
        assert count == live and info['num_str_truncated'] == 0
        assert state.get_size() == live
        assert_padded(state, live)
        shapes.append(len(state.c_array))
        if step == 2:
            cache_size = compiled._cache_size()
        if step == 3:
            assert compiled._cache_size() == cache_size
    assert shapes == [8, 8, 8, 8, 16, 16]


def test_non_power_of_two_cap_and_backward_zero_coefficient(algorithm):
    state = backend.create_op({'II': 1., 'ZI': 2., 'IZ': 3., 'ZZ': 4.})
    state = backend.reindex_spo(state, 2, [0, 1]).set_active_qubits(2, [0, 1])
    generator = backend.utils.pauli_str_to_uint('XI')
    state, count, info = backend.conjugate_pauli_rot_forward(state, generator, .3, 1e-12, 3)
    assert count == 3
    assert_padded(state, 3)
    assert len(state.c_array) == 4 and info['num_str_truncated'] > 0
    # A real physical row with c=0 must remain available to inverse rotations.
    source = backend.create_op({'Z': 0.})
    grad = backend.SparsePauliGradientOp(source.xz_array, source.c_array, jnp.array([2.]))
    grad.set_active_qubits(1, [0])
    restored, count, angle_gradient, _ = backend.conjugate_pauli_rot_backward(
        grad, backend.utils.pauli_str_to_uint('X'), .3, 0., 100)
    assert count == 2 and angle_gradient == 0
    assert_padded(restored, 2)
    assert np.linalg.norm(restored.grad_c_array) == pytest.approx(2.)


def test_mapping_projection_cancellation_and_gradient_padding():
    backend.set_precision('double')
    source = backend.create_op({'II': 2., 'ZI': -2., 'XI': 9., 'IZ': 3., 'ZZ': 4., 'IY': 5.})
    reduced = backend.contract_zero_forward(source, 2, [0], [0])
    assert_padded(reduced, 2)
    np.testing.assert_allclose(np.sort(np.asarray(reduced.c_array)), [5., 7.])
    # Three surviving rows occupy four slots. Mapping across a packed-word
    # boundary must not turn the fourth PAD row into a physical coordinate.
    source = backend.create_op({'I'*33: 1., 'Z'+'I'*32: 2., 'I'*32+'X': 3.})
    padded = backend.reindex_spo(source, 33, range(33))
    assert_padded(padded, 3)
    mapped = backend.reindex_spo(padded, 33, [32, 0])
    assert_padded(mapped, 3)
    inserted = backend.insert_identity_forward(mapped, 2, 1)
    assert_padded(inserted, 3)
    grad = backend.SparsePauliGradientOp(inserted.xz_array, jnp.zeros(4),
                                        jnp.array([1., 2., 3., 0.]))
    removed = backend.insert_identity_backward(grad, 3, 1)
    assert_padded(removed, 3)
    assert np.count_nonzero(removed.grad_c_array) == 3
    restored = backend.contract_zero_backward(removed, padded, 33, list(range(1, 32)), list(range(1, 32)))
    assert_padded(restored, 3)
    # Native/host checkpoint conversion preserves the padded allocation exactly.
    padded.set_active_qubits(33, list(range(33)))
    hooks = backend.create_checkpoint_backend(padded)
    snapshot = hooks.from_host(hooks.to_host(padded), padded.c_array.device)
    for actual, expected in zip(snapshot, padded):
        np.testing.assert_array_equal(actual, expected)


def test_merge_stacked_values_and_empty_scalar():
    pad = np.uint32(2**32 - 1)
    keys = jnp.array([[0, 0], [0, 0], [1, 0], [pad, pad]], dtype=jnp.uint32)
    values = jnp.array([[2., 1.], [-2., -1.], [0., 3.], [9., 9.]])
    merged, result = channels._merge(keys, values)
    np.testing.assert_array_equal(merged, [[1, 0]])
    np.testing.assert_array_equal(result, [[0., 3.]])
    scalar, result = channels._merge(jnp.zeros((3, 0), dtype=jnp.uint32), jnp.array([1., -1., 0.]))
    assert scalar.shape == (1, 0)
    np.testing.assert_array_equal(result, [0.])


@pytest.mark.parametrize('n', [4, 8])
def test_hva_energy_gradient_numpy_and_dense(algorithm, n):
    from spd.ansatz import binary_mera_parameter_shape
    from examples.gradient.run_1d_tfi_hva_tmera import spd_value_gradient, exact_value_gradient
    params = np.random.default_rng(19).normal(0, .1, binary_mera_parameter_shape(n))
    expected, expected_grad = exact_value_gradient(params, n, 4, .9)
    for name in ['numpy', 'jax']:
        value, grad, _ = spd_value_gradient(params, n, 4, .9, backend_name=name, cutoff=1e-14)
        assert value == pytest.approx(expected, abs=1e-10)
        np.testing.assert_allclose(grad, expected_grad, atol=1e-9)


def test_exact_zero_rotation_and_cancelled_channel_gradient(algorithm):
    from spd.circuit_ir import PauliRotation as Rot
    from test_static_channels import dense, finite_differences, evaluate
    cases = [
        (spd.CircuitIR(1, [spd.CreateZero(0), Rot('Ry', 'Y', 0.)]), {'X': 1.}),
        (spd.CircuitIR(1, [spd.ResetZero(0), Rot('Ry', 'Y', .43)]), {'I': -np.cos(.43), 'Z': 1.}),
    ]
    for circuit, observable in cases:
        value, grad, *_ = evaluate(circuit, observable, 'jax')
        assert value == pytest.approx(dense(circuit, observable), abs=1e-12)
        np.testing.assert_allclose(grad, finite_differences(circuit, observable), atol=1e-9)


def test_runner_algorithm_option_and_selection(monkeypatch):
    from examples.gradient import run_1d_tfi_hva_tmera as runner
    monkeypatch.setattr('sys.argv', ['runner', '4', '4', '0', '--backend', 'jax',
                                   '--algorithm', 'search_update_merge'])
    _, args = runner.parse_args()
    previous = backend.get_algorithm()
    try:
        backend.set_algorithm('search_update_merge_donate')
        selected = runner._execution_backend(args.backend, algorithm=args.algorithm)
        assert selected.module.get_algorithm() == 'search_update_merge'
    finally:
        backend.set_algorithm(previous)


def test_threshold_equality_diagnostics_ignore_padding(algorithm):
    state = backend.create_op({'II': 3., 'ZI': 2., 'IZ': 1., 'ZZ': .5})
    state.set_active_qubits(2, [0, 1])
    result, count, info = backend.conjugate_pauli_rot_forward(
        state, backend.utils.pauli_str_to_uint('II'), .2, .5, 100)
    assert count == 3
    assert_padded(result, 3)
    assert info['num_str_truncated'] == 1
    assert info['truncated_l1_norm'] == pytest.approx(.5)
    assert info['truncated_l2_norm'] == pytest.approx(.5)


def test_backward_contraction_preserves_zero_coefficient_and_pad():
    source = backend.create_op({'II': 0., 'ZI': 2., 'IZ': 3.})
    keys, values = channels._merge(source.xz_array, source.c_array, preserve_zero_support=True)
    checkpoint = channels._make(keys, values, lexsorted=True)
    target = backend.create_op({'I': 0., 'Z': 0.})
    downstream = backend.SparsePauliGradientOp(target.xz_array, target.c_array, jnp.array([5., 7.]))
    result = backend.contract_zero_backward(downstream, checkpoint, 2, [0], [0])
    assert_padded(result, 3)
    valid_zero = (np.asarray(result.c_array) == 0) & (np.asarray(result.xz_array)[:, 0] != 2**32-1)
    np.testing.assert_array_equal(np.asarray(result.grad_c_array)[valid_zero], [5.])
    np.testing.assert_array_equal(result.xz_array, checkpoint.xz_array)
    np.testing.assert_array_equal(result.c_array, checkpoint.c_array)
