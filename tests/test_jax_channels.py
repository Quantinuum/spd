"""JAX channel contracts on CPU and actual accelerator storage."""
from dataclasses import replace
import pickle
import weakref

import jax
import numpy as np
import pytest

import spd
from spd import jax_backend as backend
from spd.checkpoints import CheckpointStore
from spd.circuit_ir import PauliRotation as Rot, SkippedOperation
from test_static_channels import dense, finite_differences


@pytest.fixture(params=['cpu', 'gpu'])
def device(request):
    try:
        devices = jax.devices(request.param)
    except RuntimeError:
        pytest.skip(f'JAX {request.param} unavailable')
    with jax.default_device(devices[0]):
        yield devices[0]


def arrays(state):
    return tuple(state)


def assert_same(actual, expected, device):
    assert actual.active_qubits == expected.active_qubits
    assert actual.system_size == expected.system_size
    assert actual.lexsorted == expected.lexsorted
    for a, b in zip(arrays(actual), arrays(expected)):
        assert a.device == device and a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('precision', ['single', 'double'])
@pytest.mark.parametrize('tier', ['native', 'host', 'disk'])
def test_checkpoint_tiers(device, precision, tier, tmp_path, monkeypatch):
    o = spd.create_spo({'XI': 2., 'IZ': -1., 'II': 0.}, system_size=40,
                       active_qubits=[2, 35], backend_name='jax', precision=precision)
    hooks = backend.create_checkpoint_backend(o)
    size = hooks.size(o)
    assert size >= sum(a.nbytes for a in arrays(o))
    events = []
    def to_host(state):
        events.append('host')
        result = hooks.to_host(state)
        assert all(isinstance(a, np.ndarray) for a in result['arrays'])
        return result
    tracked = replace(hooks, to_host=to_host)
    store = CheckpointStore((), tmp_path, backend=tracked,
        memory_budget_bytes=0 if tier == 'disk' else max(size, hooks.size(hooks.to_host(o))),
        device_memory_budget_bytes=size if tier == 'native' else 0)
    if tier != 'disk':
        monkeypatch.setattr(pickle, 'dump', lambda *a, **kw: pytest.fail('Unexpected serialization'))
    store.save(0, o)
    if tier == 'native' or (device.platform == 'cpu' and tier == 'host'):
        assert store.load(0) is o and events == []
    else:
        assert events == ['host']
    if tier == 'disk':
        with store._snapshots[0].value.open('rb') as stream:
            host = pickle.load(stream)
        assert all(isinstance(a, np.ndarray) for a in host['arrays'])
    # Restore must not consult the currently selected coefficient precision.
    backend.set_precision('single' if precision == 'double' else 'double')
    # The current default device must not redirect restoration.
    other = jax.devices()[0] if device.platform == 'cpu' else jax.devices('cpu')[0]
    with jax.default_device(other):
        restored = store.take(0)
    assert_same(restored, o, device)
    assert not store._snapshots
    store.close()
    assert not list(tmp_path.iterdir())
    ref = weakref.ref(o)
    del o, restored
    assert ref() is None


def test_fifo_eviction(device, tmp_path):
    o = spd.create_spo({'I': 0., 'X': 1., 'Z': -1.}, system_size=1,
                       backend_name='jax', precision='double')
    hooks = backend.create_checkpoint_backend(o)
    native_size, host_size = hooks.size(o), hooks.size(hooks.to_host(o))
    store = CheckpointStore((), tmp_path, backend=hooks,
        memory_budget_bytes=native_size if device.platform == 'cpu' else host_size,
        device_memory_budget_bytes=native_size)
    for i in range(3):
        store.save(i, o)
    expected = ['disk', 'disk', 'host'] if device.platform == 'cpu' else ['disk', 'host', 'device']
    assert [store._snapshots[i].tier for i in range(3)] == expected
    for i in range(3):
        assert_same(store.take(i), o, device)
    store.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('algorithm', ['stack_sort_merge', 'search_update_merge', 'search_update_merge_donate'])
@pytest.mark.parametrize('precision', ['single', 'double'])
@pytest.mark.parametrize('disk', [False, True])
def test_dense_gradient_reconstruction(device, algorithm, precision, disk, tmp_path, monkeypatch):
    previous = backend.get_algorithm()
    backend.set_algorithm(algorithm)
    circuit = spd.CircuitIR(3, [Rot('Ry', 'YII', .2), spd.CreateZero(2),
        Rot('RXX', 'XIX', .31), spd.ResetZero(0), Rot('Ry', 'IIY', -.42),
        spd.Discard(1), spd.ResetZero(2), Rot('Ry', 'IIY', .61)])
    observable = {'IIZ': .8, 'XIX': -.3, 'III': .2}
    o = spd.create_spo(observable, system_size=3, backend_name='jax', precision=precision)
    def forbidden(*a, **kw):
        pytest.fail('CPU reference fallback')
    monkeypatch.setattr(spd.numpy_backend, 'contract_zero_forward', forbidden)
    monkeypatch.setattr(spd.numpy_backend, 'contract_zero_backward', forbidden)
    try:
        f, _ = spd.evolve(o, circuit, 0., 1000, progress=False,
            checkpoint_directory=tmp_path, checkpoint_memory_budget_bytes=0 if disk else 100000,
            checkpoint_device_memory_budget_bytes=0 if disk else 100000)
        tol = 2e-6 if precision == 'single' else 2e-9
        assert f.c_array.device == device
        assert float(f.get_expectation_value()) == pytest.approx(dense(circuit, observable), abs=tol)
        result, gradients, _ = spd.backpropagate(spd.init_gradient_spo(f), circuit, 0., 1000, progress=False)
        assert result.c_array.device == device
        np.testing.assert_allclose(gradients, finite_differences(circuit, observable), atol=tol)
        assert f._channel_checkpoints.closed and not list(tmp_path.iterdir())
    finally:
        backend.set_algorithm(previous)


@pytest.mark.parametrize('width', [1, 31, 32, 33, 65])
def test_grouped_packed_width_and_scalars(device, width):
    ops = []
    for q in range(width):
        ops.extend([spd.CreateZero(q), SkippedOperation('OpType.Barrier')])
    circuit = spd.CircuitIR(width, ops)
    o = spd.create_spo({'Z' * width: 1.}, system_size=width, backend_name='jax', precision='double')
    f, info = spd.evolve(o, circuit, 0., 100, progress=False)
    assert f.xz_array.shape == (1, 0) and f.c_array.device == device
    assert f.active_qubits == [] and float(f.get_expectation_value()) == 1.
    assert len(f._channel_checkpoints._snapshots) == 1
    assert set(info['active_widths']) == set(range(width + 1))
    result, gradients, _ = spd.backpropagate(spd.init_gradient_spo(f), circuit, 0., 100, progress=False)
    assert gradients == [] and result.active_qubits == list(range(width))


def test_complete_transpose_and_empty_scalar(device):
    adapter = spd.BackendAdapter.from_name('jax', precision='double')
    o = adapter.module.create_op({'II': 1., 'ZI': -1., 'XI': 4., 'YZ': 0., 'IZ': 2.}, num_qubits=2)
    reduced = backend.contract_zero_forward(o, 2, [0], [0])
    assert reduced.get_size() == 2
    target = backend.create_op({'I': 0., 'Z': 2.}, num_qubits=1)
    g = backend.SparsePauliGradientOp(target.xz_array, target.c_array, jax.numpy.array([3., -2.]))
    restored = backend.contract_zero_backward(g, o, 2, [0], [0])
    np.testing.assert_array_equal(restored.xz_array, o.xz_array)
    np.testing.assert_array_equal(restored.c_array, o.c_array)
    expected = {'II': 3., 'ZI': 3., 'XI': 0., 'YZ': 0., 'IZ': -2.}
    for key, gradient in zip(np.asarray(o.xz_array), np.asarray(restored.grad_c_array)):
        assert gradient == expected[backend.utils.uint_to_pauli_str(key, 32)[:2]]
    empty = backend.create_op({}, num_qubits=0)
    assert backend.insert_identity_forward(empty, 0, 0).xz_array.shape == (0, 2)
    scalar = backend.reindex_spo(empty, 0, [])
    assert scalar.xz_array.shape == (1, 0)
    hooks = backend.create_checkpoint_backend(scalar)
    assert_same(hooks.from_host(hooks.to_host(scalar), device), scalar, device)


@pytest.mark.parametrize('failure', ['to_host', 'from_host'])
def test_transfer_error_cleanup(device, tmp_path, failure):
    o = spd.create_spo({'Z': 1.}, backend_name='jax', precision='double')
    def fail(*args):
        raise RuntimeError('transfer failed')
    hooks = replace(backend.create_checkpoint_backend(o), **{failure: fail})
    store = CheckpointStore((), tmp_path, backend=hooks, memory_budget_bytes=0, device_memory_budget_bytes=0)
    with pytest.raises(RuntimeError, match='transfer failed'):
        store.save(0, o)
        store.take(0)
    assert store.closed and not list(tmp_path.iterdir())


def test_validation_and_checkpoint_readonly(device):
    o = spd.create_spo({'XIZ': 1.}, system_size=3, backend_name='jax')
    with pytest.raises(ValueError, match='discarded'):
        backend.reindex_spo(o, 3, [0, 1])
    with pytest.raises(ValueError, match='selection'):
        backend.reindex_spo(o, 3, [0, 0])
    with pytest.raises(ValueError, match='outside'):
        backend.reindex_spo(o, 2, [0, 1])
    # Saturating a tiny cap under the donation strategy must never invalidate
    # a native snapshot or the operator retained by the caller.
    previous = backend.get_algorithm()
    backend.set_algorithm('search_update_merge_donate')
    circuit = spd.CircuitIR(3, [spd.ResetZero(0), Rot('Ry', 'YII', .4), spd.ResetZero(2)])
    try:
        f, _ = spd.evolve(o, circuit, 0., 1, progress=False)
        store = f._channel_checkpoints
        retained = [store.load(key) for key in store._snapshots]
        snapshots = [[np.array(a) for a in arrays(s)] for s in retained]
        spd.backpropagate(spd.init_gradient_spo(f), circuit, 0., 1, progress=False)
        for state, expected in zip(retained, snapshots):
            for actual, before in zip(arrays(state), expected):
                np.testing.assert_array_equal(actual, before)
        assert float(o.get_norm_square()) == 1.
    finally:
        backend.set_algorithm(previous)
