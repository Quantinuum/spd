"""Device-specific static channel and native snapshot contracts."""
import pickle
import weakref

import numpy as np
import pytest

import spd
from spd.checkpoints import CheckpointStore
from spd.circuit_ir import PauliRotation as Rot, SkippedOperation
from test_static_channels import dense, finite_differences


@pytest.fixture
def gpu():
    torch = pytest.importorskip('torch')
    pytest.importorskip('triton')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from spd import triton_backend
    return triton_backend


@pytest.mark.parametrize('precision', ['single', 'double'])
@pytest.mark.parametrize('tier', ['device', 'host', 'disk'])
def test_device_snapshot_roundtrip(gpu, precision, tier, tmp_path, monkeypatch):
    o = spd.create_spo({'XI': 2., 'IZ': -1., 'II': 0.}, system_size=40,
                       active_qubits=[2, 35], backend_name='triton', precision=precision)
    hooks = gpu.create_checkpoint_backend(o)
    before = [x.clone() for x in o._raw_arrays()[:2]]
    size = hooks.size(o)
    assert size == sum(x.untyped_storage().nbytes() for x in o._raw_arrays()[:2])
    host = hooks.to_host(o)
    assert all(not hasattr(x, 'is_cuda') for x in host.values())
    assert host['coefficients'].dtype == (np.float64 if precision == 'double' else np.float32)
    assert hooks.device_of(host) is None
    host_size = hooks.size(host)
    store = CheckpointStore((), tmp_path, backend=hooks,
                            memory_budget_bytes=host_size if tier == 'host' else 0,
                            device_memory_budget_bytes=size if tier == 'device' else 0)
    if tier != 'disk':
        monkeypatch.setattr(pickle, 'dump', lambda *a, **k: pytest.fail('Unexpected serialization'))
    store.save(0, o)
    assert store._snapshots[0].tier == tier
    if tier == 'device':
        assert store.load(0) is o
    restored = store.take(0)
    assert restored._storage.device == o._storage.device
    assert restored.active_qubits == [2, 35] and restored.system_size == 40
    assert restored.precision == precision
    for actual, expected in zip(restored._raw_arrays()[:2], before):
        assert actual.equal(expected)
    for actual, expected in zip(o._raw_arrays()[:2], before):
        assert actual.equal(expected)
    assert not store._snapshots
    store.close()
    assert not list(tmp_path.iterdir())
    ref = weakref.ref(o)
    del o, restored
    assert ref() is None  # hooks must not capture the input state


def test_fifo_device_eviction(gpu, tmp_path):
    o = gpu.create_op({'I': 0., 'X': 1., 'Z': -1.}, num_qubits=1, precision='double')
    hooks = gpu.create_checkpoint_backend(o)
    size = hooks.size(o)
    host_size = hooks.size(hooks.to_host(o))
    store = CheckpointStore((), tmp_path, backend=hooks, memory_budget_bytes=host_size,
                            device_memory_budget_bytes=size)
    for i in range(3):
        store.save(i, gpu.create_op({'I': 0., 'X': 1., 'Z': -1.}, num_qubits=1, precision='double'))
    assert [store._snapshots[i].tier for i in range(3)] == ['disk', 'host', 'device']
    for i in range(3):
        restored = store.take(i)
        for a, b in zip(restored._raw_arrays()[:2], o._raw_arrays()[:2]):
            assert a.equal(b)
    store.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('precision', ['single', 'double'])
@pytest.mark.parametrize('budgets', [(0, 0), (100000, 0), (100000, 100000)])
def test_dense_gradients_and_cleanup(gpu, precision, budgets, tmp_path, monkeypatch):
    circuit = spd.CircuitIR(3, [Rot('Ry', 'YII', .2), spd.CreateZero(2),
        Rot('RXX', 'XIX', .31), spd.ResetZero(0), Rot('Ry', 'IIY', -.42),
        spd.Discard(1), spd.ResetZero(2), Rot('Ry', 'IIY', .61)])
    observable = {'IIZ': .8, 'XIX': -.3, 'III': .2}
    o = spd.create_spo(observable, system_size=3, backend_name='triton', precision=precision)
    def forbidden(*a, **kw):
        pytest.fail('CPU channel fallback')
    monkeypatch.setattr(spd.numpy_backend, 'contract_zero_forward', forbidden)
    monkeypatch.setattr(spd.numpy_backend, 'contract_zero_backward', forbidden)
    f, _ = spd.evolve(o, circuit, 0., 1000, progress=False, checkpoint_directory=tmp_path,
                      checkpoint_memory_budget_bytes=budgets[0], checkpoint_device_memory_budget_bytes=budgets[1])
    tol = 2e-6 if precision == 'single' else 2e-9
    assert f.get_expectation_value() == pytest.approx(dense(circuit, observable), abs=tol)
    _, gradients, _ = spd.backpropagate(spd.init_gradient_spo(f), circuit, 0., 1000, progress=False)
    np.testing.assert_allclose(gradients, finite_differences(circuit, observable), atol=tol)
    assert f._channel_checkpoints.closed
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('width', [1, 31, 32, 33, 65])
def test_grouped_creation_packed_width(gpu, width):
    ops = []
    for q in range(width):
        ops.extend([spd.CreateZero(q), SkippedOperation('OpType.Barrier')])
    circuit = spd.CircuitIR(width, ops)
    o = spd.create_spo({'Z' * width: 1.}, system_size=width, active_qubits=list(range(width)),
                       backend_name='triton', precision='double')
    f, info = spd.evolve(o, circuit, 0., 100, progress=False)
    assert f._raw_arrays()[0].shape == (1, 0)
    assert f.active_qubits == [] and f.get_expectation_value() == 1.
    assert len(f._channel_checkpoints._snapshots) == bool(width)
    assert set(info['active_widths']) == set(range(width + 1))
    result, gradients, _ = spd.backpropagate(spd.init_gradient_spo(f), circuit, 0., 100, progress=False)
    assert gradients == []
    assert result.active_qubits == list(range(width))


@pytest.mark.parametrize('failure', ['to_host', 'from_host'])
def test_transfer_failure_cleanup(gpu, tmp_path, failure):
    from dataclasses import replace
    o = gpu.create_op({'Z': 1.}, precision='double')
    def fail(*args):
        raise RuntimeError('transfer failed')
    hooks = replace(gpu.create_checkpoint_backend(o), **{failure: fail})
    store = CheckpointStore((), tmp_path, backend=hooks, memory_budget_bytes=0, device_memory_budget_bytes=0)
    with pytest.raises(RuntimeError, match='transfer failed'):
        store.save(0, o)
        store.take(0)
    assert store.closed and not list(tmp_path.iterdir())


def test_complete_checkpoint_transpose(gpu):
    original = gpu.create_op({'II': 1., 'ZI': -1., 'XI': 4., 'YZ': 0., 'IZ': 2.},
                             num_qubits=2, precision='double')
    reduced = gpu.contract_zero_forward(original, 2, [0], [0])
    assert reduced.get_size() == 2  # cancelled I and surviving Z
    g = gpu.create_gradient_op({'I': (0., 3.), 'Z': (2., -2.)}, num_qubits=1, precision='double')
    restored = gpu.contract_zero_backward(g, original, 2, [0], [0])
    assert restored._raw_arrays()[0].equal(original._raw_arrays()[0])
    assert restored._raw_arrays()[1].equal(original._raw_arrays()[1])
    np.testing.assert_array_equal(restored._raw_arrays()[2].cpu().numpy(), [3., 3., 0., 0., -2.])


def test_reindex_validation_and_scalar_transfer(gpu):
    o = gpu.create_op({'XIZ': 1.}, num_qubits=3)
    r = gpu.reindex_spo(o, 3, [2, 0])
    expected = gpu.create_op({'ZX': 1.}, num_qubits=2)
    assert r._raw_arrays()[0].equal(expected._raw_arrays()[0])
    with pytest.raises(ValueError, match='discarded'):
        gpu.reindex_spo(o, 3, [0, 1])
    with pytest.raises(ValueError, match='selection'):
        gpu.reindex_spo(o, 3, [0, 0])
    with pytest.raises(ValueError, match='outside'):
        gpu.reindex_spo(gpu.create_op({'IZ': 1.}, num_qubits=2), 1, [0])
    scalar = gpu.create_op({'': 0.}, num_qubits=0, precision='double')
    hooks = gpu.create_checkpoint_backend(scalar)
    restored = hooks.from_host(hooks.to_host(scalar), scalar._storage.device)
    assert restored._raw_arrays()[0].shape == (1, 0)
    assert restored.get_size() == 1
    inserted = gpu.insert_identity_forward(restored, 0, 0)
    assert inserted._raw_arrays()[0].shape == (1, 2)
