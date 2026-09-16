"""Native retention and tier transitions, independent of GPU libraries."""
from dataclasses import dataclass
import gc
import pickle
import sys
import weakref

import numpy as np
import pytest

from spd.checkpoints import (CheckpointBackend, CheckpointStore,
                             DEFAULT_CHECKPOINT_MEMORY_BUDGET_BYTES,
                             DEFAULT_CHECKPOINT_DEVICE_MEMORY_BUDGET_BYTES)
from spd.numpy_backend.sparse_pauli import checkpoint_size


@dataclass
class State:
    data: bytes
    device: str | None = None


def storage_backend(events=None, fail=None):
    events = [] if events is None else events

    def to_host(state):
        events.append(('host', state.data, state.device))
        if fail == 'transfer':
            raise RuntimeError('transfer failed')
        return State(state.data)

    def from_host(state, device):
        events.append(('restore', state.data, device))
        if fail == 'restore':
            raise RuntimeError('restore failed')
        return State(state.data, device)

    return CheckpointBackend(size=lambda state: len(state.data),
                             device_of=lambda state: state.device,
                             to_host=to_host, from_host=from_host)


def store_at(tmp_path, **kwargs):
    return CheckpointStore(None, tmp_path, backend=storage_backend(), **kwargs)


def no_serialization(*args, **kwargs):
    pytest.fail('In-budget native retention must not serialize')


@pytest.mark.parametrize('device', [None, 'gpu:0'])
def test_native_identity_without_serialization(tmp_path, monkeypatch, device):
    monkeypatch.setattr(pickle, 'dump', no_serialization)
    monkeypatch.setattr(pickle, 'dumps', no_serialization)
    events = []
    store = CheckpointStore(None, tmp_path, backend=storage_backend(events))
    state = State(b'abc', device)
    store.save(0, state)
    assert store.load(0) is state
    assert store.take(0) is state
    assert not store._snapshots
    assert store.memory_bytes == 0
    assert sum(store.device_memory_bytes.values()) == 0
    assert events == []
    assert store.directory is None
    assert list(tmp_path.iterdir()) == []
    store.close()


@pytest.mark.parametrize('budget', [0, 9, 10, 11])
def test_cpu_threshold(tmp_path, budget):
    store = store_at(tmp_path, memory_budget_bytes=budget)
    state = State(b'0123456789')
    store.save(0, state)
    assert store.memory_bytes == (10 if budget >= 10 else 0)
    assert (store.directory is None) == (budget >= 10)
    restored = store.take(0)
    assert restored == state
    assert (restored is state) == (budget >= 10)
    assert store.memory_bytes == 0
    store.close()
    store.close()
    assert list(tmp_path.iterdir()) == []


def test_cpu_oldest_first_oversize_bypass_and_reuse(tmp_path):
    store = store_at(tmp_path, memory_budget_bytes=10)
    a, b, c = State(b'a' * 6), State(b'b' * 4), State(b'c' * 6)
    store.save(0, a)
    store.save(1, b)
    store.load(0)  # Inspection does not change FIFO ordering.
    store.save(2, c)
    assert store._snapshots[0].tier == 'disk'
    assert store.load(1) is b and store.load(2) is c
    store.save(3, State(b'x' * 11))  # Bypass without evicting smaller entries.
    assert store.memory_bytes == 10
    assert store._snapshots[3].tier == 'disk'
    assert store.take(1) is b
    store.save(4, State(b'd' * 4))
    assert store.memory_bytes == 10
    assert store.take(0) == a
    assert not (store.directory / '0.pickle').exists()
    store.close()
    assert list(tmp_path.iterdir()) == []


def test_gpu_to_host_to_disk_and_independent_devices(tmp_path):
    events = []
    store = CheckpointStore(None, tmp_path, backend=storage_backend(events),
                            memory_budget_bytes=6, device_memory_budget_bytes=6)
    states = [State(bytes([i]) * 6, 'gpu:0') for i in range(4)]
    store.save(0, states[0])
    store.save(1, states[1])  # 0 -> CPU
    assert store._snapshots[0].tier == 'host'
    assert store.directory is None
    store.save(2, states[2])  # 1 -> CPU, 0 -> disk
    assert [store._snapshots[i].tier for i in range(3)] == ['disk', 'host', 'device']
    other = State(b'other!', 'gpu:1')
    store.save(3, other)
    assert store.device_memory_bytes == {'gpu:0': 6, 'gpu:1': 6}
    assert store.take(3) is other
    assert store.take(2) is states[2]
    assert store.take(1) == states[1]
    assert store.take(0) == states[0]
    assert events == [('host', states[0].data, 'gpu:0'), ('host', states[1].data, 'gpu:0'),
                      ('restore', states[1].data, 'gpu:0'), ('restore', states[0].data, 'gpu:0')]
    assert store.memory_bytes == 0
    assert sum(store.device_memory_bytes.values()) == 0
    assert list(store.directory.iterdir()) == []
    store.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('gpu_budget', [0, 5, 6, 7])
@pytest.mark.parametrize('cpu_budget', [0, 5, 6])
def test_gpu_thresholds_and_oversize(tmp_path, gpu_budget, cpu_budget):
    store = store_at(tmp_path, memory_budget_bytes=cpu_budget,
                     device_memory_budget_bytes=gpu_budget)
    state = State(b'123456', 'gpu:0')
    store.save(0, state)
    expected = 'device' if gpu_budget >= 6 else ('host' if cpu_budget >= 6 else 'disk')
    assert store._snapshots[0].tier == expected
    assert store.take(0) == state
    store.close()
    assert list(tmp_path.iterdir()) == []


def test_gpu_oversized_bypasses_without_evicting_existing(tmp_path):
    store = store_at(tmp_path, memory_budget_bytes=20, device_memory_budget_bytes=5)
    state = State(b'abc', 'gpu:0')
    store.save(0, state)
    store.save(1, State(b'123456', 'gpu:0'))
    assert store.load(0) is state
    assert store._snapshots[1].tier == 'host'
    assert store.device_memory_bytes == {'gpu:0': 3}
    store.close()


def test_defaults_and_independent_evaluations(tmp_path):
    first, second = store_at(tmp_path), store_at(tmp_path)
    assert first.memory_budget_bytes == DEFAULT_CHECKPOINT_MEMORY_BUDGET_BYTES == 4 * 1024**3
    assert first.device_memory_budget_bytes == DEFAULT_CHECKPOINT_DEVICE_MEMORY_BUDGET_BYTES == 1024**3
    first.save(0, State(b'abc'))
    assert first.memory_bytes == 3 and second.memory_bytes == 0
    first.close()
    second.close()


@pytest.mark.parametrize('argument', ['memory_budget_bytes', 'device_memory_budget_bytes'])
@pytest.mark.parametrize('budget', [-1, 1.5, True, None])
def test_invalid_budget(tmp_path, argument, budget):
    with pytest.raises(ValueError, match='nonnegative integers'):
        store_at(tmp_path, **{argument: budget})


@pytest.mark.parametrize('failure', ['transfer', 'restore', 'dump', 'load'])
def test_failures_close_all_tiers(tmp_path, monkeypatch, failure):
    def fail(*args, **kwargs):
        if failure == 'dump':
            args[1].write(b'partial')
        raise RuntimeError(f'{failure} failed')

    store = CheckpointStore(None, tmp_path, backend=storage_backend(fail=failure),
                            memory_budget_bytes=0, device_memory_budget_bytes=3)
    store.save(0, State(b'abc', 'gpu:0'))
    if failure in ('dump', 'load'):
        monkeypatch.setattr(pickle, failure, fail)
    with pytest.raises(RuntimeError, match='failed'):
        store.save(1, State(b'def', 'gpu:0'))
        store.take(0)
    assert store.closed
    assert not store._snapshots and not store.device_memory_bytes and store.memory_bytes == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('budget', [0, 100])
def test_abandoned_store_and_reference_cleanup(tmp_path, budget):
    store = store_at(tmp_path, memory_budget_bytes=budget)
    state = State(b'abc')
    state_ref, store_ref = weakref.ref(state), weakref.ref(store)
    store.save(0, state)
    del state, store
    gc.collect()
    assert state_ref() is None and store_ref() is None
    assert list(tmp_path.iterdir()) == []


def test_duplicate_and_closed_store(tmp_path):
    store = store_at(tmp_path)
    state = State(b'abc')
    store.save(0, state)
    with pytest.raises(ValueError, match='already exists'):
        store.save(0, State(b'bad'))
    assert store.take(0) is state
    store.close()
    with pytest.raises(ValueError, match='closed'):
        store.save(1, state)
    with pytest.raises(ValueError, match='closed'):
        store.take(0)


def test_numpy_size_estimation_and_large_array_round_trip(tmp_path, monkeypatch):
    from spd.numpy_backend.sparse_pauli import SparsePauliOp
    shared = np.arange(100_000, dtype=np.float64)
    once, twice = [shared], [shared, shared]
    assert checkpoint_size(twice) - checkpoint_size(once) == sys.getsizeof(twice) - sys.getsizeof(once)
    view = shared.reshape(100, 1000)
    assert checkpoint_size(view) >= shared.nbytes
    spo = SparsePauliOp({(np.uint32(0), np.uint32(1)): .3}).set_active_qubits(8, [3, 7])
    assert checkpoint_size(spo) > sys.getsizeof(spo) + sys.getsizeof(spo.active_qubits)
    store = CheckpointStore(None, tmp_path, backend=CheckpointBackend(checkpoint_size), memory_budget_bytes=0)
    store.save(0, {'array': view, 'spo': spo})
    restored = store.take(0)
    np.testing.assert_array_equal(restored['array'], view)
    assert restored['spo'] == spo and restored['spo'].active_qubits == [3, 7]
    store.close()


def test_eviction_and_take_release_native_references(tmp_path):
    store = store_at(tmp_path, memory_budget_bytes=3, device_memory_budget_bytes=3)
    device = State(b'abc', 'gpu:0')
    device_ref = weakref.ref(device)
    store.save(0, device)
    del device
    assert device_ref() is not None
    store.save(1, State(b'def', 'gpu:0'))
    assert device_ref() is None  # Only the host representation remains.
    host_ref = weakref.ref(store._snapshots[0].value)
    store.save(2, State(b'ghi', 'gpu:0'))
    assert host_ref() is None  # The CPU representation was serialized and freed.
    restored = store.take(1)
    restored_ref = weakref.ref(restored)
    del restored
    assert restored_ref() is None  # take did not retain/promote the restored state.
    store.close()
