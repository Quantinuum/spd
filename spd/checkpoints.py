"""Evaluation-owned native-object checkpoints, with serialization only on spill."""
from collections import OrderedDict
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from typing import Callable, Hashable

DEFAULT_CHECKPOINT_MEMORY_BUDGET_BYTES = 4 * 1024**3
DEFAULT_CHECKPOINT_DEVICE_MEMORY_BUDGET_BYTES = 1024**3


def _host_device(state):
    return None


def _identity(state):
    return state


def _restore_host(state, device):
    return state


@dataclass(frozen=True)
class CheckpointBackend:
    """Small backend hooks; callbacks must not retain snapshot references.

    size estimates resident bytes for native and host representations. device_of
    returns None for CPU or a hashable device identifier. to_host must finish the
    transfer before returning; from_host restores the captured device/context.
    CPU-native snapshots also use to_host before disk spill and from_host with
    device=None after disk reads. CPU backends with portable native objects only
    need size. No callback serializes or mutates its input.
    """
    size: Callable[[object], int]
    device_of: Callable[[object], Hashable | None] = _host_device
    to_host: Callable[[object], object] = _identity
    from_host: Callable[[object, Hashable], object] = _restore_host


@dataclass
class _Snapshot:
    value: object
    tier: str
    size: int
    device: Hashable | None


class CheckpointStore:
    """Read-only native snapshots under CPU and per-device retention budgets.

    save transfers read-only ownership: callers must not mutate saved objects.
    load is non-consuming inspection; take transfers ownership to backward.
    Each memory tier evicts in arrival order. Oversized snapshots bypass a tier
    without evicting its existing entries. Budgets exclude transient/live data.
    """

    def __init__(self, signature, directory=None, *, backend: CheckpointBackend,
                 memory_budget_bytes=DEFAULT_CHECKPOINT_MEMORY_BUDGET_BYTES,
                 device_memory_budget_bytes=DEFAULT_CHECKPOINT_DEVICE_MEMORY_BUDGET_BYTES):
        for budget in (memory_budget_bytes, device_memory_budget_bytes):
            if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 0:
                raise ValueError('Checkpoint memory budgets must be nonnegative integers in bytes.')
        self.signature = signature
        self.backend = backend
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.device_memory_budget_bytes = int(device_memory_budget_bytes)
        self.memory_bytes = 0
        self.device_memory_bytes = {}
        self._host = OrderedDict()
        self._devices = {}
        self._snapshots = {}
        self._parent_directory = directory
        self._temporary = None
        self.directory = None
        self.closed = False

    def _spill(self, key, entry):
        if self._temporary is None:
            self._temporary = TemporaryDirectory(prefix='spd-channels-', dir=self._parent_directory)
            self.directory = Path(self._temporary.name)
        path = self.directory / f'{key}.pickle'
        # CPU-native backends may also need a portable host representation
        # (e.g. JAX CPU arrays). Device evictions already performed this step.
        value = self.backend.to_host(entry.value) if entry.device is None else entry.value
        with path.open('wb') as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        entry.value, entry.tier, entry.size = path, 'disk', 0

    def _retain_host(self, key, entry):
        entry.tier = 'host'
        if entry.size > self.memory_budget_bytes or self.memory_budget_bytes == 0:
            self._spill(key, entry)
            return
        while self.memory_bytes + entry.size > self.memory_budget_bytes:
            oldest, _ = self._host.popitem(last=False)
            previous = self._snapshots[oldest]
            self.memory_bytes -= previous.size
            self._spill(oldest, previous)
        self._host[key] = None
        self.memory_bytes += entry.size

    def _to_host(self, key, entry):
        entry.value = self.backend.to_host(entry.value)
        entry.size = self.backend.size(entry.value)
        self._retain_host(key, entry)

    def save(self, key, state):
        if self.closed:
            raise ValueError('Checkpoint store is closed.')
        if key in self._snapshots:
            raise ValueError('Checkpoint key already exists.')
        try:
            device = self.backend.device_of(state)
            entry = _Snapshot(state, 'device' if device is not None else 'host',
                              self.backend.size(state), device)
            self._snapshots[key] = entry
            if device is None:
                self._retain_host(key, entry)
            elif entry.size > self.device_memory_budget_bytes or self.device_memory_budget_bytes == 0:
                self._to_host(key, entry)
            else:
                queue = self._devices.setdefault(device, OrderedDict())
                used = self.device_memory_bytes.get(device, 0)
                while used + entry.size > self.device_memory_budget_bytes:
                    oldest, _ = queue.popitem(last=False)
                    previous = self._snapshots[oldest]
                    used -= previous.size
                    self.device_memory_bytes[device] = used
                    self._to_host(oldest, previous)
                queue[key] = None
                self.device_memory_bytes[device] = used + entry.size
        except BaseException:
            self.close()
            raise

    def load(self, key):
        """Inspect a snapshot without consuming it; returned objects are read-only."""
        if self.closed:
            raise ValueError('Checkpoints have already been consumed or closed.')
        try:
            entry = self._snapshots[key]
            state = entry.value
            if entry.tier == 'disk':
                with state.open('rb') as stream:
                    state = pickle.load(stream)
            if entry.tier == 'disk' or (entry.tier != 'device' and entry.device is not None):
                state = self.backend.from_host(state, entry.device)
            return state
        except BaseException:
            self.close()
            raise

    def take(self, key):
        """Consume one snapshot without promoting it back into the cache."""
        state = self.load(key)
        try:
            entry = self._snapshots.pop(key)
            if entry.tier == 'device':
                del self._devices[entry.device][key]
                self.device_memory_bytes[entry.device] -= entry.size
            elif entry.tier == 'host':
                del self._host[key]
                self.memory_bytes -= entry.size
            else:
                entry.value.unlink()
            return state
        except BaseException:
            self.close()
            raise

    def close(self):
        self._snapshots.clear()
        self._host.clear()
        self._devices.clear()
        self.memory_bytes = 0
        self.device_memory_bytes.clear()
        if self._temporary is not None:
            self._temporary.cleanup()
        self.closed = True
