"""Device-native static channels with power-of-two PAD storage, without donation."""
from functools import partial
import sys

import jax
import jax.numpy as jnp
import numpy as np

from .sparse_pauli import SparsePauliOp, SparsePauliGradientOp
from ..checkpoints import CheckpointBackend

PAD_VAL = jnp.uint32(jnp.iinfo(jnp.uint32).max)


def _valid_keys(keys):
    return keys[:, 0] != PAD_VAL if keys.shape[1] else jnp.ones(len(keys), dtype=bool)


def _make(keys, coefficients, gradient=None, *, lexsorted=False):
    # Constructors use the process precision setting. Internal transformations
    # and checkpoint restores instead preserve the arrays' own precision.
    cls = SparsePauliOp if gradient is None else SparsePauliGradientOp
    result = cls.__new__(cls)
    result.xz_array, result.c_array = keys, coefficients
    result.lexsorted = lexsorted
    result._channel_storage = True
    if gradient is not None:
        result.grad_c_array = gradient
    return result


def _arrays(state, width):
    keys = state.xz_array
    if width < 0 or keys.shape[1] != 2 * ((width + 31) // 32):
        raise ValueError('Stored Pauli width disagrees with active_qubits.')
    if width % 32:
        mask = jnp.uint32((1 << (32 - width % 32)) - 1)
        last_words = keys[:, jnp.array([keys.shape[1] // 2 - 1, keys.shape[1] - 1])]
        outside = (last_words & mask) != 0
        if bool(jnp.any(outside & _valid_keys(keys)[:, None])):
            raise ValueError('Nonidentity Pauli outside the active column width.')
    return keys, state.c_array, getattr(state, 'grad_c_array', None)


@partial(jax.jit, static_argnames=('columns', 'checked', 'identity'))
def _map(keys, columns, checked=(), identity=False):
    n, words = keys.shape[0], keys.shape[1] // 2
    if checked:
        q = jnp.array(checked)
        x = (keys[:, q // 32] >> (31 - q % 32).astype(keys.dtype)) & 1
        valid = jnp.all(x == 0, axis=1)
        if identity:
            z = (keys[:, words + q // 32] >> (31 - q % 32).astype(keys.dtype)) & 1
            valid &= jnp.all(z == 0, axis=1)
    else:
        valid = jnp.ones(n, dtype=bool)
    valid &= _valid_keys(keys)
    out_words = (len(columns) + 31) // 32
    if not words or not out_words:
        return jnp.where(valid[:, None], jnp.zeros((n, 2 * out_words), dtype=keys.dtype), PAD_VAL), valid
    source = jnp.array(columns + (-1,) * (out_words * 32 - len(columns)))
    safe = jnp.maximum(source, 0)
    parts = []
    for half in range(2):
        bits = (keys[:, half * words + safe // 32] >> (31 - safe % 32).astype(keys.dtype)) & 1
        bits = jnp.where(source >= 0, bits, 0).reshape(n, out_words, 32)
        parts.append(jnp.sum(bits << jnp.arange(31, -1, -1, dtype=keys.dtype), axis=2, dtype=keys.dtype))
    return jnp.where(valid[:, None], jnp.concatenate(parts, axis=1), PAD_VAL), valid


@partial(jax.jit, static_argnames=("preserve_zero_support",))
def _merge_arrays(keys, values, *, preserve_zero_support=False):
    """Merge and compact valid nonzero rows into a fixed-shape PAD buffer."""
    n = len(keys)
    if not n:
        return keys, values, jnp.int32(0)
    if not keys.shape[1]:
        # There is only one scalar identity, including the zero scalar.
        return keys[:1], values.sum(axis=0, keepdims=True), jnp.int32(1)
    valid = _valid_keys(keys)
    values = jnp.where(valid if values.ndim == 1 else valid[:, None], values, 0)
    order = jnp.lexsort(keys.T[::-1])
    keys, values = keys[order], values[order]
    first = jnp.concatenate((jnp.ones(1, dtype=bool), jnp.any(keys[1:] != keys[:-1], axis=1)))
    groups = jnp.cumsum(first, dtype=jnp.int32) - 1
    merged = jax.ops.segment_sum(values, groups, num_segments=n)
    indices = jnp.nonzero(first, size=n, fill_value=0)[0]
    representatives = keys[indices]
    nonzero = merged != 0 if merged.ndim == 1 else jnp.any(merged != 0, axis=1)
    keep = (jnp.arange(n) <= groups[-1]) & _valid_keys(representatives) & (nonzero | preserve_zero_support)
    order = jnp.argsort(~keep, stable=True)
    keys = jnp.where(keep[:, None], representatives, PAD_VAL)[order]
    values = jnp.where(keep if merged.ndim == 1 else keep[:, None], merged, 0)[order]
    return keys, values, jnp.sum(keep)


def _merge(keys, values, *, preserve_zero_support=False):
    """Merge values and allocate a PAD bucket, pruning complete zero values.

    Exact channel differentiation can opt into retaining structural zero rows:
    unlike PAD, those coordinates may acquire a nonzero adjoint later.
    """
    keys, values, count = _merge_arrays(keys, values, preserve_zero_support=preserve_zero_support)
    size = 1 << (max(1, int(count)) - 1).bit_length()
    extra = max(0, size - len(keys))
    keys = jnp.pad(keys, ((0, extra), (0, 0)), constant_values=PAD_VAL)
    values = jnp.pad(values, ((0, extra),) + ((0, 0),) * (values.ndim - 1))
    return keys[:size], values[:size]


@jax.jit
def _aligned(query, source, values):
    if not len(source):
        return jnp.zeros((len(query),), dtype=values.dtype)
    if not query.shape[1]:
        return jnp.full((len(query),), values.sum(), dtype=values.dtype)
    from .kernels import find_row_duplications
    order = jnp.lexsort(source.T[::-1])
    found, indices = find_row_duplications(query, source[order])
    return jnp.where(found & _valid_keys(query) & _valid_keys(source[order])[indices], values[order][indices], 0)


def _contraction(keys, width, columns, removed):
    columns, removed = set(columns), set(removed)
    if not removed <= columns or any(q < 0 or q >= width for q in columns):
        raise ValueError('Invalid contraction columns.')
    selected = tuple(-1 if q in columns else q for q in range(width) if q not in removed)
    return _map(keys, selected, tuple(sorted(columns)))


def reindex_spo(spo, num_qubits, columns):
    columns = tuple(columns)
    if len(set(columns)) != len(columns) or any(q < 0 or q >= num_qubits for q in columns):
        raise ValueError('Invalid SPO column selection.')
    keys, c, _ = _arrays(spo, num_qubits)
    keys, valid = _map(keys, columns, tuple(sorted(set(range(num_qubits)) - set(columns))), True)
    if bool(jnp.any(_valid_keys(spo.xz_array) & ~valid)):
        raise ValueError('Nonidentity observable on discarded outputs.')
    keys, c = _merge(keys, jnp.where(valid, c, 0))
    return _make(keys, c, lexsorted=True)


def contract_zero_forward(spo, num_qubits, columns, remove_columns, *, preserve_zero_support=False):
    keys, c, _ = _arrays(spo, num_qubits)
    reduced, valid = _contraction(keys, num_qubits, columns, remove_columns)
    # Exact channel reconstruction needs projected zero coordinates as well:
    # a later transpose can attach a nonzero gradient to a cancelled row.
    keys, c = _merge(reduced, jnp.where(valid, c, 0),
                     preserve_zero_support=preserve_zero_support)
    return _make(keys, c, lexsorted=True)


def contract_zero_backward(spgo, checkpoint, num_qubits, columns, remove_columns):
    keys, c, _ = _arrays(checkpoint, num_qubits)
    reduced, valid = _contraction(keys, num_qubits, columns, remove_columns)
    downstream, _, g = _arrays(spgo, num_qubits - len(set(remove_columns)))
    gradient = jnp.where(valid, _aligned(reduced, downstream, g), 0)
    return _make(keys, c, gradient, lexsorted=checkpoint.lexsorted)


def insert_identity_forward(spo, num_qubits, column):
    if not 0 <= column <= num_qubits:
        raise ValueError('Invalid insertion column.')
    keys, c, _ = _arrays(spo, num_qubits)
    columns = list(range(num_qubits))
    columns.insert(column, -1)
    keys, valid = _map(keys, tuple(columns))
    keys, c = _merge(keys, jnp.where(valid, c, 0),
                     preserve_zero_support=True)
    return _make(keys, c, lexsorted=True)


def insert_identity_backward(spgo, num_qubits, column):
    if not 0 <= column < num_qubits:
        raise ValueError('Invalid insertion column.')
    keys, c, g = _arrays(spgo, num_qubits)
    keys, valid = _map(keys, tuple(q for q in range(num_qubits) if q != column), (column,), True)
    values = jnp.where(valid[:, None], jnp.stack((c, g), axis=1), 0)
    keys, values = _merge(keys, values)
    return _make(keys, values[:, 0], values[:, 1], lexsorted=True)


def _native(state):
    return isinstance(state, (SparsePauliOp, SparsePauliGradientOp))


def _device_of(state):
    if not _native(state):
        return None
    devices = set().union(*(array.devices() for array in state))
    if len(devices) != 1:
        raise ValueError('Channel checkpoints require arrays on one device.')
    device = next(iter(devices))
    return None if device.platform == 'cpu' else device


def checkpoint_size(state):
    """Estimate resident storage, deduplicating shared buffers within a snapshot."""
    if _native(state) and state.c_array.device.platform != 'cpu':
        arrays = {array.unsafe_buffer_pointer(): array.nbytes for array in state}
        return sum(arrays.values())
    seen, buffers = set(), set()
    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        total = sys.getsizeof(value)
        if isinstance(value, jax.Array):
            pointer = value.unsafe_buffer_pointer()
            if pointer not in buffers:
                buffers.add(pointer)
                total += value.nbytes
        elif _native(value):
            total += size(vars(value))
        elif isinstance(value, dict):
            total += sum(size(k) + size(v) for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            total += sum(map(size, value))
        return total
    return size(state)


def _to_host(state):
    if not _native(state):
        return state
    # device_get blocks; copies own their host allocations, independently of
    # the native snapshot. No pickling or snapshot mutation occurs here.
    arrays = tuple(np.array(array, copy=True) for array in jax.device_get(tuple(state)))
    return dict(arrays=arrays, lexsorted=state.lexsorted, system_size=state.system_size,
                active_qubits=None if state.active_qubits is None else list(state.active_qubits))


def _from_host(state, device):
    arrays = tuple(jax.device_put(array, device) for array in state['arrays'])
    jax.block_until_ready(arrays)
    result = _make(*arrays, lexsorted=state['lexsorted'])
    result.system_size = state['system_size']
    result.active_qubits = None if state['active_qubits'] is None else list(state['active_qubits'])
    return result


def create_checkpoint_backend(state):
    _device_of(state)  # Validate single-device storage before retaining anything.
    origin = state.c_array.device
    return CheckpointBackend(checkpoint_size, _device_of, _to_host,
                             lambda host, device: _from_host(host, origin if device is None else device))
