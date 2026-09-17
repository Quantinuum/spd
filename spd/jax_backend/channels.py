"""Device-native static channels and compact reconstruction, without donation."""
from functools import partial
import sys

import jax
import jax.numpy as jnp
import numpy as np

from .sparse_pauli import SparsePauliOp, SparsePauliGradientOp
from ..checkpoints import CheckpointBackend


def _make(keys, coefficients, gradient=None, *, lexsorted=False):
    # Constructors use the process precision setting. Internal transformations
    # and checkpoint restores instead preserve the arrays' own precision.
    cls = SparsePauliOp if gradient is None else SparsePauliGradientOp
    result = cls.__new__(cls)
    result.xz_array, result.c_array = keys, coefficients
    result.lexsorted = lexsorted
    if gradient is not None:
        result.grad_c_array = gradient
    return result


def _arrays(state, width):
    keys = state.xz_array
    if width < 0 or keys.shape[1] != 2 * ((width + 31) // 32):
        raise ValueError('Stored Pauli width disagrees with active_qubits.')
    if width % 32:
        mask = jnp.uint32((1 << (32 - width % 32)) - 1)
        if bool(jnp.any((keys[:, jnp.array([keys.shape[1] // 2 - 1, keys.shape[1] - 1])] & mask) != 0)):
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
    out_words = (len(columns) + 31) // 32
    if not words or not out_words:
        return jnp.zeros((n, 2 * out_words), dtype=keys.dtype), valid
    source = jnp.array(columns + (-1,) * (out_words * 32 - len(columns)))
    safe = jnp.maximum(source, 0)
    parts = []
    for half in range(2):
        bits = (keys[:, half * words + safe // 32] >> (31 - safe % 32).astype(keys.dtype)) & 1
        bits = jnp.where(source >= 0, bits, 0).reshape(n, out_words, 32)
        parts.append(jnp.sum(bits << jnp.arange(31, -1, -1, dtype=keys.dtype), axis=2, dtype=keys.dtype))
    return jnp.concatenate(parts, axis=1), valid


@jax.jit
def _merge_arrays(keys, values):
    """Merge rows without coefficient-based pruning; return a valid row count."""
    n = len(keys)
    if not n:
        return keys, values, jnp.int32(0)
    if not keys.shape[1]:
        return keys[:1], values.sum(axis=0, keepdims=True), jnp.int32(1)
    order = jnp.lexsort(keys.T[::-1])
    keys, values = keys[order], values[order]
    first = jnp.concatenate((jnp.ones(1, dtype=bool), jnp.any(keys[1:] != keys[:-1], axis=1)))
    groups = jnp.cumsum(first, dtype=jnp.int32) - 1
    merged = jax.ops.segment_sum(values, groups, num_segments=n)
    indices = jnp.nonzero(first, size=n, fill_value=0)[0]
    return keys[indices], merged, groups[-1] + 1


def _merge(keys, values):
    keys, values, count = _merge_arrays(keys, values)
    count = int(count)
    return keys[:count], values[:count]


@jax.jit
def _aligned(query, source, values):
    if not len(source):
        return jnp.zeros((len(query),), dtype=values.dtype)
    if not query.shape[1]:
        return jnp.full((len(query),), values.sum(), dtype=values.dtype)
    from .kernels import find_row_duplications
    order = jnp.lexsort(source.T[::-1])
    found, indices = find_row_duplications(query, source[order])
    return jnp.where(found, values[order][indices], 0)


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
    if not bool(jnp.all(valid)):
        raise ValueError('Nonidentity observable on discarded outputs.')
    keys, c = _merge(keys, c)
    if not len(c):
        with jax.default_device(spo.c_array.device):
            keys, c = jnp.zeros((1, keys.shape[1]), dtype=keys.dtype), jnp.zeros(1, dtype=c.dtype)
    return _make(keys, c, lexsorted=True)


def contract_zero_forward(spo, num_qubits, columns, remove_columns):
    keys, c, _ = _arrays(spo, num_qubits)
    reduced, valid = _contraction(keys, num_qubits, columns, remove_columns)
    keys, c = _merge(reduced, jnp.where(valid, c, 0))
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
    keys, _ = _map(keys, tuple(columns))
    return _make(keys, c)


def insert_identity_backward(spgo, num_qubits, column):
    if not 0 <= column < num_qubits:
        raise ValueError('Invalid insertion column.')
    keys, c, g = _arrays(spgo, num_qubits)
    keys, valid = _map(keys, tuple(q for q in range(num_qubits) if q != column), (column,), True)
    return _make(keys[valid], c[valid], g[valid])


@jax.jit
def _rotation_candidates(keys, c, g, generator, theta):
    from .kernels import pauli_product_phase_sign_second_uint
    partners, sign = pauli_product_phase_sign_second_uint(generator, keys)
    anti = sign != 0
    angle = jnp.where(anti, theta, 0).astype(c.dtype)
    values = jnp.stack((c, g), axis=1)
    own = values * jnp.cos(angle)[:, None]
    other = values * (jnp.sin(angle) * sign)[:, None]
    # Commuting rows must not introduce spurious partner coordinates.
    partners = jnp.where(anti[:, None], partners, keys)
    return jnp.concatenate((keys, partners)), jnp.concatenate((own, other)), sign


def compact_rotation(state, generator, theta, cutoff, cap, *, backward=False):
    """The usual inverse reconstruction, with exact zero coordinates and no donation."""
    from .kernels import pauli_product_phase_sign_second_uint
    keys, c = state.xz_array, state.c_array
    g = state.grad_c_array if backward else jnp.zeros_like(c)
    angle_gradient = 0.
    if backward:
        partners, sign = pauli_product_phase_sign_second_uint(generator, keys)
        angle_gradient = float(jnp.sum(c * sign * _aligned(partners, keys, g)))
    keys, values, _ = _rotation_candidates(keys, c, g, generator, -theta if backward else theta)
    keys, values = _merge(keys, values)
    magnitude = jnp.abs(values[:, 0])
    keep = magnitude >= cutoff if backward else magnitude > cutoff
    if cutoff == 0:
        keep = jnp.ones_like(keep)
    if cap is not None and len(keys) > cap:
        ranked = jnp.argsort(-magnitude, stable=True)
        selected = jnp.zeros(len(keys), dtype=bool).at[ranked[:cap]].set(True)
        keep &= selected
    removed = jnp.where(~keep, magnitude, 0)
    info = dict(num_str_truncated=int(jnp.sum(removed != 0)),
                truncated_l1_norm=float(removed.sum()),
                truncated_l2_norm=float(jnp.sqrt(jnp.sum(removed ** 2))))
    keys, values = keys[keep], values[keep]
    result = _make(keys, values[:, 0], values[:, 1] if backward else None, lexsorted=True)
    if backward:
        return result, len(keys), angle_gradient, info
    return result, len(keys), info


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
