"""Static channel maps on GPU packed rows; index metadata belongs to the runner."""
import sys

import torch

from . import SparsePauliOp
from .gradient import SparsePauliGradientOp
from .algebra import aligned, match
from .kernels import channel_columns
from ..checkpoints import CheckpointBackend


def _arrays(state, width):
    keys, c, g = state._raw_arrays()
    if keys.shape[1] != 2 * ((width + 31) // 32):
        raise ValueError('Stored Pauli width disagrees with active_qubits.')
    if width % 32:
        mask = (1 << (32 - width % 32)) - 1
        if torch.any((keys[:, [keys.shape[1] // 2 - 1, keys.shape[1] - 1]] & mask) != 0).item():
            raise ValueError('Nonidentity Pauli outside the active column width.')
    return keys, c, g


def _map(keys, columns, checked=(), *, identity=False):
    """Pack selected columns (-1 inserts I), and test contracted/omitted axes."""
    columns, checked = tuple(columns), tuple(checked)
    out = keys.new_empty((len(keys), 2 * ((len(columns) + 31) // 32)))
    keep = torch.empty(len(keys), dtype=torch.bool, device=keys.device)
    if len(keys):
        with torch.cuda.device(keys.device):
            channel_columns[((len(keys) + 127) // 128,)](
                keys, out, keep, len(keys), keys.shape[1], out.shape[1],
                columns, checked, identity, 128)
    return out, keep


def _contraction(keys, width, columns, removed):
    columns, removed = set(columns), set(removed)
    if not removed <= columns or any(q < 0 or q >= width for q in columns):
        raise ValueError('Invalid contraction columns.')
    return _map(keys, [-1 if q in columns else q for q in range(width) if q not in removed], columns)


def reindex_spo(spo, num_qubits, columns):
    """Select/reorder columns, rejecting nonidentity on omitted outputs."""
    columns = tuple(columns)
    if len(set(columns)) != len(columns) or any(q < 0 or q >= num_qubits for q in columns):
        raise ValueError('Invalid SPO column selection.')
    keys, c, _ = _arrays(spo, num_qubits)
    keys, valid = _map(keys, columns, sorted(set(range(num_qubits)) - set(columns)), identity=True)
    if not valid.all().item():
        raise ValueError('Nonidentity observable on discarded outputs.')
    if not len(c):
        keys, c = keys.new_zeros((1, keys.shape[1])), c.new_zeros(1)
    return SparsePauliOp(keys, c, len(columns))


def contract_zero_forward(spo, num_qubits, columns, remove_columns):
    """Apply joint zero-state contraction and retain every image coordinate."""
    keys, c, _ = _arrays(spo, num_qubits)
    keys, keep = _contraction(keys, num_qubits, columns, remove_columns)
    c = torch.where(keep, c, 0)
    width = num_qubits - len(set(remove_columns))
    if len(c):
        if keys.shape[1] == 0:
            keys, c = keys[:1], c.sum().reshape(1)
        else:
            keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
            merged = c.new_zeros(len(keys))
            merged.scatter_add_(0, inverse, c)
            c = merged
    # Do not prune zero coordinates, including images of killed rows.
    return SparsePauliOp(keys, c, width)


def contract_zero_backward(spgo, checkpoint, num_qubits, columns, remove_columns):
    """Restore all checkpoint rows and lift the reduced coefficient adjoint."""
    keys, c, _ = _arrays(checkpoint, num_qubits)
    reduced, keep = _contraction(keys, num_qubits, columns, remove_columns)
    downstream, _, g = _arrays(spgo, num_qubits - len(set(remove_columns)))
    indices, _ = match(reduced, downstream)
    grad = torch.where(keep, aligned(g, indices), 0)
    return SparsePauliGradientOp(keys, c, grad, num_qubits)


def insert_identity_forward(spo, num_qubits, column):
    """Discard adjoint: insert an identity column without normalization."""
    if not 0 <= column <= num_qubits:
        raise ValueError('Invalid insertion column.')
    keys, c, _ = _arrays(spo, num_qubits)
    columns = list(range(num_qubits))
    columns.insert(column, -1)
    keys, _ = _map(keys, columns)
    return SparsePauliOp(keys, c, num_qubits + 1)


def insert_identity_backward(spgo, num_qubits, column):
    """Transpose insertion by selecting I rows and removing the column."""
    if not 0 <= column < num_qubits:
        raise ValueError('Invalid insertion column.')
    keys, c, g = _arrays(spgo, num_qubits)
    keys, keep = _map(keys, [q for q in range(num_qubits) if q != column], [column], identity=True)
    return SparsePauliGradientOp(keys[keep], c[keep], g[keep], num_qubits - 1)


def checkpoint_size(state):
    """Count retained GPU allocations once, or resident host representation."""
    if isinstance(state, SparsePauliOp):
        allocations = {}
        for value in vars(state._storage).values():
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                allocations[storage.data_ptr()] = storage.nbytes()
        return sum(allocations.values())
    seen = set()
    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        total = sys.getsizeof(value)
        if isinstance(value, dict):
            total += sum(size(k) + size(v) for k, v in value.items())
        elif isinstance(value, (tuple, list)):
            total += sum(map(size, value))
        return total
    return size(state)


def _checkpoint_to_host(state):
    # Never call compact/export/to_host: retained snapshots are read-only and
    # must include zero rows. Blocking copies finish before device eviction.
    keys, c, g = state._raw_arrays()
    return dict(keys=keys.cpu().numpy().copy(), coefficients=c.cpu().numpy().copy(),
                gradient=None if g is None else g.cpu().numpy().copy(),
                num_qubits=state.num_qubits, system_size=state.system_size,
                active_qubits=None if state.active_qubits is None else list(state.active_qubits))


def _checkpoint_from_host(state, device):
    keys = torch.tensor(state['keys'], device=device)
    c = torch.tensor(state['coefficients'], device=device)
    if state['gradient'] is None:
        result = SparsePauliOp(keys, c, state['num_qubits'])
    else:
        result = SparsePauliGradientOp(keys, c, torch.tensor(state['gradient'], device=device), state['num_qubits'])
    result.system_size = state['system_size']
    result.active_qubits = None if state['active_qubits'] is None else list(state['active_qubits'])
    return result


def create_checkpoint_backend(state):
    return CheckpointBackend(size=checkpoint_size,
                             device_of=lambda obj: obj._storage.device if isinstance(obj, SparsePauliOp) else None,
                             to_host=_checkpoint_to_host, from_host=_checkpoint_from_host)
