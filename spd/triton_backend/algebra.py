"""Sparse joins on the GPU, without sorting keys or floating-point atomics."""
import math
import numbers
import numpy as np
import torch

from . import SparsePauliOp
from .gradient import SparsePauliGradientOp
from .kernels import build_index
from .auxiliary_kernels import lookup


def check_pair(a, b, *, gradient=False):
    kind = SparsePauliGradientOp if gradient else SparsePauliOp
    if not isinstance(a, kind) or not isinstance(b, kind):
        raise TypeError("Operands must be matching Triton sparse Pauli states")
    if not gradient and (isinstance(a, SparsePauliGradientOp) or isinstance(b, SparsePauliGradientOp)):
        raise TypeError("This operation requires primal SPO operands")
    if a.xz_array.shape[1] != b.xz_array.shape[1]:
        raise ValueError("Operands must have the same packed width")
    if a.c_array.device != b.c_array.device or a.c_array.dtype != b.c_array.dtype:
        raise ValueError("Operands must have the same device and precision")


def match(query, source, *, mark=False):
    """Return source indices (-1 for absent) and optionally source-presence flags."""
    n, width = query.shape
    m = len(source)
    if max(n, m) >= 2**30:
        raise ValueError("Sparse joins require fewer than 2**30 rows per operand")
    with torch.cuda.device(query.device):
        indices = torch.full((n,), -1, dtype=torch.int32, device=query.device)
        matched = torch.zeros(m, dtype=torch.bool, device=query.device) if mark else None
        if n and m:
            size = 1 << (max(4, 2*m)-1).bit_length()
            table = torch.full((size,), -1, dtype=torch.int32, device=query.device)
            build_index[((m+127)//128,)](source, table, m, size-1, width, 128)
            lookup[((n+127)//128,)](query, source, table, indices, matched, n, size-1, width, mark, 128)
        return indices, matched


def aligned(values, indices):
    if not len(values):
        return values.new_zeros(len(indices))
    return torch.where(indices >= 0, values[indices.clamp_min(0).long()], 0)


def make_like(state, keys, c, g=None, num_qubits=None):
    size = state.num_qubits if num_qubits is None else num_qubits
    if g is not None:
        return SparsePauliGradientOp(keys, c, g, size)
    return SparsePauliOp(keys, c, size)


def add(a, b):
    is_grad = isinstance(a, SparsePauliGradientOp)
    if isinstance(b, numbers.Number) and b == 0:
        return make_like(a, a.xz_array, a.c_array, a.grad_c_array if is_grad else None)
    if not isinstance(b, SparsePauliOp) or isinstance(b, SparsePauliGradientOp) != is_grad:
        return NotImplemented
    check_pair(a, b, gradient=is_grad)
    idx, matched = match(a.xz_array, b.xz_array, mark=True)
    c = torch.cat((a.c_array + aligned(b.c_array, idx), b.c_array[~matched]))
    keys = torch.cat((a.xz_array, b.xz_array[~matched]))
    g = torch.cat((a.grad_c_array + aligned(b.grad_c_array, idx), b.grad_c_array[~matched])) if is_grad else None
    keep = c != 0
    if is_grad:
        keep |= g != 0
    return make_like(a, keys[keep], c[keep], g[keep] if is_grad else None, max(a.num_qubits, b.num_qubits))


def scale(state, scalar):
    if isinstance(scalar, torch.Tensor):
        if scalar.ndim != 0 or scalar.is_complex():
            raise ValueError("Expected a real scalar")
        scalar = scalar.item()
    array = np.asarray(scalar)
    if array.ndim or np.iscomplexobj(array):
        raise ValueError("Expected a real scalar")
    scalar = float(np.float32(array)) if state.c_array.dtype == torch.float32 else float(array)
    if not math.isfinite(scalar):
        raise ValueError("Expected a finite real scalar")
    # Match the reference scalar rule (np.isclose(scalar, 0), atol=1e-8).
    empty = abs(scalar) <= 1e-8
    c = state.c_array[:0] if empty else state.c_array * scalar
    keys = state.xz_array[:0] if empty else state.xz_array
    g = None
    if isinstance(state, SparsePauliGradientOp):
        g = state.grad_c_array[:0] if empty else state.grad_c_array * scalar
    return make_like(state, keys, c, g)


def dot(a, b):
    check_pair(a, b)
    if a.get_size() > b.get_size():
        a, b = b, a
    idx, _ = match(a.xz_array, b.xz_array)
    return (a.c_array * aligned(b.c_array, idx)).sum().item()


def init_gradient_from_l2_difference(spo, target_spo):
    return _l2(spo, target_spo, union=False)


def init_gradient_from_l2_difference_union(spo, target_spo):
    return _l2(spo, target_spo, union=True)


def _l2(spo, target, union):
    check_pair(spo, target)
    live, target_live = spo.c_array != 0, target.c_array != 0
    keys, c = spo.xz_array[live], spo.c_array[live]
    target_keys, target_c = target.xz_array[target_live], target.c_array[target_live]
    idx, matched = match(keys, target_keys, mark=union)
    g = 2 * (c - aligned(target_c, idx))
    if union:
        keys = torch.cat((keys, target_keys[~matched]))
        c = torch.cat((c, torch.zeros_like(target_c[~matched])))
        g = torch.cat((g, -2 * target_c[~matched]))
    return SparsePauliGradientOp(keys, c, g, max(spo.num_qubits, target.num_qubits))
