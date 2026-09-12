"""SPD-compatible gate calls with explicit diagnostics and backward support."""

from functools import lru_cache
import math
import operator

import numpy as np
import torch

from . import SparsePauliOp, _gate_data, create_op
from .gradient import SparsePauliGradientOp
from .kernels import build_index, rotate, clifford


def _zero_info():
    return {"num_str_truncated": 0, "truncated_l1_norm": 0., "truncated_l2_norm": 0.}


@lru_cache(maxsize=4096)
def _scalars(theta, cutoff, dtype, device):
    return torch.tensor([math.cos(theta), math.sin(theta), cutoff], dtype=dtype, device=device)


@lru_cache(maxsize=4096)
def _packed_gate(words, device):
    return torch.as_tensor(np.asarray(words, dtype=np.uint32).view(np.int32), device=device)


def _prepare_gate(state, generator, theta, cutoff):
    dtype, device = state._storage.dtype, state._storage.device
    if isinstance(generator, str):
        return _gate_data(generator, theta, cutoff, state.num_qubits, dtype, device)
    if isinstance(generator, torch.Tensor):
        if generator.device != device or generator.dtype not in (torch.int32, torch.uint32):
            raise ValueError("Packed generator must be int32/uint32 on the state's device")
        gate = generator.to(torch.int32).contiguous()
    else:
        array = np.asarray(generator)
        if array.ndim != 1 or array.dtype not in (np.dtype('int32'), np.dtype('uint32')):
            raise ValueError("Packed generator must be a 1D int32/uint32 array")
        gate = _packed_gate(tuple(array.view(np.uint32).tolist()), device)
    if gate.ndim != 1 or gate.numel() != state._storage.width:
        raise ValueError("Packed generator width does not match state")
    return gate, _scalars(theta, cutoff, dtype, device)


def _validate_rotation(state, theta, cutoff, cap, backward):
    if backward:
        if not isinstance(state, SparsePauliGradientOp):
            raise TypeError("Backward rotation requires a SparsePauliGradientOp")
    elif not isinstance(state, SparsePauliOp) or isinstance(state, SparsePauliGradientOp):
        raise TypeError("Forward rotation requires a SparsePauliOp")
    if not math.isfinite(theta) or not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("theta must be finite and cutoff finite and nonnegative")
    if cap is not None:
        try:
            cap = operator.index(cap)
        except TypeError as exc:
            raise ValueError("max_num_str must be a positive integer or None") from exc
        if cap < 1:
            raise ValueError("max_num_str must be a positive integer or None")
    if state._storage.n >= 2**30:
        raise ValueError("Triton backend requires fewer than 2**30 input rows")
    return cap


def _rotation(state, generator, theta, cutoff, cap, backward):
    cap = _validate_rotation(state, theta, cutoff, cap, backward)
    keys, coeff = state.xz_array, state.c_array
    n, width = keys.shape
    device = coeff.device
    with torch.cuda.device(device):
        gate, scalars = _prepare_gate(state, generator, float(theta), float(cutoff))
        if n == 0:
            return state, 0, 0., _zero_info()
        table_size = 1 << (max(4, 2 * n) - 1).bit_length()
        table = torch.full((table_size,), -1, dtype=torch.int32, device=device)
        count = torch.zeros((), dtype=torch.int32, device=device)
        out_keys = torch.empty((2 * n, width), dtype=torch.int32, device=device)
        out_coeff = torch.empty(2 * n, dtype=coeff.dtype, device=device)
        out_grad = torch.empty_like(out_coeff) if backward else None
        blocks = (n + 127) // 128
        # Float64 block sums avoid cancellation in discarded-norm accounting.
        stats = torch.empty((blocks, 4), dtype=torch.float64, device=device)
        build_index[(blocks,)](keys, table, n, table_size - 1, width, 128)
        rotate[(blocks,)](keys, coeff, table, gate, scalars, out_keys, out_coeff, count,
                         n, table_size - 1, width, 128,
                         grad=state.grad_c_array if backward else None,
                         out_grad=out_grad, stats=stats, BACKWARD=backward, DIAGNOSTICS=True,
                         enable_fp_fusion=False)
        size = count.item()
        out_keys, out_coeff = out_keys[:size], out_coeff[:size]
        if backward:
            out_grad = out_grad[:size]
        totals = stats[:, :4 if backward else 3].sum(dim=0)
        if cap is not None and size > cap:
            selected = torch.topk(out_coeff.abs(), cap, sorted=False).indices
            removed = torch.ones(size, dtype=torch.bool, device=device)
            removed[selected] = False
            magnitudes = torch.where(removed, out_coeff.abs(), 0).to(torch.float64)
            totals[:3] += torch.stack([(magnitudes != 0).sum().to(torch.float64),
                                      magnitudes.sum(), magnitudes.square().sum()])
            out_keys, out_coeff = out_keys[selected], out_coeff[selected]
            if backward:
                out_grad = out_grad[selected]
            size = cap
        values = totals.tolist()
        info = {"num_str_truncated": int(values[0]), "truncated_l1_norm": values[1],
                "truncated_l2_norm": math.sqrt(values[2])}
        if backward:
            result = SparsePauliGradientOp(out_keys, out_coeff, out_grad, state.num_qubits)
            return result, size, values[3], info
        return SparsePauliOp(out_keys, out_coeff, state.num_qubits), size, 0., info


def conjugate_pauli_rot_forward(spo, xzk, theta, trunc_val, max_num_str=None):
    """Forward rotation returning (state, live_count, discarded diagnostics).

    Use conjugate_pauli_rotation for the existing state-only fast path.
    This direct API enforces an exact cap; runner cap rounding is separate.
    """
    _validate_rotation(spo, theta, trunc_val, max_num_str, False)
    _prepare_gate(spo, xzk, float(theta), float(trunc_val))
    if not spo.get_size():
        return spo, 0, _zero_info()
    result = spo.copy()
    _, size, _, info = result._rotate_in_place(xzk, theta, trunc_val, max_num_str)
    return result.compact(), size, info


def conjugate_pauli_rot_backward(spgo, xzk, theta, trunc_val, max_num_str=None):
    """Reverse both primal/adjoint arrays; return state, count, dL/dtheta, info."""
    return _rotation(spgo, xzk, theta, trunc_val, max_num_str, True)


def create_measurement_op(measurement_dict, padded_system_size, *, precision=None):
    terms = {}
    for qubits, value in measurement_dict.items():
        pauli = ["I"] * padded_system_size
        for q in qubits:
            q = operator.index(q)
            if not 0 <= q < padded_system_size:
                raise ValueError("Measurement qubit is outside system size")
            pauli[q] = "Z"
        key = "".join(pauli)
        terms[key] = terms.get(key, 0.) + value
    return create_op(terms, num_qubits=padded_system_size, precision=precision)


def _clifford(state, q, r, op, backward):
    is_grad = isinstance(state, SparsePauliGradientOp)
    if not isinstance(state, SparsePauliOp) or is_grad != backward:
        raise TypeError("Forward Clifford requires SPO; backward Clifford requires SPGO")
    q = operator.index(q)
    if not 0 <= q < state.num_qubits:
        raise ValueError("Qubit is outside system size")
    if r is not None:
        r = operator.index(r)
        if not 0 <= r < state.num_qubits or r == q:
            raise ValueError("Two-qubit Clifford requires distinct qubits within system size")
    else:
        r = q
    if backward and op in (1, 2):
        op = 3 - op  # Inverse S/Sdg.
    n, width = state.xz_array.shape
    if not n:
        return state
    with torch.cuda.device(state.c_array.device):
        out_keys = torch.empty_like(state.xz_array)
        out_c = torch.empty_like(state.c_array)
        out_g = torch.empty_like(state.grad_c_array) if is_grad else None
        clifford[((n + 127) // 128,)](state.xz_array, state.c_array, out_keys, out_c,
                                    n, q, r, width, op, 128,
                                    grad=state.grad_c_array if is_grad else None,
                                    out_grad=out_g, GRADIENT=is_grad)
        if is_grad:
            return SparsePauliGradientOp(out_keys, out_c, out_g, state.num_qubits)
        return SparsePauliOp(out_keys, out_c, state.num_qubits)


def _make_clifford(op, backward, two_qubit):
    if two_qubit:
        def apply(state, control_qubit, target_qubit):
            return _clifford(state, control_qubit, target_qubit, op, backward)
    else:
        def apply(state, qubit):
            return _clifford(state, qubit, None, op, backward)
    return apply


for _name, _op in {"H": 0, "S": 1, "Sdg": 2, "X": 3, "Y": 4, "Z": 5, "CX": 6, "CZ": 7, "CY": 8}.items():
    for _backward in (False, True):
        _function_name = f"conjugate_{_name}_{'backward' if _backward else 'forward'}"
        _function = _make_clifford(_op, _backward, _name in ("CX", "CY", "CZ"))
        _function.__name__ = _function_name
        globals()[_function_name] = _function


__all__ = [f"conjugate_{gate}_{direction}" for gate in
           ("H", "S", "Sdg", "X", "Y", "Z", "CX", "CY", "CZ")
           for direction in ("forward", "backward")] + [
    "conjugate_pauli_rot_forward", "conjugate_pauli_rot_backward", "create_measurement_op"]
