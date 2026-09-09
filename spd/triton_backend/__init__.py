"""Forward sparse Pauli dynamics on NVIDIA GPUs, using PyTorch + Triton.

This independent execution path supports real observables and Pauli rotations.
It does not implement SPD's gradient or Clifford APIs. Keys use SPD's packed
uint32 layout, stored as int32 tensors with identical bits.
"""

from functools import lru_cache
import math

import numpy as np
import torch

from ..circuit_ir import PauliRotation, SkippedOperation
from .kernels import build_index, rotate


def _pack(pauli, num_qubits):
    if not isinstance(pauli, str) or len(pauli) > num_qubits or set(pauli) - set("IXYZ"):
        raise ValueError("Expected an I/X/Y/Z string fitting num_qubits.")
    words = (num_qubits + 31) // 32
    key = np.zeros(2 * words, dtype=np.uint32)
    for q, p in enumerate(pauli):
        bit = np.uint32(1 << (31 - q % 32))
        if p in "XY":
            key[q // 32] |= bit
        if p in "YZ":
            key[words + q // 32] |= bit
    return key.view(np.int32)


class SparsePauliOp:
    """Unique packed Pauli rows and real coefficients resident on one GPU.

    Construct with create_op; rotations return new states and preserve inputs.
    Row order is unspecified. to_host() returns SPD-compatible uint32 keys.
    """

    def __init__(self, keys, coefficients, num_qubits):
        self.xz_array = keys
        self.c_array = coefficients
        self.num_qubits = num_qubits

    def get_size(self):
        return self.c_array.numel()

    def synchronize(self):
        torch.cuda.synchronize(self.c_array.device)

    def to_host(self):
        return (self.xz_array.cpu().numpy().view(np.uint32),
                self.c_array.cpu().numpy())

    def get_norm_square(self):
        return torch.sum(self.c_array.square()).item()

    def get_expectation_value(self, basis="Z"):
        half = self.xz_array.shape[1] // 2
        if basis in ("Z", "0"):
            part = self.xz_array[:, :half]
        elif basis in ("X", "+"):
            part = self.xz_array[:, half:]
        else:
            raise ValueError(f"Unsupported basis: {basis}")
        return torch.sum(torch.where(torch.all(part == 0, dim=1), self.c_array, 0)).item()


def create_op(pauli_dict, num_qubits=None, precision="double", device="cuda"):
    if precision not in ("single", "double"):
        raise ValueError("precision must be single or double")
    if num_qubits is None:
        num_qubits = max(map(len, pauli_dict), default=0)
    if num_qubits < 1:
        raise ValueError("num_qubits must be positive (also for an empty observable)")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("The Triton backend requires a CUDA device")
    # Merge strings which become identical after identity padding.
    merged = {}
    for pauli, c in pauli_dict.items():
        if not np.isrealobj(c) or not np.isfinite(c):
            raise ValueError("Coefficients must be finite and real")
        key = tuple(_pack(pauli, num_qubits))
        merged[key] = merged.get(key, 0.) + float(c)
    width = 2 * ((num_qubits + 31) // 32)
    keys = np.array(list(merged), dtype=np.int32).reshape(-1, width)
    dtype = torch.float64 if precision == "double" else torch.float32
    return SparsePauliOp(torch.as_tensor(keys, device=device),
                         torch.tensor(list(merged.values()), dtype=dtype, device=device),
                         num_qubits)


@lru_cache(maxsize=4096)
def _gate_data(pauli, theta, cutoff, num_qubits, dtype, device):
    gate = torch.as_tensor(_pack(pauli, num_qubits), device=device)
    params = torch.tensor([math.cos(theta), math.sin(theta), cutoff], dtype=dtype, device=device)
    return gate, params


def conjugate_pauli_rotation(spo, pauli, theta, trunc_val=0., max_num_str=None):
    """Apply exp(+i theta P/2) O exp(-i theta P/2), then truncate.

    The strict cutoff matches SPD JAX forward evolution. A binding size cap
    retains the largest magnitudes; equal-magnitude cap ties have unspecified
    order, as in the existing parallel backend. No coefficient atomics or
    probabilistic key equality are used.
    """
    if not math.isfinite(theta) or not math.isfinite(trunc_val) or trunc_val < 0:
        raise ValueError("theta must be finite and trunc_val finite and nonnegative")
    if max_num_str is not None and (not isinstance(max_num_str, int) or max_num_str < 1):
        raise ValueError("max_num_str must be a positive integer or None")
    keys, coeff = spo.xz_array, spo.c_array
    device = coeff.device
    with torch.cuda.device(device):
        gate, params = _gate_data(pauli, float(theta), float(trunc_val),
                                  spo.num_qubits, coeff.dtype, device)
        n, width = keys.shape
        if n == 0:
            return spo
        if n >= 2**30:
            raise ValueError("Triton backend requires fewer than 2**30 input rows")
        table_size = 1 << (max(4, 2 * n) - 1).bit_length()
        table = torch.full((table_size,), -1, dtype=torch.int32, device=device)
        count = torch.zeros((), dtype=torch.int32, device=device)
        out_keys = torch.empty((2 * n, width), dtype=torch.int32, device=device)
        out_coeff = torch.empty((2 * n,), dtype=coeff.dtype, device=device)
        grid = ((n + 127) // 128,)
        build_index[grid](keys, table, n, table_size - 1, width, 128)
        rotate[grid](keys, coeff, table, gate, params, out_keys, out_coeff, count,
                     n, table_size - 1, width, 128, enable_fp_fusion=False)
        size = count.item()
        out_keys, out_coeff = out_keys[:size], out_coeff[:size]
        if max_num_str is not None and size > max_num_str:
            selected = torch.topk(out_coeff.abs(), max_num_str, sorted=False).indices
            out_keys, out_coeff = out_keys[selected], out_coeff[selected]
        return SparsePauliOp(out_keys, out_coeff, spo.num_qubits)


def evolve_step(spo, operations, trunc_val=0., max_num_str=None):
    """Apply circuit operations in reverse (Heisenberg) order."""
    operations = tuple(operations)
    for op in operations:
        if not isinstance(op, (PauliRotation, SkippedOperation)):
            raise NotImplementedError("Triton evolution currently supports Pauli rotations only")
    for op in reversed(operations):
        if isinstance(op, PauliRotation):
            spo = conjugate_pauli_rotation(spo, op.pauli, op.theta, trunc_val, max_num_str)
    return spo
