"""Sparse Pauli gate kernels on NVIDIA GPUs, using PyTorch + Triton.

Supports real SPO/SPGO rotations, Clifford gates, and basis/OSE terminal losses
through the public SPD runner. Keys use SPD's packed uint32 layout, stored as
int32 tensors with identical bits.
"""

from functools import lru_cache
import math
import os

import numpy as np
import torch

# Growing supports otherwise retain many differently sized cached allocations.
# Apply once, including when torch was imported earlier; this does not initialize
# CUDA. Explicit process-wide allocator configuration always takes precedence.
if not any(name in os.environ for name in (
    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
)):
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")

from ..circuit_ir import PauliRotation, SkippedOperation
from .kernels import build_index, rotate
from . import utils
from .utils import set_precision
from .state_methods import StateMethods
from ..core import BaseSparsePauliOp


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


class SparsePauliOp(StateMethods, BaseSparsePauliOp):
    """Unique packed Pauli rows and real coefficients resident on one GPU.

    Construct with create_op; functional rotations preserve inputs.
    copy() creates independent storage; apply_in_place() explicitly mutates it.
    Row order is unspecified. to_host() returns SPD-compatible uint32 keys.
    """

    def __init__(self, keys, coefficients, num_qubits):
        from .persistent import _Storage
        self._storage = _Storage.wrap(keys, coefficients, num_qubits)
        self.num_qubits = num_qubits

    @classmethod
    def _from_storage(cls, storage):
        result = cls.__new__(cls)
        result._storage = storage
        result.num_qubits = storage.nq
        return result

    @property
    def precision(self):
        """Coefficient precision without exporting or materializing tensor views."""
        return "double" if self._storage.coeff.element_size() == 8 else "single"

    @property
    def xz_array(self):
        """Compact packed tensor view. Retained views are protected on mutation."""
        return self._storage.export()[0]

    @property
    def c_array(self):
        """Compact coefficient tensor view. Internal execution uses storage directly."""
        return self._storage.export()[1]

    def _raw_arrays(self):
        s = self._storage
        return s.keys[:s.n], s.coeff[:s.n], None if s.grad is None else s.grad[:s.n]

    def copy(self):
        """Return an independent mutable copy, including any adjoint channel."""
        from .persistent import _Storage
        with torch.cuda.device(self._storage.device):
            return type(self)._from_storage(_Storage(self, self._storage.dead_fraction))

    def _mutable_storage(self):
        if not self._storage.owned:
            self._storage = self.copy()._storage
        return self._storage

    def compact(self):
        """Release spare row capacity/dead slots; preserve the mathematical state."""
        with torch.cuda.device(self._storage.device):
            self._storage.materialize()
        return self

    def _rotate_in_place(self, generator, theta, cutoff, cap, *, diagnostics=True):
        from .operations import _validate_rotation, _prepare_gate, _zero_info
        backward = self._storage.grad is not None
        cap = _validate_rotation(self, theta, cutoff, cap, backward)
        with torch.cuda.device(self._storage.device):
            gate, params = _prepare_gate(self, generator, float(theta), float(cutoff))
            if not self._storage.n:
                return self, 0, 0. if backward else None, _zero_info() if diagnostics else None
            storage = self._mutable_storage()
            info = storage.apply(gate, params, cap, diagnostics=diagnostics)
        angle, info = info if backward else (None, info)
        return self, self.get_size(), angle, info

    def apply_in_place(self, operation, trunc_val=0., max_num_str=None, *, diagnostics=True):
        """Apply one IR gate, retaining storage: forward for SPO, backward for SPGO.

        Returns (self, count, angle_gradient_or_None, diagnostics). Copies once if
        tensors are borrowed/shared/exposed; copy() explicitly obtains private
        storage ahead of a gate loop. Skipped operations do not mutate anything.
        """
        from ..circuit_ir import SingleQubitClifford, TwoQubitClifford
        from .operations import _zero_info
        if isinstance(operation, SkippedOperation):
            return self, None, None, None
        if isinstance(operation, PauliRotation):
            return self._rotate_in_place(operation.pauli.rstrip("I"), operation.theta,
                                         trunc_val, max_num_str, diagnostics=diagnostics)
        if isinstance(operation, (SingleQubitClifford, TwoQubitClifford)):
            with torch.cuda.device(self._storage.device):
                self._mutable_storage().apply_clifford(operation)
            return self, self.get_size(), None, _zero_info() if diagnostics else None
        raise ValueError(f"Unsupported operation: {operation}")

    def get_size(self):
        return self._storage.size()

    def synchronize(self):
        torch.cuda.synchronize(self._storage.device)

    def to_host(self):
        self.compact()
        keys, c, _ = self._raw_arrays()
        return keys.cpu().numpy().view(np.uint32), c.cpu().numpy()

    def __getstate__(self):
        # Serialize the logical operator, not hash tables or execution counters.
        self.compact()
        keys, c, g = self._raw_arrays()
        state = {"xz_array": keys, "c_array": c, "num_qubits": self.num_qubits}
        if g is not None:
            state["grad_c_array"] = g
        return state

    def __setstate__(self, state):
        # Also accepts pickles written before storage became an owned object.
        from .persistent import _Storage
        self.num_qubits = state["num_qubits"]
        self._storage = _Storage.wrap(state["xz_array"], state["c_array"],
                                      self.num_qubits, state.get("grad_c_array"))

    def get_norm_square(self):
        return self._storage.coeff[:self._storage.n].square().sum().item()

    def get_operator_stabilizer_entropy(self, alpha=1.0):
        from .losses import operator_stabilizer_entropy
        return operator_stabilizer_entropy(self, alpha)

    get_OSE = get_operator_stabilizer_entropy

    def get_expectation_value(self, basis="Z"):
        keys, c, _ = self._raw_arrays()
        half = self._storage.width // 2
        if basis in ("Z", "0"):
            part = keys[:, :half]
        elif basis in ("X", "+"):
            part = keys[:, half:]
        else:
            raise ValueError(f"Unsupported basis: {basis}")
        return torch.sum(torch.where(torch.all(part == 0, dim=1), c, 0)).item()


def create_op(pauli_dict, num_qubits=None, precision=None, device="cuda"):
    precision = utils.get_precision() if precision is None else precision
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
    from .persistent import evolve_step_persistent

    operations = tuple(operations)
    if all(isinstance(op, SkippedOperation) for op in operations):
        return spo
    return evolve_step_persistent(spo, operations, trunc_val, max_num_str)

# Import after the state-only API so gradient/standard operations can reuse it.
from .gradient import SparsePauliGradientOp, create_gradient_op
from .operations import *

from .losses import init_gradient_from_basis_expectation, init_gradient_from_ose, init_gradient_spo

from .algebra import init_gradient_from_l2_difference, init_gradient_from_l2_difference_union
from .analysis import (
    get_depolarizing_susceptibility, get_one_qubit_depolarizing_susceptibility,
    get_two_qubit_depolarizing_susceptibility, pauli_product_uint,
    pauli_product_batched_second_uint,
)

from .operations import __all__ as _gate_exports

__all__ = _gate_exports + [
    'SparsePauliOp', 'SparsePauliGradientOp', 'create_op', 'create_gradient_op',
    'conjugate_pauli_rotation', 'evolve_step', 'set_precision', 'utils',
    'init_gradient_from_basis_expectation', 'init_gradient_from_ose', 'init_gradient_spo',
    'init_gradient_from_l2_difference', 'init_gradient_from_l2_difference_union',
    'get_depolarizing_susceptibility', 'get_one_qubit_depolarizing_susceptibility',
    'get_two_qubit_depolarizing_susceptibility', 'pauli_product_uint',
    'pauli_product_batched_second_uint',
]
