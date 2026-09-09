"""SPGO storage and construction. Rotation kernels are in kernels.py."""

import numpy as np
import torch

from . import SparsePauliOp, create_op, utils


class SparsePauliGradientOp(SparsePauliOp):
    """Unique packed keys with primal and adjoint arrays on one GPU.

    Arrays supplied directly must contain unique rows. Host arrays are uploaded;
    device arrays are shared when contiguous. Operations never mutate inputs.
    Qubit count defaults to the full packed width when omitted.
    """

    def __init__(self, xz_array, c_array, grad_c_array, num_qubits=None):
        device = c_array.device if isinstance(c_array, torch.Tensor) else torch.device("cuda")
        if device.type != "cuda":
            raise ValueError("SPGO requires a CUDA device")
        for array in (xz_array, grad_c_array):
            if isinstance(array, torch.Tensor) and array.device != device:
                raise ValueError("SPGO arrays must be on the same device")
        c = torch.as_tensor(c_array, device=device)
        if c.ndim != 1 or c.dtype not in (torch.float32, torch.float64):
            raise ValueError("Primal coefficients must be a 1D float32/float64 array")
        g = torch.as_tensor(grad_c_array, device=device)
        if g.shape != c.shape or g.dtype != c.dtype:
            raise ValueError("Primal and adjoint shapes and dtypes must match")
        keys = torch.as_tensor(xz_array, device=device)
        if keys.ndim != 2 or keys.shape[0] != len(c) or keys.shape[1] == 0 or keys.shape[1] % 2:
            raise ValueError("Expected packed keys of shape (N, 2 * words)")
        if keys.dtype not in (torch.int32, torch.uint32):
            raise ValueError("Packed keys must have int32/uint32 dtype")
        if num_qubits is None:
            num_qubits = keys.shape[1] // 2 * 32
        if (not isinstance(num_qubits, (int, np.integer)) or num_qubits < 1
                or 2 * ((num_qubits + 31) // 32) != keys.shape[1]):
            raise ValueError("num_qubits does not match the packed width")
        super().__init__(keys.to(torch.int32).contiguous(), c.contiguous(), num_qubits)
        self.grad_c_array = g.contiguous()

    def to_host(self):
        keys, c = super().to_host()
        return keys, c, self.grad_c_array.cpu().numpy()

    def to_spo(self):
        return SparsePauliOp(self.xz_array, self.c_array, self.num_qubits)


def create_gradient_op(pauli_dict, num_qubits=None, precision=None, device="cuda"):
    """Build an SPGO from {Pauli string: (coefficient, adjoint)}."""
    primal = create_op({p: pair[0] for p, pair in pauli_dict.items()}, num_qubits, precision, device)
    adjoint = create_op({p: pair[1] for p, pair in pauli_dict.items()}, num_qubits, precision, device)
    return SparsePauliGradientOp(primal.xz_array, primal.c_array, adjoint.c_array, primal.num_qubits)
