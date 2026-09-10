"""GPU weight, translation, product and depolarizing-noise utilities."""
import operator
import numpy as np
import torch

from . import auxiliary_kernels as kernels
from .gradient import SparsePauliGradientOp


def weight_maps(state, *, include_mass=True):
    n, width = state.xz_array.shape
    if not n:
        return {}, {}
    with torch.cuda.device(state.c_array.device):
        weight = torch.empty(n, dtype=torch.int32, device=state.c_array.device)
        kernels.weights[((n+255)//256,)](state.xz_array, weight, n, width, 256)
        bins = width//2*32 + 1
        counts = torch.bincount(weight.long(), minlength=bins)
        mass = (torch.bincount(weight.long(), weights=state.c_array.to(torch.float64).square(), minlength=bins)
                if include_mass else None)
        # Only a small histogram crosses to CPU, never all packed rows.
        counts, mass = counts.tolist(), mass.tolist() if include_mass else []
        return ({w: v for w, v in enumerate(mass) if counts[w]},
                {w: v for w, v in enumerate(counts) if v})


def translate_state(state, shift, system_size):
    from .algebra import make_like
    shift, system_size = operator.index(shift), operator.index(system_size)
    if not 1 <= system_size <= state.xz_array.shape[1]//2*32:
        raise ValueError("system_size must be positive and fit the packed width")
    n, width = state.xz_array.shape
    shift %= system_size
    keys = state.xz_array
    if n and shift:
        with torch.cuda.device(state.c_array.device):
            keys = torch.empty_like(state.xz_array)
            kernels.translate[((n+127)//128,)](state.xz_array, keys, n, shift, width, system_size, 128)
    return make_like(state, keys, state.c_array,
                     state.grad_c_array if isinstance(state, SparsePauliGradientOp) else None,
                     max(state.num_qubits, system_size))


def get_depolarizing_susceptibility(spgo, qubits):
    if not isinstance(spgo, SparsePauliGradientOp):
        raise TypeError("Noise susceptibility requires a Triton SPGO")
    qubits = tuple(operator.index(q) for q in qubits)
    if len(qubits) not in (1, 2) or len(set(qubits)) != len(qubits):
        raise ValueError("qubits must contain one or two distinct sites")
    if any(q < 0 or q >= spgo.num_qubits for q in qubits):
        raise ValueError("qubit is outside the state")
    n, width = spgo.xz_array.shape
    if not n:
        return 0.
    with torch.cuda.device(spgo.c_array.device):
        partials = torch.empty((n+255)//256, dtype=torch.float64, device=spgo.c_array.device)
        kernels.susceptibility[(len(partials),)](spgo.xz_array, spgo.c_array, spgo.grad_c_array,
                                                partials, n, qubits[0], qubits[-1], width, 256)
        return partials.sum().item()


def get_one_qubit_depolarizing_susceptibility(spgo, qubit):
    return get_depolarizing_susceptibility(spgo, (qubit,))


def get_two_qubit_depolarizing_susceptibility(spgo, qubits):
    if len(qubits) != 2:
        raise ValueError("qubits must contain exactly two sites")
    return get_depolarizing_susceptibility(spgo, qubits)


def _packed(value, device, ndim):
    if isinstance(value, torch.Tensor) and value.device != device:
        raise ValueError("Packed tensors must be on the same CUDA device")
    array = torch.as_tensor(value, device=device)
    if array.ndim != ndim or array.dtype not in (torch.int32, torch.uint32):
        raise ValueError(f"Packed input must be a rank-{ndim} int32/uint32 array")
    return array.to(torch.int32).contiguous()


def pauli_product_batched_second_uint(xz1, c1, xz2_array, c2_array):
    """Multiply one packed Pauli by a batch; return CUDA keys and complex values."""
    device = next((x.device for x in (xz2_array, xz1) if isinstance(x, torch.Tensor)), torch.device('cuda', torch.cuda.current_device()))
    if device.type != 'cuda':
        raise ValueError("Pauli products require a CUDA device")
    with torch.cuda.device(device):
        left, right = _packed(xz1, device, 1), _packed(xz2_array, device, 2)
        n, width = right.shape
        if not width or width % 2 or len(left) != width:
            raise ValueError("Packed widths must be equal, positive and even")
        # Preserve float64/complex128 input precision; scalar Python values do
        # not force single-precision batch coefficients to double precision.
        coefficients = torch.as_tensor(c2_array, device=device)
        left_dtype = c1.dtype if isinstance(c1, torch.Tensor) else (np.asarray(c1).dtype if isinstance(c1, (np.ndarray, np.generic)) else None)
        double_left = left_dtype in (torch.float64, torch.complex128, np.dtype("float64"), np.dtype("complex128"))
        dtype = torch.complex128 if double_left or coefficients.dtype in (torch.float64, torch.complex128) else torch.complex64
        c1 = torch.as_tensor(c1, device=device, dtype=dtype)
        if coefficients.shape != (n,) or c1.ndim != 0:
            raise ValueError("Expected scalar c1 and a length-N coefficient vector")
        out = torch.empty_like(right)
        phase = torch.empty(n, dtype=torch.int32, device=device)
        if n:
            kernels.product[((n+255)//256,)](left, right, out, phase, n, width, 256)
        phases = torch.tensor([1, -1j, -1, 1j], dtype=dtype, device=device)
        return out, c1 * coefficients.to(dtype) * phases[phase.long()]


def pauli_product_uint(xz1, c1, xz2, c2):
    if isinstance(xz2, torch.Tensor):
        right = xz2.unsqueeze(0)
    else:
        right = np.asarray(xz2)[None, :]
    if isinstance(c2, torch.Tensor):
        values = c2.reshape(1)
    else:
        values = np.asarray([c2])
    keys, c = pauli_product_batched_second_uint(xz1, c1, right, values)
    return keys[0], c[0]
