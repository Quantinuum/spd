import jax
import jax.numpy as jnp
from jax import lax
import time
from functools import partial
import numpy as np

if hasattr(jax, "config"):
    jax.config.update("jax_enable_x64", True)

DT_BOOL = jnp.bool_
_PRECISION = "single"
_REAL_DTYPE = jnp.float32
PHASES = jnp.array([1.0 + 0j, -1j, -1.0 + 0j, 1j], dtype=jnp.complex64)
CONJUGATION_SIGNS = jnp.array([0, 1, 0, -1], dtype=jnp.int8)

_PACKBIT = None

"""
Convention:
    We represent a Pauli string P as (x, z), where
    P = i^(x*z) X^x Z^z
    x,z are binary arrays of shape (N,) for N qubits.
    (x*z = sum_i x_i * z_i mod 4)

    So for example:
    I = (0,0)
    X = (1,0)
    Y = (1,1)
    Z = (0,1)
    The phase in Y is implicit in the formula.
"""


def set_precision(precision: str):
    global _PRECISION, _REAL_DTYPE

    if precision == "single":
        _REAL_DTYPE = jnp.float32
    elif precision == "double":
        _REAL_DTYPE = jnp.float64
    else:
        raise ValueError("precision must be one of ['single', 'double']")

    _PRECISION = precision


def get_precision() -> str:
    return _PRECISION


def get_real_dtype():
    return _REAL_DTYPE


def as_real_array(values):
    arr = jnp.asarray(values)
    if jnp.iscomplexobj(arr):
        raise ValueError("JAX SparsePauli coefficients must be real-valued.")
    return arr.astype(_REAL_DTYPE)


def real_scalar(value):
    arr = as_real_array(value)
    if arr.shape != ():
        raise ValueError("Expected a scalar value.")
    return arr

def pack_bits_to_uint8(x: jnp.ndarray) -> jnp.ndarray:
    """
    Pack a 1D boolean array into uint8 values.
    Big-endian within each uint8.
    The first element goes into the most significant bit.

    Args:
        x: boolean array of shape (n,)

    Returns:
        packed: uint8 array of shape (ceil(n/8),)
    """
    n = x.shape[0]
    n8 = (n + 7) // 8  # how many uint8 values we need

    # pad to multiple of 8
    padded = jnp.pad(x, (0, n8 * 8 - n))
    padded = padded.reshape(n8, 8)

    # bit positions [7..0] for big-endian
    bits = jnp.arange(7, -1, -1, dtype=jnp.uint8)

    # shift and sum
    # packed = jnp.sum((padded.astype(jnp.uint8) << bits), axis=1)
    packed = jnp.sum((padded.astype(jnp.uint8) << bits), axis=1, dtype=jnp.uint8)
    # packed = jnp.bitwise_or.reduce((padded.astype(jnp.uint8) << bits), axis=1)

    return packed

def pack_bits_to_uint32(x: jnp.ndarray) -> jnp.ndarray:
    """
    Pack a 1D boolean array into uint32 values.
    Big-endian within each uint32.
    The first element goes into the most significant bit.

    Args:
        x: boolean array of shape (n,)

    Returns:
        packed: uint32 array of shape (ceil(n/32),)
    """
    n = x.shape[0]
    n32 = (n + 31) // 32  # how many uint32 values we need

    # pad to multiple of 32
    padded = jnp.pad(x, (0, n32 * 32 - n))
    padded = padded.reshape(n32, 32)

    # bit positions [31..0] for big-endian
    bits = jnp.arange(31, -1, -1, dtype=jnp.uint32)

    # shift and sum
    packed = jnp.sum((padded.astype(jnp.uint32) << bits), axis=1)
    return packed

def pack_bits_to_uint64(x: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[0]
    n64 = (n + 63) // 64  # how many uint64 values we need
    padded = jnp.pad(x, (0, n64 * 64 - n))
    padded = padded.reshape(n64, 64)
    bits = jnp.arange(63, -1, -1, dtype=jnp.uint64)
    packed = jnp.sum((padded.astype(jnp.uint64) << bits), axis=1)
    return packed

def unpack_uint8_to_bits(packed: jnp.ndarray, n: int) -> jnp.ndarray:
    bits = jnp.arange(7, -1, -1, dtype=jnp.uint8)
    unpacked = ((packed[:, None] >> bits) & 1).astype(bool)
    return unpacked.reshape(-1)[:n]

def unpack_uint32_to_bits(packed: jnp.ndarray, n: int) -> jnp.ndarray:
    bits = jnp.arange(31, -1, -1, dtype=jnp.uint32)
    unpacked = ((packed[:, None] >> bits) & 1).astype(bool)
    return unpacked.reshape(-1)[:n]

def unpack_uint64_to_bits(packed: jnp.ndarray, n: int) -> jnp.ndarray:
    bits = jnp.arange(63, -1, -1, dtype=jnp.uint64)
    unpacked = ((packed[:, None] >> bits) & 1).astype(bool)
    return unpacked.reshape(-1)[:n]

def pauli_str_to_bool(pauli_str):
    """
    Convert a Pauli string to (x,z) using fixed-size jnp arrays.
    The `1j` factor in 'Y' = i X Z phase is implied.

    Parameters:
        pauli_str: string of length N, e.g. 'IXYZ'
    Returns:
        xz: jnp.bool_ array of shape (2N,)
    """
    N = len(pauli_str)
    xz = jnp.zeros(2 * N, dtype=bool)

    # Use indexing instead of append
    for i, p in enumerate(pauli_str):
        if p == 'I':
            pass  # x[i]=0, z[i]=0
        elif p == 'X':
            xz = xz.at[i].set(True)
        elif p == 'Y':
            xz = xz.at[i].set(True)
            xz = xz.at[N + i].set(True)
        elif p == 'Z':
            xz = xz.at[N + i].set(True)
        else:
            raise ValueError(f"Unknown Pauli character: {p}")

    return xz

def pauli_str_to_uint8(pauli_str):
    current_len = len(pauli_str)
    to_pad = 8 - (current_len % 8) if (current_len % 8) != 0 else 0
    pauli_str = pauli_str + 'I' * to_pad  # pad with 'I' to multiple of 8
    bool_array = pauli_str_to_bool(pauli_str)
    return pack_bits_to_uint8(bool_array)

def pauli_str_to_uint32(pauli_str):
    current_len = len(pauli_str)
    to_pad = 32 - (current_len % 32) if (current_len % 32) != 0 else 0
    pauli_str = pauli_str + 'I' * to_pad  # pad with 'I' to multiple of 32
    bool_array = pauli_str_to_bool(pauli_str)
    return pack_bits_to_uint32(bool_array)

def pauli_str_to_uint64(pauli_str):
    current_len = len(pauli_str)
    to_pad = 64 - (current_len % 64) if (current_len % 64) != 0 else 0
    pauli_str = pauli_str + 'I' * to_pad  # pad with 'I' to multiple of 64
    bool_array = pauli_str_to_bool(pauli_str)
    return pack_bits_to_uint64(bool_array)

def bool_to_pauli_str(xz):
    """
    Convert binary symplectic arrays to string.
    Done outside JAX tracing, using Python string.
    """
    assert xz.ndim == 1
    N = len(xz) // 2

    pauli_chars = []
    for i in range(N):
        xi_int = int(xz[i])
        zi_int = int(xz[N + i])
        if xi_int == 0 and zi_int == 0:
            pauli_chars.append('I')
        elif xi_int == 1 and zi_int == 0:
            pauli_chars.append('X')
        elif xi_int == 0 and zi_int == 1:
            pauli_chars.append('Z')
        elif xi_int == 1 and zi_int == 1:
            pauli_chars.append('Y')
        else:
            raise ValueError(f"Invalid combination: x={xi_int}, z={zi_int}")
    return ''.join(pauli_chars)

def uint8_to_pauli_str(packed, N):
    bool_array = unpack_uint8_to_bits(packed, 2 * N)
    return bool_to_pauli_str(bool_array)

def uint32_to_pauli_str(packed, N):
    bool_array = unpack_uint32_to_bits(packed, 2 * N)
    return bool_to_pauli_str(bool_array)

def uint64_to_pauli_str(packed, N):
    bool_array = unpack_uint64_to_bits(packed, 2 * N)
    return bool_to_pauli_str(bool_array)

def _format_coeff(coeff) -> str:
    coeff_arr = jnp.asarray(coeff)
    coeff_scalar = coeff_arr.item() if coeff_arr.shape == () else coeff

    if isinstance(coeff_scalar, complex) or jnp.iscomplexobj(coeff_scalar):
        coeff_complex = complex(coeff_scalar)
        if abs(coeff_complex.imag) < 1e-12:
            return repr(float(coeff_complex.real))
        return repr(coeff_complex)

    return repr(float(coeff_scalar))

def _get_padded_num_qubits(packed) -> int:
    return (packed.shape[-1] // 2) * _PACKBIT


def sparse_pauli_op_to_str(spo) -> str:
    lines = ["SparsePauliOp["]
    xz_rows = np.asarray(spo.xz_array)
    c_vals = np.asarray(spo.c_array)
    mask = np.abs(c_vals) > 1e-6
    xz_rows = xz_rows[mask]
    c_vals = c_vals[mask]
    for packed, coeff in zip(xz_rows, c_vals):
        pauli_str = uint_to_pauli_str(
            jnp.asarray(packed),
            _get_padded_num_qubits(packed),
        )
        lines.append(f"  {pauli_str} => {_format_coeff(coeff)}")
    lines.append("]")
    return "\n".join(lines)

def sparse_pauli_grad_op_to_str(spo) -> str:
    lines = ["SparsePauliGradientOp["]
    xz_rows = np.asarray(spo.xz_array)
    c_vals = np.asarray(spo.c_array)
    grad_vals = np.asarray(spo.grad_c_array)
    mask = (np.abs(c_vals) > 1e-6) | (np.abs(grad_vals) > 1e-6)
    xz_rows = xz_rows[mask]
    c_vals = c_vals[mask]
    grad_vals = grad_vals[mask]
    for packed, coeff, grad in zip(xz_rows, c_vals, grad_vals):
        pauli_str = uint_to_pauli_str(
            jnp.asarray(packed),
            _get_padded_num_qubits(packed),
        )
        lines.append(
            f"  {pauli_str} => coeff={_format_coeff(coeff)}, grad={_format_coeff(grad)}"
        )
    lines.append("]")
    return "\n".join(lines)

def set_packbit(n):
    global _PACKBIT
    if n not in [8, 32, 64]:
        raise ValueError("n must be one of [8, 32, 64]")

    _PACKBIT = n

def pauli_str_to_uint(*args, **kwargs):
    assert _PACKBIT is not None, "Packbit not set. Use set_packbit(n) with n in [8,32,64]."
    if _PACKBIT == 8:
        return pauli_str_to_uint8(*args, **kwargs)
    elif _PACKBIT == 32:
        return pauli_str_to_uint32(*args, **kwargs)
    elif _PACKBIT == 64:
        raise NotImplementedError("64-bit version has error")
        # return pauli_str_to_uint64(*args, **kwargs)
    else:
        raise ValueError("Packbit not set. Use set_packbit(n) with n in [8,32,64].")

def pack_bits_to_uint(x: jnp.ndarray) -> jnp.ndarray:
    assert _PACKBIT is not None, "Packbit not set. Use set_packbit(n) with n in [8,32,64]."
    if _PACKBIT == 8:
        return pack_bits_to_uint8(x)
    elif _PACKBIT == 32:
        return pack_bits_to_uint32(x)
    elif _PACKBIT == 64:
        raise NotImplementedError("64-bit version has error")
        # return pack_bits_to_uint64(x)
    else:
        raise ValueError("Packbit not set. Use set_packbit(n) with n in [8,32,64].")

def uint_to_pauli_str(*args, **kwargs):
    assert _PACKBIT is not None, "Packbit not set. Use set_packbit(n) with n in [8,32,64]."
    if _PACKBIT == 8:
        return uint8_to_pauli_str(*args, **kwargs)
    elif _PACKBIT == 32:
        return uint32_to_pauli_str(*args, **kwargs)
    elif _PACKBIT == 64:
        return uint64_to_pauli_str(*args, **kwargs)
    else:
        raise ValueError("Packbit not set. Use set_packbit(n) with n in [8,32,64].")


def _validate_translation_inputs(packed_rows, system_size: int):
    if system_size < 1:
        raise ValueError("system_size must be at least 1.")
    if _PACKBIT is None:
        raise ValueError("Packbit not set. Use set_packbit(n) before translating.")

    packed_rows = jnp.asarray(packed_rows)
    word_bits = _PACKBIT
    capacity = packed_rows.shape[-1] * word_bits
    if system_size > capacity:
        raise ValueError(
            f"system_size={system_size} exceeds the represented site capacity {capacity}."
        )
    return packed_rows, word_bits


def _prefix_mask(word_bits: int, active_bits: int, dtype):
    if active_bits <= 0:
        return jnp.asarray(0, dtype=dtype)

    full_mask = (1 << word_bits) - 1
    mask = ((1 << active_bits) - 1) << (word_bits - active_bits)
    return jnp.asarray(mask & full_mask, dtype=dtype)


def translate_packed_uint_rows_prefix_right(packed_rows, x: int, system_size: int):
    packed_rows, word_bits = _validate_translation_inputs(packed_rows, system_size)
    squeeze = packed_rows.ndim == 1
    if squeeze:
        packed_rows = packed_rows[None, :]

    x_mod = x % system_size
    if x_mod == 0:
        return packed_rows[0] if squeeze else jnp.array(packed_rows, copy=True)

    dtype = packed_rows.dtype
    n_prefix_words = (system_size + word_bits - 1) // word_bits
    source_positions = (np.arange(system_size) - x_mod) % system_size
    source_word_idx = jnp.asarray(source_positions // word_bits)
    source_shift = jnp.asarray(word_bits - 1 - (source_positions % word_bits), dtype=packed_rows.dtype)
    target_word_idx = np.arange(system_size) // word_bits
    target_shift = jnp.asarray(word_bits - 1 - (np.arange(system_size) % word_bits), dtype=packed_rows.dtype)

    prefix_masks = jnp.asarray(
        [
            int(_prefix_mask(word_bits, min(word_bits, system_size - word_idx * word_bits), dtype))
            for word_idx in range(n_prefix_words)
        ],
        dtype=dtype,
    )

    source_words = packed_rows[:, source_word_idx]
    source_bits = jnp.bitwise_and(jnp.right_shift(source_words, source_shift), jnp.asarray(1, dtype=dtype))

    translated_prefix_words = []
    for word_idx in range(n_prefix_words):
        positions = np.where(target_word_idx == word_idx)[0]
        target_word = jnp.asarray(0, dtype=dtype)
        if len(positions) > 0:
            shifted_bits = jnp.left_shift(source_bits[:, positions].astype(dtype), target_shift[positions])
            target_word = jnp.sum(shifted_bits, axis=1, dtype=dtype)
        preserved_tail = jnp.bitwise_and(
            packed_rows[:, word_idx],
            jnp.bitwise_not(prefix_masks[word_idx]),
        )
        translated_prefix_words.append(
            jnp.bitwise_or(
                preserved_tail,
                jnp.bitwise_and(target_word, prefix_masks[word_idx]),
            )
        )

    translated_prefix = jnp.stack(translated_prefix_words, axis=1)
    if n_prefix_words < packed_rows.shape[1]:
        translated_rows = jnp.concatenate(
            [translated_prefix, packed_rows[:, n_prefix_words:]],
            axis=1,
        )
    else:
        translated_rows = translated_prefix

    return translated_rows[0] if squeeze else translated_rows
