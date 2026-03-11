import numpy as np
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

def pack_bits_to_uint8(x: np.ndarray) -> np.ndarray:
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
    padded = np.pad(x, (0, n8 * 8 - n))
    padded = padded.reshape(n8, 8)

    # bit positions [7..0] for big-endian
    bits = np.arange(7, -1, -1, dtype=np.uint8)

    # shift and sum
    packed = np.sum((padded.astype(np.uint8) << bits), axis=1, dtype=np.uint8)
    return packed

def pack_bits_to_uint32(x: np.ndarray) -> np.ndarray:
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
    padded = np.pad(x, (0, n32 * 32 - n))
    padded = padded.reshape(n32, 32)

    # bit positions [31..0] for big-endian
    bits = np.arange(31, -1, -1, dtype=np.uint32)

    # shift and sum
    packed = np.sum((padded.astype(np.uint32) << bits), axis=1)
    return packed

def pack_bits_to_uint64(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    n64 = (n + 63) // 64  # how many uint64 values we need
    padded = np.pad(x, (0, n64 * 64 - n))
    padded = padded.reshape(n64, 64)
    bits = np.arange(63, -1, -1, dtype=np.uint64)
    packed = np.sum((padded.astype(np.uint64) << bits), axis=1)
    return packed

def unpack_uint8_to_bits(packed: np.ndarray, n: int) -> np.ndarray:
    bits = np.arange(7, -1, -1, dtype=np.uint8)
    unpacked = ((packed[:, None] >> bits) & 1).astype(bool)
    return unpacked.reshape(-1)[:n]

def unpack_uint32_to_bits(packed: np.ndarray, n: int) -> np.ndarray:
    bits = np.arange(31, -1, -1, dtype=np.uint32)
    unpacked = ((packed[:, None] >> bits) & 1).astype(bool)
    return unpacked.reshape(-1)[:n]

def unpack_uint64_to_bits(packed: np.ndarray, n: int) -> np.ndarray:
    bits = np.arange(63, -1, -1, dtype=np.uint64)
    unpacked = ((packed[:, None] >> bits) & 1).astype(bool)
    return unpacked.reshape(-1)[:n]

def pauli_str_to_bool(pauli_str):
    """
    Convert a Pauli string to (x,z) using fixed-size np arrays.

    Parameters:
        pauli_str: string of length N, e.g. 'IXYZ'
    Returns:
        xz: np.bool_ array of shape (2N,)
    """
    N = len(pauli_str)
    xz = np.zeros(2 * N, dtype=np.bool_)

    # Use indexing instead of append
    for i, p in enumerate(pauli_str):
        if p == 'I':
            pass  # x[i]=0, z[i]=0
        elif p == 'X':
            xz[i] = True
        elif p == 'Y':
            xz[i] = True
            xz[N + i] = True
        elif p == 'Z':
            xz[N + i] = True
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
    coeff_arr = np.asarray(coeff)
    coeff_scalar = coeff_arr.item() if coeff_arr.shape == () else coeff

    if isinstance(coeff_scalar, complex) or np.iscomplexobj(coeff_scalar):
        coeff_complex = complex(coeff_scalar)
        if abs(coeff_complex.imag) < 1e-12:
            return repr(float(coeff_complex.real))
        return repr(coeff_complex)

    return repr(float(coeff_scalar))

def sparse_pauli_op_to_str(spo) -> str:
    lines = ["SparsePauliOp["]
    for packed, coeff in spo.items():
        if np.abs(coeff) <= 1e-6:
            continue
        pauli_str = uint_to_pauli_str(np.asarray(packed), _PACKBIT)
        lines.append(f"  {pauli_str} => {_format_coeff(coeff)}")
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

def pack_bits_to_uint(x: np.ndarray) -> np.ndarray:
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
