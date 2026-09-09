"""CPU-side packing and configuration; no JAX initialization or device work."""

import numpy as np

_PRECISION = "double"


def set_packbit(packbit):
    if packbit != 32:
        raise ValueError("The Triton backend requires packbit=32")


def set_precision(precision):
    global _PRECISION
    if precision not in ("single", "double"):
        raise ValueError("precision must be single or double")
    _PRECISION = precision


def get_precision():
    return _PRECISION


def get_real_dtype():
    return np.float64 if _PRECISION == "double" else np.float32


def pauli_str_to_uint32(pauli, num_qubits=None):
    if num_qubits is None:
        num_qubits = len(pauli)
    if not isinstance(num_qubits, (int, np.integer)) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    if not isinstance(pauli, str) or len(pauli) > num_qubits or set(pauli) - set("IXYZ"):
        raise ValueError("Expected an I/X/Y/Z string fitting num_qubits")
    words = (num_qubits + 31) // 32
    key = np.zeros(2 * words, dtype=np.uint32)
    for q, p in enumerate(pauli):
        bit = np.uint32(1 << (31 - q % 32))
        if p in "XY":
            key[q // 32] |= bit
        if p in "YZ":
            key[words + q // 32] |= bit
    return key


pauli_str_to_uint = pauli_str_to_uint32


def uint32_to_pauli_str(packed, num_qubits):
    packed = np.asarray(packed, dtype=np.uint32)
    if packed.ndim != 1 or len(packed) % 2 or not 0 <= num_qubits <= len(packed) // 2 * 32:
        raise ValueError("Invalid packed key or qubit count")
    words = len(packed) // 2
    return "".join("IXZY"[((int(packed[q // 32]) >> (31 - q % 32)) & 1)
                          + 2 * ((int(packed[words + q // 32]) >> (31 - q % 32)) & 1)]
                   for q in range(num_qubits))


uint_to_pauli_str = uint32_to_pauli_str
