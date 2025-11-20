import time
import numpy as np
from . import utils

PHASES = np.array([1.0+0j, -1j, -1.0+0j, 1j])

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

class SparsePauliOp(dict):
    pass

def create_measurement_op(measurement_dict, padded_system_size):
    """
    Create a SparsePauliOp from a measurement dict.
    measurement_dict: dict
        key: tuple of int - qubit indices measured in Z basis
        val: complex - coefficient
    padded_system_size: int - total number of qubits (after padding)
    """
    spo = SparsePauliOp()
    for key, val in measurement_dict.items():
        x_array = np.zeros((1, padded_system_size), dtype=np.bool_)
        z_array = np.array([1 if i in key else 0 for i in range(padded_system_size)], dtype=bool).reshape(1, -1)
        xz_array = np.concatenate((x_array, z_array), axis=1)
        xz_array = utils.pack_bits_to_uint(xz_array.flatten())
        spo[tuple(xz_array)] = val

    return spo

def create_op(pauli_dict):
    spo = SparsePauliOp()
    for key, val in pauli_dict.items():
        xz = utils.pauli_str_to_uint(key)
        spo[tuple(xz)] = val

    return spo

def get_norm_square(sparse_pauli_op):
    val = next(iter(sparse_pauli_op.values()))
    if type(val) == tuple:
        # Gradient SPO
        return sum(np.abs(v[0]) ** 2 for v in sparse_pauli_op.values())
    else:
        return np.linalg.norm(np.fromiter(sparse_pauli_op.values(), dtype=np.float32)) ** 2

def get_size(sparse_pauli_op):
    return len(sparse_pauli_op)

def get_expectation_value(spo):
    """
    This is too slow for large arrays.
    if ( p_str.count('X') + p_str.count('Y') ) == 0:
        exp_val += pauli_dict[key]

    We just use a mask from x_array to select I and Z-only terms.

    # This can be wrong due to overflow
    # mask = jnp.sum(xz_array[:, :N], axis=1) == 0  # Select I and Z-only terms
    """
    exp_val = 0
    N = len(next(iter(spo))) // 2
    for P, P_val in spo.items():
        xz_array = np.array(P)
        # N = xz_array.shape[0] // 2
        if np.all(xz_array[:N] == 0):
            exp_val += P_val

    return exp_val

def create_gradient_spo(spo):
    gradient_spo = {}
    N = len(next(iter(spo))) // 2
    for P, P_val in spo.items():
        xz_array = np.array(P)
        if np.all(xz_array[:N] == 0):
            g_val = 1.
        else:
            g_val = 0.

        gradient_spo[P] = (P_val, g_val)

    return gradient_spo

# ---------------------------------------------------------------------- #

# # ---------- single Pauli multiply (JAX) ----------
# def pauli_product(xz1, c1, xz2, c2):
#     """
#     Multiply two Pauli strings in binary symplectic form with complex coefficients.
#
#     Parameters:
#         xz1: bool arrays, shape (2N,) - first Pauli string
#         c1: complex scalar - coefficient of first Pauli
#         xz2: bool arrays, shape (2N,) - second Pauli string
#         c2: complex scalar - coefficient of second Pauli
#
#     Returns:
#         c_new: complex scalar
#         xz_new: bool arrays - resulting Pauli string
#     """
#     N = xz1.shape[0] // 2
#     # XOR for new Pauli
#     xz_new = jnp.bitwise_xor(xz1, xz2)  # ^
#
#     x1, z1 = xz1[:N], xz1[N:]
#     x2, z2 = xz2[:N], xz2[N:]
#
#     z1_int = z1.astype(jnp.int8)
#     x1_int = x1.astype(jnp.int8)
#     z2_int = z2.astype(jnp.int8)
#     x2_int = x2.astype(jnp.int8)
#     x_new, z_new = xz_new[:N], xz_new[N:]
#
#     # jnp.sum should not overflow as promote_integers is True by default
#     count = jnp.sum(2 * x1_int * z2_int + x1_int * z1_int + x2_int * z2_int - x_new * z_new) % 4
#     phase = (-1j) ** count
#
#     # Multiply coefficients with phase
#     c_new = c1 * c2 * phase
#
#     return xz_new, c_new
#
def pauli_product_uint(xz1, c1, xz2, c2):
    """
    Multiply two Pauli strings in packed format using NumPy.
    Supports uint8/16/32/64 arrays and complex coefficients.

    """
    N = xz1.shape[0] // 2

    xz_new = xz1 ^ xz2  # XOR for new Pauli

    # population counts (same as JAX)
    pop = np.bitwise_count  # NumPy 2.1+ unified bit count

    count = ((2 * pop(xz1[:N] & xz2[N:]).astype(np.int32).sum() +
              pop(xz1[:N] & xz1[N:]).astype(np.int32).sum() +
              pop(xz2[:N] & xz2[N:]).astype(np.int32).sum() -
              pop(xz_new[:N] & xz_new[N:]).astype(np.int32).sum()) % 4)

    phase = (-1j) ** count
    c_new = c1 * c2 * phase
    return xz_new, c_new

# # ---------- per-row/batched Pauli multiply (JAX) ----------
# @jax.jit
# def pauli_product_batched(xz1_array, c1_array, xz2, c2):
#     """
#     [Deprecated] This version is not efficient, because we are using
#     floating point operations over the boolean array.
#
#     Batched version of pauli_product.
#     xz1_array: bool arrays of shape (M, 2N)
#     c1_array: complex array of shape (M,)
#     xz2: bool arrays of shape (2N,)
#     c2: complex scalar
#
#     Returns:
#         xz_new_array: bool arrays of shape (M, 2N)
#         c_new_array: complex array of shape (M,)
#     """
#     raise NotImplementedError("Use pauli_product_batched_uint instead.")
#     N = xz1_array.shape[1] // 2
#     xz_new_array = jnp.bitwise_xor(xz1_array, xz2)
#
#     # count = jnp.sum(2 * x1_int * z2_int + x1_int * z1_int + x2_int * z2_int - x_new_int * z_new_int, axis=1) % 4
#     count = jnp.sum((2 * xz1_array[:, :N] * xz2[N:] + xz1_array[:, :N] * xz1_array[:, N:] +
#                     xz2[:N] * xz2[N:] - xz_new_array[:, :N] * xz_new_array[:, N:]),
#                     axis=1) % 4
#     phase = jnp.take(PHASES, count)   # vectorized lookup
#     c_new_array = c1_array * c2 * phase
#     return xz_new_array, c_new_array
#
# @jax.jit
# def pauli_product_batched_first_uint(xz1_array, c1_array, xz2, c2):
#     """
#     Batched version of pauli_product_uint.
#     xz1_array: uint arrays of shape (M, nbytes)
#     c1_array: complex array of shape (M,)
#     xz2: uint arrays of shape (nbytes,)
#     c2: complex scalar
#
#     Returns:
#         xz_new_array: uint arrays of shape (M, nbytes)
#         c_new_array: complex array of shape (M,)
#     """
#     N = xz1_array.shape[1] // 2
#     xz_new_array = xz1_array ^ xz2
#     count = jnp.sum((2 * jax.lax.population_count(xz1_array[:, :N] & xz2[N:]) +
#                      jax.lax.population_count(xz1_array[:, :N] & xz1_array[:, N:]) +
#                      jax.lax.population_count(xz2[:N] & xz2[N:]) -
#                      jax.lax.population_count(xz_new_array[:, :N] & xz_new_array[:, N:])),
#                     axis=1) % 4
#     phase = jnp.take(PHASES, count)   # vectorized lookup
#     c_new_array = c1_array * c2 * phase
#     return xz_new_array, c_new_array
#
# @jax.jit
# def pauli_product_batched_second_uint(xz1, c1, xz2_array, c2_array):
#     """
#     Batched version of pauli_product_uint.
#     xz1: uint arrays of shape (nbytes,)
#     c1: complex scalar
#     xz2_array: uint arrays of shape (M, nbytes)
#     c2_array: complex array of shape (M,)
#
#     Returns:
#         xz_new_array: uint arrays of shape (M, nbytes)
#         c_new_array: complex array of shape (M,)
#     """
#     N = xz2_array.shape[1] // 2
#     xz_new_array = xz1 ^ xz2_array
#     count = jnp.sum((2 * jax.lax.population_count(xz1[:N] & xz2_array[:, N:]) +
#                      jax.lax.population_count(xz1[:N] & xz1[N:]) +
#                      jax.lax.population_count(xz2_array[:, :N] & xz2_array[:, N:]) -
#                      jax.lax.population_count(xz_new_array[:, :N] & xz_new_array[:, N:])),
#                     axis=1) % 4
#     phase = jnp.take(PHASES, count)   # vectorized lookup
#     c_new_array = c1 * c2_array * phase
#     return xz_new_array, c_new_array
#
#
#
# # def conjugate_pauli(xj, zj, cj, xk, zk, theta):
# #     """
# #     Conjugate Pauli string j by rotation R_k(theta):
# #     exp(i theta/2 * sigma_k) * sigma_j * exp(-i theta/2 * sigma_k)
# #     = = = = = = = = = = = = = =
# #     case1: sigma_j, if commute
# #     case2: cos(theta) sigma_j + i sin(theta) sigma_k sigma_j, if anticommute
# #     """
# #     raise NotImplementedError("Use conjugated_pauli_batched instead.")
# #     acq_val = jnp.sum(zj & xk) - jnp.sum(xj & zk)
# #     acq_val = acq_val % 2  # 0 = commute, 1 = anticommute
# #
# #     def commute():
# #         return xj, zj, cj
# #
# #     def anticommute():
# #         x_new, z_new, phase = pauli_product(xk, zk, 1., xj, zj, 1.)
# #         new_cj = cj * jnp.array([jnp.cos(theta), 1j * jnp.sin(theta) * phase], dtype=jnp.complex64)
# #         return jnp.stack([xj, x_new]), jnp.stack([zj, z_new]), new_cj
# #
# #     return jax.lax.cond(acq_val==0, commute, anticommute)
#
# def conjugated_pauli_batched(xz_array, c_array, xzk, theta):
#     """
#     Pad to a fixed size and call JIT-compiled version.
#     """
#     # return conjugated_pauli_batched_(xz_array, c_array, xzk, theta)
#     M = xz_array.shape[0]
#     max_size = 2 ** int(jax.numpy.log2(M - 1e-1) + 1)  # touch M to avoid recompile
#     pad_size = max_size - M
#     xz_padded = jnp.pad(xz_array, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
#     c_padded = jnp.pad(c_array, ((0, pad_size),), mode='constant', constant_values=0)
#     xz1, c1, xz2, c2 = conjugated_pauli_batched_(xz_padded, c_padded, xzk, theta)
#     return xz1, c1, xz2, c2
#     # return xz1[:M], c1[:M], xz2[:M], c2[:M]
#
# @jax.jit
# def conjugated_pauli_batched_(xz_array, c_array, xzk, theta):
#     """
#     Conjugate a batch of Pauli strings by rotation R_k(theta):
#     exp(i theta/2 * sigma_k) * sigma_j * exp(-i theta/2 * sigma_k)
#     = = = = = = = = = = = = = =
#     case1: sigma_j, if commute
#     case2: cos(theta) sigma_j + i sin(theta) sigma_k sigma_j, if anticommute
#
#     Parameters:
#         xz_array: bool arrays of shape (M, 2N) - M Pauli strings on N qubits
#         c_array: complex array of shape (M,) - coefficients
#         xzk: bool arrays of shape (2N,) - Pauli string for rotation
#         theta: float scalar - rotation angle
#     Returns:
#         xz_array: bool array of shape (M, 2N) - unchanged Pauli strings
#         c_array_1: complex array of shape (M,) - coefficients for sigma_j
#         xz_array_2: bool array of shape (M, 2N) - Pauli strings for sigma_k sigma_j
#         c_array_2: complex array of shape (M,) - coefficients for sigma_k sigma_j
#
#     """
#     print("Recompile: conjugated_pauli_batched", xz_array.shape, c_array.shape, xzk.shape, type(theta), theta, "\n")
#     N = xz_array.shape[1] // 2
#     # acq_val = jnp.sum(z_array & xk, axis=1) - jnp.sum(x_array & zk, axis=1)
#     acq_val = jnp.sum(xz_array[:, N:] & xzk[:N], axis=1) - jnp.sum(xz_array[:, :N] & xzk[N:], axis=1)
#     acq_val = acq_val % 2  # 0 = commute, 1 = anticommute
#     theta = theta * acq_val
#
#     c_array_1 = c_array * jnp.cos(theta)
#     xz_array_2, phase_array = pauli_product_batched(xz_array, jnp.ones_like(c_array),
#                                                     xzk, 1.)
#     c_array_2 = 1j * c_array * jnp.sin(theta) * phase_array
#
#     return xz_array, c_array_1, xz_array_2, c_array_2
#
# def conjugated_pauli_batched_uint(xz_array, c_array, xzk, theta):
#     """
#     Pad to a fixed size and call JIT-compiled version.
#     """
#     M = xz_array.shape[0]
#     max_size = 2 ** int(jax.numpy.log2(M - 1e-1) + 1)  # touch M to avoid recompile
#     pad_size = max_size - M
#     xz_padded = jnp.pad(xz_array, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
#     c_padded = jnp.pad(c_array, ((0, pad_size),), mode='constant', constant_values=0)
#     xz1, c1, xz2, c2 = conjugated_pauli_batched_uint_(xz_padded, c_padded, xzk, theta)
#     return xz1, c1, xz2, c2
#
def check_anticommute_uint(xz1, xz2):
    """
    Check if two Pauli strings in packed uint form anticommute.
    Returns 1 if anticommute, 0 if commute.
    """
    N = xz1.shape[0] // 2
    # population count of bitwise AND
    term1 = np.bitwise_count(xz1[:N] & xz2[N:]).astype(np.int32).sum()
    term2 = np.bitwise_count(xz1[N:] & xz2[:N]).astype(np.int32).sum()
    acq = (term1 - term2) % 2  # 0 = commute, 1 = anticommute
    return acq

# ---------------------------------------------------------------------- #
def conjugated_pauli_forward(spo, xzk, theta, trunc_val):
    """
    [Support uint8, uint16, uint32, uint64]
    Conjugate a batch of Pauli strings in packed uint form by rotation R_k(theta):
    exp(i theta/2 * sigma_k) * sigma_j * exp(-i theta/2 * sigma_k)
    """
    new_spo_c = {}
    new_spo_a = {}
    # 1. Split the Op into C and AC parts
    for xz_key, c_val in spo.items():
        xz = np.array(xz_key)
        acq_val = check_anticommute_uint(xz, xzk)
        if acq_val == 0:
            # commute
            new_spo_c[xz_key] = new_spo_c.get(xz_key, 0) + c_val
        else:
            # anticommute
            new_spo_a[xz_key] = new_spo_a.get(xz_key, 0) + c_val

    # 2. construct the pairs of AC parts
    new_spo_a_pairs = {}
    for xz_key, c_val in new_spo_a.items():
        P = xz_key
        P_array = np.array(P)
        Q_array, c_phase = pauli_product_uint(xzk, 1., P_array, 1.)
        Q = tuple(Q_array)

        # We want to order the pairs in [\sigma, P, Q] s.t.
        # \sigma P = i Q, P Q = i \sigma
        if np.isclose(c_phase, 1j):
            P_val, Q_val = new_spo_a_pairs.get((P, Q), (0, 0))
            new_spo_a_pairs[(P, Q)] = (P_val + c_val, Q_val + 0)
        elif np.isclose(c_phase, -1j):
            Q_val, P_val = new_spo_a_pairs.get((Q, P), (0, 0))
            new_spo_a_pairs[(Q, P)] = (Q_val + 0, P_val + c_val)
        else:
            raise ValueError("Unexpected phase in Pauli product: {}".format(c_phase))

    # 3. Apply the rotation to each AC pair
    for (P, Q), (c_P, c_Q) in new_spo_a_pairs.items():
        # _, c_phase = pauli_product_uint(xzk, 1., np.array(P), 1.)
        # plus_or_minus = np.sign(c_phase * 1j)  # ±1
        plus_or_minus = -1
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        # Update P
        new_spo_c[P] = cos_theta * c_P + plus_or_minus * sin_theta * c_Q
        # Update Q
        new_spo_c[Q] = -plus_or_minus * sin_theta * c_P + cos_theta * c_Q

    # 4. Truncate small values
    for P in list(new_spo_c.keys()):
        if np.abs(new_spo_c[P]) < trunc_val:
            new_spo_c.pop(P)

    return new_spo_c, len(new_spo_c)
# ---------------------------------------------------------------------- #
def tuple_sum(a, b):
    return tuple(a[i] + b[i] for i in range(len(a)))

def zeros_like_tuple(t):
    return tuple(0 for _ in t)

def conjugated_pauli_backward(spo_val_grad, xzk, theta, trunc_val):
    """
    [Support uint8, uint16, uint32, uint64]
    Conjugate a batch of Pauli strings in packed uint form by rotation R_k(theta):
    exp(-i theta/2 * sigma_k) * sigma_j * exp(i theta/2 * sigma_k)
    """
    new_spo_c = {}
    old_spo_a = {}
    # 1. Split the Op into C and AC parts
    for xz_key, vals in spo_val_grad.items():
        xz = np.array(xz_key)
        acq_val = check_anticommute_uint(xz, xzk)
        if acq_val == 0:
            new_spo_c[xz_key] = vals  # commute
        else:
            old_spo_a[xz_key] = vals  # anticommute

    # 2. construct the pairs of AC parts
    old_spo_a_pairs = {}
    for P, vals in old_spo_a.items():
        Q_array, c_phase = pauli_product_uint(xzk, 1., np.array(P), 1.)
        Q = tuple(Q_array)

        # We want to order the pairs in [\sigma, P, Q] s.t.
        # \sigma P = i Q, P Q = i \sigma
        if np.isclose(c_phase, 1j):
            P_vals, Q_vals = old_spo_a_pairs.get((P, Q), (zeros_like_tuple(vals), zeros_like_tuple(vals)))
            old_spo_a_pairs[(P, Q)] = (tuple_sum(P_vals, vals), Q_vals)
        elif np.isclose(c_phase, -1j):
            Q_vals, P_vals = old_spo_a_pairs.get((Q, P), (zeros_like_tuple(vals), zeros_like_tuple(vals)))
            old_spo_a_pairs[(Q, P)] = (Q_vals, tuple_sum(P_vals, vals))
        else:
            raise ValueError("Unexpected phase in Pauli product: {}".format(c_phase))

    # 2.5 Get gradient with respect to theta
    theta_grad = 0
    for (P, Q), (P_vals, Q_vals) in old_spo_a_pairs.items():
        P_val, P_grad = P_vals
        Q_val, Q_grad = Q_vals
        theta_grad += (P_val * Q_grad - Q_val * P_grad)

    # 3. Apply the rotation channel-wise to each AC pair
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    pm = +1  # backward in time

    for (P, Q), (P_vals, Q_vals) in old_spo_a_pairs.items():
        rot_P_vals = tuple(cos_theta * P_vals[i] + pm * sin_theta * Q_vals[i] for i in range(len(P_vals)))
        rot_Q_vals = tuple(-pm * sin_theta * P_vals[i] + cos_theta * Q_vals[i] for i in range(len(Q_vals)))
        new_spo_c[P] = rot_P_vals
        new_spo_c[Q] = rot_Q_vals

    # 4. Truncate small values
    for P in list(new_spo_c.keys()):
        if np.abs(new_spo_c[P][0]) < trunc_val:
            new_spo_c.pop(P)

    return new_spo_c, len(new_spo_c), theta_grad
# ---------------------------------------------------------------------- #


def conjugated_pauli_batched_uint32_H(spo, qubit):
    """
    Apply Hadamard gate on the specified qubit for a batch of packed (x,z) representations.

    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        qubit: int, qubit index to apply H on.

    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), float (±1.0 per batch)

    """
    raise NotImplementedError("numpy version not ready yet.")
    xz_array = spo.xz_array
    c_array = spo.c_array
    print("Recompile: conjugated_pauli_batched_uint_H", xz_array.shape, c_array.shape, qubit, "\n")
    N = xz_array.shape[1] // 2
    x_array = xz_array[:, :N]
    z_array = xz_array[:, N:]

    site = qubit // 32
    bit = qubit % 32
    bit_mask = jnp.uint32(1 << (31 - bit))  # big-endian

    x_word = x_array[:, site]
    z_word = z_array[:, site]

    x_bit = x_word & bit_mask
    z_bit = z_word & bit_mask
    diff = x_bit ^ z_bit

    x_word_updated = x_word ^ diff
    z_word_updated = z_word ^ diff

    x_array = x_array.at[:, site].set(x_word_updated)
    z_array = z_array.at[:, site].set(z_word_updated)

    x_bit = x_word_updated & bit_mask
    z_bit = z_word_updated & bit_mask
    and_bit = x_bit & z_bit
    phase = jnp.power(-1.0, jax.lax.population_count(and_bit))

    xz_updated = jnp.concatenate([x_array, z_array], axis=1)
    new_spo = SparsePauliOp(xz_updated, phase * c_array)
    # return xz_updated, phase * c_array
    return new_spo

def conjugated_pauli_batched_uint32_S(spo, qubit):
    """
    Apply S gate on the specified qubit for a batch of packed (x,z) representations.

    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        qubit: int, qubit index to apply S on.

    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), complex (±1, ±i per batch)

    """
    raise NotImplementedError("numpy version not ready yet.")
    xz_array = spo.xz_array
    c_array = spo.c_array
    print("Recompile: conjugated_pauli_batched_uint_S", xz_array.shape, c_array.shape, qubit, "\n")
    N = xz_array.shape[1] // 2
    x_array = xz_array[:, :N]
    z_array = xz_array[:, N:]

    site = qubit // 32
    bit = qubit % 32
    bit_mask = jnp.uint32(1 << (31 - bit))

    x_word = x_array[:, site]
    z_word = z_array[:, site]

    x_bit = x_word & bit_mask

    # Update
    z_word_updated = z_word ^ x_bit
    z_array = z_array.at[:, site].set(z_word_updated)

    # Compute phase: (-1) ^ (x_bit & z_bit)
    and_bit = x_bit & z_word_updated
    phase = jnp.power(-1.0, jax.lax.population_count(and_bit))

    xz_updated = jnp.concatenate([x_array, z_array], axis=1)
    new_spo = SparsePauliOp(xz_updated, phase * c_array)
    # return xz_updated, phase * c_array
    return new_spo

def conjugated_pauli_batched_uint32_Sdg(spo, qubit):
    """
    Apply Sdg gate on the specified qubit for a batch of packed (x,z) representations.

    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        qubit: int, qubit index to apply S on.

    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), complex (±1, ±i per batch)

    """
    raise NotImplementedError("numpy version not ready yet.")
    xz_array = spo.xz_array
    c_array = spo.c_array
    print("Recompile: conjugated_pauli_batched_uint_Sdg", xz_array.shape, c_array.shape, qubit, "\n")
    N = xz_array.shape[1] // 2
    x_array = xz_array[:, :N]
    z_array = xz_array[:, N:]

    site = qubit // 32
    bit = qubit % 32
    bit_mask = jnp.uint32(1 << (31 - bit))

    x_word = x_array[:, site]
    z_word = z_array[:, site]

    x_bit = x_word & bit_mask
    # Compute phase: (-1) ^ (x_bit & z_bit)
    and_bit = x_bit & z_word

    # Update
    z_word_updated = z_word ^ x_bit
    z_array = z_array.at[:, site].set(z_word_updated)

    phase = jnp.power(-1.0, jax.lax.population_count(and_bit))

    xz_updated = jnp.concatenate([x_array, z_array], axis=1)
    new_spo = SparsePauliOp(xz_updated, phase * c_array)
    # return xz_updated, phase * c_array
    return new_spo

def conjugated_pauli_batched_uint32_CX(spo, control_qubit, target_qubit):
    """
    Apply CX gate on the specified qubits for a batch of packed (x,z) representations.
    x_t <-- x_t XOR x_c
    z_c <-- z_c XOR z_t
    phase = (-1)^{x_c z_t (z_c \\oplus x_t)}

    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        control_qubit: int, control qubit index.
        target_qubit: int, target qubit index.

    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), complex (±1, ±i per batch)

    """
    raise NotImplementedError("numpy version not ready yet.")
    xz_array = spo.xz_array
    c_array = spo.c_array
    print("Recompile: conjugated_pauli_batched_uint_CX", xz_array.shape, c_array.shape, control_qubit, target_qubit, "\n")
    N = xz_array.shape[1] // 2
    x_array = xz_array[:, :N]
    z_array = xz_array[:, N:]

    # --- Bit/word locations ---
    control_site = control_qubit // 32
    control_bit = control_qubit % 32
    target_site = target_qubit // 32
    target_bit = target_qubit % 32

    c_bit_mask = jnp.uint32(1 << (31 - control_bit))
    t_bit_mask = jnp.uint32(1 << (31 - target_bit))

    # --- Extract words ---
    x_c_word = x_array[:, control_site]
    z_c_word = z_array[:, control_site]
    x_t_word = x_array[:, target_site]
    z_t_word = z_array[:, target_site]

    # --- Extract bits (0/1 values) ---
    x_c_bit = (x_c_word & c_bit_mask) >> (31 - control_bit)
    z_c_bit = (z_c_word & c_bit_mask) >> (31 - control_bit)
    x_t_bit = (x_t_word & t_bit_mask) >> (31 - target_bit)
    z_t_bit = (z_t_word & t_bit_mask) >> (31 - target_bit)

    # --- Update rule ---
    # X_t ← X_t XOR X_c
    x_t_word_updated = x_t_word ^ (x_c_bit << (31 - target_bit))
    # Z_c ← Z_c XOR Z_t
    z_c_word_updated = z_c_word ^ (z_t_bit << (31 - control_bit))

    x_array = x_array.at[:, target_site].set(x_t_word_updated)
    z_array = z_array.at[:, control_site].set(z_c_word_updated)

    # Compute phase: (-1)^(x_control_bit & z_target_bit & (x_target_bit ^ z_control_bit ^ 1))
    and_bit = (x_c_bit & z_t_bit) & (z_c_bit == x_t_bit)
    phase = jnp.power(-1.0, and_bit)
    # phase = jnp.power(-1.0, jax.lax.population_count(and_bit))

    xz_updated = jnp.concatenate([x_array, z_array], axis=1)
    new_spo = SparsePauliOp(xz_updated, phase * c_array)
    # return xz_updated, phase * c_array
    return new_spo

def conjugated_pauli_batched_uint32_CY(spo, control_qubit, target_qubit):
    raise NotImplementedError("numpy version not ready yet.")
    xz_array = spo.xz_array
    c_array = spo.c_array
    print("Recompile: conjugated_pauli_batched_uint_CY", xz_array.shape, c_array.shape, control_qubit, target_qubit, "\n")
    # --- Step 1: S on target ---
    xz_array, c_array = conjugated_pauli_batched_uint32_S(xz_array, c_array, target_qubit)

    # --- Step 2: CX ---
    xz_array, c_array = conjugated_pauli_batched_uint32_CX(xz_array, c_array, control_qubit, target_qubit)

    # --- Step 3: S† on target ---
    xz_array, c_array = conjugated_pauli_batched_uint32_Sdg(xz_array, c_array, target_qubit)
    new_spo = SparsePauliOp(xz_array, c_array)
    # return xz_array, c_array
    return new_spo

def conjugated_pauli_batched_uint32_CZ(spo, control_qubit, target_qubit):
    """
    Apply CZ gate on the specified qubits for a batch of packed (x,z) representations.
    z_c' = z_c XOR x_t
    z_t' = z_t XOR x_c
    phase = (-1)^( x_c * x_t * (z_c XOR z_t) )

    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        control_qubit: int, control qubit index.
        target_qubit: int, target qubit index.

    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), complex (±1, ±i per batch)
    """
    raise NotImplementedError("numpy version not ready yet.")
    xz_array = spo.xz_array
    c_array = spo.c_array
    print("Recompile: conjugated_pauli_batched_uint_CZ", xz_array.shape, c_array.shape, control_qubit, target_qubit, "\n")
    N = xz_array.shape[1] // 2
    x_array = xz_array[:, :N]
    z_array = xz_array[:, N:]

    # --- Bit/word locations ---
    control_site = control_qubit // 32
    control_bit = control_qubit % 32
    target_site = target_qubit // 32
    target_bit = target_qubit % 32

    c_bit_mask = jnp.uint32(1 << (31 - control_bit))
    t_bit_mask = jnp.uint32(1 << (31 - target_bit))

    # --- Extract words ---
    x_c_word = x_array[:, control_site]
    z_c_word = z_array[:, control_site]
    x_t_word = x_array[:, target_site]
    z_t_word = z_array[:, target_site]

    # --- Extract bits (0/1 values) ---
    x_c_bit = (x_c_word & c_bit_mask) >> (31 - control_bit)
    z_c_bit = (z_c_word & c_bit_mask) >> (31 - control_bit)
    x_t_bit = (x_t_word & t_bit_mask) >> (31 - target_bit)
    z_t_bit = (z_t_word & t_bit_mask) >> (31 - target_bit)

    # --- Update rule ---
    # Z_c ← Z_c XOR X_t
    z_c_word_updated = z_c_word ^ (x_t_bit << (31 - control_bit))
    # Z_t ← Z_t XOR X_c
    z_t_word_updated = z_t_word ^ (x_c_bit << (31 - target_bit))

    z_array = z_array.at[:, control_site].set(z_c_word_updated)
    # We need to reextract z_t_word, otherwise if control_site == target_site,
    # we would erase the previous update.
    z_t_word = z_array[:, target_site]
    z_t_word_updated = z_t_word ^ (x_c_bit << (31 - target_bit))
    z_array = z_array.at[:, target_site].set(z_t_word_updated)

    # Compute phase: (-1)^(x_control_bit & x_target_bit & (z_control_bit ^ z_target_bit))
    and_bit = (x_c_bit & x_t_bit) & (z_c_bit ^ z_t_bit)
    phase = jnp.power(-1.0, and_bit)

    xz_updated = jnp.concatenate([x_array, z_array], axis=1)
    new_spo = SparsePauliOp(xz_updated, phase * c_array)
    # return xz_updated, phase * c_array
    return new_spo

def conjugated_pauli_batched_uint32_X(spo, qubit):
    """
    Apply X gate on the specified qubit for a batch of packed (x,z) representations.
    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        qubit: int, qubit index to apply X on.
    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), float (1.0 per batch)
    """
    raise NotImplementedError("numpy version not ready yet.")
    xz_array = spo.xz_array
    c_array = spo.c_array
    print("Recompile: conjugated_pauli_batched_uint_X", xz_array.shape, c_array.shape, qubit, "\n")
    N = xz_array.shape[1] // 2
    z_array = xz_array[:, N:]

    site = qubit // 32
    bit = qubit % 32
    bit_mask = jnp.uint32(1 << (31 - bit))

    z_word = z_array[:, site]
    z_bit = (z_word & bit_mask) >> (31 - bit)

    # Compute phase = (-1)^(z_bit)
    phase = jnp.power(-1.0, z_bit)
    # return xz_array, phase * c_array
    new_spo = SparsePauliOp(xz_array, phase * c_array)
    return new_spo

def conjugated_pauli_batched_uint32_Y(spo, qubit):
    """
    Apply Y gate on the specified qubit for a batch of packed (x,z) representations.
    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        qubit: int, qubit index to apply Y on.
    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), complex (±1, ±i per batch)
    """
    raise NotImplementedError("numpy version not ready yet.")
    xz_array = spo.xz_array
    c_array = spo.c_array
    print("Recompile: conjugated_pauli_batched_uint_Y", xz_array.shape, c_array.shape, qubit, "\n")
    N = xz_array.shape[1] // 2
    x_array = xz_array[:, :N]
    z_array = xz_array[:, N:]

    site = qubit // 32
    bit = qubit % 32
    bit_mask = jnp.uint32(1 << (31 - bit))

    x_word = x_array[:, site]
    z_word = z_array[:, site]

    x_bit = (x_word & bit_mask) >> (31 - bit)
    z_bit = (z_word & bit_mask) >> (31 - bit)

    # Compute phase = i * (-1)^(x_bit & z_bit)
    nor_bit = x_bit ^ z_bit
    phase = jnp.power(-1.0, nor_bit)

    # return xz_array, phase * c_array
    new_spo = SparsePauliOp(xz_array, phase * c_array)
    return new_spo

def conjugated_pauli_batched_uint32_Z(spo, qubit):
    """
    Apply Z gate on the specified qubit for a batch of packed (x,z) representations.
    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        qubit: int, qubit index to apply Z on.
    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), float (1.0 per batch)
    """
    raise NotImplementedError("numpy version not ready yet.")
    xz_array = spo.xz_array
    c_array = spo.c_array
    print("Recompile: conjugated_pauli_batched_uint_Z", xz_array.shape, c_array.shape, qubit, "\n")
    N = xz_array.shape[1] // 2
    x_array = xz_array[:, :N]

    site = qubit // 32
    bit = qubit % 32
    bit_mask = jnp.uint32(1 << (31 - bit))

    x_word = x_array[:, site]
    x_bit = (x_word & bit_mask) >> (31 - bit)

    # Compute phase = (-1)^(x_bit)
    phase = jnp.power(-1.0, x_bit)
    # return xz_array, phase * c_array
    new_spo = SparsePauliOp(xz_array, phase * c_array)
    return new_spo













