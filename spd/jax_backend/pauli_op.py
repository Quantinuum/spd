import jax
import jax.numpy as jnp
from jax import lax
import time
import functools
from typing import NamedTuple
from . import utils
import math

DT_BOOL = jnp.bool_
DT_CPLX = jnp.complex64   # or complex128 if you need double
PHASES = jnp.array([1.0+0j, -1j, -1.0+0j, 1j], dtype=DT_CPLX)
PAD_VAL = jnp.uint32(jnp.iinfo(jnp.uint32).max)

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

class SparsePauliOp(NamedTuple):
    xz_array: jnp.ndarray  # int arrays of shape (M, 2N)
    c_array: jnp.ndarray   # real arrays of shape (M,)

class SparsePauliGradientOp(NamedTuple):
    xz_array: jnp.ndarray  # int arrays of shape (M, 2N)
    c_array: jnp.ndarray   # real arrays of shape (M,)
    grad_c_array: jnp.ndarray   # real arrays of shape (M,)

def create_measurement_op(measurement_dict, padded_system_size,):
    xz_list = []
    c_list = []
    for key, val in measurement_dict.items():
        x_array = jnp.zeros((1, padded_system_size), dtype=bool)
        z_array = jnp.array([1 if i in key else 0 for i in range(padded_system_size)], dtype=bool).reshape(1, -1)
        xz_array = jnp.concatenate((x_array, z_array), axis=1)
        xz_array = utils.pack_bits_to_uint(xz_array.flatten())
        xz_list.append(xz_array)
        c_list.append(val)

    spo = SparsePauliOp(jnp.array(xz_list), jnp.array(c_list))
    return spo

def create_op(pauli_dict):
    xz_list = []
    c_list = []
    for key, val in pauli_dict.items():
        xz = utils.pauli_str_to_uint(key)
        xz_list.append(xz)
        c_list.append(val)

    spo = SparsePauliOp(jnp.array(xz_list), jnp.array(c_list))
    return spo

def get_norm_square(sparse_pauli_op):
    return jnp.sum(jnp.abs(sparse_pauli_op.c_array) ** 2)

def get_size(sparse_pauli_op):
    return sparse_pauli_op.c_array.size

def get_expectation_value(spo, basis='0'):
    """
    This is too slow for large arrays.
    if ( p_str.count('X') + p_str.count('Y') ) == 0:
        exp_val += pauli_dict[key]

    We just use a mask from x_array to select I and Z-only terms.

    # This can be wrong due to overflow
    # mask = jnp.sum(xz_array[:, :N], axis=1) == 0  # Select I and Z-only terms
    """
    xz_array = spo.xz_array
    c_array = spo.c_array
    N = xz_array.shape[1] // 2
    if basis in ['0', 'Z']:
        mask = jnp.all(xz_array[:, :N] == 0, axis=1)
    elif basis in ['+', 'X']:
        mask = jnp.all(xz_array[:, N:] == 0, axis=1)
    else:
        raise NotImplementedError(f"Expectation value in basis {basis} not implemented.")

    exp_val = jnp.sum(c_array[mask])
    return jnp.real(exp_val)

def create_gradient_spo(spo, basis='0'):
    """
    Create a SparsePauliOp for gradient calculation.
    Only keep the terms that contribute to the expectation value
    in the specified basis.

    Parameters:
        spo: SparsePauliOp
        basis: '0'/'Z' or '+'/'X'

    Returns:
        gradient_spo: SparsePauliOp
    """
    xz_array = spo.xz_array
    c_array = spo.c_array
    N = xz_array.shape[1] // 2

    # create a 1, 0 array if the term contributes to the expectation value
    if basis in ['0', 'Z']:
        mask = jnp.all(xz_array[:, :N] == 0, axis=1)
    elif basis in ['+', 'X']:
        mask = jnp.all(xz_array[:, N:] == 0, axis=1)
    else:
        raise NotImplementedError(f"Expectation value in basis {basis} not implemented.")

    grad_c_array = jnp.where(mask, jnp.ones_like(c_array), jnp.zeros_like(c_array))
    gradient_spo = SparsePauliGradientOp(xz_array, c_array, grad_c_array)
    return gradient_spo
# ---------------------------------------------------------------------- #

# Need to provide a single function to merge
# sparse_pauli_op_1, sparse_pauli_op_2 = backend.conjugated_pauli_batched_uint_(sparse_pauli_op, xzk, theta)
# sparse_pauli_op, num_string = backend.merge_and_pad(sparse_pauli_op_1, sparse_pauli_op_2, trunc_val=trunc_val,)
# So that the memory cost can be saved under XLA compilation.

# Ideally something like:
# sparse_pauli_op, num_string = backend.conjugated_pauli_forward(sparse_pauli_op, xzk, theta, trunc_val=trunc_val)

def conjugated_pauli_forward(spo, xzk, theta, trunc_val):
    """
    Conjugate a batch of Pauli strings by rotation R_k(theta):
    exp(i theta/2 * sigma_k) * sigma_j * exp(-i theta/2 * sigma_k)

    Parameters:
        spo: SparsePauliOp
        xzk: uint arrays of shape (nbytes,) - Pauli string for rotation
        theta: float scalar - rotation angle
        trunc_val: float scalar - truncation value for coefficients

    Returns:
        new_spo: SparsePauliOp
    """
    t0 = time.time()
    x_concat, c_concat, new_size, final_valid_count = forward_jitted(spo, xzk, theta, trunc_val)
    jax.block_until_ready(new_size)
    t1 = time.time()

    x_ = slice_to_size_x_arr(x_concat, int(new_size))
    c_ = slice_to_size_c_arr(c_concat, int(new_size))
    jax.block_until_ready(c_)
    t2 = time.time()

    # print("Merge time:", (t1 - t0) * 1000, "ms, Pad time:", (t2 - t1) * 1000, "ms, Final size:", new_size, "Valid count:", final_valid_count, "Original size:", x_array_1.shape[0] + x_array_2.shape[0])
    new_spo = SparsePauliOp(x_, c_)
    return new_spo, final_valid_count

@jax.jit
def forward_jitted(spo, xzk, theta, trunc_val):
    print("Recompile: forward_jitted", spo.xz_array.shape,)
    spo_1, spo_2 = conjugated_pauli_batched_uint_(spo, xzk, theta)

    x_concat, c_concat, final_valid_count = merge_(
        spo_1.xz_array, spo_1.c_array,
        spo_2.xz_array, spo_2.c_array,
        trunc_val)

    new_size = next_pow2(final_valid_count)
    return x_concat, c_concat, new_size, final_valid_count

def conjugated_pauli_backward(spo_val_grad, xzk, theta, trunc_val):
    """
    Backward pass for conjugated_pauli.

    Parameters:
        spo_val_grad: SparsePauliGradientOp
        xzk: uint arrays of shape (nbytes,) - Pauli string for rotation
        theta: float scalar - rotation angle
        trunc_val: float scalar - truncation value for coefficients

    Returns:
        new_spo_val_grad: SparsePauliGradientOp
        num_string: int - number of Pauli strings after truncation
        grad_i: float - gradient value
    """
    grad_i = get_gradient(spo_val_grad, xzk, theta)

    x_concat, c_concat, grad_c_concat, new_size, final_valid_count = backward_jitted(spo_val_grad, xzk, theta, trunc_val)
    x_ = slice_to_size_x_arr(x_concat, int(new_size))
    c_ = slice_to_size_c_arr(c_concat, int(new_size))
    grad_c_ = slice_to_size_c_arr(grad_c_concat, int(new_size))
    new_spo_val_grad = SparsePauliGradientOp(x_, c_, grad_c_)

    return new_spo_val_grad, final_valid_count, -grad_i

@jax.jit
def backward_jitted(spo_val_grad, xzk, theta, trunc_val):
    print("Recompile: backward_jitted", spo_val_grad.xz_array.shape,)
    spo_val_grad_1, spo_val_grad_2 = conjugated_pauli_backward_batched_uint_(spo_val_grad, xzk, theta)
    x_concat, c_concat, grad_c_concat, final_valid_count = merge_val_grad_(spo_val_grad_1, spo_val_grad_2, trunc_val)
    new_size = next_pow2(final_valid_count)
    return x_concat, c_concat, grad_c_concat, new_size, final_valid_count

@jax.jit
def get_gradient(spo_val_grad, xzk, theta):
    """
    Get gradient value from SparsePauliGradientOp.

    Parameters:
        spo_val_grad: SparsePauliGradientOp
        xzk: uint arrays of shape (nbytes,) - Pauli string for rotation
        theta: float scalar - rotation angle

    Returns:
        grad_i: float - gradient value
    """
    xz_array = spo_val_grad.xz_array
    c_array = spo_val_grad.c_array
    grad_c_array = spo_val_grad.grad_c_array
    N = xz_array.shape[1] // 2

    acq_val = jnp.sum(jax.lax.population_count(xz_array[:, N:] & xzk[:N]), axis=1) - \
                jnp.sum(jax.lax.population_count(xz_array[:, :N] & xzk[N:]), axis=1)
    acq_val = acq_val % 2  # 0 = commute, 1 = anticommute

    xz_array_p = xz_array
    c_array_p = c_array
    grad_c_array_p = grad_c_array

#     # [not jittable]    sub-routine1. Select anticommute subset and padded to next_power_2 -> xz_array_p, c_array_p, grad_c_array_p
#     anticommute_size = jnp.sum(acq_val)
#     if anticommute_size == 0:
#         return 0
#
#     max_size = next_pow2(anticommute_size)
#     pad_size = max_size - anticommute_size
#     xz_array_p = jnp.pad(xz_array[acq_val.astype(bool)], ((0, pad_size), (0, 0)), constant_values=PAD_VAL)
#     c_array_p = jnp.pad(c_array[acq_val.astype(bool)], ((0, pad_size),), constant_values=0.0)
#     grad_c_array_p = jnp.pad(grad_c_array[acq_val.astype(bool)], ((0, pad_size),), constant_values=0.0)

#     # Move all jittable into a single function and jit it!
#     grad_i = get_gradient_jitted(xz_array_p, c_array_p, grad_c_array_p, xzk)
#     return grad_i
#
# @jax.jit
# def get_gradient_jitted(xz_array_p, c_array_p, grad_c_array_p, xzk):
#     print("Recompile: get_gradient_jitted", xz_array_p.shape, c_array_p.shape, grad_c_array_p.shape, xzk.shape)

    # [jittable]        sub-routine2. Get conjugated xz_array_q, phase_array; copy c_array_q, grad_c_array_q
    xz_array_q, phase_array_p = pauli_product_batched_second_uint(xzk, 1.,
                                                                  xz_array_p, jnp.ones_like(c_array_p),)
    # phase_array is an indication of the relation between sigma and p
    phase_array_p = jnp.real(phase_array_p * (-1j))
    c_array_q = c_array_p.copy()
    grad_c_array_q = grad_c_array_p.copy()

    # [jittable]        sub-routine3. Argsort xz_array_q -> apply on phase_array, c_array_copy, grad_c_array_copy
    sort_indices = jnp.lexsort(xz_array_q.T[::-1])  # sort by rows
    xz_array_q_sorted = xz_array_q[sort_indices]
    c_array_q_sorted = c_array_q[sort_indices]
    grad_c_array_q_sorted = grad_c_array_q[sort_indices]

    # [jittable]        sub-routine4. Obtain is_duplicate, indices_in_b from xz_array_p, xz_array_q
    is_duplicate, indices_in_q = find_row_duplications(xz_array_p, xz_array_q_sorted)

    # [jittable]        sub-routine5. Obtain gradient from the cross terms only
    # Formula = \sum ( a_P * grad_Q - a_Q * grad_P )
    safe_indices = jnp.minimum(indices_in_q, xz_array_q.shape[0] - 1)

    # Gather the coefficients using the discovered indices
    # Shape: (M,)
    c_array_q_aligned = c_array_q_sorted[safe_indices]
    grad_c_array_q_aligned = grad_c_array_q_sorted[safe_indices]

    # Use the formula
    raw_products = phase_array_p * ( c_array_p * grad_c_array_q_aligned - c_array_q_aligned * grad_c_array_p )
    # print(raw_products)
    # import pdb; pdb.set_trace()

    # # Apply Mask:
    # # If is_duplicate is False, we multiply by 0.
    # # If is_duplicate is True (even for PAD_VAL rows), we multiply by the product.
    # # Note: Since we pad coefficients with 0.0, the PAD_VAL matches result in 0.0 anyway.
    # valid_products = jnp.where(is_duplicate, raw_products, 0.0)

    combined_mask = is_duplicate & (acq_val.astype(bool))
    valid_products = jnp.where(combined_mask, raw_products, 0.0)

    total_sum = jnp.sum(valid_products)
    return jnp.real(total_sum / 2.)



# ---------- single Pauli multiply (JAX) ----------
def pauli_product(xz1, c1, xz2, c2):
    """
    Multiply two Pauli strings in binary symplectic form with complex coefficients.

    Parameters:
        xz1: bool arrays, shape (2N,) - first Pauli string
        c1: complex scalar - coefficient of first Pauli
        xz2: bool arrays, shape (2N,) - second Pauli string
        c2: complex scalar - coefficient of second Pauli

    Returns:
        c_new: complex scalar
        xz_new: bool arrays - resulting Pauli string
    """
    N = xz1.shape[0] // 2
    # XOR for new Pauli
    xz_new = jnp.bitwise_xor(xz1, xz2)  # ^

    x1, z1 = xz1[:N], xz1[N:]
    x2, z2 = xz2[:N], xz2[N:]

    z1_int = z1.astype(jnp.int8)
    x1_int = x1.astype(jnp.int8)
    z2_int = z2.astype(jnp.int8)
    x2_int = x2.astype(jnp.int8)
    x_new, z_new = xz_new[:N], xz_new[N:]

    # jnp.sum should not overflow as promote_integers is True by default
    count = jnp.sum(2 * x1_int * z2_int + x1_int * z1_int + x2_int * z2_int - x_new * z_new) % 4
    phase = (-1j) ** count

    # Multiply coefficients with phase
    c_new = c1 * c2 * phase

    return xz_new, c_new

def pauli_product_uint(xz1, c1, xz2, c2):
    """
    Multiply two Pauli strings in packed format,
    uint8, uint16, uint32, uint64,
    form with complex coefficients.

    Parameters:
        xz1: uint arrays, shape (nbytes,) - first Pauli string
        c1: complex scalar - coefficient of first Pauli
        xz2: uint arrays, shape (nbytes,) - second Pauli string
        c2: complex scalar - coefficient of second Pauli

    Returns:
        c_new: complex scalar
        xz_new: uint arrays - resulting Pauli string
    """
    N = xz1.shape[0] // 2
    xz_new = xz1 ^ xz2  # XOR for new Pauli

    count = jnp.sum(2 * jax.lax.population_count(xz1[:N] & xz2[N:]) +
                    jax.lax.population_count(xz1[:N] & xz1[N:]) +
                    jax.lax.population_count(xz2[:N] & xz2[N:]) -
                    jax.lax.population_count(xz_new[:N] & xz_new[N:])) % 4
    phase = (-1j) ** count
    c_new = c1 * c2 * phase
    return xz_new, c_new
# ---------- per-row/batched Pauli multiply (JAX) ----------
@jax.jit
def pauli_product_batched(xz1_array, c1_array, xz2, c2):
    """
    [Deprecated] This version is not efficient, because we are using
    floating point operations over the boolean array.

    Batched version of pauli_product.
    xz1_array: bool arrays of shape (M, 2N)
    c1_array: complex array of shape (M,)
    xz2: bool arrays of shape (2N,)
    c2: complex scalar

    Returns:
        xz_new_array: bool arrays of shape (M, 2N)
        c_new_array: complex array of shape (M,)
    """
    raise NotImplementedError("Use pauli_product_batched_uint instead.")
    N = xz1_array.shape[1] // 2
    xz_new_array = jnp.bitwise_xor(xz1_array, xz2)

    # count = jnp.sum(2 * x1_int * z2_int + x1_int * z1_int + x2_int * z2_int - x_new_int * z_new_int, axis=1) % 4
    count = jnp.sum((2 * xz1_array[:, :N] * xz2[N:] + xz1_array[:, :N] * xz1_array[:, N:] +
                    xz2[:N] * xz2[N:] - xz_new_array[:, :N] * xz_new_array[:, N:]),
                    axis=1) % 4
    phase = jnp.take(PHASES, count)   # vectorized lookup
    c_new_array = c1_array * c2 * phase
    return xz_new_array, c_new_array

@jax.jit
def pauli_product_batched_first_uint(xz1_array, c1_array, xz2, c2):
    """
    Batched version of pauli_product_uint.
    xz1_array: uint arrays of shape (M, nbytes)
    c1_array: complex array of shape (M,)
    xz2: uint arrays of shape (nbytes,)
    c2: complex scalar

    Returns:
        xz_new_array: uint arrays of shape (M, nbytes)
        c_new_array: complex array of shape (M,)
    """
    N = xz1_array.shape[1] // 2
    xz_new_array = xz1_array ^ xz2
    count = jnp.sum((2 * jax.lax.population_count(xz1_array[:, :N] & xz2[N:]) +
                     jax.lax.population_count(xz1_array[:, :N] & xz1_array[:, N:]) +
                     jax.lax.population_count(xz2[:N] & xz2[N:]) -
                     jax.lax.population_count(xz_new_array[:, :N] & xz_new_array[:, N:])),
                    axis=1) % 4
    phase = jnp.take(PHASES, count)   # vectorized lookup
    c_new_array = c1_array * c2 * phase
    return xz_new_array, c_new_array

@jax.jit
def pauli_product_batched_second_uint(xz1, c1, xz2_array, c2_array):
    """
    Batched version of pauli_product_uint.
    xz1: uint arrays of shape (nbytes,)
    c1: complex scalar
    xz2_array: uint arrays of shape (M, nbytes)
    c2_array: complex array of shape (M,)

    Returns:
        xz_new_array: uint arrays of shape (M, nbytes)
        c_new_array: complex array of shape (M,)
    """
    N = xz2_array.shape[1] // 2
    xz_new_array = xz1 ^ xz2_array
    count = jnp.sum((2 * jax.lax.population_count(xz1[:N] & xz2_array[:, N:]) +
                     jax.lax.population_count(xz1[:N] & xz1[N:]) +
                     jax.lax.population_count(xz2_array[:, :N] & xz2_array[:, N:]) -
                     jax.lax.population_count(xz_new_array[:, :N] & xz_new_array[:, N:])),
                    axis=1) % 4
    phase = jnp.take(PHASES, count)   # vectorized lookup
    c_new_array = c1 * c2_array * phase
    return xz_new_array, c_new_array



# def conjugate_pauli(xj, zj, cj, xk, zk, theta):
#     """
#     Conjugate Pauli string j by rotation R_k(theta):
#     exp(i theta/2 * sigma_k) * sigma_j * exp(-i theta/2 * sigma_k)
#     = = = = = = = = = = = = = =
#     case1: sigma_j, if commute
#     case2: cos(theta) sigma_j + i sin(theta) sigma_k sigma_j, if anticommute
#     """
#     raise NotImplementedError("Use conjugated_pauli_batched instead.")
#     acq_val = jnp.sum(zj & xk) - jnp.sum(xj & zk)
#     acq_val = acq_val % 2  # 0 = commute, 1 = anticommute
#
#     def commute():
#         return xj, zj, cj
#
#     def anticommute():
#         x_new, z_new, phase = pauli_product(xk, zk, 1., xj, zj, 1.)
#         new_cj = cj * jnp.array([jnp.cos(theta), 1j * jnp.sin(theta) * phase], dtype=jnp.complex64)
#         return jnp.stack([xj, x_new]), jnp.stack([zj, z_new]), new_cj
#
#     return jax.lax.cond(acq_val==0, commute, anticommute)

def conjugated_pauli_batched(xz_array, c_array, xzk, theta):
    """
    Pad to a fixed size and call JIT-compiled version.
    """
    # return conjugated_pauli_batched_(xz_array, c_array, xzk, theta)
    M = xz_array.shape[0]
    max_size = 2 ** int(jax.numpy.log2(M - 1e-1) + 1)  # touch M to avoid recompile
    pad_size = max_size - M
    xz_padded = jnp.pad(xz_array, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
    c_padded = jnp.pad(c_array, ((0, pad_size),), mode='constant', constant_values=0)
    xz1, c1, xz2, c2 = conjugated_pauli_batched_(xz_padded, c_padded, xzk, theta)
    return xz1, c1, xz2, c2
    # return xz1[:M], c1[:M], xz2[:M], c2[:M]

@jax.jit
def conjugated_pauli_batched_(xz_array, c_array, xzk, theta):
    """
    Conjugate a batch of Pauli strings by rotation R_k(theta):
    exp(i theta/2 * sigma_k) * sigma_j * exp(-i theta/2 * sigma_k)
    = = = = = = = = = = = = = =
    case1: sigma_j, if commute
    case2: cos(theta) sigma_j + i sin(theta) sigma_k sigma_j, if anticommute

    Parameters:
        xz_array: bool arrays of shape (M, 2N) - M Pauli strings on N qubits
        c_array: complex array of shape (M,) - coefficients
        xzk: bool arrays of shape (2N,) - Pauli string for rotation
        theta: float scalar - rotation angle
    Returns:
        xz_array: bool array of shape (M, 2N) - unchanged Pauli strings
        c_array_1: complex array of shape (M,) - coefficients for sigma_j
        xz_array_2: bool array of shape (M, 2N) - Pauli strings for sigma_k sigma_j
        c_array_2: complex array of shape (M,) - coefficients for sigma_k sigma_j

    """
    print("Recompile: conjugated_pauli_batched", xz_array.shape, c_array.shape, xzk.shape, type(theta), theta, "\n")
    N = xz_array.shape[1] // 2
    # acq_val = jnp.sum(z_array & xk, axis=1) - jnp.sum(x_array & zk, axis=1)
    acq_val = jnp.sum(xz_array[:, N:] & xzk[:N], axis=1) - jnp.sum(xz_array[:, :N] & xzk[N:], axis=1)
    acq_val = acq_val % 2  # 0 = commute, 1 = anticommute
    theta = theta * acq_val

    c_array_1 = c_array * jnp.cos(theta)
    xz_array_2, phase_array = pauli_product_batched(xz_array, jnp.ones_like(c_array),
                                                    xzk, 1.)
    c_array_2 = 1j * c_array * jnp.sin(theta) * phase_array

    return xz_array, c_array_1, xz_array_2, c_array_2

def conjugated_pauli_batched_uint(xz_array, c_array, xzk, theta):
    """
    Pad to a fixed size and call JIT-compiled version.
    """
    M = xz_array.shape[0]
    max_size = 2 ** int(jax.numpy.log2(M - 1e-1) + 1)  # touch M to avoid recompile
    pad_size = max_size - M
    xz_padded = jnp.pad(xz_array, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
    c_padded = jnp.pad(c_array, ((0, pad_size),), mode='constant', constant_values=0)
    xz1, c1, xz2, c2 = conjugated_pauli_batched_uint_(xz_padded, c_padded, xzk, theta)
    return xz1, c1, xz2, c2

# @jax.jit
def conjugated_pauli_batched_uint_(spo, xzk, theta):
    """
    [Support uint8, uint16, uint32, uint64]
    Conjugate a batch of Pauli strings in packed uint form by rotation R_k(theta):
    exp(i theta/2 * sigma_k) * sigma_j * exp(-i theta/2 * sigma_k)

    Parameters:
        xz_array: uint arrays of shape (M, nbytes) - M Pauli strings on N qubits
        c_array: complex array of shape (M,) - coefficients
        xzk: uint arrays of shape (nbytes,) - Pauli string for rotation
        theta: float scalar - rotation angle
    Returns:
        xz_array: uint array of shape (M, nbytes) - unchanged Pauli strings
        c_array_1: complex array of shape (M,) - coefficients for sigma_j
        xz_array_2: uint array of shape (M, nbytes) - Pauli strings for sigma_k sigma_j
        c_array_2: complex array of shape (M,) - coefficients for sigma_k sigma_j
    """
    xz_array = spo.xz_array
    c_array = spo.c_array
    print("Recompile: conjugated_pauli_batched_uint", xz_array.shape, c_array.shape)

    N = xz_array.shape[1] // 2
    acq_val = jnp.sum(jax.lax.population_count(xz_array[:, N:] & xzk[:N]), axis=1) - \
                jnp.sum(jax.lax.population_count(xz_array[:, :N] & xzk[N:]), axis=1)
    acq_val = acq_val % 2  # 0 = commute, 1 = anticommute
    theta = theta * acq_val

    c_array_1 = c_array * jnp.cos(theta)
    xz_array_2, phase_array = pauli_product_batched_second_uint(xzk, 1.,
                                                                xz_array, jnp.ones_like(c_array),)
    c_array_2 = 1j * c_array * jnp.sin(theta) * phase_array

    spo_1 = SparsePauliOp(xz_array, c_array_1)
    spo_2 = SparsePauliOp(xz_array_2, c_array_2)
    # return xz_array, c_array_1, xz_array_2, c_array_2
    return spo_1, spo_2

def conjugated_pauli_backward_batched_uint_(spo_val_grad, xzk, theta):
    xz_array = spo_val_grad.xz_array
    c_array = spo_val_grad.c_array
    grad_c_array = spo_val_grad.grad_c_array
    print("Recompile: conjugated_pauli_backward_batched_uint", xz_array.shape, c_array.shape)

    N = xz_array.shape[1] // 2
    acq_val = jnp.sum(jax.lax.population_count(xz_array[:, N:] & xzk[:N]), axis=1) - \
                jnp.sum(jax.lax.population_count(xz_array[:, :N] & xzk[N:]), axis=1)
    acq_val = acq_val % 2  # 0 = commute, 1 = anticommute
    theta = theta * acq_val

    # Going backward
    theta = -theta

    c_array_1 = c_array * jnp.cos(theta)
    grad_c_array_1 = grad_c_array * jnp.cos(theta)
    xz_array_2, phase_array = pauli_product_batched_second_uint(xzk, 1.,
                                                                xz_array, jnp.ones_like(c_array),)
    c_array_2 = 1j * c_array * jnp.sin(theta) * phase_array
    grad_c_array_2 = 1j * grad_c_array * jnp.sin(theta) * phase_array

    spo_val_grad_1 = SparsePauliGradientOp(xz_array, c_array_1, grad_c_array_1)
    spo_val_grad_2 = SparsePauliGradientOp(xz_array_2, c_array_2, grad_c_array_2)
    return spo_val_grad_1, spo_val_grad_2

@jax.jit
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

@jax.jit
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

@jax.jit
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

@jax.jit
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

@jax.jit
def conjugated_pauli_batched_uint32_CY(spo, control_qubit, target_qubit):
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

@jax.jit
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

@jax.jit
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

@jax.jit
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

@jax.jit
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














# - CONJUGATE: xz_array, c_array, P, theta -> xz_array_1, c_array_1, xz_array_2, c_array_2
# - MERGE: xz_array, c_array_1, c_array_2, P, trunc_val -> dict_xz_array: c_array_3
# - UNPACK: dict_xz_array: c_array_3 -> xz_array, c_array_3

@jax.jit
def merge_pauli_batched_part_1(x_array_1, z_array_1, c_array_1,
                               x_array_2, z_array_2, c_array_2,
                               ):
    # concatenate
    x_array_merge = jnp.concatenate([x_array_1, x_array_2], axis=0).astype(jnp.uint8)
    z_array_merge = jnp.concatenate([z_array_1, z_array_2], axis=0).astype(jnp.uint8)
    c_array_merge = jnp.concatenate([c_array_1, c_array_2], axis=0).astype(DT_CPLX)


    """
    # truncate small coefficients
    mask = jnp.abs(c_array_merge) > eps
    x_array_merge = x_array_merge[mask]
    z_array_merge = z_array_merge[mask]
    c_array_merge = c_array_merge[mask]
    """

    # 1. pack bits -> (M, nbytes)
    xz_packed = jnp.packbits(jnp.concatenate([x_array_merge, z_array_merge], axis=1), axis=1, bitorder='big')  # (M, nbytes)

    # 2. lexsort: reverse columns so first byte is most significant
    order = jnp.lexsort(xz_packed.T[::-1])   # reverse rows to make first byte most significant
    xz_packed_sorted = xz_packed[order]
    C_sorted = c_array_merge[order]

    # 3. group boundaries: True where a new row begins
    # compare with previous row
    diff = jnp.any(xz_packed_sorted[1:] != xz_packed_sorted[:-1], axis=1)
    boundaries = jnp.concatenate([jnp.ones((1,), dtype=bool), diff])
    group_ids = jnp.cumsum(boundaries) - 1  # group index per row

    # 4. segment sum
    num_groups = group_ids[-1] + jnp.int32(1)
    return xz_packed_sorted, C_sorted, group_ids, boundaries, num_groups

def merge_pauli_batched_part_2(xz_packed_sorted, C_sorted, group_ids, boundaries,
                               num_groups, trunc_val,
                               ):
    # 5. pick one representative per group (first occurrence is fine)
    # idx = jnp.nonzero(boundaries, size=num_groups, fill_value=0)[0]
    idx = jnp.nonzero(boundaries)[0]   # dynamic-length integer array of indices

    xz_packed_unique = xz_packed_sorted[idx]
    c_array_new = jnp.ufunc(jnp.add, 2, 1).reduceat(C_sorted, idx)

    # 6. truncate small coeffs
    keep = jnp.abs(c_array_new) > trunc_val
    xz_packed_unique = xz_packed_unique[keep]
    c_array_new = c_array_new[keep]
    return xz_packed_unique, c_array_new

def merge_pauli_batched_part3_(xz_packed_unique, c_array_new, N):
    # 7. unpack back to bits and split to X,Z
    xz_new = jnp.unpackbits(xz_packed_unique, axis=1, bitorder='big')[:, :N*2]
    x_array_merge = xz_new[:, :N].astype(DT_BOOL)
    z_array_merge = xz_new[:, N:].astype(DT_BOOL)
    return x_array_merge, z_array_merge, c_array_new

merge_pauli_batched_part3 = jax.jit(merge_pauli_batched_part3_, static_argnames='N')


def merge_pauli_batched(x_array_1, z_array_1, c_array_1,
                        x_array_2, z_array_2, c_array_2,
                        trunc_val=1e-12, eps=1e-8,):
    """
    Merge two batches of Pauli strings into one batch, optionally truncating small coefficients.

    Parameters:
        x_array_1, z_array_1: bool arrays of shape (M1,N)
        c_array_1: complex array of shape (M1,)
        x_array_2, z_array_2: bool arrays of shape (M2,N)
        c_array_2: complex array of shape (M2,)
        trunc_val: float, minimum coefficient magnitude to keep

    Returns:
        x_array_merge, z_array_merge: bool arrays of shape (M1+M2,N)
        c_array_merge: complex array of shape (M1+M2,)
    """
    N = x_array_1.shape[1]
    t0 = time.time()
    xz_packed_sorted, C_sorted, group_ids, boundaries, num_groups = merge_pauli_batched_part_1(
        x_array_1, z_array_1, c_array_1,
        x_array_2, z_array_2, c_array_2)

    t1 = time.time()
    xz_packed_unique, c_array_new = merge_pauli_batched_part_2(
        xz_packed_sorted, C_sorted, group_ids, boundaries,
        num_groups, trunc_val)

    t2 = time.time()
    x_array_merge, z_array_merge, c_array_new = merge_pauli_batched_part3(
        xz_packed_unique, c_array_new, N)

    t3 = time.time()
    # print(f"pack {t1-t0:.4f}, group {t2-t1:.4f}, unpack {t3-t2:.4f}, total {t3-t0:.4f}")
    return x_array_merge, z_array_merge, c_array_new


@jax.jit
def merge_pauli_batched_2_part1(x_array_1, z_array_1, c_array_1,
                                x_array_2, z_array_2, c_array_2,
                                trunc_val=1e-12, eps=1e-8):
    """
    JAX-friendly merge with static shapes. Returns possibly zero-padded arrays.
    """
    # ---- 1. concatenate ----
    x_array = jnp.concatenate([x_array_1, x_array_2], axis=0).astype(jnp.uint8)
    z_array = jnp.concatenate([z_array_1, z_array_2], axis=0).astype(jnp.uint8)
    c_array = jnp.concatenate([c_array_1, c_array_2], axis=0).astype(DT_CPLX)

    # M, N are static shapes known to JAX at compile time
    M = x_array.shape[0]
    N = x_array.shape[1]

    # ---- 2. truncate eps ----
    mask = jnp.abs(c_array) > eps
    c_array = c_array * mask.astype(c_array.dtype)

    # ---- 3. pack bits ----
    xz = jnp.concatenate([x_array, z_array], axis=1)
    xz_packed = jnp.packbits(xz, axis=1, bitorder="big")

    # ---- 4. lexsort ----
    order = jnp.lexsort(xz_packed.T[::-1])
    xz_packed_sorted = xz_packed[order]
    c_sorted = c_array[order]

    # ---- 5. group boundaries ----
    same_as_prev = jnp.all(xz_packed_sorted[1:] == xz_packed_sorted[:-1], axis=1)
    boundaries = jnp.concatenate([jnp.array([True]), ~same_as_prev])
    group_ids = jnp.cumsum(boundaries) - 1

    # ---- 6. segment sum ----
    num_groups = jnp.max(group_ids) + 1
    c_new = jnp.zeros((num_groups,), dtype=c_sorted.dtype).at[group_ids].add(c_sorted)
    # c_new = jax.ops.segment_sum(c_sorted, group_ids, num_segments=num_groups)

    # ---- 7. representatives ----
    idx = jnp.where(boundaries, size=num_groups, fill_value=0)[0]
    xz_packed_unique = xz_packed_sorted[idx]

    # ---- 8. truncate small coeffs (still static) ----
    keep2 = jnp.abs(c_new) > trunc_val
    c_new = c_new * keep2

    # ---- 9. unpack ----
    xz_unpacked = jnp.unpackbits(xz_packed_unique, axis=1, bitorder="big")[:, :2*N]
    x_new = xz_unpacked[:, :N].astype(DT_BOOL)
    z_new = xz_unpacked[:, N:].astype(DT_BOOL)

    return x_new, z_new, c_new

def compact_paulis(x_new, z_new, c_new):
    """Remove zero-coefficient Paulis after JIT execution."""
    mask = jnp.abs(c_new) > 0
    idx = jnp.nonzero(mask, size=mask.shape[0])[0]  # dynamic length allowed here
    return x_new[idx], z_new[idx], c_new[idx]

def merge_pauli_batched_2(x_array_1, z_array_1, c_array_1,
                          x_array_2, z_array_2, c_array_2,
                          trunc_val=1e-12, eps=1e-8):
    """
    Merge two batches of Pauli strings into one batch, optionally truncating small coefficients.
    JAX-friendly version with static shapes. Returns compacted arrays.
    Parameters:
        x_array_1, z_array_1: bool arrays of shape (M1,N)
        c_array_1: complex array of shape (M1,)
        x_array_2, z_array_2: bool arrays of shape (M2,N)
        c_array_2: complex array of shape (M2,)
        trunc_val: float, minimum coefficient magnitude to keep
    Returns:
        x_array_merge, z_array_merge: bool arrays of shape (M_merge,N)
        c_array_merge: complex array of shape (M_merge,)
    """
    x_new, z_new, c_new = merge_pauli_batched_2_part1(
        x_array_1, z_array_1, c_array_1,
        x_array_2, z_array_2, c_array_2,
        # N,
        trunc_val, eps)
    return compact_paulis(x_new, z_new, c_new)


def merge_pauli_batches_fast(x_array_1, z_array_1, c_array_1,
                             x_array_2, z_array_2, c_array_2,
                             trunc_val=1e-12):
    """
    Efficiently merge two large batches of Pauli strings with deduplication.
    Identical (x,z) rows have their coefficients summed.

    Parameters:
        x_array_1, z_array_1: bool arrays of shape (M1,N)
        c_array_1: complex array of shape (M1,)
        x_array_2, z_array_2: bool arrays of shape (M2,N)
        c_array_2: complex array of shape (M2,)
        trunc_val: float, minimum coefficient magnitude to keep

    Returns:
        x_array_merge, z_array_merge: bool arrays of shape (M_merge,N)
        c_array_merge: complex array of shape (M_merge,)
    """
    # Concatenate batches
    x_all = jnp.concatenate([x_array_1, x_array_2], axis=0)
    z_all = jnp.concatenate([z_array_1, z_array_2], axis=0)
    c_all = jnp.concatenate([c_array_1, c_array_2], axis=0)

    # Convert bool -> uint8 and combine x and z for lexicographic sorting
    xz_all = jnp.concatenate([x_all.astype(jnp.uint8), z_all.astype(jnp.uint8)], axis=1)

    # Lexicographical sort of rows
    sort_idx = jnp.lexsort(jnp.flip(xz_all.T, axis=0))  # sort by last column first
    xz_sorted = xz_all[sort_idx]
    c_sorted = c_all[sort_idx]

    # Identify boundaries where rows change
    diff = jnp.any(xz_sorted[1:] != xz_sorted[:-1], axis=1)
    boundaries = jnp.concatenate([jnp.array([0]), jnp.where(diff)[0] + 1, jnp.array([xz_sorted.shape[0]])])

    # Assign segment ids
    segment_ids = jnp.zeros_like(c_sorted, dtype=jnp.int32)
    segment_ids = segment_ids.at[boundaries[1:-1]].set(jnp.arange(1, len(boundaries)-1))
    segment_ids = jnp.cumsum(segment_ids)

    # Sum coefficients for each segment
    c_merge = jax.ops.segment_sum(c_sorted, segment_ids, num_segments=len(boundaries)-1)

    # Take one representative xz per segment
    xz_merge = xz_sorted[boundaries[:-1]]

    # Split back x and z
    N = x_array_1.shape[1]
    x_merge = xz_merge[:, :N].astype(bool)
    z_merge = xz_merge[:, N:].astype(bool)

    # Truncate small coefficients
    mask = jnp.abs(c_merge) > trunc_val
    return x_merge[mask], z_merge[mask], c_merge[mask]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def benchmark_pauli_product_batched():
    M, N = 300000, 96
    key = jax.random.PRNGKey(0)
    # Random binary arrays
    xz1_array = jax.random.randint(key, (M, 2*N), 0, 2).astype(bool)
    c1_array = jax.random.normal(key, (M,)) + 1j * jax.random.normal(key, (M,))
    xz2 = jax.random.randint(key, (2*N,), 0, 2).astype(bool)
    c2 = 1.0 + 0j

    # Warm-up
    pauli_product_batched(xz1_array, c1_array, xz2, c2)
    jax.random.normal(key, (10,)).block_until_ready()  # ensure all computations are done

    # Timing
    start = time.time()
    xz_new, c_new = pauli_product_batched(xz1_array, c1_array, xz2, c2)
    jax.random.normal(key, (10,)).block_until_ready()  # ensure all computations are done
    end = time.time()
    print(f"Pauli product batched: M={M}, N={N}, time={end-start:.4f} s")
    print(f"Throughput: {(M)/(end-start)/1e6:.2f} M rows/s")
    print("Output shapes:", xz_new.shape, c_new.shape)


def benchmark_conjugated_pauli_batched():
    M, N = 300000, 96
    key = jax.random.PRNGKey(0)
    # Random binary arrays
    xz_array = jax.random.randint(key, (M, 2*N), 0, 2).astype(bool)
    c_array = jax.random.normal(key, (M,)) + 1j * jax.random.normal(key, (M,))
    xzk = jax.random.randint(key, (2*N,), 0, 2).astype(bool)
    theta = jnp.pi / 4

    # Warm-up
    conjugated_pauli_batched(xz_array, c_array, xzk, theta)
    jax.random.normal(key, (10,)).block_until_ready()  # ensure all computations are done

    # Timing
    start = time.time()
    xz1, c1, xz2, c2 = conjugated_pauli_batched(xz_array, c_array, xzk, theta)
    jax.random.normal(key, (10,)).block_until_ready()  # ensure all computations are done
    end = time.time()
    print(f"Conjugated Pauli batched: M={M}, N={N}, time={end-start:.4f} s")
    print(f"Throughput: {(M)/(end-start)/1e6:.2f} M rows/s")
    print("Output shapes:", xz1.shape, c1.shape, xz2.shape, c2.shape)

def benchmark_merge_pauli_batched():
    M1, M2, N = 32000, 32000, 96
    key = jax.random.PRNGKey(0)
    # Random binary arrays
    x1_array = jax.random.randint(key, (M1, N), 0, 2).astype(bool)
    z1_array = jax.random.randint(key, (M1, N), 0, 2).astype(bool)
    c1_array = jax.random.normal(key, (M1,)) + 1j * jax.random.normal(key, (M1,))
    x2_array = jax.random.randint(key, (M2, N), 0, 2).astype(bool)
    z2_array = jax.random.randint(key, (M2, N), 0, 2).astype(bool)
    c2_array = jax.random.normal(key, (M2,)) + 1j * jax.random.normal(key, (M2,))

    # x2_array = x1_array.copy()  # make identical to x1_array to test merging
    # z2_array = z1_array.copy()

    # Warm-up
    merge_pauli_batched(x1_array, z1_array, c1_array,
                        x2_array, z2_array, c2_array,
                        trunc_val=1e-12)
    jax.random.normal(key, (10,)).block_until_ready()  # ensure all computations are done

    # Timing
    start = time.time()
    x_new, z_new, c_new = merge_pauli_batched(x1_array, z1_array, c1_array,
                                              x2_array, z2_array, c2_array,
                                              trunc_val=1e-12)
    jax.random.normal(key, (10,)).block_until_ready()  # ensure all computations are done
    end = time.time()
    print(f"Merge Pauli batched: M1={M1}, M2={M2}, N={N}, time={end-start:.4f} s")
    print(f"Throughput: {((M1+M2))/(end-start)/1e6:.2f} M rows/s")
    print("Output shapes:", x_new.shape, z_new.shape, c_new.shape)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# @jax.jit
def merge_(x_array_1, c_array_1, x_array_2, c_array_2, trunc_val):
    print("Recompiling merge_...", x_array_1.shape, x_array_2.shape)
    x_concat = jnp.concatenate([x_array_1, x_array_2], axis=0)
    c_concat = jnp.concatenate([c_array_1, c_array_2], axis=0)
    # Don't need to hash as we are using uint8 array
    total_size = x_concat.shape[0]

    # sort_keys = [multi_hash[:, i] for i in reversed(range(N_CHUNKS))]
    # Lexicographic sort
    sort_indices = jnp.lexsort([x_concat[:, i] for i in range(x_concat.shape[1]-1, -1, -1)])

    c_concat = c_concat[sort_indices]
    x_concat = x_concat[sort_indices]
    # return sorted_c, sorted_hashes, sort_indices

    hash_changes = jnp.any(x_concat[1:] != x_concat[:-1],
                           axis=1)

    # Boundaries: first row is always a boundary, then where hashes change
    boundaries = jnp.concatenate([jnp.array([True]), hash_changes])

    # Assign group IDs: cumulative sum gives unique ID to each group
    group_ids = jnp.cumsum(boundaries) - 1
    # return boundaries, group_ids

    # Use total_size as num_segments (safe, concrete int)
    c_concat = jax.ops.segment_sum(c_concat, group_ids,
                                   num_segments=total_size,
                                   indices_are_sorted=True)
    # Automatic padding to total_size
    # Array([ 3,  7, 18,  8,  9], dtype=int32)
    # Array([ 3,  7, 18,  8,  9,  0,  0,  0,  0,  0], dtype=int32)

    # Now sorted according to c_concat before truncation
    c_sort_indices = jnp.argsort(-jnp.abs(c_concat))  # Descending order))
    c_concat = c_concat[c_sort_indices]

    mask = jnp.abs(c_concat) > trunc_val
    final_valid_count = jnp.sum(mask.astype(jnp.int32))
    c_concat = c_concat * mask.astype(c_concat.dtype)

    """
    # ---------------------------------------------------------------
    # [version 1]
    # Representatives for x (aligned with c_concat)
    x_concat = x_concat * boundaries[:, None].astype(x_concat.dtype)
    x_concat = jax.ops.segment_sum(x_concat, group_ids,
                                   num_segments=x_concat.shape[0],
                                   indices_are_sorted=True)
    # ---------------------------------------------------------------
    """
    # ---------------------------------------------------------------
    # [version 2]
    # Use scatter with boundaries as a mask
    x_concat = x_concat * boundaries[:, None].astype(x_concat.dtype)
    # Then use advanced indexing which might be more JIT-friendly
    x_concat = jnp.zeros_like(x_concat).at[group_ids].add(x_concat)
    # ---------------------------------------------------------------

    # x_concat = x_concat[c_sort_indices]
    # x_concat = x_concat * mask[:, None].astype(x_concat.dtype)
    # ---------------------------------------------------------------
    # x_final = jnp.zeros_like(x_concat)
    # valid_indices = c_sort_indices[mask]  # Only the indices we actually want
    # x_concat = x_final.at[:jnp.sum(mask)].set(x_concat[valid_indices])
    x_concat = x_concat[c_sort_indices] * mask[:, None].astype(x_concat.dtype)

    return x_concat, c_concat, final_valid_count

def merge_val_grad_(spo_val_grad_1, spo_val_grad_2, trunc_val):
    x_array_1, c_array_1, grad_c_array_1 = spo_val_grad_1.xz_array, spo_val_grad_1.c_array, spo_val_grad_1.grad_c_array
    x_array_2, c_array_2, grad_c_array_2 = spo_val_grad_2.xz_array, spo_val_grad_2.c_array, spo_val_grad_2.grad_c_array
    print("Recompiling merge_val_grad_ ...", x_array_1.shape, x_array_2.shape)
    x_concat = jnp.concatenate([x_array_1, x_array_2], axis=0)
    c_concat = jnp.concatenate([c_array_1, c_array_2], axis=0)
    grad_c_concat = jnp.concatenate([grad_c_array_1, grad_c_array_2], axis=0)
    # Don't need to hash as we are using uint8 array
    total_size = x_concat.shape[0]

    # sort_keys = [multi_hash[:, i] for i in reversed(range(N_CHUNKS))]
    # Lexicographic sort
    sort_indices = jnp.lexsort([x_concat[:, i] for i in range(x_concat.shape[1]-1, -1, -1)])

    c_concat = c_concat[sort_indices]
    x_concat = x_concat[sort_indices]
    grad_c_concat = grad_c_concat[sort_indices]
    # return sorted_c, sorted_hashes, sort_indices

    hash_changes = jnp.any(x_concat[1:] != x_concat[:-1],
                           axis=1)

    # Boundaries: first row is always a boundary, then where hashes change
    boundaries = jnp.concatenate([jnp.array([True]), hash_changes])

    # Assign group IDs: cumulative sum gives unique ID to each group
    group_ids = jnp.cumsum(boundaries) - 1
    # return boundaries, group_ids

    # Use total_size as num_segments (safe, concrete int)
    c_concat = jax.ops.segment_sum(c_concat, group_ids,
                                   num_segments=total_size,
                                   indices_are_sorted=True)
    # Automatic padding to total_size
    # Array([ 3,  7, 18,  8,  9], dtype=int32)
    # Array([ 3,  7, 18,  8,  9,  0,  0,  0,  0,  0], dtype=int32)
    grad_c_concat = jax.ops.segment_sum(grad_c_concat, group_ids,
                                        num_segments=total_size,
                                        indices_are_sorted=True)

    # Now sorted according to c_concat before truncation
    c_sort_indices = jnp.argsort(-jnp.abs(c_concat))  # Descending order))
    c_concat = c_concat[c_sort_indices]
    grad_c_concat = grad_c_concat[c_sort_indices]

    mask = jnp.abs(c_concat) > trunc_val
    final_valid_count = jnp.sum(mask.astype(jnp.int32))
    c_concat = c_concat * mask.astype(c_concat.dtype)
    grad_c_concat = grad_c_concat * mask.astype(grad_c_concat.dtype)

    # ---------------------------------------------------------------
    # [version 2]
    # Use scatter with boundaries as a mask
    x_concat = x_concat * boundaries[:, None].astype(x_concat.dtype)
    # Then use advanced indexing which might be more JIT-friendly
    x_concat = jnp.zeros_like(x_concat).at[group_ids].add(x_concat)
    # ---------------------------------------------------------------
    x_concat = x_concat[c_sort_indices] * mask[:, None].astype(x_concat.dtype)

    return x_concat, c_concat, grad_c_concat, final_valid_count



def merge_all(x_array_1, c_array_1, x_array_2, c_array_2):
    """
    Complete merge pipeline using all steps.
    """
    x_concat, c_concat, boundaries, group_ids = merge_(
        x_array_1, c_array_1, x_array_2, c_array_2)

    valid_count = jnp.max(group_ids) + 1

    return x_concat[boundaries], c_concat[:valid_count]


def next_pow2(x):
    """Return next power of 2 >= x (x is a JAX scalar)."""
    return (1 << jnp.ceil(jnp.log2(x)).astype(int))

def next_pow2_min16(x):
    num = (1 << jnp.ceil(jnp.log2(x)).astype(int))
    return jnp.maximum(num, 16)

@functools.partial(jax.jit, static_argnums=1)
def slice_to_size_x_arr(x_arr, size):
    """Slice arrays to the given size."""
    print("recompiling slice_to_size...", x_arr.shape, size, type(size))
    x_ = jax.lax.dynamic_slice(x_arr, (0, 0), (size, x_arr.shape[1]))
    return x_

@functools.partial(jax.jit, static_argnums=1)
def slice_to_size_c_arr(c_arr, size):
    """Slice arrays to the given size."""
    print("recompiling slice_to_size...", c_arr.shape, size, type(size))
    c_ = jax.lax.dynamic_slice(c_arr, (0,), (size,))
    return c_

def merge_and_pad(spo_1, spo_2, trunc_val):
    """
    Complete merge pipeline using all steps.
    Pads output to the closest fixed sizes in terms of power of 2.
    Also combine the trunction step
    This avoids recompilation both in the current and the downstream.
    """
    x_array_1, c_array_1 = spo_1.xz_array, spo_1.c_array
    x_array_2, c_array_2 = spo_2.xz_array, spo_2.c_array

    t0 = time.time()
    x_concat, c_concat, final_valid_count = merge_(
        x_array_1, c_array_1, x_array_2, c_array_2, trunc_val)

    jax.block_until_ready((x_concat, c_concat))
    t1 = time.time()

    new_size = next_pow2(final_valid_count)

    x_, c_ = slice_to_size(x_concat, c_concat, int(new_size))
    jax.block_until_ready((x_, c_))
    t2 = time.time()

    # print("Merge time:", (t1 - t0) * 1000, "ms, Pad time:", (t2 - t1) * 1000, "ms, Final size:", new_size, "Valid count:", final_valid_count, "Original size:", x_array_1.shape[0] + x_array_2.shape[0])
    new_spo = SparsePauliOp(x_, c_)
    # return x_, c_, final_valid_count
    return new_spo, final_valid_count

@jax.jit
def find_row_duplications(a, b):
    """
    Optimized Binary Search for Static Shapes.
    Replaces while_loop with a fixed unrolled loop for GPU efficiency.
    """
    # Assumes 'b' is the haystack.
    # M is the size of the haystack (b).
    M = b.shape[0]

    # # Calculate max iterations needed: log2(M) + buffer
    # # For M=4096, this is 12. We can just use a safe constant like 32
    # # or compute it dynamically if M is static.
    # num_steps = int(jnp.ceil(jnp.log2(M))) + 2
    # --- FIX IS HERE ---
    # Use Python's math library to calculate this at compile-time.
    # jnp.log2 creates a tracer; math.log2 creates a concrete int.
    num_steps = int(math.ceil(math.log2(M))) + 2
    # -------------------

    def lexical_gt(row1, row2):
        not_eq = row1 != row2
        first_diff_idx = jnp.argmax(not_eq)
        is_gt = row1[first_diff_idx] > row2[first_diff_idx]
        are_equal = jnp.all(row1 == row2)
        return is_gt & (~are_equal)

    def binary_search_row(needle, haystack):
        # Unrolled Binary Search
        # Instead of while(low < high), we iterate fixed times.

        def body_fun(i, state):
            low, high = state

            # Standard Binary Search logic
            mid = (low + high) // 2
            mid_row = haystack[mid]

            # Check condition
            go_right = lexical_gt(needle, mid_row)

            # Update bounds
            # Note: logic must ensure high doesn't get stuck if low=mid
            new_low = jnp.where(go_right, mid + 1, low)
            new_high = jnp.where(go_right, high, mid)

            return (new_low, new_high)

        # Use fori_loop (which JAX unrolls better than while_loop)
        # or simple python loop if num_steps is small constant.
        final_low, final_high = jax.lax.fori_loop(0, num_steps, body_fun, (0, M))

        return final_low

    # 3. Vectorize
    indices_in_b = jax.vmap(binary_search_row, in_axes=(0, None))(a, b)

    # 4. Verify matches (Same as your code)
    indices_in_b = jnp.minimum(indices_in_b, M - 1)
    potential_matches = b[indices_in_b]
    is_duplicate = jnp.all(a == potential_matches, axis=1)

    return is_duplicate, indices_in_b

@jax.jit
def find_row_duplications_old(a, b):
    """
    Finds rows in 'a' that also exist in 'b'.
    Assumes 'a' and 'b' are lexically sorted (M, N) int32 arrays.
    """
    print("Recompiling find_row_duplications...", a.shape, b.shape)

    # 1. Define Lexical Comparison Logic
    # We need to determine if row1 > row2 lexically.
    def lexical_gt(row1, row2):
        # Find where elements differ
        not_eq = row1 != row2

        # Find the first index where they differ.
        # If rows are identical, argmax returns 0 (but we handle equality later).
        first_diff_idx = jnp.argmax(not_eq)

        # Check the value at that differing index
        is_gt = row1[first_diff_idx] > row2[first_diff_idx]

        # If rows are exactly equal, is_gt might be False/True based on index 0.
        # We ensure strict inequality: if they are equal, it is NOT greater.
        are_equal = jnp.all(row1 == row2)
        return is_gt & (~are_equal)

    # 2. Define Binary Search (Lower Bound)
    # Finds the first index in 'haystack' where 'needle' could be inserted.
    def binary_search_row(needle, haystack):
        m = haystack.shape[0]

        def cond_fun(state):
            low, high = state
            return low < high

        def body_fun(state):
            low, high = state
            mid = (low + high) // 2
            mid_row = haystack[mid]

            # If mid_row < needle, we search the right half (low = mid + 1)
            # This is equivalent to: needle > mid_row
            go_right = lexical_gt(needle, mid_row)

            low = jnp.where(go_right, mid + 1, low)
            high = jnp.where(go_right, high, mid)

            return (low, high)

        # Run the loop
        idx, _ = jax.lax.while_loop(cond_fun, body_fun, (0, m))
        return idx

    # 3. Vectorize the search over all rows of 'a'
    # in_axes=(0, None) means: iterate over rows of 'a', keep 'b' fixed.
    indices_in_b = jax.vmap(binary_search_row, in_axes=(0, None))(a, b)

    # 4. Verify matches
    # The binary search gives us the insertion point. We must check if
    # the row at that point in 'b' is actually equal to the row in 'a'.

    # Handle out of bounds (if insertion point is at the end of b)
    indices_in_b = jnp.minimum(indices_in_b, b.shape[0] - 1)

    potential_matches = b[indices_in_b]
    is_duplicate = jnp.all(a == potential_matches, axis=1)

    # If the insertion index was out of bounds (size of b), it's not a match.
    # (Though our minimum clamp handles the crash, we need logic correctness).
    # If a row is greater than all rows in b, idx will be len(b).
    # With the clamp, we compare to the last element. If that's not equal, is_duplicate is False.

    return is_duplicate, indices_in_b

if __name__ == "__main__":
    print(" ==== Benchmarking Pauli Operations ==== ")
    benchmark_pauli_product_batched()
    print(" ==== Benchmarking Conjugation ==== ")
    benchmark_conjugated_pauli_batched()

    print(" ==== Benchmarking Merge ==== ")
    benchmark_merge_pauli_batched()

    # pauli_str_1 = "XIII"
    # pauli_str_2 = "YIII"
    # c_1 = 0.5 + 0j
    # c_2 = 1.0 + 0j
    # print("Input 1:", pauli_str_1, c_1)
    # print("Input 2:", pauli_str_2, c_2)
    # x1, z1 = pauli_str_to_binary(pauli_str_1)
    # x2, z2 = pauli_str_to_binary(pauli_str_2)
    # print("X1, Z1:", x1, z1)
    # print("X2, Z2:", x2, z2)

    # x_new, z_new, c_new = pauli_product(x1, z1, c_1, x2, z2, c_2)
    # pauli_str_new = binary_to_pauli_str(x_new, z_new)
    # print("Output:", pauli_str_new, c_new)

    # print("Rotation test:")
    # theta = np.pi / 4
    # x_k, z_k, c_k = conjugate_pauli(x1, z1, c_1, x2, z2, theta)
    # print("Conjugated:", x_k, z_k, c_k)

