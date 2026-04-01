import jax
import jax.numpy as jnp
from jax import lax
import time
import functools
from .sparse_pauli import SparsePauliGradientOp, SparsePauliOp
from . import utils
import math
import numpy as np

DT_BOOL = jnp.bool_
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


def set_precision(precision: str):
    utils.set_precision(precision)

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

    spo = SparsePauliOp(jnp.asarray(xz_list), utils.as_real_array(c_list))
    return spo

def create_op(pauli_dict):
    xz_list = []
    c_list = []
    for key, val in pauli_dict.items():
        xz = utils.pauli_str_to_uint(key)
        xz_list.append(xz)
        c_list.append(val)

    spo = SparsePauliOp(jnp.asarray(xz_list), utils.as_real_array(c_list))
    return spo

def init_gradient_from_basis_expectation(spo, basis='0'):
    xz_array = spo.xz_array
    c_array = spo.c_array
    N = xz_array.shape[1] // 2

    if basis in ['0', 'Z']:
        mask = jnp.all(xz_array[:, :N] == 0, axis=1)
    elif basis in ['+', 'X']:
        mask = jnp.all(xz_array[:, N:] == 0, axis=1)
    else:
        raise NotImplementedError(f"Expectation value in basis {basis} not implemented.")

    grad_c_array = jnp.where(mask, jnp.ones_like(c_array), jnp.zeros_like(c_array))
    gradient_spo = SparsePauliGradientOp(xz_array, c_array, grad_c_array)
    return gradient_spo

def init_gradient_from_ose(spo, alpha=1.0):
    c_array = spo.c_array
    probabilities = jnp.abs(c_array) ** 2
    eps = utils.as_real_array(1e-12)

    if alpha == 1:
        grad_c_array = -2.0 * c_array * (jnp.log(probabilities + eps) + 1.0)
    else:
        denom = jnp.sum((probabilities + eps) ** alpha) + eps
        grad_c_array = (
            2.0
            * alpha
            * c_array
            * (probabilities + eps) ** (alpha - 1.0)
            / ((1.0 - alpha) * denom)
        )

    return SparsePauliGradientOp(spo.xz_array, c_array, grad_c_array)

def _sort_rows_and_coeffs(xz_array, c_array):
    if xz_array.shape[0] == 0:
        return xz_array, c_array
    sort_indices = jnp.lexsort([xz_array[:, i] for i in range(xz_array.shape[1] - 1, -1, -1)])
    return xz_array[sort_indices], c_array[sort_indices]

def _filter_nonzero_spo_arrays(spo):
    mask = jnp.abs(spo.c_array) > 0
    return spo.xz_array[mask], spo.c_array[mask]

def _align_coeffs_to_support(support_xz, source_xz, source_c):
    if support_xz.shape[0] == 0 or source_xz.shape[0] == 0:
        return jnp.zeros((support_xz.shape[0],), dtype=utils.get_real_dtype())

    source_xz, source_c = _sort_rows_and_coeffs(source_xz, source_c)
    is_duplicate, indices_in_source = find_row_duplications(support_xz, source_xz)
    safe_indices = jnp.minimum(indices_in_source, source_xz.shape[0] - 1)
    aligned = source_c[safe_indices]
    return jnp.where(is_duplicate, aligned, jnp.zeros((support_xz.shape[0],), dtype=source_c.dtype))

def _union_support_xz(spo, target_spo):
    spo_xz, spo_c = _filter_nonzero_spo_arrays(spo)
    target_xz, target_c = _filter_nonzero_spo_arrays(target_spo)
    if spo_xz.shape[0] == 0:
        return target_xz
    if target_xz.shape[0] == 0:
        return spo_xz

    support_marker = jnp.ones_like(spo_c)
    target_marker = jnp.ones_like(target_c)
    x_union, _, union_size = merge_(
        spo_xz,
        support_marker,
        target_xz,
        target_marker,
        0.0,
    )
    return x_union[:int(union_size)]

def init_gradient_from_l2_difference(spo, target_spo):
    support_xz = _union_support_xz(spo, target_spo)
    current_xz, current_c = _filter_nonzero_spo_arrays(spo)
    target_xz, target_c = _filter_nonzero_spo_arrays(target_spo)
    current_coeffs = _align_coeffs_to_support(support_xz, current_xz, current_c)
    target_coeffs = _align_coeffs_to_support(support_xz, target_xz, target_c)
    grad_coeffs = 2.0 * (current_coeffs - target_coeffs)
    return SparsePauliGradientOp(support_xz, current_coeffs, grad_coeffs)

def init_gradient_spo(
    spo,
    *,
    loss_type='basis_expectation',
    basis='0',
    target_spo=None,
    lambda_ose=0.0,
    alpha=1.0,
):
    """Canonical gradient initializer for terminal losses on the JAX backend."""
    if loss_type == 'basis_expectation':
        gradient_spo = init_gradient_from_basis_expectation(spo, basis=basis)
    elif loss_type == 'l2_difference':
        if target_spo is None:
            raise ValueError("target_spo must be provided when loss_type='l2_difference'.")
        gradient_spo = init_gradient_from_l2_difference(spo, target_spo)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    if lambda_ose != 0.0:
        gradient_spo = gradient_spo + lambda_ose * init_gradient_from_ose(spo, alpha=alpha)

    return gradient_spo

# ---------------------------------------------------------------------- #

def conjugated_pauli_forward(spo, xzk, theta, trunc_val, max_num_str):
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

    slice_size = min(int(new_size), max_num_str, x_concat.shape[0])
    final_valid_count = min(int(final_valid_count), slice_size)

    x_ = slice_to_size_x_arr(x_concat, slice_size)
    c_ = slice_to_size_c_arr(c_concat, slice_size)
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

def conjugated_pauli_backward(spo_val_grad, xzk, theta, trunc_val, max_num_str):
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

    x_concat, c_concat, grad_c_concat, new_size, final_valid_count = backward_jitted(
        spo_val_grad, xzk, theta, trunc_val
    )
    slice_size = min(int(new_size), max_num_str, x_concat.shape[0])
    final_valid_count = min(int(final_valid_count), slice_size)

    x_ = slice_to_size_x_arr(x_concat, slice_size)
    c_ = slice_to_size_c_arr(c_concat, slice_size)
    grad_c_ = slice_to_size_c_arr(grad_c_concat, slice_size)
    new_spo_val_grad = SparsePauliGradientOp(x_, c_, grad_c_)

    return new_spo_val_grad, final_valid_count, grad_i

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


    #- # [jittable]        sub-routine2. Get conjugated xz_array_q, phase_array; copy c_array_q, grad_c_array_q
    #- xz_array_q, phase_array_p = pauli_product_batched_second_uint(xzk, 1.,
    #-                                                               xz_array_p, jnp.ones_like(c_array_p),)
    #- # phase_array is an indication of the relation between sigma and p
    #- phase_array_p = jnp.real(phase_array_p * (-1j))

    # [jittable]        sub-routine2. Get conjugated xz_array_q, sign_array_p; copy c_array_q, grad_c_array_q
    xz_array_q, sign_array_p = pauli_product_phase_sign_second_uint(xzk, xz_array_p)
    # Here sign_array_p = phase_array_p * (1j), still miss one minus sign for the formula, so we negate it here.
    sign_array_p = -sign_array_p
    # This is the c_array_p_of_the_corresponding_q from (sigma, p, q). It is not the coefficient of q itself.
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
    # The q_aligned now points to the corresponding q for each p, if exists. If not exists, it points to the last row, which will be filtered out.
    c_array_q_aligned = c_array_q_sorted[safe_indices]
    grad_c_array_q_aligned = grad_c_array_q_sorted[safe_indices]

    # Use the formula
    raw_products = sign_array_p * ( -c_array_p * grad_c_array_q_aligned + c_array_q_aligned * grad_c_array_p )
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


# ---------------------------------------------------------------------------
# ---------- Pauli Products (JAX) ----------
# ---------------------------------------------------------------------------

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

@jax.jit
def pauli_product_phase_sign_second_uint(xz1, xz2_array):
    """
    Return the product Pauli string together with the real sign used by
    conjugation updates.

    Pauli multiplication may produce phases in {1, -i, -1, i}. For
    anti-commuting terms, the conjugation formula multiplies that phase by an
    extra i, so the effective coefficient update is always real (+1 or -1).
    """
    N = xz2_array.shape[1] // 2
    xz_new_array = xz1 ^ xz2_array
    count = jnp.sum((2 * jax.lax.population_count(xz1[:N] & xz2_array[:, N:]) +
                     jax.lax.population_count(xz1[:N] & xz1[N:]) +
                     jax.lax.population_count(xz2_array[:, :N] & xz2_array[:, N:]) -
                     jax.lax.population_count(xz_new_array[:, :N] & xz_new_array[:, N:])),
                    axis=1) % 4
    sign = jnp.take(utils.CONJUGATION_SIGNS, count)
    return xz_new_array, sign

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
    phase = jnp.take(utils.PHASES, count)   # vectorized lookup
    c_new_array = c1 * c2_array * phase
    return xz_new_array, c_new_array

# ---------------------------------------------------------------------------
# ---------- Conjugate Pauli String (JAX) ----------
# ---------------------------------------------------------------------------

# @jax.jit
def conjugated_pauli_batched_uint_(spo, xzk, theta):
    """
    [Support uint8, uint16, uint32, uint64]
    Conjugate a batch of Pauli strings in packed uint form by rotation R_k(theta):
    exp(i theta/2 * sigma_k) * sigma_j * exp(-i theta/2 * sigma_k)

    Parameters:
        xz_array: uint arrays of shape (M, nbytes) - M Pauli strings on N qubits
        c_array: real array of shape (M,) - coefficients
        xzk: uint arrays of shape (nbytes,) - Pauli string for rotation
        theta: float scalar - rotation angle
    Returns:
        xz_array: uint array of shape (M, nbytes) - unchanged Pauli strings
        c_array_1: real array of shape (M,) - coefficients for sigma_j
        xz_array_2: uint array of shape (M, nbytes) - Pauli strings for sigma_k sigma_j
        c_array_2: real array of shape (M,) - coefficients for sigma_k sigma_j
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
# -    xz_array_2, phase_array = pauli_product_batched_second_uint(xzk, 1.,
# -                                                                xz_array, jnp.ones_like(c_array),)
# -    c_array_2 = 1j * c_array * jnp.sin(theta) * phase_array
# -    grad_c_array_2 = 1j * grad_c_array * jnp.sin(theta) * phase_array

    # sign_array = phase_array * (1j)
    xz_array_2, sign_array = pauli_product_phase_sign_second_uint(xzk, xz_array)
    c_array_2 = c_array * jnp.sin(theta) * sign_array

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

    # [old convention]
    # xz_array_2, phase_array = pauli_product_batched_second_uint(xzk, 1.,
    #                                                             xz_array, jnp.ones_like(c_array),)
    # c_array_2 = 1j * c_array * jnp.sin(theta) * phase_array
    # grad_c_array_2 = 1j * grad_c_array * jnp.sin(theta) * phase_array
    xz_array_2, sign_array = pauli_product_phase_sign_second_uint(xzk, xz_array)
    c_array_2 = c_array * jnp.sin(theta) * sign_array
    grad_c_array_2 = grad_c_array * jnp.sin(theta) * sign_array

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





# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

@jax.jit
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

@jax.jit
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


def next_pow2(x):
    """Return next power of 2 >= x (x is a JAX scalar)."""
    x = jnp.maximum(x, 1)
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

@jax.jit
def find_row_duplications(a, b):
    """
    Checks which rows of 'a' exist in 'b' using an optimized binary search.

    This function performs a vectorized, lexically-aware binary search. It is
    designed for JAX JIT-compilation by using a fixed number of iterations
    (based on the log2 size of 'b') to avoid dynamic branching on the GPU.

    # Old Doc-String - Optimized Binary Search for Static Shapes.
    # Replaces while_loop with a fixed unrolled loop for GPU efficiency.

    Args:
        a: jax.Array of shape (N, D). The rows to search for (needles).
        b: jax.Array of shape (M, D). The search space (haystack).
           MUST be lexically sorted.

    Returns:
        is_duplicate: A boolean array of shape (N,) where True indicates the
            row in 'a' exists in 'b'.
        indices_in_b: An int32 array of shape (N,) containing the indices in 'b'
            where the rows of 'a' are located (or where they would be inserted).

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
