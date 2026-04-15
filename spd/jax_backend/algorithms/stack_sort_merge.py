"""Wrapper for the current JAX stack/sort/merge algorithm."""

import jax
import jax.numpy as jnp
# import time

from .. import kernels
from ..sparse_pauli import SparsePauliGradientOp, SparsePauliOp


def forward_step(spo, xzk, theta, trunc_val, max_num_str):
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
    # t0 = time.time()
    x_concat, c_concat, new_size, final_valid_count = forward_stack_sort_merge_jitted(spo, xzk, theta, trunc_val)
    jax.block_until_ready(new_size)
    # t1 = time.time()

    slice_size = min(int(new_size), max_num_str, x_concat.shape[0])
    final_valid_count = min(int(final_valid_count), slice_size)

    x_ = kernels.slice_to_size_x_arr(x_concat, slice_size)
    c_ = kernels.slice_to_size_c_arr(c_concat, slice_size)
    jax.block_until_ready(c_)
    # t2 = time.time()

    # print("Merge time:", (t1 - t0) * 1000, "ms, Pad time:", (t2 - t1) * 1000,
    #       "ms, Final size:", new_size, "Valid count:", final_valid_count,
    #       "Original size:", x_array_1.shape[0] + x_array_2.shape[0])
    new_spo = SparsePauliOp(x_, c_)
    return new_spo, final_valid_count

@jax.jit
def forward_stack_sort_merge_jitted(spo, xzk, theta, trunc_val):
    print("Recompile: forward_jitted", spo.xz_array.shape,)
    spo_1, spo_2 = kernels.conjugated_pauli_batched_uint_(spo, xzk, theta)

    x_concat, c_concat, final_valid_count = kernels.merge_(
        spo_1.xz_array, spo_1.c_array,
        spo_2.xz_array, spo_2.c_array,
        trunc_val)

    new_size = kernels.next_pow2(final_valid_count)
    return x_concat, c_concat, new_size, final_valid_count


def backward_step(spo_val_grad, xzk, theta, trunc_val, max_num_str):
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

    x_ = kernels.slice_to_size_x_arr(x_concat, slice_size)
    c_ = kernels.slice_to_size_c_arr(c_concat, slice_size)
    grad_c_ = kernels.slice_to_size_c_arr(grad_c_concat, slice_size)
    new_spo_val_grad = SparsePauliGradientOp(x_, c_, grad_c_)

    return new_spo_val_grad, final_valid_count, grad_i

@jax.jit
def backward_jitted(spo_val_grad, xzk, theta, trunc_val):
    print("Recompile: backward_jitted", spo_val_grad.xz_array.shape,)
    spo_val_grad_1, spo_val_grad_2 = kernels.conjugate_pauli_rot_backward_batched_uint_(spo_val_grad, xzk, theta)
    x_concat, c_concat, grad_c_concat, final_valid_count = kernels.merge_val_grad_(spo_val_grad_1, spo_val_grad_2, trunc_val)
    new_size = kernels.next_pow2(final_valid_count)
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

    # [jittable]        sub-routine2. Get conjugated xz_array_q, sign_array_p; copy c_array_q, grad_c_array_q
    xz_array_q, sign_array_p = kernels.pauli_product_phase_sign_second_uint(xzk, xz_array_p)
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
    is_duplicate, indices_in_q = kernels.find_row_duplications(xz_array_p, xz_array_q_sorted)

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
