"""JAX search/update/merge algorithm with lexicographically sorted storage."""

from functools import partial

import jax
import jax.numpy as jnp

from .. import kernels
from ..sparse_pauli import SparsePauliGradientOp, SparsePauliOp


@jax.jit
def _step_info_values_from_tail(c_concat, slice_size):
    magnitudes = jnp.abs(c_concat)
    indices = jnp.arange(c_concat.shape[0])
    removed_mask = indices >= slice_size
    removed_coeffs = jnp.where(removed_mask, magnitudes, 0.0)
    return (
        jnp.sum(removed_mask),
        jnp.sum(removed_coeffs),
        jnp.sqrt(jnp.sum(removed_coeffs ** 2)),
    )


def _step_info_from_tail(c_concat, slice_size):
    num_str_truncated, truncated_l1_norm, truncated_l2_norm = _step_info_values_from_tail(
        c_concat, slice_size
    )
    return {
        "num_str_truncated": int(num_str_truncated),
        "truncated_l1_norm": float(truncated_l1_norm),
        "truncated_l2_norm": float(truncated_l2_norm),
    }


@jax.jit
def _step_info_values_from_removed(c_concat, trunc_val, slice_size):
    magnitudes = jnp.abs(c_concat)
    indices = jnp.arange(c_concat.shape[0])
    removed_mask = ((magnitudes <= trunc_val) | (indices >= slice_size)) & (magnitudes > 0)
    removed_coeffs = jnp.where(removed_mask, magnitudes, 0.0)
    return (
        jnp.sum(removed_mask),
        jnp.sum(removed_coeffs),
        jnp.sqrt(jnp.sum(removed_coeffs ** 2)),
    )


def _step_info_from_removed(c_concat, trunc_val, slice_size):
    num_str_truncated, truncated_l1_norm, truncated_l2_norm = _step_info_values_from_removed(
        c_concat, trunc_val, slice_size
    )
    return {
        "num_str_truncated": int(num_str_truncated),
        "truncated_l1_norm": float(truncated_l1_norm),
        "truncated_l2_norm": float(truncated_l2_norm),
    }


def _apply_hard_cutoff(c_array, trunc_val):
    keep_mask = jnp.abs(c_array) > trunc_val
    return jnp.where(keep_mask, c_array, 0.0), keep_mask


def _step_info_dict(num_str_truncated, truncated_l1_norm, truncated_l2_norm):
    return {
        "num_str_truncated": int(num_str_truncated),
        "truncated_l1_norm": float(truncated_l1_norm),
        "truncated_l2_norm": float(truncated_l2_norm),
    }


def _top_k_step_info_values(magnitudes, final_keep_mask):
    magnitudes, final_keep_mask = jax.lax.optimization_barrier(
        (magnitudes, final_keep_mask)
    )
    removed_mask = (magnitudes > 0) & (~final_keep_mask)
    count_values = removed_mask.astype(jnp.int32)
    l1_values = jnp.where(removed_mask, magnitudes, 0.0)
    l2_values = l1_values * l1_values

    def add(x, y):
        return tuple(a + b for a, b in zip(x, y))

    count, l1_norm, l2_square = jax.lax.reduce(
        (count_values, l1_values, l2_values),
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=magnitudes.dtype),
            jnp.asarray(0.0, dtype=magnitudes.dtype),
        ),
        add,
        dimensions=(0,),
    )
    return count, l1_norm, jnp.sqrt(l2_square)


def forward_step(spo, xzk, theta, trunc_val, max_num_str):
    """
    Conjugate a sparse-Pauli operator using the lexicographic search/update path.
    """
    (
        x_concat,
        c_concat,
        new_size,
        final_valid_count,
        num_str_truncated,
        truncated_l1_norm,
        truncated_l2_norm,
    ) = forward_search_update_merge_top_k_jitted(
        spo,
        xzk,
        theta,
        trunc_val,
        max_num_str,
    )
    jax.block_until_ready(new_size)

    slice_size = min(int(new_size), max_num_str, x_concat.shape[0])
    x_ = kernels.slice_to_size_x_arr(x_concat, slice_size)
    c_ = kernels.slice_to_size_c_arr(c_concat, slice_size)
    jax.block_until_ready(c_)

    new_spo = SparsePauliOp(x_, c_, lexsorted=True)
    return (
        new_spo,
        min(int(final_valid_count), slice_size),
        _step_info_dict(num_str_truncated, truncated_l1_norm, truncated_l2_norm),
    )


@jax.jit
def forward_search_update_merge_jitted(spo, xzk, theta, trunc_val):
    """
    Parameters:
        spo: SparsePauliOp
        xzk: uint arrays of shape (nbytes,) - Pauli string for rotation
        theta: float scalar - rotation angle
        trunc_val: float scalar - truncation value for coefficients

    Returns:
        x_concat
        c_concat
        new_size
        final_valid_count
    """
    print("Recompile: forward_search_update_merge_jitted", spo.xz_array.shape,)
    xz_array = spo.xz_array
    c_array = spo.c_array
    # xz_array has to be sorted for the search-update-merge algorithm to work correctly.


    # 1. Compute Commutation Mask
    # ---------------------------------------------------
    # mask_anti_commute = True if {P, sigma} = 0 (Anti-commute)
    # mask_anti_commute = False if [P, sigma] = 0 (Commute)
    N = xz_array.shape[1] // 2
    comm_val = jnp.sum(jax.lax.population_count(xz_array[:, N:] & xzk[:N]), axis=1) - \
               jnp.sum(jax.lax.population_count(xz_array[:, :N] & xzk[N:]), axis=1)

    # Check parity. Odd = Anti-commute.
    mask_anti_commute = (comm_val % 2).astype(bool)


    # 2. Compute Conjugated Paulis (Q candidates)
    # ---------------------------------------------------
    # We compute this for ALL rows.
    # For commuting rows (R, S), this result is garbage/irrelevant,
    # but vectorization is faster than branching.

    # Returns conj_P such that: sigma * P = phase * conj_P
    # Definition: phase = i means "P-type", -i means "Q-type"
    # Taking phase * (1j) = sign
    # Definition: sign = -1 means "P-type", sign = +1 means "Q-type"
    xz_array_conj, sign_array = kernels.pauli_product_phase_sign_second_uint(xzk, xz_array,)


    # 3. Search for Partners in Existing Array
    # ---------------------------------------------------
    # haystack = xz_array (Sorted)
    # needle = xz_array_conj (Not sorted, preserves index mapping)

    is_duplicate, indices_in_existing = kernels.find_row_duplications(xz_array_conj, xz_array)


    # 4. Prepare Rotation Values
    # ---------------------------------------------------
    # We handle the rotation for every row 'i'.
    # coefficient self = c_array[i]
    # coefficient pair = c_array[indices_in_existing[i]] (if exists)

    # Get value of the partner (Q if we are P, P if we are Q)
    # If partner doesn't exist (is_duplicate=False), value is 0.0
    val_self = c_array

    # Safe gather: clamp index to valid range, then mask result
    safe_indices = jnp.minimum(indices_in_existing, xz_array.shape[0]-1)
    val_pair_raw = c_array[safe_indices]
    val_pair = jnp.where(is_duplicate, val_pair_raw, 0.0)

    # Effective Theta:
    # If commuting, theta = 0. (cos=1, sin=0) -> No change.
    # If anti-commuting, theta = theta.
    theta_eff = jnp.where(mask_anti_commute, theta, 0.0)

    cos_t = jnp.cos(theta_eff)
    sin_t = jnp.sin(theta_eff)


    # 5. Update Existing Coefficients (In-Place)
    # ---------------------------------------------------
    # Formula derivation:
    # a_P^{t+1} = a_P^{t} cos + a_Q^{t} sin  (Assuming sigma*P = iQ)
    # The coefficient update rule is:
    # a_self_new = a_self * cos - a_pair * sin * sign
    # (The sign depends on whether we are P or Q, handled by sign_array)

    # Note: If partner doesn't exist (val_pair=0), this correctly reduces to:
    # a_self * cos

    c_array_updated = val_self * cos_t - val_pair * sin_t * sign_array


    # 6. Handle New Terms (Insertions)
    # ---------------------------------------------------
    # We need to insert a new term Q if:
    # 1. We are anti-commuting (mask_anti_commute)
    # 2. Our partner is NOT in the list (~is_duplicate)

    mask_insert = mask_anti_commute & (~is_duplicate)

    # The Pauli string to insert is xz_array_conj
    new_xz = jnp.where(mask_insert[:, None], xz_array_conj, kernels.PAD_VAL)

    # The coefficient to insert:
    # Contribution from P to Q is: -a_P * sin(t)
    # Be careful with signs.
    # If we are P (sign=-1): we generate Q with coeff -a_P * sin
    # If we are Q (sign=+1): we generate P with coeff +a_Q * sin
    # Formula: new_coeff = val_self * sin_t * sign_array
    new_c_val = val_self * sin_t * sign_array
    new_c = jnp.where(mask_insert, new_c_val, 0.0)

    # 7. Merge, Filter, Sort
    # ---------------------------------------------------
    # Combine old updated array and new candidates
    merged_xz = jnp.concatenate([xz_array, new_xz], axis=0)
    merged_c  = jnp.concatenate([c_array_updated, new_c], axis=0)

    # Apply Truncation (Sparsity)
    # Set effectively zero coefficients to kernels.PAD_VAL/0.0 so they sort to the bottom
    # We also filter out kernels.PAD_VAL rows that might have been carried over
    is_large_enough = jnp.abs(merged_c) > trunc_val
    # is_valid_row = (merged_xz[:, 0] != 255) # Assuming 255/0xFF is PAD byte

    keep_mask = is_large_enough # & is_valid_row

    final_xz_masked = jnp.where(keep_mask[:, None], merged_xz, kernels.PAD_VAL)
    ## [Logic change] [Need to double check]
    ## We don't strictly need to zero out the coefficients.
    ## We are doing a lexsort based on xz and the PAD_VAL rows will sort to the bottom.
    ## Keeping the coefficients let us track the truncation correctly in the step info.
    # final_c_masked  = jnp.where(keep_mask, merged_c, 0.0)
    final_c_masked = merged_c


    final_valid_count = jnp.sum(keep_mask.astype(jnp.int32))
    new_size = kernels.next_pow2(final_valid_count)

    # Sort
    # kernels.PAD_VAL (highest int) will naturally go to the bottom
    sort_indices = jnp.lexsort(final_xz_masked.T[::-1])

    sorted_xz = final_xz_masked[sort_indices]
    sorted_c = final_c_masked[sort_indices]

    return sorted_xz, sorted_c, new_size, final_valid_count


@partial(jax.jit, static_argnames=("max_num_str",))
def forward_search_update_merge_top_k_jitted(spo, xzk, theta, trunc_val, max_num_str):
    """
    Forward search/update with cap truncation by largest coefficient magnitude.

    The output shapes match forward_search_update_merge_jitted. Rows not kept by
    either the coefficient threshold or the max_num_str top-k cap are PAD/zeroed
    before the final lexsort, and the removed coefficient magnitudes are returned
    for step-info accounting.
    """
    print("Recompile: forward_search_update_merge_top_k_jitted", spo.xz_array.shape,)
    xz_array = spo.xz_array
    c_array = spo.c_array

    N = xz_array.shape[1] // 2
    comm_val = jnp.sum(jax.lax.population_count(xz_array[:, N:] & xzk[:N]), axis=1) - \
               jnp.sum(jax.lax.population_count(xz_array[:, :N] & xzk[N:]), axis=1)
    mask_anti_commute = (comm_val % 2).astype(bool)

    xz_array_conj, sign_array = kernels.pauli_product_phase_sign_second_uint(xzk, xz_array,)
    is_duplicate, indices_in_existing = kernels.find_row_duplications(xz_array_conj, xz_array)

    val_self = c_array
    safe_indices = jnp.minimum(indices_in_existing, xz_array.shape[0]-1)
    val_pair_raw = c_array[safe_indices]
    val_pair = jnp.where(is_duplicate, val_pair_raw, 0.0)

    theta_eff = jnp.where(mask_anti_commute, theta, 0.0)
    cos_t = jnp.cos(theta_eff)
    sin_t = jnp.sin(theta_eff)

    c_array_updated = val_self * cos_t - val_pair * sin_t * sign_array

    mask_insert = mask_anti_commute & (~is_duplicate)
    new_xz = jnp.where(mask_insert[:, None], xz_array_conj, kernels.PAD_VAL)
    new_c_val = val_self * sin_t * sign_array
    new_c = jnp.where(mask_insert, new_c_val, 0.0)

    merged_xz = jnp.concatenate([xz_array, new_xz], axis=0)
    merged_c = jnp.concatenate([c_array_updated, new_c], axis=0)

    magnitudes = jnp.abs(merged_c)
    live_mask = magnitudes > trunc_val
    scores = jnp.where(live_mask, magnitudes, -jnp.inf)
    k = min(int(max_num_str), merged_c.shape[0])
    _, top_indices = jax.lax.top_k(scores, k)
    selected_mask = jnp.zeros(merged_c.shape[0], dtype=bool).at[top_indices].set(True)
    final_keep_mask = live_mask & selected_mask

    final_xz_masked = jnp.where(final_keep_mask[:, None], merged_xz, kernels.PAD_VAL)
    final_c_masked = jnp.where(final_keep_mask, merged_c, 0.0)

    final_valid_count = jnp.sum(final_keep_mask.astype(jnp.int32))
    num_str_truncated, truncated_l1_norm, truncated_l2_norm = _top_k_step_info_values(
        magnitudes,
        final_keep_mask,
    )
    new_size = kernels.next_pow2(final_valid_count)

    sort_indices = jnp.lexsort(final_xz_masked.T[::-1])
    sorted_xz = final_xz_masked[sort_indices]
    sorted_c = final_c_masked[sort_indices]

    return (
        sorted_xz,
        sorted_c,
        new_size,
        final_valid_count,
        num_str_truncated,
        truncated_l1_norm,
        truncated_l2_norm,
    )


def backward_step(spo_val_grad, xzk, theta, trunc_val, max_num_str):
    """
    Backward conjugation for the lexicographic search/update path.

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
    (
        x_concat,
        c_concat,
        grad_c_concat,
        new_size,
        final_valid_count,
        grad_i,
        num_str_truncated,
        truncated_l1_norm,
        truncated_l2_norm,
    ) = (
        backward_search_update_merge_top_k_jitted(
            spo_val_grad, xzk, theta, trunc_val, max_num_str
        )
    )
    jax.block_until_ready(new_size)

    slice_size = min(int(new_size), max_num_str, x_concat.shape[0])

    x_ = kernels.slice_to_size_x_arr(x_concat, slice_size)
    c_ = kernels.slice_to_size_c_arr(c_concat, slice_size)
    grad_c_ = kernels.slice_to_size_c_arr(grad_c_concat, slice_size)
    jax.block_until_ready(grad_c_)

    new_spo_val_grad = SparsePauliGradientOp(x_, c_, grad_c_, lexsorted=True)
    return (
        new_spo_val_grad,
        min(int(final_valid_count), slice_size),
        grad_i,
        _step_info_dict(num_str_truncated, truncated_l1_norm, truncated_l2_norm),
    )


@jax.jit
def backward_search_update_merge_jitted(spo_val_grad, xzk, theta, trunc_val):
    """
    Return the lexicographically sorted backward-update result before Python-side
    slicing, together with the fused parameter gradient.

    Parameters:
        spo_val_grad: SparsePauliGradientOp
        xzk: uint arrays of shape (nbytes,) - Pauli string for rotation
        theta: float scalar - rotation angle
        trunc_val: float scalar - truncation value for coefficients

    Returns:
        sorted_xz
        sorted_c
        sorted_grad_c
        new_size
        final_valid_count
        grad_i
    """
    print("Recompile: backward_search_update_merge_jitted", spo_val_grad.xz_array.shape,)
    xz_array = spo_val_grad.xz_array
    c_array = spo_val_grad.c_array
    grad_c_array = spo_val_grad.grad_c_array

    # 1. Compute Commutation Mask
    N = xz_array.shape[1] // 2
    comm_val = jnp.sum(jax.lax.population_count(xz_array[:, N:] & xzk[:N]), axis=1) - \
               jnp.sum(jax.lax.population_count(xz_array[:, :N] & xzk[N:]), axis=1)
    mask_anti_commute = (comm_val % 2).astype(bool)

    # 2. Compute Conjugated Paulis (Q candidates)
    xz_array_conj, sign_array = kernels.pauli_product_phase_sign_second_uint(xzk, xz_array)

    # 3. Search for Partners in Existing Array
    is_duplicate, indices_in_existing = kernels.find_row_duplications(xz_array_conj, xz_array)
    safe_indices = jnp.minimum(indices_in_existing, xz_array.shape[0] - 1)

    # 4. Gather Values and Gradients
    val_self = c_array
    grad_self = grad_c_array
    val_pair_raw = c_array[safe_indices]
    grad_pair_raw = grad_c_array[safe_indices]
    val_pair = jnp.where(is_duplicate, val_pair_raw, 0.0)
    grad_pair = jnp.where(is_duplicate, grad_pair_raw, 0.0)

    # 5. Fused gradient computation from discovered PQ pairs
    # grad_theta = \sum_i - a_pi \partial a_qi + a_qi \partial a_pi
    # Recall P sign is -1, Q sign is +1, so we can unify the formula with sign_array:
    grad_sign = -sign_array
    raw_grad_products = grad_sign * (-val_self * grad_pair + val_pair * grad_self)
    valid_grad_products = jnp.where(mask_anti_commute & is_duplicate, raw_grad_products, 0.0)
    grad_i = jnp.real(jnp.sum(valid_grad_products) / 2.0)

    # 6. Backward rotation uses the negative angle on anti-commuting rows
    theta_eff = jnp.where(mask_anti_commute, -theta, 0.0)
    cos_t = jnp.cos(theta_eff)
    sin_t = jnp.sin(theta_eff)

    # 7. Update existing rows
    c_array_updated = val_self * cos_t - val_pair * sin_t * sign_array
    grad_c_array_updated = grad_self * cos_t - grad_pair * sin_t * sign_array

    # 8. Prepare insertions for missing partners
    mask_insert = mask_anti_commute & (~is_duplicate)
    new_xz = jnp.where(mask_insert[:, None], xz_array_conj, kernels.PAD_VAL)
    new_c_val = val_self * sin_t * sign_array
    new_grad_c_val = grad_self * sin_t * sign_array
    new_c = jnp.where(mask_insert, new_c_val, 0.0)
    new_grad_c = jnp.where(mask_insert, new_grad_c_val, 0.0)

    # 9. Merge, filter by coefficient magnitude, and sort lexicographically
    merged_xz = jnp.concatenate([xz_array, new_xz], axis=0)
    merged_c = jnp.concatenate([c_array_updated, new_c], axis=0)
    merged_grad_c = jnp.concatenate([grad_c_array_updated, new_grad_c], axis=0)

    keep_mask = jnp.abs(merged_c) > trunc_val
    final_xz_masked = jnp.where(keep_mask[:, None], merged_xz, kernels.PAD_VAL)

    ## We lexsort the XZ. The PAD_VAL in rows still sort to the bottom.
    ## We don't mask the c_array.
    ## Keep the discarded coefficient magnitudes in the tail so step-info can
    ## read them after sorting.
    # final_c_masked = jnp.where(keep_mask, merged_c, 0.0)
    # final_grad_c_masked = jnp.where(keep_mask, merged_grad_c, 0.0)

    final_valid_count = jnp.sum(keep_mask.astype(jnp.int32))
    new_size = kernels.next_pow2(final_valid_count)

    sort_indices = jnp.lexsort(final_xz_masked.T[::-1])
    sorted_xz = final_xz_masked[sort_indices]
    sorted_c = merged_c[sort_indices]
    sorted_grad_c = merged_grad_c[sort_indices]

    return sorted_xz, sorted_c, sorted_grad_c, new_size, final_valid_count, grad_i


@partial(jax.jit, static_argnames=("max_num_str",))
def backward_search_update_merge_top_k_jitted(spo_val_grad, xzk, theta, trunc_val, max_num_str):
    """
    Backward search/update with cap truncation by largest coefficient magnitude.

    The selected top-k mask is applied consistently to xz, c, and grad_c before
    lexsorting, and removed coefficient magnitudes are returned for step info.
    """
    print("Recompile: backward_search_update_merge_top_k_jitted", spo_val_grad.xz_array.shape,)
    xz_array = spo_val_grad.xz_array
    c_array = spo_val_grad.c_array
    grad_c_array = spo_val_grad.grad_c_array

    N = xz_array.shape[1] // 2
    comm_val = jnp.sum(jax.lax.population_count(xz_array[:, N:] & xzk[:N]), axis=1) - \
               jnp.sum(jax.lax.population_count(xz_array[:, :N] & xzk[N:]), axis=1)
    mask_anti_commute = (comm_val % 2).astype(bool)

    xz_array_conj, sign_array = kernels.pauli_product_phase_sign_second_uint(xzk, xz_array)
    is_duplicate, indices_in_existing = kernels.find_row_duplications(xz_array_conj, xz_array)
    safe_indices = jnp.minimum(indices_in_existing, xz_array.shape[0] - 1)

    val_self = c_array
    grad_self = grad_c_array
    val_pair_raw = c_array[safe_indices]
    grad_pair_raw = grad_c_array[safe_indices]
    val_pair = jnp.where(is_duplicate, val_pair_raw, 0.0)
    grad_pair = jnp.where(is_duplicate, grad_pair_raw, 0.0)

    grad_sign = -sign_array
    raw_grad_products = grad_sign * (-val_self * grad_pair + val_pair * grad_self)
    valid_grad_products = jnp.where(mask_anti_commute & is_duplicate, raw_grad_products, 0.0)
    grad_i = jnp.real(jnp.sum(valid_grad_products) / 2.0)

    theta_eff = jnp.where(mask_anti_commute, -theta, 0.0)
    cos_t = jnp.cos(theta_eff)
    sin_t = jnp.sin(theta_eff)

    c_array_updated = val_self * cos_t - val_pair * sin_t * sign_array
    grad_c_array_updated = grad_self * cos_t - grad_pair * sin_t * sign_array

    mask_insert = mask_anti_commute & (~is_duplicate)
    new_xz = jnp.where(mask_insert[:, None], xz_array_conj, kernels.PAD_VAL)
    new_c_val = val_self * sin_t * sign_array
    new_grad_c_val = grad_self * sin_t * sign_array
    new_c = jnp.where(mask_insert, new_c_val, 0.0)
    new_grad_c = jnp.where(mask_insert, new_grad_c_val, 0.0)

    merged_xz = jnp.concatenate([xz_array, new_xz], axis=0)
    merged_c = jnp.concatenate([c_array_updated, new_c], axis=0)
    merged_grad_c = jnp.concatenate([grad_c_array_updated, new_grad_c], axis=0)

    magnitudes = jnp.abs(merged_c)
    live_mask = magnitudes > trunc_val
    scores = jnp.where(live_mask, magnitudes, -jnp.inf)
    k = min(int(max_num_str), merged_c.shape[0])
    _, top_indices = jax.lax.top_k(scores, k)
    selected_mask = jnp.zeros(merged_c.shape[0], dtype=bool).at[top_indices].set(True)
    final_keep_mask = live_mask & selected_mask

    final_xz_masked = jnp.where(final_keep_mask[:, None], merged_xz, kernels.PAD_VAL)
    final_c_masked = jnp.where(final_keep_mask, merged_c, 0.0)
    final_grad_c_masked = jnp.where(final_keep_mask, merged_grad_c, 0.0)

    final_valid_count = jnp.sum(final_keep_mask.astype(jnp.int32))
    num_str_truncated, truncated_l1_norm, truncated_l2_norm = _top_k_step_info_values(
        magnitudes,
        final_keep_mask,
    )
    new_size = kernels.next_pow2(final_valid_count)

    sort_indices = jnp.lexsort(final_xz_masked.T[::-1])
    sorted_xz = final_xz_masked[sort_indices]
    sorted_c = final_c_masked[sort_indices]
    sorted_grad_c = final_grad_c_masked[sort_indices]

    return (
        sorted_xz,
        sorted_c,
        sorted_grad_c,
        new_size,
        final_valid_count,
        grad_i,
        num_str_truncated,
        truncated_l1_norm,
        truncated_l2_norm,
    )
