"""Experimental donated full-step wrapper for search_update_merge."""

from functools import partial

import jax
import jax.numpy as jnp

from .. import kernels
from ..sparse_pauli import SparsePauliGradientOp, SparsePauliOp
from . import search_update_merge


def _step_info_from_scalars(num_str_truncated, truncated_l1_norm, truncated_l2_norm):
    return {
        "num_str_truncated": int(num_str_truncated),
        "truncated_l1_norm": float(truncated_l1_norm),
        "truncated_l2_norm": float(truncated_l2_norm),
    }


def _removed_stats(c_concat, trunc_val, slice_size):
    magnitudes = jnp.abs(c_concat)
    indices = jnp.arange(c_concat.shape[0])
    removed_mask = ((magnitudes <= trunc_val) | (indices >= slice_size)) & (magnitudes > 0)
    removed_coeffs = jnp.where(removed_mask, magnitudes, 0.0)
    return (
        jnp.sum(removed_mask),
        jnp.sum(removed_coeffs),
        jnp.sqrt(jnp.sum(removed_coeffs ** 2)),
    )


def _apply_hard_cutoff(c_array, trunc_val):
    keep_mask = jnp.abs(c_array) > trunc_val
    return jnp.where(keep_mask, c_array, 0.0), keep_mask


def _use_donated_step(state_size, max_num_str):
    return state_size == max_num_str


def forward_step(spo, xzk, theta, trunc_val, max_num_str):
    """Conjugate with donation once storage has reached the max size."""
    if not _use_donated_step(spo.get_size(), max_num_str):
        return search_update_merge.forward_step(spo, xzk, theta, trunc_val, max_num_str)

    new_spo, final_valid_count, new_size, num_truncated, tail_l1, tail_l2 = (
        forward_fullstep_donate_jitted(spo, xzk, theta, trunc_val, max_num_str)
    )
    jax.block_until_ready(new_size)

    slice_size = min(int(new_size), max_num_str, new_spo.get_size())
    if slice_size < new_spo.get_size():
        x_ = kernels.slice_to_size_x_arr(new_spo.xz_array, slice_size)
        c_ = kernels.slice_to_size_c_arr(new_spo.c_array, slice_size)
        jax.block_until_ready(c_)
        new_spo = SparsePauliOp(x_, c_, lexsorted=True)
    else:
        jax.block_until_ready(new_spo.c_array)

    return (
        new_spo,
        min(int(final_valid_count), slice_size),
        _step_info_from_scalars(num_truncated, tail_l1, tail_l2),
    )


@partial(jax.jit, donate_argnums=(0,), static_argnums=(4,))
def forward_fullstep_donate_jitted(spo, xzk, theta, trunc_val, max_num_str):
    x_concat, c_concat, new_size, final_valid_count = (
        search_update_merge.forward_search_update_merge_jitted(
            spo,
            xzk,
            theta,
            trunc_val,
        )
    )
    slice_size = jnp.minimum(new_size, max_num_str)
    num_truncated, tail_l1, tail_l2 = _removed_stats(c_concat, trunc_val, slice_size)
    x_out = kernels.slice_to_size_x_arr(x_concat, max_num_str)
    c_out = kernels.slice_to_size_c_arr(c_concat, max_num_str)
    c_out, _ = _apply_hard_cutoff(c_out, trunc_val)
    return (
        SparsePauliOp(x_out, c_out, lexsorted=True),
        final_valid_count,
        new_size,
        num_truncated,
        tail_l1,
        tail_l2,
    )


def backward_step(spo_val_grad, xzk, theta, trunc_val, max_num_str):
    """Backward conjugation with donation once storage has reached the max size."""
    if not _use_donated_step(spo_val_grad.get_size(), max_num_str):
        return search_update_merge.backward_step(
            spo_val_grad,
            xzk,
            theta,
            trunc_val,
            max_num_str,
        )

    new_spo_val_grad, final_valid_count, grad_i, new_size, num_truncated, tail_l1, tail_l2 = (
        backward_fullstep_donate_jitted(spo_val_grad, xzk, theta, trunc_val, max_num_str)
    )
    jax.block_until_ready(new_size)

    slice_size = min(int(new_size), max_num_str, new_spo_val_grad.get_size())
    if slice_size < new_spo_val_grad.get_size():
        x_ = kernels.slice_to_size_x_arr(new_spo_val_grad.xz_array, slice_size)
        c_ = kernels.slice_to_size_c_arr(new_spo_val_grad.c_array, slice_size)
        grad_c_ = kernels.slice_to_size_c_arr(new_spo_val_grad.grad_c_array, slice_size)
        jax.block_until_ready(grad_c_)
        new_spo_val_grad = SparsePauliGradientOp(x_, c_, grad_c_, lexsorted=True)
    else:
        jax.block_until_ready(new_spo_val_grad.grad_c_array)

    return (
        new_spo_val_grad,
        min(int(final_valid_count), slice_size),
        grad_i,
        _step_info_from_scalars(num_truncated, tail_l1, tail_l2),
    )


@partial(jax.jit, donate_argnums=(0,), static_argnums=(4,))
def backward_fullstep_donate_jitted(spo_val_grad, xzk, theta, trunc_val, max_num_str):
    x_concat, c_concat, grad_c_concat, new_size, final_valid_count, grad_i = (
        search_update_merge.backward_search_update_merge_jitted(
            spo_val_grad,
            xzk,
            theta,
            trunc_val,
        )
    )
    slice_size = jnp.minimum(new_size, max_num_str)
    num_truncated, tail_l1, tail_l2 = _removed_stats(c_concat, trunc_val, slice_size)
    x_out = kernels.slice_to_size_x_arr(x_concat, max_num_str)
    c_out = kernels.slice_to_size_c_arr(c_concat, max_num_str)
    grad_c_out = kernels.slice_to_size_c_arr(grad_c_concat, max_num_str)
    c_out, keep_mask = _apply_hard_cutoff(c_out, trunc_val)
    grad_c_out = jnp.where(keep_mask, grad_c_out, 0.0)
    return (
        SparsePauliGradientOp(x_out, c_out, grad_c_out, lexsorted=True),
        final_valid_count,
        grad_i,
        new_size,
        num_truncated,
        tail_l1,
        tail_l2,
    )
