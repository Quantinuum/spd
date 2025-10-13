import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import time
import functools

# Fixed configuration for N=192 (32*6 chunks)
CHUNK_SIZE = 32
# N_FIXED = 192
N_CHUNKS = 6


@jit
def merge_(x_array_1, c_array_1, x_array_2, c_array_2, trunc_val):
    """
    uint8 version
    """
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

@functools.partial(jit, static_argnums=2)
def slice_to_size(x_arr, c_arr, size):
    """Slice arrays to the given size."""
    print("recompiling slice_to_size...", x_arr.shape, size, type(size))
    x_ = jax.lax.dynamic_slice(x_arr, (0, 0), (size, x_arr.shape[1]))
    c_ = jax.lax.dynamic_slice(c_arr, (0,), (size,))
    return x_, c_


# @jit
def merge_and_pad(x_array_1, c_array_1, x_array_2, c_array_2, trunc_val):
    """
    Complete merge pipeline using all steps.
    Pads output to the closest fixed sizes in terms of power of 2.
    Also combine the trunction step
    This avoids recompilation both in the current and the downstream.
    """

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
    return x_, c_, final_valid_count

    # x_concat, c_concat, boundaries, group_ids = merge_(
    #     x_array_1, c_array_1, x_array_2, c_array_2)

    # valid_count = jnp.max(group_ids) + 1

    # # new_size = 1 << (int(valid_count) - 1).bit_length()   # next power of two
    # new_size = next_pow2(valid_count)
    # # assert new_size == next_pow2(valid_count)
    # pad_size = new_size - valid_count

    # x_ = jnp.pad(x_concat[boundaries], ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
    # c_ = jnp.pad(c_concat[:valid_count], (0, pad_size), mode='constant', constant_values=0.0)

    # mask = jnp.abs(c_) > trunc_val
    # x_ = jnp.where(mask[:, None], x_, 0)
    # c_ = jnp.where(mask, c_, 0.0)
    # return x_, c_

