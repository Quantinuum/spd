import time

import jax
import jax.numpy as jnp

from . import utils
from .kernels import DT_BOOL

# Legacy module: archived prototypes, benchmarks, and reference implementations
# that are not used by the current runtime path in `kernels.py`.


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
"""
1) Difference between the four conjugated_pauli_batched* variants
In spd/jax_backend/kernels.py:

conjugated_pauli_batched
    Input: unpacked/bool-like row arrays (xz_array, c_array, xzk, theta)
    Role: wrapper that pads to power-of-2, then calls conjugated_pauli_batched_
    Status: legacy wrapper path, not used by runtime

conjugated_pauli_batched_
    Input: unpacked/bool-like arrays
    Role: core jitted implementation for unpacked representation
    Uses pauli_product_phase_sign_first
    Status: legacy core for unpacked path, not used by runtime

conjugated_pauli_batched_uint
    Input: packed uint row arrays (xz_array, c_array, xzk, theta)
    Role: wrapper that pads, then calls conjugated_pauli_batched_uint_
    Status: legacy wrapper, not used by runtime

conjugated_pauli_batched_uint_
    Input: SparsePauliOp + packed xzk, theta
    Role: packed uint core split used by current runtime path
    Called by forward_jitted and therefore by conjugated_pauli_forward
    Status: keep (runtime-critical)

We don't work with bool-like and unpacked array anymore, so we move it here.
"""

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
        c_array: real array of shape (M,) - coefficients
        xzk: bool arrays of shape (2N,) - Pauli string for rotation
        theta: float scalar - rotation angle
    Returns:
        xz_array: bool array of shape (M, 2N) - unchanged Pauli strings
        c_array_1: real array of shape (M,) - coefficients for sigma_j
        xz_array_2: bool array of shape (M, 2N) - Pauli strings for sigma_k sigma_j
        c_array_2: real array of shape (M,) - coefficients for sigma_k sigma_j

    """
    print("Recompile: conjugated_pauli_batched", xz_array.shape, c_array.shape, xzk.shape, type(theta), theta, "\n")
    N = xz_array.shape[1] // 2
    # acq_val = jnp.sum(z_array & xk, axis=1) - jnp.sum(x_array & zk, axis=1)
    acq_val = jnp.sum(xz_array[:, N:] & xzk[:N], axis=1) - jnp.sum(xz_array[:, :N] & xzk[N:], axis=1)
    acq_val = acq_val % 2  # 0 = commute, 1 = anticommute
    theta = theta * acq_val

    c_array_1 = c_array * jnp.cos(theta)

    # -    xz_array_2, phase_array = pauli_product_batched(xz_array, jnp.ones_like(c_array),
    # -                                                    xzk, 1.)
    # -    c_array_2 = 1j * c_array * jnp.sin(theta) * phase_array
    # Keep the original batching order here: this path batches over xz_array
    # in the first argument, so its sign convention must match P * sigma_k.
    xz_array_2, sign_array = pauli_product_phase_sign_first(xz_array, xzk)
    c_array_2 = c_array * jnp.sin(theta) * sign_array

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

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------



"""
Functions present:

pauli_product (single, bool/unpacked): legacy, not used in runtime
pauli_product_uint (single, packed): useful for tests and debug; not runtime-critical
pauli_product_batched (bool batched): explicitly deprecated (NotImplementedError)
pauli_product_batched_first_uint: not used by runtime
pauli_product_batched_second_uint: not used by runtime, but used in tests / parity checks
pauli_product_phase_sign_first: only used by legacy conjugated_pauli_batched_
pauli_product_phase_sign_second_uint: used by runtime (conjugated_pauli_batched_uint_, backward variant)
"""


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
    phase = jnp.take(utils.PHASES, count)   # vectorized lookup
    c_new_array = c1_array * c2 * phase
    return xz_new_array, c_new_array

@jax.jit
def pauli_product_batched_first_uint(xz1_array, c1_array, xz2, c2):
    """
    Batched version of pauli_product_uint over first argument.
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
    phase = jnp.take(utils.PHASES, count)   # vectorized lookup
    c_new_array = c1_array * c2 * phase
    return xz_new_array, c_new_array

@jax.jit
def pauli_product_phase_sign_first(xz1_array, xz2):
    """
    Batched first-argument version of the real sign used by conjugation updates.
    """
    N = xz1_array.shape[1] // 2
    xz_new_array = jnp.bitwise_xor(xz1_array, xz2)
    count = jnp.sum((2 * xz1_array[:, :N] * xz2[N:] +
                     xz1_array[:, :N] * xz1_array[:, N:] +
                     xz2[:N] * xz2[N:] -
                     xz_new_array[:, :N] * xz_new_array[:, N:]),
                    axis=1) % 4
    sign = jnp.take(utils.CONJUGATION_SIGNS, count)
    return xz_new_array, sign


# ---------------------------------------------------------------------------
# old merge prototype code
# Didn't have xz_array, but instead x_array and z_array separately.
# ---------------------------------------------------------------------------

@jax.jit
def merge_pauli_batched_part_1(x_array_1, z_array_1, c_array_1,
                               x_array_2, z_array_2, c_array_2,
                               ):
    # concatenate
    x_array_merge = jnp.concatenate([x_array_1, x_array_2], axis=0).astype(jnp.uint8)
    z_array_merge = jnp.concatenate([z_array_1, z_array_2], axis=0).astype(jnp.uint8)
    c_array_merge = jnp.concatenate([c_array_1, c_array_2], axis=0).astype(utils.get_real_dtype())


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
        c_array_1: real array of shape (M1,)
        x_array_2, z_array_2: bool arrays of shape (M2,N)
        c_array_2: real array of shape (M2,)
        trunc_val: float, minimum coefficient magnitude to keep

    Returns:
        x_array_merge, z_array_merge: bool arrays of shape (M1+M2,N)
        c_array_merge: real array of shape (M1+M2,)
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

# ---------------------------------------------------------------------------
# Previous try on getting the code to jitted.
# Not sure whether the following code can actually be jitted.
#
# ---------------------------------------------------------------------------

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
    c_array = jnp.concatenate([c_array_1, c_array_2], axis=0).astype(utils.get_real_dtype())

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
        c_array_1: real array of shape (M1,)
        x_array_2, z_array_2: bool arrays of shape (M2,N)
        c_array_2: real array of shape (M2,)
        trunc_val: float, minimum coefficient magnitude to keep
    Returns:
        x_array_merge, z_array_merge: bool arrays of shape (M_merge,N)
        c_array_merge: real array of shape (M_merge,)
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
        c_array_1: real array of shape (M1,)
        x_array_2, z_array_2: bool arrays of shape (M2,N)
        c_array_2: real array of shape (M2,)
        trunc_val: float, minimum coefficient magnitude to keep

    Returns:
        x_array_merge, z_array_merge: bool arrays of shape (M_merge,N)
        c_array_merge: real array of shape (M_merge,)
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



# ---------------------------------------------------------------------------
# OLD BENCHMARK CODE
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
# Legacy row-duplication search
# ---------------------------------------------------------------------------
@jax.jit
def find_row_duplications_old(a, b):
    """
    Legacy `while_loop` binary-search implementation for row duplication checks.

    Compared with the active `find_row_duplications` in `kernels.py`, this
    version uses `jax.lax.while_loop(low < high)` instead of a fixed-iteration
    `fori_loop`. The fixed-step version is generally more JIT/XLA-friendly and
    gives more predictable performance, especially on accelerators.
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
