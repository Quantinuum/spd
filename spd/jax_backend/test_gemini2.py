import jax
import jax.numpy as jnp

# Constant for padding (Max int32)
PAD_VAL = jnp.iinfo(jnp.int32).max

def find_duplicates_and_sum(a, b, a_coeffs, b_coeffs):
    """
    1. Finds lexical duplications between a and b.
    2. Computes sum(a_coeff * b_coeff) for matching rows.

    Assumes inputs are padded to fixed shapes (powers of 2)
    and sorted such that padding (INT_MAX) is at the end.
    """

    # --- 1. Lexical Comparison Logic (Same as before) ---
    def lexical_gt(row1, row2):
        not_eq = row1 != row2
        first_diff_idx = jnp.argmax(not_eq)
        is_gt = row1[first_diff_idx] > row2[first_diff_idx]
        are_equal = jnp.all(row1 == row2)
        return is_gt & (~are_equal)

    # --- 2. Binary Search ---
    def binary_search_row(needle, haystack):
        m = haystack.shape[0]

        def cond_fun(state):
            low, high = state
            return low < high

        def body_fun(state):
            low, high = state
            mid = (low + high) // 2
            mid_row = haystack[mid]
            go_right = lexical_gt(needle, mid_row)
            low = jnp.where(go_right, mid + 1, low)
            high = jnp.where(go_right, high, mid)
            return (low, high)

        idx, _ = jax.lax.while_loop(cond_fun, body_fun, (0, m))
        return idx

    # --- 3. Vectorized Search ---
    # Find where every row of 'a' would fit in 'b'
    indices_in_b = jax.vmap(binary_search_row, in_axes=(0, None))(a, b)

    # Clamp indices to be safe for extraction (bounds check)
    # If binary search returns len(b) (not found/greater than all), clamp to last idx
    safe_indices = jnp.minimum(indices_in_b, b.shape[0] - 1)

    # Check if it is an actual match
    potential_matches = b[safe_indices]
    is_duplicate = jnp.all(a == potential_matches, axis=1)

    # --- 4. The Summation (Q3 Solution) ---

    # Gather the coefficients from B using the discovered indices
    # Shape: (M,)
    b_coeffs_aligned = b_coeffs[safe_indices]

    # Multiply A coeffs * B coeffs
    raw_products = a_coeffs * b_coeffs_aligned

    # Apply Mask:
    # If is_duplicate is False, we multiply by 0.
    # If is_duplicate is True (even for PAD_VAL rows), we multiply by the product.
    # Note: Since we pad coefficients with 0.0, the PAD_VAL matches result in 0.0 anyway.
    valid_products = jnp.where(is_duplicate, raw_products, 0.0)

    total_sum = jnp.sum(valid_products)

    return total_sum, is_duplicate, indices_in_b

# JIT Compile
# Because the input shapes are fixed (via your padding strategy),
# this compiles once and runs fast.
jit_find_and_sum = jax.jit(find_duplicates_and_sum)

# --- Helper for your Padding Strategy ---

def pad_and_sort_inputs(a, b, a_c, b_c):
    """
    Pads arrays to the next power of 2.
    A/B padded with INT_MAX.
    Coeffs padded with 0.0.
    """
    current_m = a.shape[0]
    # Calculate next power of 2
    next_pow2 = 1 << (current_m - 1).bit_length()

    pad_len = next_pow2 - current_m

    if pad_len > 0:
        # Pad A and B with INT_MAX (so they stay at the bottom after sort/if sorted)
        a_pad = jnp.pad(a, ((0, pad_len), (0, 0)), constant_values=PAD_VAL)
        b_pad = jnp.pad(b, ((0, pad_len), (0, 0)), constant_values=PAD_VAL)

        # Pad coeffs with 0.0
        ac_pad = jnp.pad(a_c, (0, pad_len), constant_values=0.0)
        bc_pad = jnp.pad(b_c, (0, pad_len), constant_values=0.0)

        return a_pad, b_pad, ac_pad, bc_pad
    return a, b, a_c, b_c

# --- Example Usage ---

M_real = 5
N = 4

# Mock data
a = jnp.array([[1, 2, 3, 4], [5, 5, 5, 5], [0, 1, 0, 1], [9, 9, 9, 9], [8,8,8,8]], dtype=jnp.int32)
b = jnp.array([[0, 1, 0, 1], [1, 2, 3, 4], [9, 9, 9, 9], [2, 2, 2, 2], [3,3,3,3]], dtype=jnp.int32)

# Coefficients
ac = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]) # Coeffs for A
bc = jnp.array([ 1.0,  2.0,  3.0,  4.0,  5.0]) # Coeffs for B

# 1. Sort (Preprocessing step - usually done before padding if data is raw)
# Remember: inputs must be sorted for binary search
order_a = jnp.lexsort(a.T[::-1])
order_b = jnp.lexsort(b.T[::-1])

a_sorted = a[order_a]
b_sorted = b[order_b]
ac_sorted = ac[order_a]
bc_sorted = bc[order_b]

# 2. Pad (to power of 2, e.g., 5 -> 8)
a_final, b_final, ac_final, bc_final = pad_and_sort_inputs(a_sorted, b_sorted, ac_sorted, bc_sorted)

print(f"Padded Shape: {a_final.shape}") # Should be (8, 4)

# 3. Run JIT function
total, is_dup, locs = jit_find_and_sum(a_final, b_final, ac_final, bc_final)
import pdb;pdb.set_trace()

print(f"\nTotal Sum: {total}")

# Verification:
# Overlap:
# [1, 2, 3, 4] (A index 0, val 10.0) matches [1, 2, 3, 4] (B index 0, val 2.0) -> 20.0
# [0, 1, 0, 1] (A index 0, val 30.0) matches [0, 1, 0, 1] (B index 0, val 1.0) -> 30.0
# [9, 9, 9, 9] (A index 3, val 40.0) matches [9, 9, 9, 9] (B index 4, val 3.0) -> 120.0
# Total should be 170.0
