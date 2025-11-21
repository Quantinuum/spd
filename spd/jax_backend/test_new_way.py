import jax
import jax.numpy as jnp
from jax import lax


# ------------------------------------------------------------
# 2. Lexicographic comparison for 3-word uint64 keys
#    Returns:
#        -1 if a < b
#         0 if a == b
#         1 if a > b
# ------------------------------------------------------------
def lex_compare_words(a, b):
    """
    a, b: shape (3,) uint64
    returns -1, 0, or +1
    """
    # Compare word by word
    less = jnp.where(a < b, 1, 0)
    greater = jnp.where(a > b, 1, 0)
    # find the first position where a != b
    diff = less + greater  # nonzero at first differing word
    # index of first differing word
    idx = jnp.argmax(diff)

    # If all equal
    return jnp.where(jnp.all(a == b), 0,
                     jnp.where(a[idx] < b[idx], -1, 1))


# ------------------------------------------------------------
# 3. multi-word searchsorted via lax.while_loop
# ------------------------------------------------------------
def searchsorted_one(keys_b, key):
    """
    keys_b: (M_b, 3)
    key:   (3,)
    Binary search for insertion index of key in keys_b.
    """

    def cond_fun(state):
        left, right = state
        return left < right

    def body_fun(state):
        left, right = state
        mid = (left + right) // 2

        cmp = lex_compare_words(key, keys_b[mid])   # -1, 0 or +1

        # If key < mid → move right
        new_right = jnp.where(cmp == -1, mid, right)
        # else key >= mid → move left up
        new_left  = jnp.where(cmp == -1, left, mid + 1)

        return (new_left, new_right)

    left0 = 0
    right0 = keys_b.shape[0]

    left, right = lax.while_loop(cond_fun, body_fun, (left0, right0))
    return left


# vmap to apply searchsorted over all rows in A
searchsorted_vec = jax.vmap(searchsorted_one, in_axes=(None, 0))


# ------------------------------------------------------------
# 4. FINAL duplicate finder
# ------------------------------------------------------------
# @jax.jit
def find_row_duplicates_(a, b):
    """
    a, b: shape (M, 6) uint32
    must be lexicographically sorted
    """

    a_words = a # pack_rows_to_words(a)   # (M, 3)
    b_words = b # pack_rows_to_words(b)   # (M, 3)

    # Step 1 — searchsorted for each row in a
    idxs = searchsorted_vec(b_words, a_words)  # (M,)

    # Step 2 — check exact match (avoids all false positives)
    valid = (idxs < b_words.shape[0]) & jnp.all(a_words == b_words[idxs], axis=1)

    dup_a = jnp.where(valid)[0]
    dup_b = idxs[valid]

    return dup_a, dup_b

def find_row_duplicates(a, b):
    """
    Finds rows in 'a' that also exist in 'b'.
    Assumes 'a' and 'b' are lexically sorted (M, N) int32 arrays.
    """

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
    a = jnp.array([[1,2,3,4,5,6],
                   [2,3,4,5,6,7],
                   [9,9,9,9,9,9]], dtype=jnp.uint32)

    b = jnp.array([[2,3,4,5,6,7],
                   [4,4,4,4,4,4],
                   [9,9,9,9,9,9]], dtype=jnp.uint32)

    dup_a_idx, dup_b_idx = find_row_duplicates(a, b)

    print(dup_a_idx)   # [1, 2]
    print(dup_b_idx)   # [0, 2]

    # # Example usage
    # a = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.uint32)
    # b = jnp.array([[3, 4], [5, 6], [7, 8]], dtype=jnp.uint32)

    # dup_a_idx, dup_b_idx = find_row_duplicates(a, b)
    # print("Duplicate indices in a:", dup_a_idx)
    # print("Duplicate indices in b:", dup_b_idx)
