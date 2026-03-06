import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def find_row_duplications(a, b):
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

# --- Example Usage ---

# Mock Data setup
M = 100 # Using smaller M for demo; logic holds for 1e9
N = 10
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)

# Create sorted arrays
# (In reality, you provided sorted inputs, so we sort them here just to generate valid data)
a = jax.random.randint(k1, (M, N), 0, 100)
b = jax.random.randint(k2, (M, N), 0, 100)

# Inject a known duplicate for testing
a = a.at[5].set(b[10])
a = a.at[15].set(b[33])

# JAX lexsort uses keys in reverse order (last column is primary key in the call signature)
# We transpose to sort correctly by col 0, then col 1...
a = a[jnp.lexsort(a.T[::-1])]
b = b[jnp.lexsort(b.T[::-1])]

# Run the function
is_dup, locs = jax.jit(find_row_duplications)(a, b)

print(f"Row 5 is duplicate? {is_dup[5]}")
print(f"Found at index in B: {locs[5]}")

print("---"*30)
print(is_dup)
print("---"*30)
print(locs)


# If A is too large, process it in chunks using jax.lax.map
def batched_find_duplicates(a, b, batch_size=10000):
    num_batches = a.shape[0] // batch_size

    # Reshape A to (num_batches, batch_size, N)
    # Note: Handle remainder rows if M is not perfectly divisible
    a_reshaped = a[:num_batches*batch_size].reshape(num_batches, batch_size, -1)

    def scan_fn(carry, a_batch):
        is_dup, locs = find_row_duplications(a_batch, b)
        return carry, (is_dup, locs)

    _, (all_dups, all_locs) = jax.lax.scan(scan_fn, None, a_reshaped)

    return all_dups.reshape(-1), all_locs.reshape(-1)
