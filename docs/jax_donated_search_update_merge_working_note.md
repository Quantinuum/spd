# Donated Search/Update/Merge Working Note

This note tracks the experimental donated JAX algorithm work. It is separate
from the memory-efficiency handoff notes.

## Current Choice

Add a new algorithm:

```text
search_update_merge_donate
```

The current version keeps the selected `search_update_merge` behavior while
the state is still growing. It uses a donated full-step JIT only when
`state.get_size() == max_num_str`. The donated full-step calls the same top-k
cap helpers as the non-donated path, so cap truncation keeps the largest live
coefficients before lexsorted storage is returned.

## Why This Shape

The memory benchmark showed that donating the old inner JIT did not help much.
The useful case was donating the whole logical step:

```text
M input rows -> internal 2M work -> M output rows
```

Keeping the `2M` arrays inside the donated JIT avoids returning them to Python.

## Correctness Details

Even when the input size is already `max_num_str`, truncation can make the next
logical size smaller. The donated wrapper handles this by:

- slicing to `max_num_str` inside the donated JIT,
- computing truncation stats inside the donated JIT from the same top-k keep
  mask used to PAD/zero removed rows,
- shrinking the returned `max_num_str` state outside the JIT if `new_size` is
  smaller.

This preserves the current `search_update_merge` results while still avoiding
the large `2M` arrays at the Python boundary.

Because this path uses JAX buffer donation, callers should treat capped input
states as consumed by the step. This matches the forward/backward runner flow,
where each step replaces the previous state.

## Files

- `spd/jax_backend/algorithms/search_update_merge_donate.py`
- `spd/jax_backend/kernels.py`
- `tests/test_jax_search_update_merge_donate.py`

The `kernels.py` change is only algorithm registration.
