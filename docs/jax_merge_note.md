# JAX `merge_` Note

## What `merge_` does

At a high level, `merge_` in the JAX backend does four things:

1. concatenate two batches of Pauli strings
2. sort them lexicographically so identical rows become adjacent
3. merge duplicate rows by summing their coefficients
4. sort the merged result by coefficient magnitude and apply truncation

The important point is that there are two different filtering stages:

- deduplication of identical `xz` rows
- truncation of small coefficients

These are not the same step.

## Deduplication step

After lexicographic sorting, identical rows are adjacent. For example:

```python
x_concat = [A, A, A, B, B, C]
c_concat = [1, 2, 3, 4, 5, 6]
```

The code builds:

```python
group_ids = [0, 0, 0, 1, 1, 2]
```

Then:

```python
c_concat = segment_sum(c_concat, group_ids, num_segments=total_size)
```

produces:

```python
c_concat = [6, 9, 6, 0, 0, 0]
```

The matching `xz` representatives are packed in the same order:

```python
x_concat = [A, B, C, 0, 0, 0]
```

At this stage, everything is consistent:

```python
[A, B, C, invalid, invalid, invalid]
[6, 9, 6, 0, 0, 0]
```

There is no bug here.

## What the scatter block does

This code:

```python
x_concat = x_concat * boundaries[:, None].astype(x_concat.dtype)
x_concat = jnp.zeros_like(x_concat).at[group_ids].add(x_concat)
```

does two things:

1. keep only the first row of each duplicate group
2. pack those surviving representatives into the front so they align with the
   packed `segment_sum` output

So yes, it zeroes non-representative duplicate rows first, then repacks the
unique rows.

## Where truncation enters

After deduplication, the code sorts by descending `|c|`:

```python
c_sort_indices = jnp.argsort(-jnp.abs(c_concat))
c_concat = c_concat[c_sort_indices]
x_concat = x_concat[c_sort_indices]
```

Then it builds the truncation mask:

```python
mask = jnp.abs(c_concat) > trunc_val
```

This is a different stage from deduplication. Even though the rows are already
unique, some of them are now below the requested threshold.

Example:

```python
x_concat = [A, B, C, D, 0, 0]
c_concat = [0.90, 0.20, 0.06, 0.04, 0.00, 0.00]
trunc_val = 0.05
mask = [True, True, True, False, False, False]
```

Row `D` is unique, but it is below the requested threshold.

## Hard vs soft truncation

There are two possible interpretations of the JAX behavior.

### Hard truncation

If `trunc_val` is a hard cutoff, then a row below threshold is no longer part
of the physical state.

In that interpretation, the state should satisfy:

```python
invalid row  <=>  zero coefficient
```

If a row fails the threshold mask, one of these must happen:

- remove both the `xz` row and its coefficient
- or keep a padded slot, but force its coefficient to zero

### Soft truncation

If `trunc_val` is only a target used to control storage pressure, then keeping a
few extra below-threshold terms is acceptable as long as the returned operator
is still mathematically valid.

In that interpretation, this is fine:

```python
x_concat = [A, B, C, D]
c_concat = [0.90, 0.20, 0.06, 0.04]
```

even if `0.04 < trunc_val`, because the extra term is still represented by a
valid Pauli string and coefficient pair.

The invalid case is mixing the two interpretations:

```python
x_concat = [A, B, C, invalid]
c_concat = [0.90, 0.20, 0.06, 0.04]
```

That is not "soft truncation". That is a broken state representation, because a
coefficient survives after its matching Pauli string has been invalidated.

This mixed case is the bug we discussed.

## Where the mismatch comes from

The mismatch does not come from `segment_sum`.

It comes later, after truncation, if the code does something like:

```python
x_concat = x_concat * mask[:, None]
```

but leaves:

```python
c_concat
```

unchanged.

Then we can end up with:

```python
x_concat = [A, B, C, 0, 0, 0]
c_concat = [0.90, 0.20, 0.06, 0.04, 0.00, 0.00]
```

The `0.04` coefficient is below threshold, but still numerically present. If the
wrapper later slices a padded prefix that includes that slot, then:

- this is fine under the soft-truncation interpretation if the matching `xz`
  row is also kept
- this is wrong if the `xz` row has already been invalidated

## About `PAD_VAL`

`PAD_VAL` is used in the codebase as a sentinel invalid row that sorts to the
bottom under lexicographic ordering.

It is not the main issue in this bug.

The main issue is whether the coefficient in an invalid slot is guaranteed to be
zero.

Using `PAD_VAL` for invalid `xz` rows is fine, but it does not by itself solve
the problem if the matching coefficient is still nonzero.

## What outer slicing does

`slice_to_size_x_arr` and `slice_to_size_c_arr` do not implement the threshold
decision. They only keep the first `slice_size` slots, where `slice_size`
depends on `next_pow2(final_valid_count)` and `max_num_str`.

So there are two separate choices:

1. which terms are above threshold
2. how many slots to keep in the padded returned state

Under the soft-truncation interpretation, it is acceptable for the returned
state to contain:

- all above-threshold terms
- plus some additional below-threshold terms that still fit inside the padded
  prefix

But if we choose that interpretation, then both `xz` and `c` must remain valid
together for those extra terms.

## Current stack/sort behavior

The current `stack_sort_merge` path now follows the soft-truncation
interpretation:

- `trunc_val` is treated as a storage target, not a hard mathematical cutoff
- power-of-two padding may keep some additional valid low-weight terms
- `num_above_trunc_val` means "how many terms are above threshold"
- the wrapper computes `num_string` from the actual returned sliced state

So in the current stack/sort path:

1. `merge_` and `merge_val_grad_` keep `xz` rows and coefficients aligned after
   the magnitude sort
2. below-threshold terms may still be present in the returned padded state
3. the returned operator is still mathematically valid because every nonzero
   coefficient still has a matching `xz` row

## Difference from `search_update_merge`

The two JAX algorithms do not currently have the same truncation semantics.

### `stack_sort_merge`

- below-threshold terms may survive in the padded returned state
- the wrapper reports the actual number of stored returned terms
- `num_above_trunc_val` is kept separate from the returned-state size

### `search_update_merge`

- rows below threshold are still converted to `PAD_VAL` before sorting
- coefficients can still be kept for step-info accounting
- the wrapper still uses `final_valid_count` as if it were the returned-state
  size

So right now:

- `stack_sort_merge` uses soft-cutoff semantics
- `search_update_merge` still follows the older threshold-masking behavior

They are not yet equivalent.

## Remaining alignment plan

If the goal is to make both JAX algorithms behave the same way, the remaining
work is:

1. decide whether `search_update_merge` should also move to soft-cutoff
   semantics
2. if yes, stop invalidating `xz` rows below threshold there as well
3. rename the threshold count there to `num_above_trunc_val`
4. make the wrapper report the actual stored returned size, as done for
   `stack_sort_merge`
5. add parity tests that compare not only coefficients but also truncation/count
   semantics between the two JAX paths
