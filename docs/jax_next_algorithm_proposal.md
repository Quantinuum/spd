# JAX Next-Algorithm Proposal

## Goal

This note proposes a faster JAX forward/backward algorithm than the current
`stack_sort_merge` and `search_update_merge` paths.

The main target is not only semantic cleanup. The main target is to reduce the
amount of full-array sorting and reshuffling done at every Pauli rotation step.

## What the current two methods are doing

### `stack_sort_merge`

Current shape:

1. split one operator into two rotated branches
2. concatenate both branches
3. lexicographically sort all rows
4. merge duplicates with `segment_sum`
5. sort again by coefficient magnitude
6. keep a power-of-two prefix

Strengths:

- simple merge logic after the split
- easy to reason about threshold-by-magnitude
- easy to compute truncation info from the tail

Weaknesses:

- it pays for two global reorderings of the whole state:
  - one `lexsort`
  - one `argsort(-abs(c))`
- it always materializes both rotated branches before merging
- it moves a lot of memory even when only part of the operator changes

### `search_update_merge`

Current shape:

1. keep the stored operator lexicographically sorted
2. generate conjugated partner rows
3. find partners with binary-search-style duplicate lookup
4. update existing coefficients in place
5. append only missing partner rows
6. lexsort the merged result again

Strengths:

- avoids explicitly splitting into two full branch operators
- uses the sorted structure of the stored state
- naturally fits the "search/update" idea

Weaknesses:

- still does full-array concatenation and lexsort
- still rebuilds and resorts the full merged state each step
- current truncation handling is semantically different from
  `stack_sort_merge`

## What is likely the bottleneck now

The main bottlenecks are the global data-movement operations, not the small
masking logic:

- `jnp.lexsort(...)` over all rows
- `jnp.argsort(-jnp.abs(c_concat))` over all coefficients
- full-array `concatenate(...)`
- `segment_sum(...)` over a fully materialized concatenated array
- scatter/repack of all `xz` rows
- shape changes across powers of two, which trigger recompiles in several
  jitted functions

In short:

- sorting the whole state is expensive
- rebuilding the whole state is expensive
- changing array shapes is expensive

Any faster design should reduce at least one of those three.

## Main observation

The current two methods each preserve only one useful property:

- `stack_sort_merge` preserves coefficient-ranked storage
- `search_update_merge` preserves lexicographically sorted storage

For performance, lexicographically sorted storage is more useful than
coefficient-ranked storage, because:

- partner lookup depends on row identity, not on coefficient order
- duplicate merging depends on row identity, not on coefficient order
- keeping lexicographic order avoids one global reorder in future steps

So the next algorithm should start from the `search_update_merge` idea, not from
the `stack_sort_merge` idea.

## Proposed direction

### Name

`search_update_compact`

### Core idea

Keep the operator in lexicographic order permanently, and avoid global
coefficient sorting. Instead, do:

1. update matched rows in place
2. append only genuinely new rows
3. merge duplicates only for the appended part versus the base part
4. compact the state with a selection-based cutoff instead of full sorting by
   magnitude

The key change is:

- do not rank the entire operator by `|c|`
- only decide which rows to keep when storage pressure requires compaction

## Proposed forward flow

For one rotation step:

1. start from a lexicographically sorted `(xz_array, c_array)`
2. compute conjugated partner rows `xz_array_conj`
3. find existing partners with `find_row_duplications(...)`
4. update coefficients of existing rows directly
5. build only the missing inserted rows
6. sort the inserted rows lexicographically
7. merge the base rows and inserted rows with a two-way merge
8. if the state size is still under capacity, stop here
9. if the state size exceeds capacity, compact using a selection-style cutoff

The important change is that step 7 is not "concatenate everything and lexsort
again". It should be a merge of two already sorted arrays:

- base rows remain lexicographically sorted
- inserted rows are sorted once
- then merged like a merge-sort merge

That should be cheaper than a full lexsort of the whole state.

## Proposed backward flow

Use the same structural plan as forward:

1. update `c_array`
2. update `grad_c_array`
3. build only missing inserted rows for both value and grad
4. sort inserted rows once
5. merge base and inserted arrays together
6. compact only if capacity pressure requires it

This keeps value and gradient arrays structurally aligned without a second
independent ordering pass.

## Compaction proposal

If performance matters, compaction should not be:

- full `argsort(-abs(c))` of the whole state every step

Instead, it should be something closer to:

1. if `num_terms <= max_num_str`, do no compaction
2. if `num_terms > max_num_str`, estimate a cutoff with selection / top-k
3. keep all rows above that cutoff
4. break ties only if needed

The main point is to separate:

- normal update steps
- occasional compaction steps

Most steps should not pay the price of globally re-ranking the whole operator.

## Why this can be faster

Compared with `stack_sort_merge`, this proposal removes:

- the explicit two-branch materialization
- the full lexsort of the concatenated two-branch state
- the full global sort by coefficient magnitude on every step

Compared with the current `search_update_merge`, this proposal reduces:

- full-state lexsort after every append
- unnecessary treatment of all rows as newly reorderable data

The intended pattern is:

- small inserted set
- large stable base set
- merge small sorted delta into large sorted base

This is the usual place where a delta-based algorithm wins.

## What would have to change in practice

### 1. Add a two-way merge kernel

Instead of:

- concatenate base and inserted rows
- lexsort the whole thing

we would want:

- inserted rows sorted once
- merge two already sorted arrays

This is probably the most important structural change.

### 2. Keep capacity fixed more often

Shape changes currently cause recompiles across powers of two.

A faster version should try to keep one fixed capacity for longer stretches:

- allocate a fixed working capacity
- update within that capacity
- only grow when truly necessary

This can reduce recompilation churn.

### 3. Separate threshold statistics from physical compaction

If truncation remains "soft", then:

- threshold stats should be tracked separately
- physical compaction should happen only when capacity requires it

This avoids turning the threshold into a global reorder trigger.

### 4. Compact lazily

Instead of compacting every step:

- compact only when the number of live rows exceeds a target capacity
- otherwise keep the lexicographically sorted live state as-is

This should save work in long runs where growth is gradual.

## Risks

This design is not free.

### Implementation complexity

It is more complex than either current path because it depends on:

- a stable sorted-state invariant
- a fast merge of base and inserted rows
- delayed compaction

### Tie handling

If compaction uses a cutoff rather than full sort, tie behavior near the
boundary must be defined carefully.

### JAX constraints

A theoretically good algorithm can still be bad in JAX if it introduces:

- too much control flow
- hard-to-fuse scatter/gather patterns
- shape polymorphism that forces recompilation

So the design has to stay "array programming friendly".

### CPU vs GPU evaluation risk

The real target metric is GPU performance, but current development happens on
CPU.

That means CPU benchmarking is useful but limited:

- CPU measurements are good for correctness work
- CPU measurements are good for spotting obvious full-sort / full-copy
  bottlenecks
- CPU measurements are good for comparing rough scaling trends

But CPU measurements are not the final truth for JAX performance, because the
tradeoffs can flip on GPU:

- GPU often prefers large fused array operations
- GPU is more sensitive to shape churn and recompilation
- GPU can punish irregular update-heavy logic more than CPU
- a method that looks slightly better on CPU may still lose on GPU

So CPU should be treated as a development filter, not as the final winner
selection environment.

## Minimal experimental plan

I would not replace either current path immediately.

Instead I would prototype a third path in stages:

1. keep the current `search_update_merge` update logic
2. replace full concat+lexsort with "sort inserted rows + merge two sorted arrays"
3. benchmark that alone
4. only then test lazy compaction
5. only then test selection-based cutoff instead of full coefficient sort

This isolates which change actually buys performance.

## Benchmarking guidance

During development:

1. use CPU benchmarks to reject obviously bad designs
2. use CPU benchmarks to track scaling and recompilation behavior
3. do not treat CPU timings as the final ranking between close candidates

Before choosing the long-term algorithm:

1. run the final benchmark on GPU
2. compare end-to-end runtime, not only micro-kernels
3. include compile time and steady-state runtime separately
4. compare memory growth and shape stability as well as raw speed

## Recommendation

If the priority is performance, I would not try to make `search_update_merge`
behave exactly like `stack_sort_merge`.

I would instead:

1. keep `search_update_merge` as the semantic starting point
2. move toward a `search_update_compact` design
3. reduce full-array sorts first
4. reduce recompiles second
5. revisit truncation semantics only after the performance shape is settled

The main bet is simple:

- the next speedup is more likely to come from "delta merge + lazy compaction"
  than from another variation of "materialize everything, then globally sort it".
