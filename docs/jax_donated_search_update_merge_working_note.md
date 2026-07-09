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

## Step-Info Memory

The first exact top-k step-info implementation built full `2M` removed masks
and removed coefficient arrays. At large sizes this caused much higher allocator
pressure and can OOM even when the minimal donated top-k step fits.

The winning implementation uses:

```text
jax.lax.optimization_barrier((magnitudes, final_keep_mask))
then a fused jax.lax.reduce over removed count, l1, and l2-square
```

This keeps exact truncation norms while changing XLA's buffer scheduling enough
to recover the low-memory donated behavior in the full sparse-Pauli update.

## Files

- `spd/jax_backend/algorithms/search_update_merge_donate.py`
- `spd/jax_backend/kernels.py`
- `examples/benchmark_jax_memory_donation.py`
- `tests/test_jax_search_update_merge_donate.py`

The `kernels.py` change is only algorithm registration.

## Benchmark Command

Use this variant to measure the integrated donated algorithm:

```bash
python examples/benchmark_jax_memory_donation.py \
    --mode gradient-eval \
    --variant search-update-merge-donate \
    --rows 33554432 \
    --system-size 64 \
    --precision double \
    --max-num-str 33554432 \
    --loop-steps 4 \
    --diagnostics both \
    --lifetime keep-all
```

Compare against:

```bash
python examples/benchmark_jax_memory_donation.py \
    --mode gradient-eval \
    --variant search-current \
    --rows 33554432 \
    --system-size 64 \
    --precision double \
    --max-num-str 33554432 \
    --loop-steps 4 \
    --diagnostics both \
    --lifetime keep-all
```

To isolate the old non-top-k search/update/merge path without donation:

```bash
python examples/benchmark_jax_memory_donation.py \
    --mode gradient-eval \
    --variant search-legacy-current \
    --rows 33554432 \
    --system-size 64 \
    --precision double \
    --max-num-str 33554432 \
    --loop-steps 4 \
    --diagnostics both \
    --lifetime keep-all
```

To isolate the top-k inner path with a benchmark-local minimal donated wrapper:

```bash
python examples/benchmark_jax_memory_donation.py \
    --mode gradient-eval \
    --variant search-top-k-fullstep-donate \
    --rows 33554432 \
    --system-size 64 \
    --precision double \
    --max-num-str 33554432 \
    --loop-steps 4 \
    --diagnostics both \
    --lifetime keep-all
```

To confirm whether forcing the top-k diagnostic outputs causes the extra
allocator pressure:

```bash
python examples/benchmark_jax_memory_donation.py \
    --mode gradient-eval \
    --variant search-top-k-fullstep-donate-stats \
    --rows 33554432 \
    --system-size 64 \
    --precision double \
    --max-num-str 33554432 \
    --loop-steps 4 \
    --diagnostics both \
    --lifetime keep-all
```

To test the same full donated top-k path with a fused `jax.lax.reduce` stats
formulation:

```bash
python examples/benchmark_jax_memory_donation.py \
    --mode gradient-eval \
    --variant search-top-k-fullstep-donate-lax-stats \
    --rows 33554432 \
    --system-size 64 \
    --precision double \
    --max-num-str 33554432 \
    --loop-steps 4 \
    --diagnostics both \
    --lifetime keep-all
```

To test the same path with the promising barrier-mask stats schedule in the
original donation benchmark:

```bash
python examples/benchmark_jax_memory_donation.py \
    --mode gradient-eval \
    --variant search-top-k-fullstep-donate-barrier-stats \
    --rows 33554432 \
    --system-size 64 \
    --precision double \
    --max-num-str 33554432 \
    --loop-steps 4 \
    --diagnostics both \
    --lifetime keep-all
```

To benchmark exact top-k stat formulations without the sparse-Pauli update:

```bash
python examples/benchmark_jax_topk_stats_memory.py \
    --rows 536870912 \
    --k 268435456 \
    --precision double
```

The stats-only variants are:

- `top-scores-only`: avoid selected masks and compute stats from top-k values.
- `selected-mask`: build selected/final masks but avoid `removed_coeffs`.
- `selected-mask-removed`: close to the original removed-coefficients formula.
- `final-keep-sums`: compute total minus kept sums using `final_keep_mask`.
- `removed-where-direct`: put `where(...)` directly inside reductions.
- `removed-mul-square`: use multiplication instead of `** 2`.
- `inline-abs-removed`: recompute `abs(c)` inline instead of naming magnitudes.
- `lax-reduce-removed`: fuse count/l1/l2 reductions with `jax.lax.reduce`.
- `sort-scores`: use full sorting instead of `top_k`.

To benchmark full donated top-k sparse-Pauli update variants:

```bash
python examples/benchmark_jax_fullstep_stats_variants.py \
    --mode gradient-eval \
    --variant lax \
    --rows 33554432 \
    --system-size 64 \
    --precision double \
    --max-num-str 33554432 \
    --loop-steps 4 \
    --diagnostics both
```

The full-step variants are:

- `minimal`: top-k donated full step without exact stats.
- `current`: current algebraic stats from top-k scores.
- `removed`: explicit removed-mask/removed-coefficients stats.
- `lax`: removed-mask stats with fused `jax.lax.reduce`.
- `lax-stats-first`: compute lax stats before final output construction.
- `lax-stats-last`: compute lax stats after final output construction in code order.
- `lax-barrier-mask`: add an optimization barrier around stat inputs.
- `lax-barrier-output`: add an optimization barrier around sorted outputs before stats.

## Current Benchmark Findings

At `rows=33554432`, `system_size=64`, double precision, `max_num_str=33554432`,
and `gradient-eval` with 4 loop steps:

| variant | gpu_peak_mb | jax_in_use_peak_mb | note |
| --- | ---: | ---: | --- |
| `minimal` | 6841 | 4731.2 | no exact stats |
| `current` | 15033 | 5435.8 | exact stats from top-k scores |
| `removed` | 15033 | 5435.8 | exact removed-mask stats |
| `lax` | 15033 | 5435.8 | fused exact removed-mask stats |
| `lax-stats-first` | 15033 | 5435.8 | stats before output construction |
| `lax-stats-last` | 15033 | 5435.8 | stats after output construction |
| `lax-barrier-mask` | 6841 | 4731.2 | exact stats with barrier on stat inputs |
| `lax-barrier-output` | 6841 | 5167.4 | exact stats with barrier on sorted outputs |

At `rows=268435456`, `system_size=64`, double precision, `max_num_str=268435456`,
and the same gradient-eval setup:

| variant | result |
| --- | --- |
| stats without barrier | OOM in backward donated top-k stats |
| `lax-barrier-mask` | runs, `gpu_peak_mb=50747`, `jax_in_use_peak_mb=37849.4` |

This suggests exact stats are viable if `jax.lax.optimization_barrier` is used
around `(magnitudes, final_keep_mask)` before the fused `jax.lax.reduce` stats.
The barrier appears to improve XLA buffer scheduling/lifetime in the full
sparse-Pauli update.
