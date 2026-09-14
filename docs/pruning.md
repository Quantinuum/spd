# Circuit pruning

Pruning skips gates outside the support reachable from the input observable.
It is opt-in and is most useful for local observables and local circuits.

| `pruning` | Behavior |
|---|---|
| `None` (default) | Execute the complete circuit. |
| `"light-cone"` | Plan once from the initial SPO. |
| `"light-cone-barrier"` | Refresh from the current SPO before each nonempty barrier-delimited block. |

```python
final_spo, info = spd.evolve(
    initial_spo, circuit, trunc_val, max_num_str,
    pruning="light-cone-barrier", progress=False,
)
print(info["pruning"])  # method and total/retained/pruned gate counts

terminal = spd.init_gradient_spo(final_spo, basis="0")
initial_spgo, gradients, backward_info = spd.backpropagate(
    terminal, circuit, trunc_val, max_num_str, progress=False,
)
```

Both modes propagate observables in reverse circuit order. Barrier refresh can
find tighter support after commutation and truncation. It uses one forward call
and one combined record: backward replays exactly the selected forward gates,
without deriving a cone from the gradient operator. Use the full original circuit,
including the same angles, for backward. Gradient and diagnostic entries retain
their original indices; omitted rotations have zero gradients.

Pytket and OpenQASM barriers are supported. Direct IR represents them as
`SkippedOperation("barrier")` or `SkippedOperation("OpType.Barrier")`.
Consecutive barriers and measurement-only blocks add no scans. With no barriers,
barrier mode uses one whole-circuit plan. No gates are reordered.

Both modes work on NumPy, JAX and Triton, including Triton's `evolve_step`.
Triton reduces support on GPU, transfers only the packed mask and validity flag,
and avoids state copies between blocks. Scans still depend on state size;
packed key widths and memory usage still depend on the number of qubits.

Skipped gates also skip truncation and term caps. There is no initial cleanup:
a sub-threshold input coefficient can survive if all gates are omitted. Retained
gates use the requested numerical policy. The returned backward SPGO describes
the reduced circuit; its adjoint coefficients need not equal the unpruned result.

Keep the forward result unmodified until gradient initialization. Another forward
call replaces its record, so chaining calls does not assemble a backward record
for the concatenated circuit. Use barrier mode within one call for that workflow.
Pruned noise analysis is not supported.

See the [benchmark note](../benchmarks/light_cone/README.md) for timings and reproduction.
