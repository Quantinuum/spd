# TODO Plan

This file tracks intentionally deferred follow-up work after the JAX algorithm
switch and backend-object runner cleanup.

## Open Items

### 1. Decide long-term algorithm-selection UX
Current state:
- JAX algorithm choice is controlled through `spd.jax_backend.set_algorithm(...)`
- advanced users can also construct a `BackendAdapter` and configure
  `backend.module.set_algorithm(...)` before passing `backend=...` into runners

Deferred decision:
- whether algorithm selection should remain documented as a low-level advanced
  setting
- or whether it should eventually gain a more explicit runner/backend-construction
  API

### 2. Run the full suite before release/merge
Focused regression and conformance suites have been run during development, but
the repository still benefits from a final full-suite pass before a release or
merge point.

### 3. Revisit curated low-level public API
The note in [`docs/low_level_api_scope_note.md`](low_level_api_scope_note.md)
is still relevant. If power-user low-level operations become more important,
consider defining a small supported low-level public module instead of exposing
runtime internals indirectly.

### 4. Decide long-term fate of `stack_sort_merge`
Current state:
- `stack_sort_merge` remains available as a reference and fallback algorithm

Deferred decision:
- keep it permanently as a supported alternate path
- or deprecate/remove it after the new JAX path has enough long-term confidence

### 5. Separate IR from padded backend storage width
Current state:
- parsed OpenQASM / pytket operations are lowered with packbit-padded Pauli strings

Possible refactor:
- keep the IR logical and unpadded
- move padding/packing to the backend-lowering step instead of the frontend parser
