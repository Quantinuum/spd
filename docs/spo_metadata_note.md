# SPO Metadata Note

`SparsePauliOp.translate(x, system_size)` currently relies on two pieces of
state that are not stored on the `SPO` object itself:

- `packbit`, taken from the backend-global utils configuration
- `system_size`, provided by the caller

This is acceptable for the current codebase because the project is effectively
using `packbit=32` throughout, and translation is explicitly defined on the
physical system size. However, it leaves a small long-term design risk:

- if an `SPO` is created under one `packbit` setting and translated after the
  global `packbit` changes, the packed layout could be misinterpreted
- callers must still carry `system_size` separately even though translation is a
  structural operation on the operator itself

Possible future cleanup:

- store `packbit` on `SparsePauliOp`
- optionally store the physical `system_size` on `SparsePauliOp`
- let translation validate against object-local metadata instead of backend-global state

This is a design improvement note, not a blocker for the current translation
feature.
