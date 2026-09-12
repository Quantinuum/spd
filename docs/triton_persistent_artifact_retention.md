# Experiment artifact retention (2026-09-11)

The persistent-storage experiments remain on `experiment/triton-persistent-options`.
No main integration or commit was performed during this cleanup.

Retain source code, tests, benchmark/reproduction scripts, experiment notes,
per-run JSON metadata and timing samples, correctness-comparison JSONs, validation
source hashes, profile kernel summaries, and useful logs. These are the reviewable
record of the experiments.

Removed 37 generated artifacts totaling **36,601,095,973 bytes (36.60 decimal GB)**:
full NumPy state snapshots under `persistent_generalization_20260910`, and the
two full Chrome profiler traces under `persistent_generalization_20260911_profile`.
The exact paths, sizes and SHA-256 hashes are retained in
`benchmarks/results/persistent_artifact_retention_20260911.json`. None was tracked
by Git. Other result directories and user optimization outputs were not removed.

The full benchmark-results tree is now approximately 50 MiB. Git exclusions cover
new `.npz` snapshots and full trace JSONs under `benchmarks/results`; compact JSON
measurements and summaries remain eligible for committing. Small historical
pickle timing records are retained, because they contain original measurements.

Full coefficient comparisons and trace inspection require regenerating the raw
artifacts using the recorded arguments and source versions. Existing compact
comparison records document completed checks, not checks repeated after cleanup.
The generalization summarizer preserves prior coefficient results when matching
run metadata is unchanged and snapshots are absent, explicitly labelling them
as retained results. To perform a new validation, regenerate snapshots.
