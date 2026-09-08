"""Aggregate independently saved Figure 7 R-SPD runs.

Re-run this script after adding runs to the input directory.  By
default it reports prefix ensembles of size 1, 2, 4, ..., R, where R is the
number of completed runs. Historical ``trajectory_*`` filenames are accepted
for compatibility; each file contains one run, not a single-Pauli trajectory.
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


def parse_run_counts(text):
    if text is None:
        return None
    try:
        counts = sorted({int(value.strip()) for value in text.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "run counts must be comma-separated integers"
        ) from error
    if not counts or counts[0] < 1:
        raise argparse.ArgumentTypeError("run counts must be positive")
    return counts


def default_run_counts(num_available):
    counts = []
    count = 1
    while count < num_available:
        counts.append(count)
        count *= 2
    counts.append(num_available)
    return counts


def load_runs(input_dir, max_runs=None):
    paths = sorted(input_dir.glob("trajectory_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No completed R-SPD runs found in {input_dir}")

    loaded = []
    expected_metadata = None
    seen_indices = set()
    for path in paths:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        metadata = data.get("metadata")
        if expected_metadata is None:
            expected_metadata = metadata
        elif metadata != expected_metadata:
            raise ValueError(f"Run metadata mismatch: {path}")

        trajectory_index = int(data["trajectory_index"])
        if trajectory_index in seen_indices:
            raise ValueError(f"Duplicate run index {trajectory_index}")
        seen_indices.add(trajectory_index)
        run = data["run"]
        if int(run["trajectory_index"]) != trajectory_index:
            raise ValueError(f"Run index mismatch: {path}")
        loaded.append((trajectory_index, run, path))

    loaded.sort(key=lambda item: item[0])
    if max_runs is not None:
        if max_runs < 1:
            raise ValueError("max runs must be positive")
        loaded = loaded[:max_runs]
    return expected_metadata, loaded


def load_deterministic_reference(path, num_values):
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict) and "all_results" in data:
        values = data["all_results"]
    else:
        values = data
    reference = np.asarray(values, dtype=float)
    if reference.ndim != 1:
        raise ValueError("Deterministic reference must be a one-dimensional sequence")
    if len(reference) < num_values:
        raise ValueError(
            f"Deterministic reference has {len(reference)} values; need {num_values}"
        )
    return reference[:num_values]


def aggregate_prefix(estimates, count, deterministic):
    sample = estimates[:count]
    mean = np.mean(sample, axis=0)
    if count > 1:
        sample_variance = np.var(sample, axis=0, ddof=1)
        standard_error = np.sqrt(sample_variance / count)
    else:
        sample_variance = np.full(mean.shape, np.nan)
        standard_error = np.full(mean.shape, np.nan)
    delta = mean - deterministic
    z_score = np.divide(
        delta,
        standard_error,
        out=np.full(mean.shape, np.nan),
        where=standard_error > 0.0,
    )
    inside_three_se = np.less_equal(np.abs(delta), 3.0 * standard_error)
    inside_three_se = np.where(np.isnan(standard_error), False, inside_three_se)
    return {
        "num_runs": count,
        # Compatibility key for statistics pickles written by the prototype.
        "num_populations": count,
        "mean": mean,
        "sample_variance": sample_variance,
        "standard_error": standard_error,
        "delta_from_deterministic": delta,
        "z_score": z_score,
        "inside_three_standard_errors": inside_three_se,
    }


def atomic_pickle_dump(data, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(data, handle)
    temporary_path.replace(path)


def write_csv(path, statistics):
    fieldnames = [
        "num_runs",
        "pauli_budget",
        "total_string_budget",
        "step",
        "time",
        "mean",
        "sample_variance",
        "standard_error",
        "deterministic",
        "delta_from_deterministic",
        "z_score",
        "inside_three_standard_errors",
    ]
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        pauli_budget = statistics["metadata"].get(
            "pauli_budget", statistics["metadata"]["population_size"]
        )
        dt = statistics["metadata"]["dt"]
        deterministic = statistics["deterministic"]
        for result in statistics["by_run_count"]:
            count = result["num_runs"]
            for step, mean in enumerate(result["mean"]):
                writer.writerow(
                    {
                        "num_runs": count,
                        "pauli_budget": pauli_budget,
                        "total_string_budget": count * pauli_budget,
                        "step": step,
                        "time": step * dt,
                        "mean": mean,
                        "sample_variance": result["sample_variance"][step],
                        "standard_error": result["standard_error"][step],
                        "deterministic": deterministic[step],
                        "delta_from_deterministic": result[
                            "delta_from_deterministic"
                        ][step],
                        "z_score": result["z_score"][step],
                        "inside_three_standard_errors": bool(
                            result["inside_three_standard_errors"][step]
                        ),
                    }
                )
    temporary_path.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate completed R-SPD runs and compare to SPD."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Deterministic pickle containing all_results, or a pickled 1D sequence.",
    )
    parser.add_argument(
        "--run-counts",
        "--population-counts",
        dest="run_counts",
        type=parse_run_counts,
        default=None,
        help="Comma-separated prefix sizes; default: 1,2,4,...,all.",
    )
    parser.add_argument(
        "--max-runs",
        "--max-populations",
        dest="max_runs",
        type=int,
        default=None,
        help="Use at most this many completed runs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path without extension; default: INPUT_DIR/statistics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata, run_files = load_runs(
        args.input_dir, max_runs=args.max_runs
    )
    indices = [item[0] for item in run_files]
    estimate_rows = [
        np.asarray(item[1]["estimates"], dtype=float) for item in run_files
    ]
    lengths = {len(row) for row in estimate_rows}
    if len(lengths) != 1:
        raise ValueError("Completed runs have inconsistent step counts")
    estimates = np.stack(estimate_rows)
    if not np.all(np.isfinite(estimates)):
        raise FloatingPointError("A run contains a non-finite estimate")

    num_available = len(run_files)
    run_counts = args.run_counts or default_run_counts(
        num_available
    )
    if run_counts[-1] > num_available:
        raise ValueError(
            f"Requested {run_counts[-1]} runs, but only "
            f"{num_available} are complete"
        )
    deterministic = load_deterministic_reference(args.reference, estimates.shape[1])
    by_count = [
        aggregate_prefix(estimates, count, deterministic)
        for count in run_counts
    ]
    statistics = {
        "metadata": metadata,
        "run_indices": tuple(indices),
        "run_paths": tuple(str(item[2]) for item in run_files),
        # Compatibility keys for readers of the first persisted schema.
        "trajectory_indices": tuple(indices),
        "trajectory_paths": tuple(str(item[2]) for item in run_files),
        "deterministic_reference_path": str(args.reference),
        "deterministic": deterministic,
        "estimates": estimates,
        "by_run_count": by_count,
        "by_population_count": by_count,
    }

    output_prefix = args.output_prefix or args.input_dir / "statistics"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pickle_path = output_prefix.with_suffix(".pkl")
    csv_path = output_prefix.with_suffix(".csv")
    atomic_pickle_dump(statistics, pickle_path)
    write_csv(csv_path, statistics)

    final_step = estimates.shape[1] - 1
    pauli_budget = metadata.get("pauli_budget", metadata["population_size"])
    print(
        f"loaded {num_available} runs with K={pauli_budget}; "
        f"final step={final_step}, deterministic={deterministic[-1]:.10g}"
    )
    for result in by_count:
        count = result["num_runs"]
        variance = result["sample_variance"][-1]
        standard_error = result["standard_error"][-1]
        print(
            f"R={count:6d} R*K={count * pauli_budget:12d} "
            f"mean={result['mean'][-1]: .10g} variance={variance:.6g} "
            f"SE={standard_error:.6g} delta={result['delta_from_deterministic'][-1]: .6g}"
        )
    print(f"wrote {pickle_path}")
    print(f"wrote {csv_path}")


# Compatibility aliases for callers of the first statistics helper API.
parse_population_counts = parse_run_counts
default_population_counts = default_run_counts


def load_trajectories(input_dir, max_populations=None):
    return load_runs(input_dir, max_runs=max_populations)


if __name__ == "__main__":
    main()
