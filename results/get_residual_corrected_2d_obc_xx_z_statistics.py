"""Aggregate Phase 3 diagnostics from residual-corrected R-SPD runs."""

import argparse
import csv
from math import sqrt
import pickle
from pathlib import Path

import numpy as np


def default_run_counts(num_available):
    counts = []
    count = 1
    while count < num_available:
        counts.append(count)
        count *= 2
    counts.append(num_available)
    return counts


def load_runs(input_dir):
    paths = sorted(input_dir.glob("run_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No completed residual-corrected runs in {input_dir}")

    metadata = None
    loaded = []
    seen = set()
    for path in paths:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        if metadata is None:
            metadata = data["metadata"]
        elif data["metadata"] != metadata:
            raise ValueError(f"Run metadata mismatch: {path}")
        run = data["run"]
        run_index = int(run["run_index"])
        if run_index in seen:
            raise ValueError(f"Duplicate run index: {run_index}")
        seen.add(run_index)
        loaded.append(run)
    return metadata, sorted(loaded, key=lambda run: run["run_index"])


def load_reference(path, step):
    with path.open("rb") as handle:
        data = pickle.load(handle)
    values = data["all_results"] if isinstance(data, dict) and "all_results" in data else data
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) <= step:
        raise ValueError(f"Reference must contain values through step {step}")
    return float(values[step])


def aggregate_ensemble(runs, count, reference=None):
    prefix = runs[:count]
    estimates = np.asarray([run["estimate"] for run in prefix], dtype=float)
    corrections = np.asarray(
        [run["correction_estimate"] for run in prefix], dtype=float
    )
    norm_squares = np.asarray(
        [run["correction_norm_sq"] for run in prefix], dtype=float
    )
    elapsed = np.asarray([run["elapsed_seconds"] for run in prefix], dtype=float)
    variance = float(np.var(estimates, ddof=1)) if count > 1 else float("nan")
    mean_elapsed = float(np.mean(elapsed))
    row = {
        "runs": count,
        "mean": float(np.mean(estimates)),
        "backbone_estimate": float(prefix[0]["backbone_estimate"]),
        "correction_mean": float(np.mean(corrections)),
        "sample_variance": variance,
        "standard_error": sqrt(variance / count) if count > 1 else float("nan"),
        "terminal_nonzero_hit_fraction": float(np.count_nonzero(corrections) / count),
        "estimate_q0": float(np.quantile(estimates, 0.0)),
        "estimate_q25": float(np.quantile(estimates, 0.25)),
        "estimate_q50": float(np.quantile(estimates, 0.5)),
        "estimate_q75": float(np.quantile(estimates, 0.75)),
        "estimate_q100": float(np.quantile(estimates, 1.0)),
        "correction_q0": float(np.quantile(corrections, 0.0)),
        "correction_q25": float(np.quantile(corrections, 0.25)),
        "correction_q50": float(np.quantile(corrections, 0.5)),
        "correction_q75": float(np.quantile(corrections, 0.75)),
        "correction_q100": float(np.quantile(corrections, 1.0)),
        "correction_norm_sq_min": float(np.min(norm_squares)),
        "correction_norm_sq_median": float(np.median(norm_squares)),
        "correction_norm_sq_max": float(np.max(norm_squares)),
        "mean_elapsed_seconds": mean_elapsed,
        "variance_times_seconds": variance * mean_elapsed,
    }
    if reference is not None:
        row.update(
            reference=reference,
            backbone_bias=row["backbone_estimate"] - reference,
            corrected_bias=row["mean"] - reference,
            rmse=float(np.sqrt(np.mean((estimates - reference) ** 2))),
        )
    return row


def aggregate_steps(runs, metadata):
    gates_per_step = int(metadata["num_gates_per_step"])
    correction_budget = int(metadata["correction_budget"])
    rows = []
    for step in range(1, int(metadata["num_steps"]) + 1):
        end_norm_squares = []
        predicted_mse = []
        truncation_counts = []
        heavy_counts = []
        heavy_fractions = []
        maximum_candidates = []
        start = 1 + (step - 1) * gates_per_step
        stop = 1 + step * gates_per_step
        for run in runs:
            records = run["diagnostics"].records[start:stop]
            if len(records) != gates_per_step:
                raise ValueError("Run diagnostics do not match the circuit metadata")
            end_norm_squares.append(records[-1].sampled_correction_l2_norm_sq)
            truncations = [
                record.randomized_truncation_stats
                for record in records
                if record.randomized_truncation_stats.support_before
                > correction_budget
            ]
            predicted_mse.append(sum(item.predicted_mse for item in truncations))
            truncation_counts.append(len(truncations))
            heavy_counts.extend(item.heavy_count for item in truncations)
            heavy_fractions.extend(
                item.heavy_count / correction_budget for item in truncations
            )
            maximum_candidates.append(
                max(record.correction_candidate_support for record in records)
            )
        rows.append(
            {
                "step": step,
                "correction_norm_sq_min": float(np.min(end_norm_squares)),
                "correction_norm_sq_median": float(np.median(end_norm_squares)),
                "correction_norm_sq_max": float(np.max(end_norm_squares)),
                "heavy_count_median": (
                    float(np.median(heavy_counts)) if heavy_counts else float("nan")
                ),
                "heavy_fraction_mean": (
                    float(np.mean(heavy_fractions))
                    if heavy_fractions
                    else float("nan")
                ),
                "predicted_mse_median": float(np.median(predicted_mse)),
                "randomized_truncations_median": float(
                    np.median(truncation_counts)
                ),
                "maximum_correction_candidate_support": int(
                    np.max(maximum_candidates)
                ),
            }
        )
    return rows


def write_csv(path, rows):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 3 residual-corrected R-SPD diagnostics."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional deterministic reference pickle containing all_results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata, runs = load_runs(args.input_dir)
    reference = (
        load_reference(args.reference, metadata["num_steps"])
        if args.reference is not None
        else None
    )
    ensemble_rows = [
        aggregate_ensemble(runs, count, reference)
        for count in default_run_counts(len(runs))
    ]
    step_rows = aggregate_steps(runs, metadata)
    write_csv(args.input_dir / "phase3_ensemble.csv", ensemble_rows)
    write_csv(args.input_dir / "phase3_steps.csv", step_rows)
    with (args.input_dir / "phase3_statistics.pkl").open("wb") as handle:
        pickle.dump(
            {
                "metadata": metadata,
                "reference": reference,
                "ensemble": ensemble_rows,
                "steps": step_rows,
            },
            handle,
        )

    final = ensemble_rows[-1]
    print(
        f"R={final['runs']}: mean={final['mean']:.10g}, "
        f"variance={final['sample_variance']:.6g}, "
        f"SE={final['standard_error']:.6g}, "
        f"hit_fraction={final['terminal_nonzero_hit_fraction']:.3g}"
    )
    print(f"wrote {args.input_dir / 'phase3_ensemble.csv'}")
    print(f"wrote {args.input_dir / 'phase3_steps.csv'}")


if __name__ == "__main__":
    main()
