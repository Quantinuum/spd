"""Run appendable residual-corrected R-SPD realizations for 2D OBC XX+Z.

Every completed run is stored independently as ``run_XXXXXX.pkl``. Increasing
``--runs`` starts only missing run indices. Seeds depend only on the master
seed and run index, so requested ensemble size and worker count do not change
existing realizations.

Each realization currently executes its full circuit in one call. Completed
runs are reusable, but an interrupted realization restarts from its beginning.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import sqrt
import pickle
import time
from pathlib import Path

import numpy as np

import spd
from spd.circuit_ir import CircuitIR
from spd.pytket_frontend import parse_pytket_circuit
from spd.residual_corrected import ALGORITHM_VERSION

try:
    from .benchmark_rspd_2d_obc_xx_z_stepwise import (
        PACKBIT,
        build_initial_observable,
        build_one_step_circuit,
        neighbor_list,
        padded_system_size,
    )
except ImportError:
    from benchmark_rspd_2d_obc_xx_z_stepwise import (
        PACKBIT,
        build_initial_observable,
        build_one_step_circuit,
        neighbor_list,
        padded_system_size,
    )


SCRIPT_VERSION = "residual-corrected-2d-obc-xx-z-v1"


def run_seed(master_seed, run_index):
    return int(
        np.random.SeedSequence(master_seed, spawn_key=(run_index,)).generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )


def initialize_worker_precision(precision):
    spd.numpy_backend.set_precision(precision)


def run_one_realization(
    run_index,
    observable,
    circuit_ir,
    backbone_budget,
    correction_budget,
    master_seed,
    basis,
    numerical_zero_tolerance,
):
    seed = run_seed(master_seed, run_index)
    started = time.perf_counter()
    result = spd.run_residual_corrected_spd(
        observable,
        circuit_ir,
        backbone_budget,
        correction_budget,
        seed=seed,
        basis=basis,
        numerical_zero_tolerance=numerical_zero_tolerance,
    )
    elapsed = time.perf_counter() - started
    evolution = result.evolution
    return {
        "run_index": run_index,
        "seed": seed,
        "estimate": result.estimate,
        "raw_estimate": result.raw_estimate,
        "backbone_estimate": result.backbone_estimate,
        "correction_estimate": result.correction_estimate,
        "imaginary_leakage": result.imaginary_leakage,
        "backbone_support": len(evolution.final_backbone),
        "correction_support": len(evolution.final_sampled_correction),
        "backbone_norm_sq": float(evolution.final_backbone.get_norm_square()),
        "correction_norm_sq": float(
            evolution.final_sampled_correction.get_norm_square()
        ),
        "elapsed_seconds": elapsed,
        "diagnostics": result.diagnostics,
    }


def _run_path(output_dir, run_index):
    return output_dir / f"run_{run_index:06d}.pkl"


def _atomic_pickle(path, data):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(data, handle)
    temporary_path.replace(path)


def save_run(output_dir, metadata, run):
    path = _run_path(output_dir, run["run_index"])
    _atomic_pickle(path, {"metadata": metadata, "run": run})
    return path


def load_runs(output_dir, metadata):
    runs = []
    seen = set()
    for path in sorted(output_dir.glob("run_*.pkl")):
        with path.open("rb") as handle:
            data = pickle.load(handle)
        if data.get("metadata") != metadata:
            raise ValueError(f"Run metadata mismatch: {path}")
        run = data["run"]
        run_index = int(run["run_index"])
        if run_index in seen:
            raise ValueError(f"Duplicate run index: {run_index}")
        seen.add(run_index)
        runs.append(run)
    return sorted(runs, key=lambda run: run["run_index"])


def _prefix_counts(total):
    counts = []
    count = 1
    while count <= total:
        counts.append(count)
        count *= 2
    if total and counts[-1] != total:
        counts.append(total)
    return counts


def write_statistics(output_dir, metadata, runs):
    rows = []
    for count in _prefix_counts(len(runs)):
        prefix = runs[:count]
        estimates = np.asarray([run["estimate"] for run in prefix], dtype=np.float64)
        corrections = np.asarray(
            [run["correction_estimate"] for run in prefix],
            dtype=np.float64,
        )
        sample_variance = (
            float(np.var(estimates, ddof=1, dtype=np.float64))
            if count > 1
            else float("nan")
        )
        rows.append(
            {
                "runs": count,
                "mean": float(np.mean(estimates, dtype=np.float64)),
                "backbone_estimate": float(prefix[0]["backbone_estimate"]),
                "correction_mean": float(np.mean(corrections, dtype=np.float64)),
                "sample_variance": sample_variance,
                "standard_error": (
                    sqrt(sample_variance / count) if count > 1 else float("nan")
                ),
                "min_estimate": float(np.min(estimates)),
                "max_estimate": float(np.max(estimates)),
                "mean_elapsed_seconds": float(
                    np.mean([run["elapsed_seconds"] for run in prefix])
                ),
            }
        )

    _atomic_pickle(
        output_dir / "statistics.pkl",
        {"metadata": metadata, "rows": rows},
    )
    csv_path = output_dir / "statistics.csv"
    temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(csv_path)
    return rows


def default_output_dir(args):
    name = (
        f"residual_corrected_2d_obc_xx_z_n{args.n}_Kd{args.backbone_budget}_"
        f"Kc{args.correction_budget}_steps{args.num_steps}_seed{args.master_seed}"
    )
    return Path(__file__).resolve().parents[1] / "results" / name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run appendable residual-corrected 2D OBC XX+Z realizations."
    )
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument("--h", type=float, default=3.044382)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--backbone-budget", type=int, default=1_000)
    parser.add_argument("--correction-budget", type=int, default=1_000)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--additional-runs", type=int, default=None)
    parser.add_argument("--run-index", type=int, default=None)
    parser.add_argument("--master-seed", type=int, default=20260818)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--precision", choices=["single", "double"], default="double")
    parser.add_argument("--basis", choices=["0", "Z", "+", "X"], default="Z")
    parser.add_argument("--numerical-zero-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(
        args.n,
        args.num_steps,
        args.backbone_budget,
        args.correction_budget,
        args.workers,
    ) < 1:
        raise ValueError("n, steps, budgets, and workers must be positive.")
    if args.run_index is not None and args.run_index < 0:
        raise ValueError("run index must be nonnegative.")
    if args.run_index is not None and args.additional_runs is not None:
        raise ValueError("Choose either --run-index or --additional-runs.")
    if args.numerical_zero_tolerance < 0.0:
        raise ValueError("numerical-zero tolerance must be nonnegative.")

    num_qubits, pairs = neighbor_list(args.n)
    one_step_ir = parse_pytket_circuit(
        build_one_step_circuit(args.n, args.h, args.dt),
        padded_system_size(num_qubits),
    )
    circuit_ir = CircuitIR(
        one_step_ir.system_size,
        one_step_ir.operations * args.num_steps,
    )
    observable = spd.create_spo(
        build_initial_observable(args.n),
        backend_name="numpy",
        precision=args.precision,
    )
    metadata = {
        "script_version": SCRIPT_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "n": args.n,
        "h": args.h,
        "dt": args.dt,
        "num_steps": args.num_steps,
        "num_qubits": num_qubits,
        "num_gates_per_step": len(pairs) + num_qubits,
        "backbone_budget": args.backbone_budget,
        "correction_budget": args.correction_budget,
        "master_seed": args.master_seed,
        "precision": args.precision,
        "basis": args.basis,
        "numerical_zero_tolerance": args.numerical_zero_tolerance,
    }

    output_dir = args.output_dir or default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_runs = load_runs(output_dir, metadata)
    completed = {run["run_index"] for run in completed_runs}

    if args.run_index is not None:
        requested = [args.run_index]
    elif args.additional_runs is not None:
        if args.additional_runs < 1:
            raise ValueError("additional-runs must be positive.")
        start = max(completed, default=-1) + 1
        requested = list(range(start, start + args.additional_runs))
    else:
        if args.runs < 1:
            raise ValueError("runs must be positive.")
        requested = list(range(args.runs))
    pending = [run_index for run_index in requested if run_index not in completed]

    print(
        f"residual-corrected R-SPD: {args.n}x{args.n}, steps={args.num_steps}, "
        f"Kd={args.backbone_budget}, Kc={args.correction_budget}, "
        f"completed={len(completed)}, pending={len(pending)}"
    )
    print(f"output directory: {output_dir}")

    def save_completed(run):
        path = save_run(output_dir, metadata, run)
        print(
            "saved run {index} to {path} "
            "(estimate={estimate:.10g}, correction_norm_sq={norm:.6g}, "
            "elapsed={elapsed:.2f}s)".format(
                index=run["run_index"],
                path=path,
                estimate=run["estimate"],
                norm=run["correction_norm_sq"],
                elapsed=run["elapsed_seconds"],
            ),
            flush=True,
        )

    if args.workers == 1:
        for run_index in pending:
            save_completed(
                run_one_realization(
                    run_index,
                    observable,
                    circuit_ir,
                    args.backbone_budget,
                    args.correction_budget,
                    args.master_seed,
                    args.basis,
                    args.numerical_zero_tolerance,
                )
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker_precision,
            initargs=(args.precision,),
        ) as executor:
            futures = {
                executor.submit(
                    run_one_realization,
                    run_index,
                    observable,
                    circuit_ir,
                    args.backbone_budget,
                    args.correction_budget,
                    args.master_seed,
                    args.basis,
                    args.numerical_zero_tolerance,
                ): run_index
                for run_index in pending
            }
            for future in as_completed(futures):
                save_completed(future.result())

    all_runs = load_runs(output_dir, metadata)
    rows = write_statistics(output_dir, metadata, all_runs)
    final = rows[-1]
    print(
        "R={runs}: mean={mean:.10g}, standard_error={se:.6g}, "
        "sample_variance={variance:.6g}".format(
            runs=final["runs"],
            mean=final["mean"],
            se=final["standard_error"],
            variance=final["sample_variance"],
        )
    )


if __name__ == "__main__":
    main()
