"""Run an appendable one-Trotter-step-per-block Phase 4 pilot."""

import argparse
import csv
from math import sqrt
import pickle
from pathlib import Path
import time

import numpy as np

import spd
from spd.circuit_ir import CircuitIR, SkippedOperation
from spd.pytket_frontend import parse_pytket_circuit
from spd.residual_corrected import _combine_zero_stats
from spd.time_blocked_residual import (
    ALGORITHM_VERSION,
    _prepare_time_blocked_backbone,
    _run_time_blocked_residual_block,
    _warn_numerical_zeros,
    summarize_time_blocked_block_runs,
    time_blocked_run_seed,
)

try:
    from .benchmark_rspd_2d_obc_xx_z_stepwise import (
        build_initial_observable,
        build_one_step_circuit,
        neighbor_list,
        padded_system_size,
    )
except ImportError:
    from benchmark_rspd_2d_obc_xx_z_stepwise import (
        build_initial_observable,
        build_one_step_circuit,
        neighbor_list,
        padded_system_size,
    )


SCRIPT_VERSION = "time-blocked-residual-2d-obc-xx-z-v1"


def _atomic_pickle(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(data, handle)
    temporary.replace(path)


def _run_path(output_dir, block_index, run_index):
    return output_dir / "pilot" / f"block_{block_index:04d}" / f"run_{run_index:06d}.pkl"


def _save_run(output_dir, metadata, block_index, run_index, elapsed, result):
    path = _run_path(output_dir, block_index, run_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_pickle(
        path,
        {
            "metadata": metadata,
            "block_index": block_index,
            "run_index": run_index,
            "elapsed_seconds": elapsed,
            "result": result,
        },
    )
    return path


def _load_runs(output_dir, metadata):
    runs = {}
    for path in sorted((output_dir / "pilot").glob("block_*/run_*.pkl")):
        with path.open("rb") as handle:
            data = pickle.load(handle)
        if data.get("metadata") != metadata:
            raise ValueError(f"Pilot metadata mismatch: {path}")
        key = (int(data["block_index"]), int(data["run_index"]))
        if key in runs:
            raise ValueError(f"Duplicate pilot run: {key}")
        runs[key] = data
    return runs


def _write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_pilot_summary(output_dir, metadata, plan, runs):
    summaries = []
    costs = []
    rows = []
    for block_index in range(plan.num_blocks):
        block_data = sorted(
            (data for (block, _), data in runs.items() if block == block_index),
            key=lambda data: data["run_index"],
        )
        if not block_data:
            continue
        summary = summarize_time_blocked_block_runs(
            data["result"] for data in block_data
        )
        elapsed = np.asarray(
            [data["elapsed_seconds"] for data in block_data], dtype=np.float64
        )
        summaries.append(summary)
        costs.append(float(np.mean(elapsed)))
        diagnostics = [data["result"].diagnostics for data in block_data]
        rows.append(
            {
                "block_index": block_index,
                "reverse_gate_start": plan.block_starts[block_index],
                "reverse_gate_stop": (
                    plan.block_starts[block_index] + plan.block_sizes[block_index]
                ),
                "correction_budget": summary.correction_budget,
                "runs": len(summary.runs),
                "mean": summary.mean,
                "second_moment": summary.second_moment,
                "sample_variance": summary.sample_variance,
                "standard_error": summary.standard_error,
                "terminal_hit_fraction": summary.terminal_hit_fraction,
                "false_convergence": summary.false_convergence,
                "mean_elapsed_seconds": costs[-1],
                "max_sampled_correction_norm_sq": max(
                    item.max_sampled_correction_l2_norm_sq for item in diagnostics
                ),
                "max_norm_inflation_ratio": max(
                    item.max_norm_inflation_ratio for item in diagnostics
                ),
                "min_heavy_fraction": min(
                    item.min_heavy_fraction for item in diagnostics
                ),
                "first_degeneracy_warning_gate": min(
                    (
                        item.first_degeneracy_warning_gate
                        for item in diagnostics
                        if item.first_degeneracy_warning_gate is not None
                    ),
                    default="",
                ),
                "max_candidate_support": max(
                    item.max_correction_candidate_support for item in diagnostics
                ),
            }
        )

    complete = len(summaries) == plan.num_blocks
    allocation_blocked = any(summary.false_convergence for summary in summaries)

    if complete:
        raw_backbone = spd.estimate_expectation(plan.final_backbone, basis="Z")
        correction = sum(summary.mean for summary in summaries)
        variances = [
            summary.sample_variance / len(summary.runs) for summary in summaries
        ]
        standard_error = (
            sqrt(sum(variances)) if all(np.isfinite(variances)) else float("nan")
        )
        total = {
            "backbone_estimate": float(raw_backbone.real),
            "correction_estimate": correction,
            "corrected_estimate": float(raw_backbone.real) + correction,
            "standard_error": standard_error,
            "any_false_convergence": any(
                summary.false_convergence for summary in summaries
            ),
        }
    else:
        total = None

    _write_csv(output_dir / "pilot_blocks.csv", rows)
    _atomic_pickle(
        output_dir / "pilot_summary.pkl",
        {
            "metadata": metadata,
            "block_summaries": summaries,
            "allocation_blocked_by_false_convergence": allocation_blocked,
            "total": total,
        },
    )
    return rows, total


def default_output_dir(args):
    name = (
        f"time_blocked_residual_2d_obc_xx_z_n{args.n}_"
        f"Kd{args.backbone_budget}_Kc{args.correction_budget}_"
        f"steps{args.num_steps}_seed{args.master_seed}"
    )
    return Path(__file__).resolve().parents[1] / "results" / name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an appendable time-blocked residual-correction pilot."
    )
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument("--h", type=float, default=3.044382)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--backbone-budget", type=int, default=1_000)
    parser.add_argument("--correction-budget", type=int, default=1_000)
    parser.add_argument("--pilot-runs", type=int, default=4)
    parser.add_argument(
        "--blocks",
        type=str,
        default=None,
        help="Comma-separated reverse-time block indices; default is every block.",
    )
    parser.add_argument("--master-seed", type=int, default=20260819)
    parser.add_argument("--precision", choices=["single", "double"], default="double")
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
        args.pilot_runs,
    ) < 1:
        raise ValueError("n, steps, budgets, and pilot runs must be positive.")
    if args.numerical_zero_tolerance < 0.0:
        raise ValueError("numerical-zero tolerance must be nonnegative.")
    if args.blocks is None:
        requested_blocks = tuple(range(args.num_steps))
    else:
        requested_blocks = tuple(int(value) for value in args.blocks.split(","))
        if (
            not requested_blocks
            or len(set(requested_blocks)) != len(requested_blocks)
            or any(not 0 <= value < args.num_steps for value in requested_blocks)
        ):
            raise ValueError("blocks must be distinct indices in [0, num_steps).")

    spd.numpy_backend.set_precision(args.precision)
    num_qubits, pairs = neighbor_list(args.n)
    one_step_ir = parse_pytket_circuit(
        build_one_step_circuit(args.n, args.h, args.dt),
        padded_system_size(num_qubits),
    )
    active_per_step = sum(
        not isinstance(operation, SkippedOperation)
        for operation in one_step_ir.operations
    )
    expected_per_step = len(pairs) + num_qubits
    if active_per_step != expected_per_step:
        raise ValueError(
            f"Expected {expected_per_step} active gates per step, got {active_per_step}."
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
    block_sizes = (active_per_step,) * args.num_steps
    metadata = {
        "script_version": SCRIPT_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "n": args.n,
        "h": args.h,
        "dt": args.dt,
        "num_steps": args.num_steps,
        "num_qubits": num_qubits,
        "active_gates_per_step": active_per_step,
        "backbone_budget": args.backbone_budget,
        "correction_budget": args.correction_budget,
        "master_seed": args.master_seed,
        "precision": args.precision,
        "basis": "Z",
        "numerical_zero_tolerance": args.numerical_zero_tolerance,
        "block_order": "reverse Heisenberg; block 0 is the last physical step",
    }

    output_dir = args.output_dir or default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    backbone_path = output_dir / "backbone.pkl"
    if backbone_path.exists():
        with backbone_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint.get("metadata") != metadata:
            raise ValueError(f"Backbone metadata mismatch: {backbone_path}")
        plan = checkpoint["plan"]
        print(f"loaded deterministic backbone: {backbone_path}")
    else:
        started = time.perf_counter()
        plan = _prepare_time_blocked_backbone(
            observable,
            circuit_ir,
            args.backbone_budget,
            block_sizes,
            numerical_zero_tolerance=args.numerical_zero_tolerance,
            rebase=False,
            emit_warning=False,
        )
        _atomic_pickle(
            backbone_path,
            {
                "metadata": metadata,
                "elapsed_seconds": time.perf_counter() - started,
                "plan": plan,
            },
        )
        print(f"saved deterministic backbone: {backbone_path}", flush=True)

    completed = _load_runs(output_dir, metadata)
    for block_index in requested_blocks:
        for run_index in range(args.pilot_runs):
            key = (block_index, run_index)
            if key in completed:
                continue
            seed = time_blocked_run_seed(
                args.master_seed,
                block_index,
                args.correction_budget,
                run_index,
                stage=0,
            )
            started = time.perf_counter()
            result = _run_time_blocked_residual_block(
                plan,
                block_index,
                args.correction_budget,
                seed=seed,
                basis="Z",
                state_overlap=None,
                emit_warning=False,
            )
            elapsed = time.perf_counter() - started
            path = _save_run(
                output_dir, metadata, block_index, run_index, elapsed, result
            )
            completed[key] = {
                "metadata": metadata,
                "block_index": block_index,
                "run_index": run_index,
                "elapsed_seconds": elapsed,
                "result": result,
            }
            print(
                f"saved block {block_index} run {run_index}: {path} "
                f"(estimate={result.correction_estimate:.8g}, "
                f"hit={result.terminal_hit}, elapsed={elapsed:.2f}s)",
                flush=True,
            )

    zero_stats = _combine_zero_stats(
        plan.numerical_zero_stats,
        *(data["result"].diagnostics.numerical_zero_stats for data in completed.values()),
    )
    _warn_numerical_zeros(zero_stats, args.numerical_zero_tolerance)
    _, total = write_pilot_summary(output_dir, metadata, plan, completed)
    print(f"wrote {output_dir / 'pilot_blocks.csv'}")
    if total is not None:
        print(
            "pilot corrected estimate={:.10g}, standard error={:.3g}, "
            "false convergence={}".format(
                total["corrected_estimate"],
                total["standard_error"],
                total["any_false_convergence"],
            )
        )


if __name__ == "__main__":
    main()
