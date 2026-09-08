"""Generate independent R-SPD runs for the Figure 7 benchmark.

Every run is saved independently after it finishes. Worker-local state is
checkpointed after every timestep, so interrupted runs resume with the same
per-step seeds. Increasing ``--runs`` later only starts missing run indices.

The historical filename and persisted ``trajectory_*`` schema are retained so
existing benchmark directories remain resumable. In R-SPD terminology these
files contain independent runs, not single-Pauli trajectories.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import pickle
import time
from pathlib import Path

import numpy as np
from pytket import Circuit

import spd
from spd.pytket_frontend import parse_pytket_circuit
from spd.randomized import ALGORITHM_VERSION


PACKBIT = 32
SCRIPT_VERSION = "fpspp-2d-obc-xx-z-trajectories-v1"


def neighbor_list(n):
    """Return OBC nearest-neighbor pairs for an ``n x n`` square lattice."""
    num_qubits = n * n
    pairs = []
    for row in range(n):
        for col in range(n):
            qubit = row * n + col
            if col + 1 < n:
                pairs.append((qubit, qubit + 1))
            if row + 1 < n:
                pairs.append((qubit, qubit + n))
    return num_qubits, pairs


def build_one_step_circuit(n, h, dt):
    """Build one Trotter step with the Figure 7 gate convention."""
    num_qubits, pairs = neighbor_list(n)
    circuit = Circuit(num_qubits)
    xx_parameter = (2.0 * dt * -1.0) / math.pi
    z_parameter = (2.0 * dt * -h) / math.pi
    for qubit in range(num_qubits):
        circuit.Rz(z_parameter, qubit)
    circuit.add_barrier(list(range(num_qubits)))
    for q0, q1 in pairs:
        circuit.XXPhase(xx_parameter, q0, q1)
    return circuit


def build_initial_observable(n):
    """Return the center-site Z observable used by the Figure 7 benchmark."""
    num_qubits = n * n
    center_qubit = (num_qubits - 1) // 2
    pauli = ["I"] * num_qubits
    pauli[center_qubit] = "Z"
    return {"".join(pauli): 1.0}


def padded_system_size(num_qubits):
    return PACKBIT * ((num_qubits + PACKBIT - 1) // PACKBIT)


def run_step_seeds(master_seed, run_index, num_steps):
    """Derive seeds that do not depend on the requested ensemble size."""
    run_stream = np.random.SeedSequence(
        master_seed,
        spawn_key=(run_index,),
    )
    return tuple(
        int(step_stream.generate_state(1, dtype=np.uint64)[0])
        for step_stream in run_stream.spawn(num_steps)
    )


def initialize_worker_precision(precision):
    """Set NumPy precision before a worker unpickles an SPO task argument."""
    spd.numpy_backend.set_precision(precision)


def _new_run(run_index, step_seeds, initial_observable):
    return {
        # Historical persisted key retained for existing result directories.
        "trajectory_index": run_index,
        "step_seeds": tuple(step_seeds),
        "estimates": [float(initial_observable.get_expectation_value(basis="Z"))],
        "supports": [initial_observable.get_size()],
        "norm_squares": [float(initial_observable.get_norm_square())],
        "elapsed_seconds": [],
        "throughput_work": [],
        "compression_counts": [],
        "predicted_mse": [],
        "raw_children": [],
        "merged_children": [],
        "compression_records": [],
    }


def _save_progress(progress_path, metadata, current_population, run):
    progress = {
        "metadata": metadata,
        "current_population": current_population,
        "run": run,
    }
    temporary_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(progress, handle)
    temporary_path.replace(progress_path)


def run_one_realization(
    run_index,
    initial_observable,
    operations,
    pauli_budget,
    step_seeds,
    num_gates,
    metadata,
    progress_path,
    quiet_steps,
):
    """Evolve one independent R-SPD realization through every timestep."""
    if progress_path.exists():
        with progress_path.open("rb") as handle:
            progress = pickle.load(handle)
        if progress.get("metadata") != metadata:
            raise ValueError(f"Checkpoint metadata mismatch: {progress_path}")
        current = progress["current_population"]
        run = progress["run"]
        if (
            run.get("trajectory_index") != run_index
            or tuple(run.get("step_seeds", ())) != tuple(step_seeds)
        ):
            raise ValueError(f"Checkpoint run mismatch: {progress_path}")
        start_step = len(run["elapsed_seconds"])
        print(
            f"resuming run {run_index} at step {start_step + 1}",
            flush=True,
        )
    else:
        current = initial_observable
        run = _new_run(run_index, step_seeds, initial_observable)
        start_step = 0

    for step_index in range(start_step, len(step_seeds)):
        support_before = current.get_size()
        start = time.perf_counter()
        result = spd.run_randomized_spd(
            current,
            operations,
            pauli_budget,
            seed=step_seeds[step_index],
            basis="Z",
        )
        elapsed = time.perf_counter() - start
        current = result.final_sampled_spo
        diagnostics = result.diagnostics

        run["estimates"].append(result.estimate)
        run["supports"].append(current.get_size())
        run["norm_squares"].append(float(current.get_norm_square()))
        run["elapsed_seconds"].append(elapsed)
        run["throughput_work"].append(
            0.5 * (support_before + current.get_size()) * num_gates
        )
        run["compression_counts"].append(diagnostics.num_compressions)
        run["predicted_mse"].append(diagnostics.accumulated_predicted_mse)
        run["raw_children"].append(diagnostics.total_raw_children)
        run["merged_children"].append(diagnostics.total_merged_children)
        run["compression_records"].append(diagnostics.compression_records)
        _save_progress(progress_path, metadata, current, run)

        if not quiet_steps:
            print(
                "run {run_index} step {step}: estimate={estimate:.10g} "
                "support={support} compressions={compressions} elapsed={elapsed:.3f}s".format(
                    run_index=run_index,
                    step=step_index + 1,
                    estimate=result.estimate,
                    support=current.get_size(),
                    compressions=diagnostics.num_compressions,
                    elapsed=elapsed,
                ),
                flush=True,
            )
    return run


def _run_path(output_dir, run_index):
    # Historical filename retained for resumability of active result directories.
    return output_dir / f"trajectory_{run_index:06d}.pkl"


def _progress_path(output_dir, run_index):
    return output_dir / ".progress" / f"trajectory_{run_index:06d}.pkl"


def save_run(output_dir, metadata, run):
    """Atomically save one completed R-SPD run independently of the others."""
    run_index = run["trajectory_index"]
    output_path = _run_path(output_dir, run_index)
    data = {
        "metadata": metadata,
        "trajectory_index": run_index,
        "run": run,
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(data, handle)
    temporary_path.replace(output_path)
    _progress_path(output_dir, run_index).unlink(missing_ok=True)
    return output_path


def load_completed_indices(output_dir, metadata):
    completed = set()
    for path in sorted(output_dir.glob("trajectory_*.pkl")):
        with path.open("rb") as handle:
            data = pickle.load(handle)
        if data.get("metadata") != metadata:
            raise ValueError(f"Run metadata mismatch: {path}")
        run_index = int(data["trajectory_index"])
        if run_index in completed:
            raise ValueError(f"Duplicate run index: {run_index}")
        completed.add(run_index)
    return completed


def default_output_dir(args, num_steps):
    name = (
        f"rspd_2d_obc_xx_z_n{args.n}_K{args.pauli_budget}_"
        f"steps{num_steps}_dt{args.dt}_seed{args.master_seed}"
    )
    return Path(__file__).resolve().parents[1] / "results" / name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate appendable NumPy R-SPD runs for Figure 7."
    )
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument("--h", type=float, default=3.044382)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--total-t", type=float, default=0.92)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument(
        "--pauli-budget",
        "--population-size",
        dest="pauli_budget",
        type=int,
        default=100_000,
        help="Persistent Pauli-string budget K.",
    )
    parser.add_argument(
        "--runs",
        "--num-populations",
        dest="runs",
        type=int,
        default=1,
        help="Ensure independent run indices 0 through R-1 exist.",
    )
    parser.add_argument(
        "--additional-runs",
        "--additional-populations",
        dest="additional_runs",
        type=int,
        default=None,
        help="Append this many runs after the largest existing index.",
    )
    parser.add_argument(
        "--run-index",
        "--trajectory-index",
        dest="run_index",
        type=int,
        default=None,
        help="Execute only this run index, useful for external schedulers.",
    )
    parser.add_argument("--master-seed", type=int, default=20260812)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--precision", choices=["single", "double"], default="double")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quiet-steps", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n < 1 or args.pauli_budget < 1 or args.workers < 1:
        raise ValueError("n, Pauli-string budget, and workers must be positive.")
    if args.run_index is not None and args.run_index < 0:
        raise ValueError("run index must be nonnegative.")
    if args.run_index is not None and args.additional_runs is not None:
        raise ValueError("Choose either --run-index or --additional-runs.")

    num_steps = int(args.total_t / args.dt) if args.num_steps is None else args.num_steps
    if num_steps < 1:
        raise ValueError("number of steps must be positive.")

    num_qubits, pairs = neighbor_list(args.n)
    num_gates = len(pairs) + num_qubits
    expected_num_gates = 2 * args.n * (args.n - 1) + args.n * args.n
    if num_gates != expected_num_gates:
        raise AssertionError(f"Expected {expected_num_gates} gates, got {num_gates}.")

    operations = parse_pytket_circuit(
        build_one_step_circuit(args.n, args.h, args.dt),
        padded_system_size(num_qubits),
    )
    initial_observable = spd.create_spo(
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
        "total_t": args.total_t,
        "num_steps": num_steps,
        "num_qubits": num_qubits,
        "num_gates_per_step": num_gates,
        # Historical persisted key retained for active benchmark checkpoints.
        "population_size": args.pauli_budget,
        "master_seed": args.master_seed,
        "precision": args.precision,
    }

    output_dir = args.output_dir or default_output_dir(args, num_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".progress").mkdir(parents=True, exist_ok=True)
    completed = load_completed_indices(output_dir, metadata)
    known_indices = completed | {
        int(path.stem.split("_")[-1])
        for path in (output_dir / ".progress").glob("trajectory_*.pkl")
    }

    if args.run_index is not None:
        requested_indices = [args.run_index]
    elif args.additional_runs is not None:
        if args.additional_runs < 1:
            raise ValueError("additional runs must be positive.")
        start_index = max(known_indices, default=-1) + 1
        requested_indices = list(
            range(start_index, start_index + args.additional_runs)
        )
    else:
        if args.runs < 1:
            raise ValueError("runs must be positive.")
        requested_indices = list(range(args.runs))

    pending_indices = [index for index in requested_indices if index not in completed]
    print(
        f"Figure 7 R-SPD runs: {args.n}x{args.n}, K={args.pauli_budget}, "
        f"steps={num_steps}, completed={len(completed)}, pending={len(pending_indices)}"
    )
    print(f"output directory: {output_dir}")
    if not pending_indices:
        print("All requested runs already exist.")
        return

    def save_completed_run(run):
        path = save_run(output_dir, metadata, run)
        print(
            f"saved run {run['trajectory_index']} to {path} "
            f"(final estimate={run['estimates'][-1]:.10g})",
            flush=True,
        )

    if args.workers == 1:
        for run_index in pending_indices:
            run = run_one_realization(
                run_index,
                initial_observable,
                operations,
                args.pauli_budget,
                run_step_seeds(args.master_seed, run_index, num_steps),
                num_gates,
                metadata,
                _progress_path(output_dir, run_index),
                args.quiet_steps,
            )
            save_completed_run(run)
    else:
        print(
            f"using {args.workers} workers; peak live support can approach "
            f"{args.workers * args.pauli_budget} strings"
        )
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker_precision,
            initargs=(args.precision,),
        ) as executor:
            futures = {
                executor.submit(
                    run_one_realization,
                    run_index,
                    initial_observable,
                    operations,
                    args.pauli_budget,
                    run_step_seeds(args.master_seed, run_index, num_steps),
                    num_gates,
                    metadata,
                    _progress_path(output_dir, run_index),
                    True,
                ): run_index
                for run_index in pending_indices
            }
            for future in as_completed(futures):
                save_completed_run(future.result())


if __name__ == "__main__":
    main()
