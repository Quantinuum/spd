"""Benchmark stepwise OBC observable evolution against the legacy SPD script.

This example intentionally uses SPD internals so it can evolve a single
observable one timestep at a time while reusing the previous SparsePauliOp.
That matches the benchmarking style of the older SPD implementation more
closely than repeatedly calling the public runner on deeper circuits.
"""

import argparse
import math
import pickle
import time
from pathlib import Path

import numpy as np
from pytket import Circuit

import spd.jax_backend as jax_backend
from spd.backend_adapter import BackendAdapter
from spd.pytket_frontend import parse_pytket_circuit


PACKBIT = 32


def neighbor_list(n):
    """Return the OBC nearest-neighbor pairs for an n x n square lattice."""
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
    """Build one Trotter step matching exp(-i * dt * H) for the benchmark model."""
    num_qubits, pairs = neighbor_list(n)
    circ = Circuit(num_qubits)

    # pytket stores Pauli rotations as exp(-i * pi * param * P / 2), while SPD
    # lowers them to exp(-i * theta * P / 2). To match exp(-i * dt * c * P),
    # we therefore need theta = 2 * dt * c and param = theta / pi.
    xx_param = (2.0 * dt * -1.0) / math.pi
    z_param = (2.0 * dt * -h) / math.pi

    for q0, q1 in pairs:
        circ.XXPhase(xx_param, q0, q1)

    circ.add_barrier(list(range(num_qubits)))

    for qubit in range(num_qubits):
        circ.Rz(z_param, qubit)

    return circ


def build_initial_observable(n):
    """Return the center-site Z observable as a Pauli-string dict."""
    num_qubits = n * n
    center_qubit = (num_qubits - 1) // 2
    pauli = ["I"] * num_qubits
    pauli[center_qubit] = "Z"
    return {"".join(pauli): 1.0}


def padded_system_size(num_qubits, packbit=PACKBIT):
    return packbit * ((num_qubits + packbit - 1) // packbit)


def run_one_step(spo, operations, adapter, trunc_val, max_num_str):
    """Apply one pre-parsed timestep to an existing SparsePauliOp."""
    current = spo
    for operation in operations[::-1]:
        current, _ = adapter.apply_forward(
            current,
            operation,
            trunc_val=trunc_val,
            max_num_str=max_num_str,
        )
    return current


def benchmark_size(spo, backend_name, trunc_val):
    """Return the benchmark-relevant number of retained Pauli strings."""
    if backend_name == "jax":
        coeffs = np.asarray(spo.c_array)
        return int(np.count_nonzero(np.abs(coeffs) > trunc_val))
    return spo.get_size()


def evolve_step(spo, operations, adapter, trunc_val, max_num_str):
    """Advance one timestep, with a warm pass for JAX timing stability."""
    if adapter.name != "jax":
        t0 = time.time()
        next_spo = run_one_step(
            spo,
            operations,
            adapter,
            trunc_val=trunc_val,
            max_num_str=max_num_str,
        )
        return next_spo, time.time() - t0

    # JAX compilation is shape-sensitive, so warm each step on the same input
    # state and only log the second pass.
    _ = run_one_step(
        spo,
        operations,
        adapter,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
    )

    t0 = time.time()
    next_spo = run_one_step(
        spo,
        operations,
        adapter,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
    )
    return next_spo, time.time() - t0


def configure_jax_algorithm(algorithm):
    """Apply the current internal JAX algorithm selection."""
    jax_backend.set_algorithm(algorithm)
    return jax_backend.get_algorithm()


def benchmark_filename(dt, total_t, threshold_log, backend, algorithm_tag=None):
    algorithm_prefix = ""
    if backend == "jax" and algorithm_tag is not None:
        algorithm_prefix = f"{algorithm_tag}_"
    return (
        f"{backend}_{algorithm_prefix}benchmark_data_dt_{dt}_"
        f"total_t_{total_t}_threshold_log_{threshold_log}.pkl"
    )


def save_benchmark_data(output_path, data_dict):
    with output_path.open("wb") as handle:
        pickle.dump(data_dict, handle)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark stepwise 2D OBC XX+Z observable evolution using SPD internals."
        )
    )
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument("--h", type=float, default=3.044382)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--total-t", type=float, default=0.92)
    parser.add_argument("--threshold-log", type=int, default=18)
    parser.add_argument("--backend", choices=["numpy", "jax"], default="jax")
    parser.add_argument("--precision", choices=["single", "double"], default="double")
    parser.add_argument(
        "--jax-algorithm",
        choices=["stack_sort_merge", "search_update_merge"],
        default="stack_sort_merge",
        help=(
            "JAX forward/merge algorithm to benchmark."
        ),
    )
    parser.add_argument("--max-num-str", type=int, default=int(1e9))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override the number of timesteps for quick sanity checks.",
    )
    args = parser.parse_args()

    threshold = 2.0 ** (-args.threshold_log)
    num_qubits, pairs = neighbor_list(args.n)
    num_gate = len(pairs) + num_qubits
    expected_num_gate = (2 * args.n * (args.n - 1)) + (args.n * args.n)
    if num_gate != expected_num_gate:
        raise AssertionError(
            f"Expected {expected_num_gate} gates for OBC, got {num_gate}."
        )

    total_steps = int(args.total_t / args.dt)
    if args.num_steps is not None:
        total_steps = args.num_steps
    if total_steps < 1:
        raise ValueError("Number of timesteps must be positive.")

    algorithm_tag = None
    if args.backend == "jax":
        algorithm_tag = configure_jax_algorithm(args.jax_algorithm)

    step_circuit = build_one_step_circuit(args.n, args.h, args.dt)
    adapter = BackendAdapter.from_name(
        args.backend,
        packbit=PACKBIT,
        precision=args.precision,
    )
    operations = parse_pytket_circuit(
        step_circuit,
        padded_system_size(num_qubits),
    )
    observable = adapter.create_initial_spo(
        build_initial_observable(args.n),
        padded_system_size(num_qubits),
    )

    initial_exp_val = float(np.asarray(observable.get_expectation_value(basis="Z")))

    all_results = [initial_exp_val]
    num_paulis = []
    avg_num_paulis = []
    avg_speeds = []
    times = []
    norms = []

    print(
        f"Running {total_steps} steps on {args.n}x{args.n} OBC lattice "
        f"with backend={args.backend}, precision={args.precision}"
    )
    print(
        f"dt={args.dt}, total_t={args.total_t}, threshold={threshold}, "
        f"max_num_str={args.max_num_str}"
    )
    print(f"per-step gate count={num_gate}")
    if args.backend == "jax":
        print("JAX timing mode: warm one pass per step and log the second pass only")
        print(f"JAX algorithm: {algorithm_tag}")
    print(
        f"initial observable expectation={initial_exp_val} "
        f"size={benchmark_size(observable, args.backend, threshold)}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / benchmark_filename(
        args.dt,
        args.total_t,
        args.threshold_log,
        args.backend,
        algorithm_tag,
    )

    for step_idx in range(total_steps):
        prev_num_string = benchmark_size(observable, args.backend, threshold)
        observable, elapsed = evolve_step(
            observable,
            operations,
            adapter,
            trunc_val=threshold,
            max_num_str=args.max_num_str,
        )

        exp_val = float(np.asarray(observable.get_expectation_value(basis="Z")))
        current_num_string = benchmark_size(observable, args.backend, threshold)
        current_norm = float(np.asarray(observable.get_norm_square()) ** 0.5)

        all_results.append(exp_val)
        num_paulis.append(current_num_string)
        norms.append(current_norm)
        times.append(elapsed)

        print(
            f"step {step_idx + 1}/{total_steps}: "
            f"exp_val={exp_val} size={current_num_string}"
        )

        if step_idx > 0:
            avg_num_string = 0.5 * (prev_num_string + current_num_string)
            avg_throughput = (avg_num_string * num_gate) / elapsed
            avg_num_paulis.append(avg_num_string)
            avg_speeds.append(avg_throughput)
            print(
                f"{prev_num_string} --> {current_num_string} "
                f"Time taken: {elapsed} s Norm: {current_norm}"
            )
            print(
                f"Avg throughput (# string / gate / s): {avg_throughput}"
            )
        else:
            print(f"initial --> {current_num_string} Time taken: {elapsed} s Norm: {current_norm}")

        data_dict = {
            "num_paulis": num_paulis,
            "avg_num_paulis": avg_num_paulis,
            "avg_speeds": avg_speeds,
            "times": times,
            "norms": norms,
            "all_results": all_results,
        }
        save_benchmark_data(output_path, data_dict)
        print(f"checkpoint saved to {output_path}")

    data_dict = {
        "num_paulis": num_paulis,
        "avg_num_paulis": avg_num_paulis,
        "avg_speeds": avg_speeds,
        "times": times,
        "norms": norms,
        "all_results": all_results,
    }

    save_benchmark_data(output_path, data_dict)

    print(f"Wrote benchmark data to {output_path}")
    print("final results =", all_results)


if __name__ == "__main__":
    main()
