"""MonoProp counterpart of the standard 23-step benchmark.

Uses MonoProp 0.9.0 coefficient-branch truncation with no weight cutoff.
Each step warms a deep copy, then times propagation of the original state.
Copies and observations are outside the timer. Output has the standard six
pickle fields; thread count is included in the filename.

Install monoprop separately, then run with the SPD repository on PYTHONPATH.
"""

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time
import sys

# Historical drivers share the original example circuit builder.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np

from benchmark_2d_obc_xx_z_stepwise import (
    benchmark_filename,
    build_initial_observable,
    build_one_step_circuit,
    neighbor_list,
    padded_system_size,
    save_benchmark_data,
)
from spd.circuit_ir import PauliRotation
from spd.pytket_frontend import parse_pytket_circuit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument("--h", type=float, default=3.044382)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--total-t", type=float, default=0.92)
    parser.add_argument("--threshold-log", type=int, default=18)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.threads < 1 or args.n < 1 or args.dt <= 0:
        parser.error("threads, n and dt must be positive")
    steps = int(args.total_t / args.dt) if args.num_steps is None else args.num_steps
    if steps < 1:
        parser.error("number of steps must be positive")

    # Both variables are explicit so an inherited partition setting cannot
    # silently change the requested CPU parallelism.
    os.environ["monoprop_NUM_THREADS"] = str(args.threads)
    os.environ["monoprop_PARTITIONS"] = str(args.threads)
    from monoprop import Circuit, ExpGate, PauliOperator, PauliPropagator

    nq, edges = neighbor_list(args.n)
    ir = parse_pytket_circuit(
        build_one_step_circuit(args.n, args.h, args.dt), padded_system_size(nq)
    )
    rotations = [op for op in ir.operations if isinstance(op, PauliRotation)]
    assert len(rotations) == nq + len(edges)
    # MonoProp reverses authoring order in Heisenberg propagation. Preserve
    # pytket's parsed order and angles, including its wrapping of negatives.
    circuit = Circuit(
        gates=[ExpGate(PauliOperator({op.pauli[:nq]: -0.5}, num_qubits=nq))
               for op in rotations],
        system_size=nq,
        parameters=[op.theta for op in rotations],
    )
    threshold = 2.0 ** -args.threshold_log
    observable = PauliPropagator(
        PauliOperator(build_initial_observable(args.n), num_qubits=nq),
        [], cutoff=nq, lower_atol=threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / benchmark_filename(
        args.dt, args.total_t, args.threshold_log, f"monoprop_{args.threads}threads"
    )
    data = dict(num_paulis=[], avg_num_paulis=[], avg_speeds=[], times=[],
                norms=[], all_results=[float(observable.expectation_value())])
    metadata = dict(
        backend="monoprop", version=importlib.metadata.version("monoprop"),
        n=args.n, h=args.h, dt=args.dt, total_t=args.total_t, num_steps=steps,
        threshold=threshold, weight_cutoff=nq, max_num_str=None,
        precision="double", threads=args.threads, partitions=args.threads,
        cpu_affinity=sorted(os.sched_getaffinity(0)), per_step_gates=len(rotations),
        truncation="MonoProp sine-contribution lower_atol; no binding weight or term cap",
        timing="warm deep copy per step; time original propagation; copies/observations excluded",
        norm="Euclidean coefficient norm from native contract_partially([], False)",
        completed_steps=0,
    )

    def save():
        save_benchmark_data(output, data)
        metadata["completed_steps"] = len(data["times"])
        metadata["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    save()
    print(f"Running {steps} steps on {args.n}x{args.n} OBC lattice with backend=monoprop, threads={args.threads}", flush=True)
    print(f"dt={args.dt}, total_t={args.total_t}, threshold={threshold}; no weight/top-k cap", flush=True)
    print(f"per-step gate count={len(rotations)}; warm copy then timed original", flush=True)
    for step in range(steps):
        previous = observable.size()
        warm = copy.deepcopy(observable)
        warm.propagate(circuit)
        del warm
        start = time.perf_counter()
        observable.propagate(circuit)
        elapsed = time.perf_counter() - start
        count = observable.size()
        value = float(observable.expectation_value())
        # Avoid decoding millions of Pauli keys into Python objects just to
        # compute a norm. This native call returns only unrounded coefficients.
        coeffs = np.asarray(observable._simulator.contract_partially([], False))
        norm = float(np.sqrt(np.dot(coeffs, coeffs)))
        del coeffs
        data["num_paulis"].append(count)
        data["times"].append(elapsed)
        data["norms"].append(norm)
        data["all_results"].append(value)
        print(f"step {step+1}/{steps}: exp_val={value} size={count}", flush=True)
        print(f"{previous} --> {count} Time taken: {elapsed} s Norm: {norm}", flush=True)
        if step > 0:
            average = 0.5 * (previous + count)
            throughput = average * len(rotations) / elapsed
            data["avg_num_paulis"].append(average)
            data["avg_speeds"].append(throughput)
            print(f"Avg throughput (# string / gate / s): {throughput}", flush=True)
        save()
        print(f"checkpoint saved to {output}", flush=True)
    print(f"Wrote benchmark data to {output}", flush=True)
    print("final results =", data["all_results"], flush=True)


if __name__ == "__main__":
    main()
