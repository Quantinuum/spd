"""cuPauliProp counterpart of benchmark_2d_obc_xx_z_stepwise.py.

Uses the same defaults, parsed gate order, per-step warmup and six pickle fields.
The coefficient cutoff matches SPD forward propagation. There is no top-k cap;
the original default cap (1e9) does not bind on this 23-step workload.
The async allocator is the default so all 23 steps fit on the tested H100.
Use --allocator default to reproduce CuPy's usual pool (faster here, but OOM at step 23).

Install cuquantum-python-cu12 and cupy-cuda12x separately from SPD if desired.
In this Studio: PYTHONPATH=/tmp/spd-m5-packages:$PYTHONPATH python <this script>.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from benchmark_2d_obc_xx_z_stepwise import (
    benchmark_filename,
    build_one_step_circuit,
    neighbor_list,
    padded_system_size,
    save_benchmark_data,
)
from spd.circuit_ir import PauliRotation
from spd.pytket_frontend import parse_pytket_circuit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument("--h", type=float, default=3.044382)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--total-t", type=float, default=0.92)
    parser.add_argument("--threshold-log", type=int, default=18)
    parser.add_argument("--precision", choices=["single", "double"], default="double")
    parser.add_argument("--num-steps", type=int)
    parser.add_argument(
        "--allocator",
        choices=["default", "async"],
        default="async",
        help="async (default) fits the full H100 run; default selects CuPy's caching pool",
    )
    parser.add_argument(
        "--memory-limit",
        default="80%",
        help="cuPauliProp scratch-workspace budget, not total device memory",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args()
    if (
        args.n < 1
        or args.dt <= 0
        or not np.isfinite([args.h, args.dt, args.total_t]).all()
    ):
        parser.error("n must be positive; h/dt/total-t must be finite and dt positive")
    steps = int(args.total_t / args.dt) if args.num_steps is None else args.num_steps
    if steps < 1:
        parser.error("number of steps must be positive")

    import cupy as cp
    from cuquantum.pauliprop.experimental import (
        LibraryHandle,
        PauliExpansion,
        PauliExpansionOptions,
        PauliRotationGate,
        Truncation,
        get_num_packed_integers,
    )

    if args.allocator == "async":
        cp.cuda.set_allocator(cp.cuda.malloc_async)

    nq, edges = neighbor_list(args.n)
    num_gate = nq + len(edges)
    threshold = 2.0 ** (-args.threshold_log)
    ir = parse_pytket_circuit(
        build_one_step_circuit(args.n, args.h, args.dt), padded_system_size(nq)
    )
    gates = []
    for operation in reversed(ir.operations):
        if isinstance(operation, PauliRotation):
            qubits = [q for q, p in enumerate(operation.pauli[:nq]) if p != "I"]
            gates.append(
                PauliRotationGate(
                    operation.theta, [operation.pauli[q] for q in qubits], qubits
                )
            )
    assert len(gates) == num_gate
    truncation = Truncation(pauli_coeff_cutoff=threshold)
    handle = LibraryHandle()
    words = get_num_packed_integers(nq)
    keys = cp.zeros((1, 2 * words), dtype=cp.uint64)
    center = (nq - 1) // 2
    keys[0, words + center // 64] = np.uint64(1 << (center % 64))
    observable = PauliExpansion(
        handle,
        nq,
        1,
        keys,
        cp.ones(1, dtype=cp.float64 if args.precision == "double" else cp.float32),
        has_duplicates=False,
        options=PauliExpansionOptions(memory_limit=args.memory_limit, blocking=True),
    )

    def advance(state):
        for gate in gates:
            state = state.apply_gate(
                gate,
                truncation=truncation,
                adjoint=True,
                sort_order=None,
                keep_duplicates=False,
            )
        return state

    def expectation(state):
        significand, exponent = state.trace_with_zero_state()
        return float(significand * np.exp2(exponent))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / benchmark_filename(
        args.dt, args.total_t, args.threshold_log, "cupauliprop"
    )
    data = dict(
        num_paulis=[],
        avg_num_paulis=[],
        avg_speeds=[],
        times=[],
        norms=[],
        all_results=[expectation(observable)],
    )
    metadata = dict(
        backend="cupauliprop",
        n=args.n,
        h=args.h,
        dt=args.dt,
        total_t=args.total_t,
        num_steps=steps,
        threshold=threshold,
        precision=args.precision,
        max_num_str=None,
        per_step_gates=num_gate,
        memory_limit=args.memory_limit,
        allocator=args.allocator,
        timing="one warm pass per step; synchronized second pass; observations excluded",
        versions={
            p: importlib.metadata.version(p)
            for p in [
                "cuquantum-python-cu12",
                "cupauliprop-cu12",
                "cupy-cuda12x",
                "pytket",
            ]
        },
    )
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        f"Running {steps} steps on {args.n}x{args.n} OBC lattice with "
        f"backend=cupauliprop, precision={args.precision}",
        flush=True,
    )
    print(f"dt={args.dt}, total_t={args.total_t}, threshold={threshold}; no top-k cap")
    print(f"per-step gate count={num_gate}")
    print(f"allocator={args.allocator}, workspace memory limit={args.memory_limit}")
    print("cuPauliProp timing mode: warm one pass per step; synchronized second pass")
    print(f'initial observable expectation={data["all_results"][0]} size=1', flush=True)

    for step in range(steps):
        previous = observable.num_terms
        warm = advance(observable)
        cp.cuda.runtime.deviceSynchronize()
        del warm
        cp.cuda.runtime.deviceSynchronize()
        start = time.perf_counter()
        observable = advance(observable)
        cp.cuda.runtime.deviceSynchronize()
        elapsed = time.perf_counter() - start
        count = observable.num_terms
        value = expectation(observable)
        norm = float(cp.sqrt(cp.sum(observable.coeffs[:count] ** 2)))
        data["num_paulis"].append(count)
        data["times"].append(elapsed)
        data["norms"].append(norm)
        data["all_results"].append(value)
        print(f"step {step+1}/{steps}: exp_val={value} size={count}")
        print(f"{previous} --> {count} Time taken: {elapsed} s Norm: {norm}")
        # Preserve the original convention: throughput starts at step 2.
        if step > 0:
            average = 0.5 * (previous + count)
            throughput = average * num_gate / elapsed
            data["avg_num_paulis"].append(average)
            data["avg_speeds"].append(throughput)
            print(f"Avg throughput (# string / gate / s): {throughput}")
        save_benchmark_data(output, data)
        print(f"checkpoint saved to {output}", flush=True)
    print(f"Wrote benchmark data to {output}")
    print("final results =", data["all_results"])


if __name__ == "__main__":
    main()
