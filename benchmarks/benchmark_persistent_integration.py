"""Forward comparison; small disposable compilation, one timed pass.

Run from the repository root: python benchmarks/benchmark_persistent_integration.py
Both paths include private copying/allocation, truncation, and materialization.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples' / 'gradient'))
import numpy as np
import torch
import triton
import heisenberg_setup
import tfi_setup
from spd import triton_backend as gpu
from spd.ansatz.tfi import tfi_1d_hva
from spd.circuit_ir import PauliRotation, SingleQubitClifford, TwoQubitClifford
from spd.pytket_frontend import parse_pytket_circuit


def reference(state, ops, cutoff, cap):
    for op in reversed(ops):
        if isinstance(op, PauliRotation):
            state = gpu.conjugate_pauli_rotation(state, op.pauli, op.theta, cutoff, cap)
        elif isinstance(op, SingleQubitClifford):
            state = getattr(gpu, f"conjugate_{op.gate_name.split('.')[-1]}_forward")(state, op.qubit)
        elif isinstance(op, TwoQubitClifford):
            state = getattr(gpu, f"conjugate_{op.gate_name.split('.')[-1]}_forward")(
                state, op.control_qubit, op.target_qubit)
    return state


def measure(fn, state, ops, cutoff, cap, basis):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = fn(state, ops, cutoff, cap)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    info = None
    if isinstance(result, tuple):
        result, info = result
    metrics = dict(seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   peak_extra_allocated_bytes=torch.cuda.max_memory_allocated() - initial,
                   terms=result.get_size())
    # Observations and host coefficient comparisons are outside timing/memory.
    if info is not None:
        metrics['diagnostics'] = info
    metrics['energy'] = result.get_expectation_value(basis)
    keys, coeff = result.to_host()
    host = {tuple(k): float(c) for k, c in zip(keys, coeff)}
    return metrics, host


def reference_diagnostics(state, ops, cutoff, cap):
    from spd.backend_adapter import BackendAdapter
    from spd.run_circuit import _run_operation_loop
    from spd.triton_backend.operations import _rotation
    adapter = BackendAdapter.from_name('triton', precision='double')
    def apply(state, op):
        if isinstance(op, PauliRotation):
            return _rotation(state, op.pauli, op.theta, cutoff, cap, False)
        return adapter.apply_forward(state, op, cutoff, cap)
    state, _, _, info = _run_operation_loop(ops[::-1], state, apply, time.time(), progress=False)
    return state, info


def persistent_diagnostics(state, ops, cutoff, cap):
    import spd
    from spd.circuit_ir import CircuitIR
    return spd.evolve(state, CircuitIR(state.num_qubits, ops), cutoff, cap, progress=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', action='store_true', help='Also measure public forward diagnostics/history')
    parser.add_argument('--cap', type=int, help='Default: 4096 forward-only, 65536 with diagnostics to avoid ambiguous cap ties')
    args = parser.parse_args()
    cutoff, cap, nq = 1e-4, args.cap if args.cap is not None else (65536 if args.diagnostics else 4096), 8
    rng = np.random.RandomState(0)
    cases = [
        ('tfi_1d_8_3layers', tfi_1d_hva(rng.uniform(size=6) * .3, nq).circuit,
         tfi_setup.gen_1d_Hamiltonian_dict(nq, 1.), '+'),
        ('afh_1d_8_3layers', heisenberg_setup.gen_1d_AFH_ansatz_circuit(rng.uniform(size=12) * .3, nq),
         heisenberg_setup.gen_1d_Hamiltonian_dict(nq), '0'),
    ]
    # Same packed width/dtype and gate specializations on tiny disposable support.
    tiny = gpu.create_op({'X'*nq: .7, 'Y'*nq: .2, 'Z'*nq: .3}, precision='double')
    warm_ops = [PauliRotation('rotation', 'X', .31), PauliRotation('rotation', 'Y', .27),
                SingleQubitClifford('OpType.X', 1)]
    for fn in (reference, gpu.evolve_step):
        fn(tiny, warm_ops, cutoff, cap).synchronize()
    if args.diagnostics:
        for fn in (reference_diagnostics, persistent_diagnostics):
            warmed, _ = fn(tiny, warm_ops, cutoff, cap)
            warmed.synchronize()
            del warmed
    del tiny
    results = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                   precision='double', cutoff=cutoff, cap=cap, timed_passes=1, workloads={})
    for name, circuit, ham, basis in cases:
        ops = parse_pytket_circuit(circuit, nq).operations
        state = gpu.create_op(ham, precision='double')
        # Prepare the shared cached gate tensors before either measurement, so
        # the first path does not pay all cache misses or retain less cache memory.
        for op in ops:
            if isinstance(op, PauliRotation):
                gpu._gate_data(op.pauli, float(op.theta), cutoff, nq,
                               state.c_array.dtype, state.c_array.device)
                if args.diagnostics:
                    gpu._gate_data(op.pauli.rstrip('I'), float(op.theta), cutoff, nq,
                                   state.c_array.dtype, state.c_array.device)
        old, expected = measure(reference, state, ops, cutoff, cap, basis)
        new, actual = measure(gpu.evolve_step, state, ops, cutoff, cap, basis)
        shared = actual.keys() & expected.keys()
        comparison = dict(same_keys=actual.keys() == expected.keys(),
                          max_coefficient_error=max((abs(actual[k]-expected[k]) for k in shared), default=0.),
                          energy_error=abs(new['energy']-old['energy']))
        results['workloads'][name] = dict(operations=len(ops), reference=old, persistent=new, correctness=comparison)
        assert comparison['same_keys']
        assert comparison['max_coefficient_error'] < 1e-12
        if args.diagnostics:
            ref, expected = measure(reference_diagnostics, state, ops, cutoff, cap, basis)
            current, actual = measure(persistent_diagnostics, state, ops, cutoff, cap, basis)
            assert actual.keys() == expected.keys()
            coefficient_error = max((abs(actual[k] - expected[k]) for k in expected), default=0.)
            assert coefficient_error < 1e-12
            for field in ref['diagnostics']['history']:
                np.testing.assert_allclose(current['diagnostics']['history'][field],
                                           ref['diagnostics']['history'][field], atol=1e-12, rtol=1e-12)
            for field in ref['diagnostics'].keys() - {'history'}:
                np.testing.assert_allclose(current['diagnostics'][field], ref['diagnostics'][field], atol=1e-12, rtol=1e-12)
            results['workloads'][name]['diagnostic_forward'] = dict(
                reference=ref, persistent=current, max_coefficient_error=coefficient_error,
                energy_error=abs(current['energy'] - ref['energy']), history_matches=True)

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
