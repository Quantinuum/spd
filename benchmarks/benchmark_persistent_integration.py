"""Stage-one forward comparison; small disposable compilation, one timed pass.

Run from the repository root: python benchmarks/benchmark_persistent_integration.py
Both paths include private copying/allocation, truncation, and materialization.
"""
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
    metrics = dict(seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   peak_extra_allocated_bytes=torch.cuda.max_memory_allocated() - initial,
                   terms=result.get_size())
    # Observations and host coefficient comparisons are outside timing/memory.
    metrics['energy'] = result.get_expectation_value(basis)
    keys, coeff = result.to_host()
    host = {tuple(k): float(c) for k, c in zip(keys, coeff)}
    return metrics, host


def main():
    cutoff, cap, nq = 1e-4, 4096, 8
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
        old, expected = measure(reference, state, ops, cutoff, cap, basis)
        new, actual = measure(gpu.evolve_step, state, ops, cutoff, cap, basis)
        shared = actual.keys() & expected.keys()
        comparison = dict(same_keys=actual.keys() == expected.keys(),
                          max_coefficient_error=max((abs(actual[k]-expected[k]) for k in shared), default=0.),
                          energy_error=abs(new['energy']-old['energy']))
        results['workloads'][name] = dict(operations=len(ops), reference=old, persistent=new, correctness=comparison)
        assert comparison['same_keys']
        assert comparison['max_coefficient_error'] < 1e-12
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
