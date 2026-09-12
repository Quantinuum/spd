"""Profile unchanged forward kernels on real step inputs; never use as a baseline.

Run from the repository root. Profiles one extra warmed pass at selected steps;
CUDA kernel durations and CPU API durations overlap and must not be added.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, nargs='+', default=[3, 14, 23])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('stepwise', ROOT / 'examples/benchmark_2d_obc_xx_z_stepwise.py')
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    import spd.triton_backend as tb
    from spd.circuit_ir import PauliRotation
    operations = example.parse_pytket_circuit(example.build_one_step_circuit(11, 3.044382, .04), 128).operations
    gate_list = [op for op in reversed(operations) if isinstance(op, PauliRotation)]
    result = dict(
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [ROOT / 'spd/triton_backend/__init__.py', ROOT / 'spd/triton_backend/kernels.py', ROOT / 'examples/benchmark_2d_obc_xx_z_stepwise.py']},
        versions={p: importlib.metadata.version(p) for p in ['torch', 'triton', 'numpy', 'pytket']},
        python=platform.python_version(), gpu=torch.cuda.get_device_name(),
        host=platform.platform(), affinity=sorted(os.sched_getaffinity(0)),
        allocator_backend=torch.cuda.get_allocator_backend(),
        allocator_env={k: os.environ.get(k) for k in ['PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF']},
        allocator_effective_default='expandable_segments:True when neither allocator environment variable is set',
        gpu_status=subprocess.check_output(['nvidia-smi','-q'], text=True),
        gate_order_sha256=hashlib.sha256(repr([(g.pauli, g.theta) for g in gate_list]).encode()).hexdigest(),
        gate_count=len(gate_list), profiles=[], trajectory=[])
    def save():
        (args.output_dir / 'profile.json').write_text(json.dumps(result, indent=2) + '\n')
    save()
    state = tb.create_op(example.build_initial_observable(11), 128, 'double')
    for step in range(1, max(args.steps) + 1):
        # Retain each step input, as in the standard warm-from-input benchmark.
        warm = tb.evolve_step(state, operations, 2**-18, 10**9)
        warm.synchronize()
        del warm
        if step in args.steps:
            torch.cuda.reset_peak_memory_stats()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                start = time.perf_counter()
                output = tb.evolve_step(state, operations, 2**-18, 10**9)
                output.synchronize()
                elapsed = time.perf_counter() - start
            prof.export_chrome_trace(str(args.output_dir / f'step_{step}_trace.json'))
            events = []
            for e in prof.key_averages():
                events.append(dict(name=e.key, calls=e.count, cpu_total_us=e.cpu_time_total,
                                   cpu_self_us=e.self_cpu_time_total, device_total_us=e.device_time_total,
                                   device_self_us=e.self_device_time_total))
            record = dict(step=step, input_rows=state.get_size(), output_rows=output.get_size(),
                          instrumented_wall_seconds=elapsed, events=events,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            # Diagnostic only: fixed-stride output snapshot, outside profiler.
            stride = max(1, (output.get_size() + 65535) // 65536)
            sample = output.xz_array[::stride].contiguous().cpu().numpy().view(np.uint32)
            anti_by_kind = {'Z': [], 'XX': []}
            for gate in gate_list:
                parity = np.zeros(len(sample), dtype=np.uint32)
                kind = ''.join(p for p in gate.pauli if p != 'I')
                for q, p in enumerate(gate.pauli):
                    if p in 'XY':
                        parity ^= (sample[:, 4 + q // 32] >> (31 - q % 32)) & 1
                    if p in 'YZ':
                        parity ^= (sample[:, q // 32] >> (31 - q % 32)) & 1
                anti_by_kind[kind].append(float(parity.mean()))
            record['output_snapshot_sample_rows'] = len(sample)
            record['output_snapshot_anticommuting_fraction_by_gate'] = anti_by_kind
            result['profiles'].append(record)
        else:
            output = tb.evolve_step(state, operations, 2**-18, 10**9)
        state = output
        del output
        record = dict(step=step, count=state.get_size(), expectation=state.get_expectation_value('Z'),
                      norm=state.get_norm_square()**.5)
        result['trajectory'].append(record)
        save()
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
