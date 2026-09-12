"""Cross-workload forward experiments; no production dispatch changes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'examples'), str(ROOT/'examples/gradient')]
from spd.triton_backend import create_op
from spd.triton_backend.persistent_options import evolve_options, OptionStorage
from spd.triton_backend import persistent
from spd.backend_adapter import BackendAdapter
from spd.circuit_ir import PauliRotation, SkippedOperation
from spd.pytket_frontend import parse_pytket_circuit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workload', choices=['xxz','afh'], required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--size', type=int, default=21)
    p.add_argument('--steps', type=int, default=18)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--random-scale', type=float, default=.1)
    p.add_argument('--cutoff', type=float)
    p.add_argument('--cap', type=int, default=1000000000)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--stop-rows', type=int, default=65000000)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--save-state', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--extra-padding-qubits',type=int,default=0,help='Experimental identity-qubit padding; multiple of 32.')
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.workload == 'xxz':
        import benchmark_2d_obc_xx_z_stepwise as setup
        nq = a.size**2
        circ = setup.build_one_step_circuit(a.size, 3.044382, .04)
        ham = setup.build_initial_observable(a.size)
        cutoff = 2.**-18
        thetas = None
    else:
        import heisenberg_setup as setup
        import run_utils
        nq = a.size**3
        np.random.seed(0)
        thetas, _ = run_utils.init_thetas(num_params=4*a.layers, init_mode='random', random_scale=a.random_scale)
        circ = setup.gen_3d_AFH_ansatz_circuit(thetas, a.size, a.size, a.size)
        ham = setup.gen_3d_Hamiltonian_dict(a.size, a.size, a.size, full=False)
        cutoff = .001
    if a.cutoff is not None:
        if not np.isfinite(a.cutoff) or a.cutoff < 0:
            raise ValueError('cutoff must be finite and nonnegative')
        cutoff = a.cutoff
    if a.extra_padding_qubits<0 or a.extra_padding_qubits%32:
        p.error('--extra-padding-qubits must be a nonnegative multiple of 32')
    padded = 32*((nq+31)//32)+a.extra_padding_qubits
    operations = parse_pytket_circuit(circ, padded).operations
    # Partition in execution order, retaining exact Clifford dispatch and order.
    chunks, pending = [], []
    for op in reversed(operations):
        if isinstance(op, (PauliRotation, SkippedOperation)):
            pending.append(op)
        else:
            if pending:
                chunks.append(tuple(reversed(pending))); pending = []
            chunks.append(op)
    if pending:
        chunks.append(tuple(reversed(pending)))
    adapter = BackendAdapter.from_name('triton', packbit=32, precision='double')
    gate_counts = Counter(op.gate_name for op in operations)
    result = dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        qubits=nq, padded_qubits=padded, key_words=2*(padded//32), cutoff=cutoff,
        thetas=None if thetas is None else thetas.tolist(), gate_counts=dict(gate_counts),
        circuit_sha256=hashlib.sha256(repr(operations).encode()).hexdigest(),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
            ['spd/triton_backend/persistent.py','spd/triton_backend/persistent_options.py',
             'spd/triton_backend/experiment_kernels.py','benchmarks/benchmark_persistent_generalization.py',
             'examples/gradient/heisenberg_setup.py','examples/gradient/run_utils.py']},
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, steps=[])
    tracking = None
    def wrap(original):
        def apply(self, gate, params, cap):
            if tracking is not None:
                tracking['input_live_sum'] += self.live
                tracking['input_slots_sum'] += self.n
            original(self, gate, params, cap)
            if tracking is not None:
                tracking['gates'] += 1
                tracking['peak_live'] = max(tracking['peak_live'], self.live)
                tracking['peak_slots'] = max(tracking['peak_slots'], self.n)
                tracking['at_cap'] += int(cap is not None and self.live == cap)
                if tracking['gates'] % 2048 == 0:
                    print('warm progress', tracking['gates'], 'gates', self.live, 'live rows', flush=True)
        return apply
    persistent._Storage.apply = wrap(persistent._Storage.apply)
    OptionStorage.apply = wrap(OptionStorage.apply)
    def evaluate(state):
        for chunk in chunks:
            if isinstance(chunk, tuple):
                if a.variant == 'reference':
                    from spd.triton_backend import evolve_step
                    state = evolve_step(state, chunk, cutoff, a.cap)
                else:
                    stats = {} if tracking is not None else None
                    state = evolve_options(state, chunk, cutoff, a.cap, mode=a.variant, stats=stats)
                    if stats is not None:
                        for key, value in stats.items():
                            tracking[key] = tracking.get(key, 0) + value
            else:
                state, *_ = adapter.apply_forward(state, chunk, cutoff, a.cap)
        state.synchronize()
        return state
    state = create_op(ham, padded, 'double')
    print(json.dumps({k:result[k] for k in ['qubits','key_words','gate_counts','thetas']}),flush=True)
    def save():
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    try:
        for step in range(a.steps if a.workload=='xxz' else 1):
            tracking = dict(input_live_sum=0,input_slots_sum=0,gates=0,peak_live=0,peak_slots=0,at_cap=0)
            warm = evaluate(state)
            warm.get_expectation_value('0')
            work = tracking
            tracking = None
            del warm
            torch.cuda.reset_peak_memory_stats()
            times, eval_times, observations = [], [], []
            for repeat in range(a.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = evaluate(state)
                elapsed = time.perf_counter()-start
                energy = output.get_expectation_value('0')
                eval_elapsed = time.perf_counter()-start
                norm2 = output.get_norm_square()
                times.append(elapsed); eval_times.append(eval_elapsed)
                observations.append(dict(rows=output.get_size(), energy=energy, norm2=norm2))
                if repeat+1 < a.repeats:
                    del output
            row = dict(step=step+1, rows=output.get_size(), energy=energy,norm2=norm2,
                seconds=times,energy_seconds=eval_times,observations=observations,work=work,
                live_row_rotations_per_second=work['input_live_sum']/np.median(times) if work['gates'] else None,
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                allocator_retries=torch.cuda.memory_stats()['num_alloc_retries'])
            result['steps'].append(row); save(); print(json.dumps(row),flush=True)
            if a.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    start=time.perf_counter()
                    profiled=evaluate(state)
                    elapsed=time.perf_counter()-start
                trace_path=a.output.with_suffix('.trace.json')
                prof.export_chrome_trace(str(trace_path))
                result['profile']=dict(instrumented_seconds=elapsed,rows=profiled.get_size(),
                    energy=profiled.get_expectation_value('0'),norm2=profiled.get_norm_square(),
                    trace=str(trace_path))
                del profiled
            state = output
            del output
            if state.get_size() >= a.stop_rows and a.workload=='xxz':
                result['stop_reason']='row safety boundary after complete timestep'; break
        if a.save_state:
            keys, coeff = state.to_host()
            np.savez(a.output.with_suffix('.npz'), keys=keys,coeff=coeff)
        result['status']='complete'
    except torch.OutOfMemoryError as exc:
        result.update(status='out_of_memory', error=str(exc),
                      last_work=tracking if tracking is not None else locals().get('work'),
                      peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved())
        raise
    finally:
        save()

if __name__ == '__main__':
    main()
