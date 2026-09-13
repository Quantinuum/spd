"""Trace the 23-step TFI workload; run each revision in a separate process.

Only tiny disposable inputs are compiled first. Trace timings include tracing
cost and must not be treated as uninstrumented performance measurements.
Example: python benchmarks/trace_persistent_allocator.py --output /tmp/current
Use --repo /path/to/historical/checkout to compare a committed implementation.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '12')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
p.add_argument('--output', type=Path, required=True)
p.add_argument('--steps', type=int, default=23)
p.add_argument('--no-history', action='store_true', help='Measure without allocation-event tracing')
p.add_argument('--diagnostics', action='store_true', help='Use public evolve with history')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(a.repo), str(a.repo / 'examples')]
import torch
import triton
from spd import triton_backend as gpu
import spd
from spd.circuit_ir import PauliRotation, CircuitIR
from spd.pytket_frontend import parse_pytket_circuit
import benchmark_2d_obc_xx_z_stepwise as setup

assert Path(gpu.__file__).resolve().is_relative_to(a.repo.resolve())
nq, cutoff, cap = 128, 2**-18, 10**9
ops = parse_pytket_circuit(setup.build_one_step_circuit(11, 3.044382, .04), nq).operations
state = gpu.create_op(setup.build_initial_observable(11), nq, 'double')
# Old and new forward entry points use different but equivalent label padding.
for op in ops:
    if not isinstance(op, PauliRotation):
        continue
    for label in [op.pauli, op.pauli.rstrip('I')]:
        gpu._gate_data(label, float(op.theta), cutoff, nq, torch.float64, torch.device('cuda:0'))
def execute(state, operations, trunc_val, max_num_str):
    if a.diagnostics:
        return spd.evolve(state, CircuitIR(nq, operations), trunc_val, max_num_str, progress=False)
    return gpu.evolve_step(state, operations, trunc_val, max_num_str), None

tiny = gpu.create_op({'X': .7, 'Y': .2, 'Z': .1}, nq, 'double')
execute(tiny, (PauliRotation('rotation', 'Z', .31),), .05, 2)[0].synchronize()
execute(tiny, (PauliRotation('rotation', 'Z', .31),), .3, 16)[0].synchronize()
del tiny
gc.collect()
torch.cuda.synchronize()

def metrics():
    stats = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                inactive_split=stats['inactive_split_bytes.all.current'],
                allocation_retries=stats['num_alloc_retries'], ooms=stats['num_ooms'],
                driver_free=free, driver_total=total)

result = dict(repo=str(a.repo.resolve()), torch=torch.__version__, triton=triton.__version__,
              device=torch.cuda.get_device_name(), steps=[], diagnostics=a.diagnostics,
              timing='synchronized evolution; measurements outside timing' if a.no_history else 'instrumented; not a runtime benchmark',
              allocator_env={k:os.environ.get(k) for k in ['PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF']})
torch.cuda.reset_peak_memory_stats()
result['before'] = metrics()
if not a.no_history:
    torch.cuda.memory._record_memory_history(stacks='python', max_entries=250000)
with (a.output / 'initial_snapshot.pickle').open('wb') as f:
    pickle.dump(torch.cuda.memory._snapshot(), f)
for step in range(a.steps):
    start = time.perf_counter()
    state, info = execute(state, ops, cutoff, cap)
    state.synchronize()
    row = dict(step=step+1, seconds=time.perf_counter()-start, rows=state.get_size(), **metrics())
    row.update(energy=state.get_expectation_value('0'), norm2=state.get_norm_square())
    if info is not None:
        row['history'] = info
    result['steps'].append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='history'}), flush=True)
result['before_release'] = metrics()
snapshot = torch.cuda.memory._snapshot()
with (a.output / 'snapshot.pickle').open('wb') as f:
    pickle.dump(snapshot, f)
blocks = [b for s in snapshot['segments'] for b in s['blocks']]
result['end_blocks'] = {state:dict(count=sum(b['state']==state for b in blocks),
                                  bytes=sum(b['size'] for b in blocks if b['state']==state),
                                  largest=max((b['size'] for b in blocks if b['state']==state), default=0))
                        for state in {b['state'] for b in blocks}}
result['trace_event_counts'] = [len(t) for t in snapshot['device_traces']]
torch.cuda.memory._record_memory_history(enabled=None)
torch.cuda.empty_cache()
result['after_release'] = metrics()
(a.output / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
