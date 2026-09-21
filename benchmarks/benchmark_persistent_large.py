"""One-pass large TFI/AFH checks with tiny disposable compilation first.

Forward paths report counts/energy/norm. Backward compares both paths on the
same terminal SPGO. Reference backward is the retained main-era per-gate kernel
with current wrappers, not a separate main checkout. Caps here are exact, using
the shared operation loop; public power-of-two cap normalization is excluded.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '12')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'examples'), str(ROOT/'examples/gradient')]
import numpy as np
import torch
import triton
import spd
from spd import triton_backend as gpu
from spd.circuit_ir import PauliRotation, SingleQubitClifford
from spd.pytket_frontend import parse_pytket_circuit
from spd.run_circuit import _run_operation_loop
from spd.triton_backend.operations import _prepare_gate
from benchmark_persistent_backward import reference


def persistent(state, ops, cutoff, cap, backward=False):
    state = state.copy()
    angles = []
    def apply(state, op):
        result = state.apply_in_place(op, cutoff, cap)
        if result[2] is not None:
            angles.append(result[2])
        return result
    state, _, _, info = _run_operation_loop(ops if backward else ops[::-1], state,
                                            apply, time.time(), progress=False)
    return state.compact(), angles, info


def measured(fn):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    answer, angles, history = fn()
    answer.synchronize()
    metrics = dict(seconds=time.perf_counter()-start,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   peak_extra_allocated_bytes=torch.cuda.max_memory_allocated()-before,
                   terms=answer.get_size(), angles=angles, diagnostics=history)
    return answer, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['tfi', 'afh'], required=True)
    p.add_argument('--phase', choices=['state', 'diagnostics', 'backward'], required=True)
    p.add_argument('--cap', type=int)
    p.add_argument('--cutoff', type=float)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.case == 'tfi':
        import benchmark_2d_obc_xx_z_stepwise as setup
        nq, steps = 128, 10
        cutoff = 2**-18 if a.cutoff is None else a.cutoff
        ham = setup.build_initial_observable(11)
        ops = parse_pytket_circuit(setup.build_one_step_circuit(11, 3.044382, .04)).operations
    else:
        import heisenberg_setup as setup
        nq, steps = 224, 1
        cutoff = 3e-4 if a.cutoff is None else a.cutoff
        theta = np.random.RandomState(0).uniform(-.5, .5, 8)
        ham = setup.gen_3d_Hamiltonian_dict(6, 6, 6, full=False)
        ops = parse_pytket_circuit(setup.gen_3d_AFH_ansatz_circuit(theta, 6, 6, 6)).operations
    tiny = gpu.create_gradient_op({'X':(.7,.2), 'Y':(.2,.3), 'Z':(.1,.4)}, num_qubits=nq, precision='double')
    warm = (PauliRotation('rotation','Z',.31), SingleQubitClifford('OpType.X',1))
    for cap in [2,16]:
        gpu.evolve_step(tiny.to_spo(), warm, .3, cap).synchronize()
        persistent(tiny.to_spo(), warm, .3, cap)[0].synchronize()
        if a.phase == 'backward':
            for fn in [reference, lambda s,o,c,m:persistent(s,o,c,m,True)]:
                fn(tiny, warm, .3, cap)[0].synchronize()
    for op in ops:
        if isinstance(op, PauliRotation):
            for label in [op.pauli, op.pauli.rstrip('I')]:
                _prepare_gate(tiny, label, op.theta, cutoff)
    del tiny
    initial = gpu.create_op(ham, nq, 'double')
    result = dict(case=a.case, phase=a.phase, steps=steps, cutoff=cutoff, cap=a.cap,
                  torch=torch.__version__, triton=triton.__version__, device=torch.cuda.get_device_name(),
                  timing='one measured pass per path; tiny compilation; observations excluded')
    if a.phase != 'backward':
        fn = (lambda: (gpu.evolve_step(initial,ops*steps,cutoff,a.cap),[],None)) if a.phase=='state' else (
            lambda: persistent(initial,ops*steps,cutoff,a.cap))
        answer, result['measurement'] = measured(fn)
        result.update(energy=answer.get_expectation_value('0'), norm2=answer.get_norm_square())
    else:
        state = initial
        for _ in range(steps):
            state = gpu.evolve_step(state, ops, cutoff, a.cap)
        terminal = spd.init_gradient_spo(state, basis='0', lambda_ose=.13, alpha=2.)
        result.update(terminal_terms=terminal.get_size(), energy=state.get_expectation_value('0'))
        arrays = {}
        for name,fn in [('reference',reference),('persistent',lambda s,o,c,m:persistent(s,o,c,m,True))]:
            answer, result[name] = measured(lambda: fn(terminal,ops*steps,cutoff,a.cap))
            keys,c,g = answer.to_host()
            order = np.lexsort(keys.T[::-1])
            arrays[name] = (keys[order],c[order],g[order])
            del answer
        x,y = arrays['reference'],arrays['persistent']
        assert np.array_equal(x[0],y[0])
        errors = dict(same_keys=True, coefficient=float(np.max(np.abs(x[1]-y[1]),initial=0)),
                      adjoint=float(np.max(np.abs(x[2]-y[2]),initial=0)),
                      angle=float(np.max(np.abs(np.array(result['reference']['angles'])-result['persistent']['angles']),initial=0)))
        errors['history'] = {k:float(np.max(np.abs(np.array(v)-result['persistent']['diagnostics']['history'][k]),initial=0))
                             for k,v in result['reference']['diagnostics']['history'].items()}
        assert max(errors[k] for k in ['coefficient','adjoint','angle'])<1e-12
        assert errors['history']['num_str_truncated']==0
        result['correctness'] = errors
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(a.output)


if __name__ == '__main__':
    main()
