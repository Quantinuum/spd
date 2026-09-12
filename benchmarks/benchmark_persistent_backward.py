"""Compare persistent backward with retained main-era per-gate kernels.

Compile only on tiny disposable inputs, then measure each workload once.
Reference uses current state wrappers with the original per-gate rotation kernel.
"""
import gc
import json
import time
import numpy as np
import torch

from benchmark_persistent_integration import gpu, heisenberg_setup, tfi_setup, tfi_1d_hva
from benchmark_persistent_integration import parse_pytket_circuit
import spd
from spd.circuit_ir import CircuitIR, PauliRotation, SingleQubitClifford
from spd.triton_backend.operations import _rotation, _prepare_gate
from spd.run_circuit import _run_operation_loop


def reference(state, ops, cutoff, cap):
    adapter=spd.BackendAdapter.from_name('triton',precision=state.precision)
    angles=[]
    def apply(state,op):
        result=(_rotation(state,op.pauli,op.theta,cutoff,cap,True)
                if isinstance(op,PauliRotation) else adapter.apply_backward(state,op,cutoff,cap))
        if result[2] is not None:angles.append(result[2])
        return result
    state,_,_,info=_run_operation_loop(ops,state,apply,time.time(),progress=False)
    return state,angles,info


def persistent(state,ops,cutoff,cap):
    return spd.backpropagate(state,CircuitIR(state.num_qubits,ops),cutoff,cap,progress=False)


def measure(fn,state,ops,cutoff,cap):
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    result,angles,info=fn(state,ops,cutoff,cap)
    result.synchronize()
    metrics=dict(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 peak_extra_allocated_bytes=torch.cuda.max_memory_allocated()-before,
                 terms=result.get_size(),angles=angles,diagnostics=info)
    k,c,g=result.to_host()
    host={tuple(row):(float(a),float(b)) for row,a,b in zip(k,c,g)}
    return metrics,host


def main():
    nq=8;cutoff=1e-4;cap=65536
    rng=np.random.default_rng(867)
    cases=[('tfi_1d_8_3layers',tfi_1d_hva(rng.uniform(size=6)*.3,nq).circuit,
            tfi_setup.gen_1d_Hamiltonian_dict(nq,1.),'+'),
           ('afh_1d_8_3layers',heisenberg_setup.gen_1d_AFH_ansatz_circuit(rng.uniform(size=12)*.3,nq),
            heisenberg_setup.gen_1d_Hamiltonian_dict(nq),'0')]
    tiny=gpu.create_gradient_op({'X'*nq:(.7,.2),'Y'*nq:(.2,.3),'Z'*nq:(.3,.4)},precision='double')
    warm=(PauliRotation('rotation','X',.31),SingleQubitClifford('OpType.S',1))
    for fn in [reference,persistent]:fn(tiny,warm,cutoff,cap)[0].synchronize()
    del tiny
    results=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,
                 precision='double',cutoff=cutoff,cap=cap,
                 timing='one pass per backward path; tiny compilation; terminal initialization and observations excluded',
                 workloads={})
    for name,circuit,ham,basis in cases:
        ops=parse_pytket_circuit(circuit,nq).operations
        for op in ops:
            if isinstance(op,PauliRotation):
                # Prepare both frontend-normalized and original labels before timing.
                temp=gpu.create_op({'I'*nq:1.},precision='double')
                _prepare_gate(temp,op.pauli,op.theta,cutoff)
                _prepare_gate(temp,op.pauli.rstrip('I'),op.theta,cutoff)
                del temp
        initial=gpu.create_op(ham,nq,precision='double')
        forward=gpu.evolve_step(initial,ops,cutoff,cap)
        terminal=spd.init_gradient_spo(forward,basis=basis,lambda_ose=.13,alpha=2.)
        old,expected=measure(reference,terminal,ops,cutoff,cap)
        new,actual=measure(persistent,terminal,ops,cutoff,cap)
        assert expected.keys()==actual.keys()
        coefficient_error=max(abs(actual[k][0]-expected[k][0]) for k in expected)
        adjoint_error=max(abs(actual[k][1]-expected[k][1]) for k in expected)
        assert coefficient_error<1e-12 and adjoint_error<1e-12
        np.testing.assert_allclose(new['angles'],old['angles'],atol=1e-12,rtol=1e-12)
        for field in old['diagnostics']['history']:
            np.testing.assert_allclose(new['diagnostics']['history'][field],old['diagnostics']['history'][field],atol=1e-12,rtol=1e-12)
        results['workloads'][name]=dict(operations=len(ops),terminal_terms=terminal.get_size(),
            energy=forward.get_expectation_value(basis),reference=old,persistent=new,
            correctness=dict(same_keys=True,max_coefficient_error=coefficient_error,max_adjoint_error=adjoint_error,
                             max_angle_error=max(abs(a-b) for a,b in zip(new['angles'],old['angles'])),history_matches=True))
        del initial,forward,terminal
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
