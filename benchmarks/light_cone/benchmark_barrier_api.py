"""Benchmark one public forward/backward call for the 10x10 TFI HVA."""
import argparse,gc,importlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'examples/gradient')]
import numpy as np
import torch
import spd
import run_tfi_gs as tfi
from spd.circuit_ir import CircuitIR,SkippedOperation
import spd.pruning as pruning_module
runner=importlib.import_module('spd.run_circuit')

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['baseline','whole','barrier'],required=True);p.add_argument('--scale',type=float,default=1);p.add_argument('--output',type=Path,required=True);p.add_argument('--dump',type=Path);a=p.parse_args()
assert not a.output.exists()
b=spd.BackendAdapter.from_name('triton',packbit=32,precision='double')
np.random.seed(0);params=(np.random.rand(8)-.5)*.1*a.scale
ansatz=tfi.make_tfi_ansatz(params,2,10)
full=runner._normalize_input_circuit(ansatz.circuit,b,False)
ops=full.operations
blocks=[];start=0
for i,op in enumerate(ops):
    if isinstance(op,SkippedOperation) and op.gate_name=='OpType.Barrier':
        if any(not isinstance(x,SkippedOperation) for x in ops[start:i]):
            blocks.append((start,CircuitIR(100,ops[start:i+1])))
        start=i+1
if any(not isinstance(x,SkippedOperation) for x in ops[start:]):
    blocks.append((start,CircuitIR(100,ops[start:])))
assert len(blocks)==20,(len(blocks),[o.gate_name for o in ops if isinstance(o,SkippedOperation)])
initial=lambda:spd.create_spo(tfi.make_local_tfi_hamiltonian(2,10,3.1),backend=b)
# Common tiny public warmup, outside timing.
w,_=spd.evolve(initial(),blocks[-1][1],1e-8,10485760,progress=False,backend=b,pruning='light-cone')
g=spd.init_gradient_spo(w,basis='+',backend=b)
spd.backpropagate(g,blocks[-1][1],1e-8,10485760,progress=False,backend=b)
torch.cuda.synchronize();del w,g;gc.collect()
support_times=[]
original_support=pruning_module._support
def measured_support(*args,**kw):
    t=time.perf_counter();v=original_support(*args,**kw)
    support_times.append(time.perf_counter()-t)
    return v
pruning_module._support=measured_support
state=initial();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();retained=[];rows=[]
method={'baseline':None,'whole':'light-cone','barrier':'light-cone-barrier'}[a.mode]
state,info=spd.evolve(state,full,1e-8,10485760,progress=False,backend=b,in_place=True,pruning=method)
retained=list(state._pruning_record.retained_indices) if method else [i for i,o in enumerate(ops) if not isinstance(o,SkippedOperation)]
torch.cuda.synchronize();forward=time.perf_counter()-t
energy=float(state.get_expectation_value('+'));count=state.get_size()

torch.cuda.synchronize();t=time.perf_counter()
g=spd.init_gradient_spo(state,basis='+',backend=b)
_,gate_gradients,_=spd.backpropagate(g,full,1e-8,10485760,progress=False,backend=b)
gradients=ansatz.parameter_gradients(gate_gradients)
torch.cuda.synchronize();backward=time.perf_counter()-t
# Dump after all timing: to_host compacts storage and must not affect backward.
if a.dump:
    a.dump.mkdir(parents=True,exist_ok=True);keys,c=state.to_host();np.save(a.dump/'keys.npy',keys);np.save(a.dump/'coeffs.npy',c);del keys,c


result=dict(mode=a.mode,scale=a.scale,parameters=params.tolist(),num_qubits=100,blocks=len(blocks),total_gates=sum(not isinstance(o,SkippedOperation) for o in ops),retained_gates=len(retained),forward_s=forward,backward_s=backward,total_s=forward+backward,support_s=sum(support_times),support_calls=len(support_times),energy=energy,num_terms=count,gradients=np.asarray(gradients).tolist(),effective_cap=1<<24,peak_gpu_bytes=torch.cuda.max_memory_allocated(),rows=rows,gpu=torch.cuda.get_device_name(0),timing='public diagnostics-enabled runner; IR construction and energy excluded; forward in_place=True in all modes; backward includes adjoint initialization; single public call in every mode; production composite record; support_s measures only support extraction, not all planning')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
