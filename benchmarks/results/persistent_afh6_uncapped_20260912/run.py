import os,sys,time,json,argparse,hashlib,resource
os.environ['monoprop_NUM_THREADS']='12';os.environ['monoprop_PARTITIONS']='12'
os.environ['OMP_NUM_THREADS']='12';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT='/teamspace/studios/this_studio/spd'
sys.path[:0]=[ROOT,ROOT+'/examples/gradient']
import numpy as np
from spd.circuit_ir import PauliRotation,SkippedOperation,SingleQubitClifford
from spd.pytket_frontend import parse_pytket_circuit
import heisenberg_setup as setup,run_utils
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['mono','cu','persistent','anti','main'],required=True);p.add_argument('--prefix',type=int,default=0);p.add_argument('--cutoff',type=float,default=1e-4);a=p.parse_args()
np.random.seed(0);theta,_=run_utils.init_thetas(num_params=8,init_mode='random',random_scale=1.)
circ=setup.gen_3d_AFH_ansatz_circuit(theta,6,6,6);ham=setup.gen_3d_Hamiltonian_dict(6,6,6,full=False)
ops=list(reversed(parse_pytket_circuit(circ,224).operations));rots=[o for o in ops if isinstance(o,PauliRotation)];cliffs=[o for o in ops if isinstance(o,SingleQubitClifford)]
assert all(str(o.gate_name).split('.')[-1]=='X' for o in cliffs)
assert [o for o in ops if not isinstance(o,SkippedOperation)]==rots+cliffs
nq=216;cap=None;cutoff=a.cutoff
out=f'/tmp/afh6-uncapped/{a.backend}_{cutoff:g}.json';result=dict(backend=a.backend,qubits=nq,padded_qubits=224,rotations=len(rots),preparation_gates=len(cliffs),cutoff=cutoff,cap=None if a.backend=='mono' else cap,threads=12,thetas=theta.tolist(),circuit_sha256=hashlib.sha256(repr(ops).encode()).hexdigest(),blocks=[],status='running',timing='one pass; small synthetic Triton compilation preflight excluded; block synchronization included; observations/logging excluded')
def save():open(out,'w').write(json.dumps(result,indent=2))
sync=lambda:None
if a.backend=='mono':
 from monoprop import PauliOperator,PauliPropagator,Circuit,ExpGate
 circuits=[Circuit(gates=[ExpGate(PauliOperator({o.pauli[:nq]:-.5},num_qubits=nq))],system_size=nq,parameters=[o.theta]) for o in rots]
 state=PauliPropagator(PauliOperator(ham,num_qubits=nq),[o.qubit for o in cliffs],cutoff=nq,lower_atol=cutoff)
 def apply(i):state.propagate(circuits[i])
 def count():return state.size()
 def finish():pass
 def observe():return dict(energy=float(state.expectation_value()))
 result['preparation']='exact computational-basis reference via occupied qubit indices; no X propagation required'
elif a.backend=='cu':
 import cupy as cp,torch
 from cuquantum.pauliprop.experimental import LibraryHandle,PauliExpansion,PauliExpansionOptions,PauliRotationGate,CliffordGate,Truncation
 pool=cp.cuda.MemoryAsyncPool();cp.cuda.set_allocator(pool.malloc)
 handle=LibraryHandle();options=PauliExpansionOptions(memory_limit='50%',blocking=True);w=4
 keys=np.zeros((len(ham),2*w),dtype=np.uint64)
 for i,label in enumerate(ham):
  for q,c in enumerate(label):
   if c in 'XY':keys[i,q//64]|=np.uint64(1<<q%64)
   if c in 'YZ':keys[i,w+q//64]|=np.uint64(1<<q%64)
 state=PauliExpansion(handle,nq,len(ham),cp.asarray(keys),cp.asarray(list(ham.values()),dtype=cp.float64),has_duplicates=False,options=options)
 gates=[]
 for o in rots:
  qs=[q for q,c in enumerate(o.pauli[:nq]) if c!='I'];gates.append(PauliRotationGate(o.theta,[o.pauli[q] for q in qs],qs))
 cg=[CliffordGate('X',[o.qubit]) for o in cliffs];trunc=Truncation(pauli_coeff_cutoff=cutoff)
 sync=cp.cuda.runtime.deviceSynchronize
 def apply(i):
  global state
  state=state.apply_gate(gates[i],truncation=trunc,adjoint=True,sort_order=None,keep_duplicates=False)
 def count():return state.num_terms
 def finish():
  global state
  for g in cg:state=state.apply_gate(g,adjoint=True,sort_order=None,keep_duplicates=False)
 def observe():
  v,e=state.trace_with_zero_state();return dict(energy=float(v*np.exp2(e)),norm2=float(cp.sum(state.coeffs[:state.num_terms]**2)))
 result['cap_method']='none; native coefficient cutoff only'
else:
 import torch
 from spd import triton_backend as tb
 from spd.triton_backend import persistent as ps
 from spd.backend_adapter import BackendAdapter
 from spd.triton_backend.kernels import build_index
 if a.backend=='main':
  class FullIndexLaunch:
   def __getitem__(self,grid):
    def launch(keys,table,gate,n,mask,width,block):return build_index[grid](keys,table,n,mask,width,block)
    return launch
  tb.build_anticommuting_index=FullIndexLaunch()
 sync=torch.cuda.synchronize;adapter=BackendAdapter.from_name('triton',packbit=32,precision='double')
 state=tb.create_op(ham,224,'double')
 descriptors=[tb._gate_data(o.pauli,float(o.theta),cutoff,224,state.c_array.dtype,state.c_array.device) for o in rots]
 t=time.perf_counter()
 # Compile on tiny disposable support, not another benchmark evaluation.
 test=tb.create_op({'X'+'I'*223:1.,'Z'+'I'*223:.1,'Y'+'I'*223:.3},224,'double')
 warmops=[PauliRotation('preflight',c+'I'*223,.43) for c in 'XYZ']
 if a.backend=='persistent':
  test=ps.evolve_step_persistent(test,warmops,.05,2)
  temp=ps._Storage(test,.1);temp.compact(final=True);del temp
 else:
  for o in warmops:test=tb.conjugate_pauli_rotation(test,o.pauli,o.theta,.05,2)
 for o in cliffs[:1]:test,*_=adapter.apply_forward(test,o,cutoff,cap)
 sync();del test;result['preflight_seconds']=time.perf_counter()-t
 storage=None
 def apply(i):
  global state,storage
  if a.backend=='persistent':
   if storage is None:storage=ps._Storage(state,.1)
   storage.apply(*descriptors[i],cap)
  else:
   o=rots[i];state=tb.conjugate_pauli_rotation(state,o.pauli,o.theta,cutoff,cap)
 def count():return storage.live if a.backend=='persistent' and storage is not None else state.get_size()
 def finish():
  global state,storage
  if a.backend=='persistent':
   storage.compact(final=True);state=tb.SparsePauliOp(storage.keys,storage.coeff,224);storage=None
  for o in cliffs:state,*_=adapter.apply_forward(state,o,cutoff,cap)
 def observe():return dict(energy=state.get_expectation_value('0'),norm2=state.get_norm_square())
 result['source_sha256']={f:hashlib.sha256(open(ROOT+'/'+f,'rb').read()).hexdigest() for f in ['spd/triton_backend/persistent.py','spd/triton_backend/kernels.py','spd/triton_backend/__init__.py']}
save();sync();start=time.perf_counter();last=0;peak=count();first_cap=None
try:
 for i in range(len(rots)):
  apply(i);n=count();peak=max(peak,n)
  if n>=50000000 and first_cap is None:first_cap=i+1
  stopped=n>50000000
  if (i+1)%216==0 or i+1==a.prefix or i+1==len(rots) or stopped:
   sync();elapsed=time.perf_counter()-start
   row=dict(start_gate=last+1,end_gate=i+1,seconds=elapsed,rows=n,peak_rows=peak)
   result['blocks'].append(row);result['rotation_seconds']=sum(r['seconds'] for r in result['blocks']);result['completed_rotations']=i+1;result['safety_boundary_gate']=first_cap
   save();print(json.dumps(row),flush=True)
   if stopped:
    result['status']='stopped_at_50m_safety_boundary_not_capped';break
   last=i+1;peak=n;sync();start=time.perf_counter()
 else:result['status']='complete'
 sync();t=time.perf_counter()
 if result['status']=='complete':finish()
 sync();result['finalization_seconds']=time.perf_counter()-t
 result['total_seconds']=result['rotation_seconds']+result['finalization_seconds'];result['rows']=count();result.update(observe());result['rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 save();print(json.dumps({k:v for k,v in result.items() if k!='blocks'}),flush=True)
except Exception as e:result['status']='failed';result['error']=repr(e);save();raise
