"""Persistent adjoints against the retained per-gate backward implementation."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires NVIDIA GPU')

import spd
from spd import triton_backend as gpu
from spd.circuit_ir import CircuitIR, PauliRotation, SingleQubitClifford, TwoQubitClifford, SkippedOperation
from spd.triton_backend.operations import _rotation
from spd.triton_backend.persistent import _Storage
from tests.test_triton_backward import terms
from tests.test_triton_persistent_diagnostics import assert_info


def reference(state, op, cutoff, cap):
    if isinstance(op, PauliRotation):
        return _rotation(state, op.pauli, op.theta, cutoff, cap, True)
    return spd.BackendAdapter.from_name('triton', precision=state.precision).apply_backward(state, op, cutoff, cap)


def snapshot(state):
    k,c,g=state._raw_arrays()
    keep=(c!=0)|(g!=0)
    return gpu.SparsePauliGradientOp(k[keep],c[keep],g[keep],state.num_qubits)


def assert_states(a,b,tol):
    actual,expected=terms(snapshot(a)),terms(snapshot(b))
    assert actual.keys()==expected.keys()
    np.testing.assert_allclose([actual[k] for k in expected],list(expected.values()),rtol=tol,atol=tol)


@pytest.mark.parametrize('nq',[3,65,121])
@pytest.mark.parametrize('precision,tol',[('single',5e-6),('double',5e-13)])
@pytest.mark.parametrize('cutoff,cap',[(0.,None),(.15,None),(.15,17)])
def test_mixed_backward_per_gate(nq,precision,tol,cutoff,cap):
    rng=np.random.default_rng(775)
    labels=[''.join(row) for row in rng.choice(list('IXYZ'),(129,nq))]
    initial=gpu.create_gradient_op(dict(zip(labels,rng.normal(size=(129,2)))),precision=precision)
    before=terms(initial)
    work,expected=initial.copy(),initial
    storage=work._storage
    for name in ['H','S','Sdg','X','Y','Z','CX','CY','CZ']:
        rotation=PauliRotation('rotation',''.join(rng.choice(list('IXYZ'),nq)),float(rng.uniform(-2,2)))
        clifford=(TwoQubitClifford('OpType.'+name,0,nq-1) if name.startswith('C')
                  else SingleQubitClifford('OpType.'+name,nq-1))
        for op in [rotation,clifford]:
            expected,count,angle,info=reference(expected,op,cutoff,cap)
            result,actual_count,actual_angle,actual_info=work.apply_in_place(op,cutoff,cap)
            assert result is work and work._storage is storage
            assert actual_count==count
            assert actual_angle==pytest.approx(angle,abs=tol,rel=tol) if angle is not None else actual_angle is None
            assert_info(actual_info,info,tol)
            assert_states(work,expected,tol)
            assert gpu.get_depolarizing_susceptibility(work,(0,nq-1))==pytest.approx(
                gpu.get_depolarizing_susceptibility(expected,(0,nq-1)),abs=tol,rel=tol)
    assert terms(initial)==before


def test_gradient_only_growth_reuses_storage_and_preserves_source():
    source=gpu.create_gradient_op({'XXX':(0.,1.)},precision='double')
    work=source.copy();storage=work._storage
    for p in ['ZII','IZI','IIZ']:
        work.apply_in_place(PauliRotation('rotation',p,.31),0.)
    assert work.get_size()==8
    assert storage is work._storage and storage.growths==0
    # The index grows as support doubles; row capacity already accommodates it.
    assert storage.n==8 and storage.capacity==16
    assert torch.count_nonzero(storage.coeff[:storage.n])==0
    assert torch.count_nonzero(storage.grad[:storage.n])==8
    assert source.get_size()==1


def test_cap_keeps_live_zero_primal_adjoints_instead_of_dead_slots():
    # Dead zeros must rank below gradient-only rows when a zero-magnitude cap binds.
    state=gpu.create_gradient_op({'II':(0.,0.),'XI':(0.,1.),'IX':(0.,2.),'XX':(0.,3.)},precision='double').copy()
    state.apply_in_place(PauliRotation('rotation','II',0.),0.,2)
    assert state.get_size()==2
    assert torch.all(state._raw_arrays()[2]!=0)


def test_dead_owner_reactivation_angle_and_inclusive_cutoff():
    state=gpu.create_gradient_op({'X':(.01,0.),'Y':(.8,.3),'Z':(.5,.2)},precision='double')
    expected=state;work=state.copy()
    for p,t,cut in [('I',0.,.5),('Z',.31,0.),('X',-.4,0.),('I',0.,.5)]:
        op=PauliRotation('rotation',p,t)
        expected,count,angle,info=reference(expected,op,cut,None)
        _,actual_count,actual_angle,actual_info=work.apply_in_place(op,cut)
        assert count==actual_count
        assert angle==pytest.approx(actual_angle,abs=1e-14)
        assert_info(actual_info,info,1e-14)
        assert_states(work,expected,1e-14)


@pytest.mark.parametrize('analysis',[False,True])
def test_public_backward_uses_one_copy_and_retains_storage(monkeypatch,analysis):
    source=gpu.create_gradient_op({'XI':(.7,.2),'YI':(.3,.8),'IZ':(.1,.3)},precision='double')
    before=terms(source)
    ops=tuple([PauliRotation('rotation','ZI',.31)]*20)
    copies=[];original=_Storage.__init__
    def track(self,*args,**kwargs):
        original(self,*args,**kwargs);copies.append(self)
    monkeypatch.setattr(_Storage,'__init__',track)
    fn=spd.backpropagate_noise_analysis if analysis else spd.backpropagate
    result=fn(source,CircuitIR(2,ops),0.,64,progress=False)
    assert len(copies)==1 and result[0]._storage is copies[0]
    assert copies[0].rebuilds==1
    assert len(result[1])==20 and result[-1]['num_steps_tracked']==20
    assert terms(source)==before


def test_shared_terminal_source_and_skipped_backward():
    spo=gpu.create_op({'X':.7,'Y':.3,'Z':.1},precision='double').copy()
    grad=gpu.init_gradient_from_basis_expectation(spo,'X')
    before=tuple(a.clone() for a in spo._raw_arrays()[:2])
    result,_,_=spd.backpropagate(grad,CircuitIR(1,(SkippedOperation('barrier'),)),0.,8,progress=False)
    assert result is grad
    grad.apply_in_place(PauliRotation('rotation','Z',.31))
    for a,b in zip(spo._raw_arrays()[:2],before):torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_backward_without_diagnostics_preserves_angle_and_state():
    source=gpu.create_gradient_op({'X':(.7,.2),'Y':(.3,.8),'Z':(.01,.2)},precision='double')
    a,b=source.copy(),source.copy()
    op=PauliRotation('rotation','Z',.31)
    _,count,angle,info=a.apply_in_place(op,.1,2)
    _,other_count,other_angle,other_info=b.apply_in_place(op,.1,2,diagnostics=False)
    assert count==other_count and angle==other_angle and other_info is None
    assert_states(a,b,0.)


def test_noise_history_matches_original_after_each_mixed_gate():
    from spd.run_circuit import get_operation_qubits
    source=gpu.create_gradient_op({'XI':(.7,.2),'YI':(.3,.8),'IZ':(.01,.3),'XY':(.8,-.4)},precision='double')
    ops=(PauliRotation('rotation','ZI',.31),SingleQubitClifford('OpType.S',0),
         SkippedOperation('barrier'),TwoQubitClifford('OpType.CX',0,1),
         PauliRotation('rotation','XY',-.23),SingleQubitClifford('OpType.H',1))
    expected=source;angles=[];one=[];two=[]
    for op in ops:
        expected,_,angle,_=reference(expected,op,.1,16)
        if angle is not None:angles.append(angle)
        qubits=get_operation_qubits(op)
        one.append(gpu.get_depolarizing_susceptibility(expected,qubits) if len(qubits)==1 else 0.)
        two.append(gpu.get_depolarizing_susceptibility(expected,qubits) if len(qubits)==2 else 0.)
    actual,actual_angles,noise,_=spd.backpropagate_noise_analysis(source,CircuitIR(2,ops),.1,16,progress=False)
    assert_states(actual,expected,1e-13)
    np.testing.assert_allclose(actual_angles,angles,atol=1e-13,rtol=1e-13)
    np.testing.assert_allclose(noise['one_qubit_depolarizing'],one,atol=1e-13,rtol=1e-13)
    np.testing.assert_allclose(noise['two_qubit_depolarizing'],two,atol=1e-13,rtol=1e-13)


def test_forward_sequence_does_not_dispatch_spgo_backward():
    state=gpu.create_gradient_op({'X':(.7,.2)},precision='double')
    with pytest.raises(TypeError,match='Forward evolution'):
        gpu.evolve_step(state,(PauliRotation('rotation','Z',.31),))


@pytest.mark.parametrize('analysis',[False,True])
def test_public_in_place_backward_retains_object_and_storage(analysis):
    source=gpu.create_gradient_op({'X':(.7,.2),'Y':(.3,.8),'Z':(.01,.2)},precision='double')
    before=terms(source);work=source.copy();storage=work._storage;expected=source
    fn=spd.backpropagate_noise_analysis if analysis else spd.backpropagate
    for theta in [.31,-.12,.17]:
        circuit=CircuitIR(1,(PauliRotation('rotation','Z',theta),))
        ref=fn(expected,circuit,.1,8,progress=False);expected=ref[0]
        actual=fn(work,circuit,.1,8,progress=False,in_place=True)
        assert actual[0] is work and work._storage is storage
        assert actual[1:]==ref[1:]
    assert storage.capacity>storage.n and storage.table is not None
    assert_states(work,expected,1e-13)
    assert terms(source)==before


@pytest.mark.parametrize('backend',['numpy','jax'])
@pytest.mark.parametrize('analysis',[False,True])
def test_in_place_backward_rejects_other_backends(backend,analysis):
    initial=spd.create_spo({'X':1.},backend_name=backend)
    terminal=spd.init_gradient_spo(initial,basis='X')
    fn=spd.backpropagate_noise_analysis if analysis else spd.backpropagate
    with pytest.raises(NotImplementedError,match='only for Triton'):
        fn(terminal,CircuitIR(1,()),0.,8,progress=False,in_place=True)


def test_in_place_backward_detaches_shared_terminal_primal():
    spo=gpu.create_op({'X':.7,'Y':.3,'Z':.01},precision='double').copy()
    terminal=spd.init_gradient_spo(spo,basis='X')
    before=tuple(a.clone() for a in spo._raw_arrays()[:2])
    shared=terminal._storage
    result,_,_=spd.backpropagate(terminal,CircuitIR(1,(PauliRotation('rotation','Z',.31),)),.1,8,progress=False,in_place=True)
    assert result is terminal and terminal._storage is not shared
    for a,b in zip(spo._raw_arrays()[:2],before):torch.testing.assert_close(a,b,atol=0,rtol=0)
