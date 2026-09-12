"""Object ownership, explicit mutation, alias protection, and zero-copy adjoints."""
import pickle
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires NVIDIA GPU')
from spd import triton_backend as gpu
from spd.circuit_ir import PauliRotation, SingleQubitClifford
from spd.triton_backend.persistent import _Storage
from tests.test_triton_backend import as_dict


def rotate(state, cutoff=.1):
    state.apply_in_place(PauliRotation('rotation', 'Z', .31), cutoff)
    return state


def test_copy_is_independent_and_reuses_private_storage():
    initial=gpu.create_op({'X':.7,'Y':.4,'Z':.01,'I':0.},precision='double')
    before=as_dict(initial)
    work=initial.copy();storage=work._storage
    assert work is not initial and storage is not initial._storage
    assert storage.keys.data_ptr()!=initial._storage.keys.data_ptr()
    rotate(work)
    for _ in range(10):rotate(work)
    assert work._storage is storage
    assert as_dict(initial)==before
    assert work.get_size()==2


def test_tensor_views_and_constructor_aliases_are_preserved():
    initial=gpu.create_op({'X':.7,'Y':.4,'Z':.01},precision='double')
    work=initial.copy()
    keys,c=work.xz_array,work.c_array
    before_k,before_c=keys.clone(),c.clone()
    # A separately constructed object and retained tensor views keep their values.
    alias=gpu.SparsePauliOp(keys,c,1)
    rotate(work)
    torch.testing.assert_close(keys,before_k,rtol=0,atol=0)
    torch.testing.assert_close(c,before_c,rtol=0,atol=0)
    assert as_dict(alias)==as_dict(initial)
    # Externally changed key tensors cannot leave a stale cached index in use.
    alias.xz_array.copy_(gpu.create_op({'Y':1.,'X':1.,'Z':1.}).xz_array)
    expected=gpu.conjugate_pauli_rotation(alias,'Z',.31,.1)
    rotate(alias)
    assert as_dict(alias)==as_dict(expected)


@pytest.mark.parametrize('initializer,kwargs', [
    ('init_gradient_from_basis_expectation', {'basis':'0'}),
    ('init_gradient_from_ose', {'alpha':2.}),
    ('init_gradient_spo', {'basis':'X','lambda_ose':.13,'alpha':1.}),
])
def test_gradient_initialization_shares_primal_without_copy_or_compaction(monkeypatch,initializer,kwargs):
    state=gpu.create_op({'X':.7,'Y':.4,'Z':.01,'I':0.},precision='double').copy()
    rotate(state)
    assert state._storage.n>state.get_size()  # Dead slots remain in storage.
    keys,c,_=state._raw_arrays()
    def unexpected(*args,**kw):
        raise AssertionError('Gradient initialization copied/compacted persistent storage')
    with monkeypatch.context() as m:
        m.setattr(_Storage,'__init__',unexpected)
        m.setattr(_Storage,'compact',unexpected)
        grad=getattr(gpu,initializer)(state,**kwargs)
    gkeys,gc,gg=grad._raw_arrays()
    assert gkeys.data_ptr()==keys.data_ptr() and gc.data_ptr()==c.data_ptr()
    assert grad.get_size()==state.get_size()
    assert torch.all(gg[c==0]==0)
    # Mutating the still-live SPO must not corrupt the shared SPGO.
    before=tuple(x.clone() for x in grad._raw_arrays())
    state.apply_in_place(SingleQubitClifford('OpType.H',0))
    for actual,expected in zip(grad._raw_arrays(),before):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_spgo_copy_and_to_spo_aliases():
    initial=gpu.create_gradient_op({'X':(.7,.2),'Y':(.4,-.1),'Z':(0.,.8)},precision='double')
    clone=initial.copy()
    assert isinstance(clone,gpu.SparsePauliGradientOp)
    for a,b in zip(initial._raw_arrays(),clone._raw_arrays()):
        assert a.data_ptr()!=b.data_ptr()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    view=clone.to_spo()
    assert view.c_array.data_ptr()==clone.c_array.data_ptr()
    before=tuple(a.copy() for a in clone.to_host())
    rotate(view,0.)
    for a,b in zip(clone.to_host(),before):np.testing.assert_array_equal(a,b)


def test_compact_tensor_exports_are_aligned_for_gradient_only_rows():
    initial=gpu.create_gradient_op({'X':(1.,0.),'Y':(0.,.2),'Z':(0.,0.)},precision='double').copy()
    # This is the representation the persistent backward kernel will produce.
    initial._storage.pruned=True;initial._storage.live=2
    assert initial.get_size()==2
    assert initial.c_array.shape==initial.grad_c_array.shape==(2,)
    assert initial.xz_array.shape==(2,2)
    assert initial.to_spo().get_size()==2


@pytest.mark.parametrize('gradient',[False,True])
def test_serialization_omits_execution_cache_and_reads_legacy_fields(gradient):
    state=(gpu.create_gradient_op({'X':(.7,.1),'Y':(.4,.8)},precision='double') if gradient
           else gpu.create_op({'X':.7,'Y':.4,'Z':.01},precision='double')).copy()
    if not gradient:rotate(state)
    fields=state.__getstate__()
    assert set(fields)==({'xz_array','c_array','num_qubits','grad_c_array'} if gradient
                         else {'xz_array','c_array','num_qubits'})
    result=pickle.loads(pickle.dumps(state))
    legacy=type(state).__new__(type(state));legacy.__setstate__(fields)
    for other in [result,legacy]:
        assert other._storage.table is None
        for a,b in zip(state.to_host(),other.to_host()):np.testing.assert_array_equal(a,b)


def test_materializing_shared_storage_does_not_retain_a_stale_index():
    state=gpu.create_op({'X':.7,'Y':.4,'Z':.2},precision='double').copy()
    grad=gpu.init_gradient_from_basis_expectation(state)
    grad.xz_array.copy_(gpu.create_op({'Y':1.,'X':1.,'Z':1.}).xz_array)
    state.compact()
    keys,c,_=state._raw_arrays()
    independent=gpu.SparsePauliOp(keys.clone(),c.clone(),1)
    expected=gpu.conjugate_pauli_rotation(independent,'Z',.31,.1)
    rotate(state)
    assert as_dict(state)==as_dict(expected)


def test_public_gradient_initialization_does_not_export_primal_arrays(monkeypatch):
    import spd
    state=gpu.create_op({'X':.7,'Y':.4,'Z':.01,'I':0.},precision='double').copy()
    rotate(state)
    keys,c,_=state._raw_arrays()
    def unexpected(*args,**kwargs):
        raise AssertionError('Backend inference or gradient initialization materialized the SPO')
    monkeypatch.setattr(_Storage,'compact',unexpected)
    grad=spd.init_gradient_spo(state,basis='X',lambda_ose=.13)
    assert grad._storage.keys.data_ptr()==keys.data_ptr()
    assert grad._storage.coeff.data_ptr()==c.data_ptr()
    assert grad.precision==state.precision=='double'


def test_skipped_public_evolution_preserves_persistent_storage():
    import spd
    from spd.circuit_ir import CircuitIR, SkippedOperation
    state=gpu.create_op({'X':.7,'Y':.4,'Z':.01,'I':0.},precision='double').copy()
    rotate(state)
    storage=state._storage
    before=(storage.keys.data_ptr(),storage.coeff.data_ptr(),storage.n,storage.capacity)
    result,info=spd.evolve(state,CircuitIR(1,(SkippedOperation('barrier'),)),.1,8,progress=False)
    assert result is state and result._storage is storage
    assert (storage.keys.data_ptr(),storage.coeff.data_ptr(),storage.n,storage.capacity)==before
    assert info['num_steps_tracked']==0


def test_public_in_place_retains_storage_across_circuits():
    import spd
    from spd.circuit_ir import CircuitIR
    initial=gpu.create_op({'X':.7,'Y':.4,'Z':.01,'I':0.},precision='double')
    before=as_dict(initial)
    expected=initial
    work=initial.copy(); storage=work._storage
    circuits=[CircuitIR(1,(PauliRotation('rotation','Z',angle),)) for angle in [.31,.27,.12]]
    for circuit in circuits:
        expected,expected_info=spd.evolve(expected,circuit,.1,8,progress=False)
        actual,info=spd.evolve(work,circuit,.1,8,progress=False,in_place=True)
        assert actual is work and work._storage is storage
        assert info==expected_info
    assert storage.capacity>storage.n
    assert storage.table is not None
    assert as_dict(work)==as_dict(expected)
    assert as_dict(initial)==before


@pytest.mark.parametrize('backend',['numpy','jax'])
def test_public_in_place_rejects_other_backends(backend):
    import spd
    from spd.circuit_ir import CircuitIR
    state=spd.create_spo({'X':1.},backend_name=backend)
    with pytest.raises(NotImplementedError,match='only for Triton'):
        spd.evolve(state,CircuitIR(1,()),0.,8,progress=False,in_place=True)
