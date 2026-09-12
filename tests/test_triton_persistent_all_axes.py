"""Axis-complete local parity, phase, indexing and adjoint checks."""
import itertools
import numpy as np
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('triton')
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires NVIDIA GPU')
from spd import triton_backend as tb
from spd.circuit_ir import PauliRotation
from spd.triton_backend.persistent_options import evolve_options,OptionStorage,descriptor,local_descriptor
from spd.triton_backend.experiment_kernels import update_variant
from tests.test_triton_backend import as_dict

MODES=['local_all','select_all','inverted_all','select_all+local_all','inverted_all+local_all']
AXES=['XI','YI','ZI']+[''.join(p) for p in itertools.product('XYZ',repeat=2)]


def label(pair,width):
    chars=['I']*width
    chars[31],chars[width-1]=pair
    return ''.join(chars)


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('width',[33,448,512])
@pytest.mark.parametrize('cutoff,cap',[(0.,None),(.07,None),(.05,37)])
def test_all_axis_sequence(mode,width,cutoff,cap):
    rng=np.random.default_rng(7001)
    labels=[''.join(p) for p in rng.choice(list('IXYZ'),(129,width))]
    state=tb.create_op(dict(zip(labels,rng.normal(size=len(labels)))),precision='double')
    original=as_dict(state)
    ops=[PauliRotation('test',label(pair,width),.43 if j%2 else -.31) for j,pair in enumerate(AXES)]
    # A three-site gate takes the generic fallback, including under selection.
    fallback=list('I'*width)
    fallback[0],fallback[31],fallback[-1]='XYZ'
    fallback=''.join(fallback)
    assert local_descriptor(fallback,state.c_array.device)==(0,None)
    ops.append(PauliRotation('test',fallback,.19))
    expected=as_dict(tb.evolve_step(state,ops,cutoff,cap))
    actual=as_dict(evolve_options(state,ops,cutoff,cap,mode=mode))
    assert actual.keys()==expected.keys()
    np.testing.assert_allclose([actual[k] for k in expected],list(expected.values()),rtol=5e-13,atol=5e-13)
    assert as_dict(state)==original


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('axes',AXES)
@pytest.mark.parametrize('missing',[False,True])
def test_all_axis_adjoint_and_selection(mode,axes,missing):
    width=512
    rng=np.random.default_rng(114)
    logical=list(itertools.product('IXYZ',repeat=2))
    def anti(pair):
        return sum(a!='I' and b!='I' and a!=b for a,b in zip(pair,axes))%2
    lone=next(pair for pair in logical if anti(pair))
    pairs={label(pair,width):(float(rng.normal()),float(rng.normal())) for pair in logical}
    pairs[label(lone,width)]=(0.,1.)
    if missing:
        pairs={label(lone,width):(0.,1.)}
    state=tb.create_gradient_op(pairs,precision='double')
    gate_label=label(axes,width)
    expected,_,dtheta,_=tb.conjugate_pauli_rot_backward(state,gate_label,.43,0.)
    gate,scalars=tb._gate_data(gate_label,.43,0.,width,torch.float64,state.c_array.device)
    old={gate.data_ptr():descriptor(gate_label,state.c_array.device)}
    local={gate.data_ptr():local_descriptor(gate_label,state.c_array.device)}
    arity,sites=local[gate.data_ptr()]
    assert arity==sum(p!='I' for p in axes)
    storage=OptionStorage(state,.1,mode,old,local)
    adjoint=torch.zeros(storage.capacity,dtype=torch.float64,device='cuda')
    adjoint[:state.get_size()].copy_(state.grad_c_array)
    active,count=(None,storage.n)
    if 'select' in mode or 'inverted' in mode:
        active,count=storage.active_rows(gate)
        assert active is not None
        assert count==(1 if missing else sum(anti(pair) for pair in logical))
    blocks=(count+127)//128
    dgrad=torch.zeros(blocks,dtype=torch.float64,device='cuda')
    storage.counts.zero_()
    update_variant[(blocks,)](storage.keys,storage.coeff,storage.table,gate,scalars,
        storage.counts,storage.n,storage.table_size-1,storage.width,128,
        **storage.key_args(),active=active,active_n=count,ACTIVE=active is not None,
        sites=sites,LOCAL_ARITY=arity if 'local_all' in mode else 0,
        grad=adjoint,gradient_stats=dgrad,BACKWARD=True,enable_fp_fusion=False)
    n=storage.n+int(storage.counts[0].item())
    assert state.get_size()+int(storage.counts[1].item())==expected.get_size()
    keys=storage.keys[:n].cpu().numpy().view(np.uint32)
    c=storage.coeff[:n].cpu().numpy();g=adjoint[:n].cpu().numpy()
    actual={tuple(k):(cv,gv) for k,cv,gv in zip(keys,c,g) if cv!=0 or gv!=0}
    ek,ec,eg=expected.to_host()
    reference={tuple(k):(cv,gv) for k,cv,gv in zip(ek,ec,eg)}
    assert actual.keys()==reference.keys()
    np.testing.assert_allclose([actual[k] for k in reference],list(reference.values()),rtol=4e-13,atol=4e-13)
    assert dgrad.sum().item()==pytest.approx(dtheta,abs=4e-13)
