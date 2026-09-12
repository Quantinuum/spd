"""Wide-key and mixed-axis coverage for cross-workload experiments."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires NVIDIA GPU')
from spd import triton_backend as tb
from spd.circuit_ir import PauliRotation
from spd.triton_backend.persistent_options import evolve_options
from tests.test_triton_backend import as_dict


@pytest.mark.parametrize('width', [448,512])
@pytest.mark.parametrize('mode', ['control','local','inverted','select','fingerprint','soa','select+local'])
@pytest.mark.parametrize('cap', [None,97])
def test_wide_mixed_axis_sequence(width, mode, cap):
    rng = np.random.default_rng(812)
    labels = [''.join(row) for row in rng.choice(list('IXYZ'),(133,width))]
    state = tb.create_op(dict(zip(labels,rng.normal(size=len(labels)))),precision='double')
    ops=[]
    for axis,sites in [('Z',[width-1]),('X',[0,width-1]),('Y',[31,32]),
                       ('Z',[32,width-1]),('X',[31,width-2]),('Z',[0])]:
        label=['I']*width
        for q in sites:
            label[q]=axis
        ops.append(PauliRotation('test',''.join(label),.37))
    before=as_dict(state)
    expected=as_dict(tb.evolve_step(state,ops,.03,cap))
    actual=as_dict(evolve_options(state,ops,.03,cap,mode=mode))
    assert actual.keys()==expected.keys()
    np.testing.assert_allclose([actual[k] for k in expected],list(expected.values()),rtol=3e-13,atol=3e-13)
    assert as_dict(state)==before


@pytest.mark.parametrize('mode', ['inverted','select','soa','fingerprint','select+local'])
@pytest.mark.parametrize('axis', ['Z','XX'])
@pytest.mark.parametrize('missing', [False,True])
def test_wide_adjoint_pair(mode, axis, missing):
    import itertools
    from spd.triton_backend.persistent_options import OptionStorage, descriptor
    from spd.triton_backend.experiment_kernels import update_variant
    width=512
    def label(pair):
        chars=['I']*width
        chars[31],chars[511]=pair
        return ''.join(chars)
    rng=np.random.default_rng(144)
    pairs={label(pair):(float(rng.normal()),float(rng.normal()))
           for pair in itertools.product('IXYZ',repeat=2)}
    lone=label('XI' if axis=='Z' else 'ZI')
    pairs[lone]=(0.,1.)
    if missing:
        pairs={lone:(0.,1.)}
    gate_label=label('ZI' if axis=='Z' else 'XX')
    original=tb.create_gradient_op(pairs,precision='double')
    expected,_,dtheta,_=tb.conjugate_pauli_rot_backward(original,gate_label,.43,0.)
    gate,scalars=tb._gate_data(gate_label,.43,0.,width,torch.float64,original.c_array.device)
    desc={gate.data_ptr():descriptor(gate_label,original.c_array.device)}
    storage=OptionStorage(original,.1,mode,desc)
    grad=torch.zeros(storage.capacity,dtype=torch.float64,device='cuda')
    grad[:original.get_size()].copy_(original.grad_c_array)
    active,count=(None,storage.n)
    if 'inverted' in mode or 'select' in mode:
        active,count=storage.active_rows(gate)
    storage.counts.zero_()
    blocks=(count+127)//128
    dgrad=torch.zeros(blocks,dtype=torch.float64,device='cuda')
    kind,info=desc[gate.data_ptr()]
    update_variant[(blocks,)](storage.keys,storage.coeff,storage.table,gate,scalars,
        storage.counts,storage.n,storage.table_size-1,storage.width,128,
        **storage.key_args(),fingerprints=storage.fingerprints,FP=mode=='fingerprint',
        info=info,LOCAL=kind if 'local' in mode else 0,
        active=active,active_n=count,ACTIVE=active is not None,
        grad=grad,gradient_stats=dgrad,BACKWARD=True,enable_fp_fusion=False)
    n=storage.n+int(storage.counts[0].item())
    assert original.get_size()+int(storage.counts[1].item())==expected.get_size()
    keys=storage.keys[:n].cpu().numpy().view(np.uint32)
    coeff=storage.coeff[:n].cpu().numpy()
    grads=grad[:n].cpu().numpy()
    actual={tuple(k):(c,g) for k,c,g in zip(keys,coeff,grads) if c!=0 or g!=0}
    ek,ec,eg=expected.to_host()
    reference={tuple(k):(c,g) for k,c,g in zip(ek,ec,eg)}
    assert actual.keys()==reference.keys()
    np.testing.assert_allclose([actual[k] for k in reference],list(reference.values()),rtol=3e-13,atol=3e-13)
    assert dgrad.sum().item()==pytest.approx(dtheta,abs=3e-13)
