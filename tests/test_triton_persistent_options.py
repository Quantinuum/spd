"""Coefficient and adjoint probes for independent storage experiments."""
import itertools
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires NVIDIA GPU')
from spd import triton_backend as tb
from spd.circuit_ir import PauliRotation
from spd.triton_backend.persistent_options import evolve_options, MODES, OptionStorage, descriptor
from spd.triton_backend.experiment_kernels import update_variant
from tests.test_triton_backend import as_dict


@pytest.mark.parametrize('mode', MODES[1:])
@pytest.mark.parametrize('cutoff,cap', [(0.,None),(.07,None),(.2,17)])
def test_sequence(mode, cutoff, cap):
    rng = np.random.default_rng(2718)
    labels = [''.join(x) for x in rng.choice(list('IXYZ'), (129, 33))]
    state = tb.create_op(dict(zip(labels, rng.normal(size=129))), precision='double')
    before = as_dict(state)
    strings = ['I'*q+'Z'+'I'*(32-q) for q in (0,31,32)]
    strings += ['X'+'I'*30+'XX', 'Y'+'I'*32, 'XX'+'I'*31]
    ops = [PauliRotation('rotation', p, .37 if i%2 else -.71) for i,p in enumerate(strings*2)]
    expected = as_dict(tb.evolve_step(state, ops, cutoff, cap))
    actual = as_dict(evolve_options(state, ops, cutoff, cap, mode=mode))
    assert actual.keys() == expected.keys()
    np.testing.assert_allclose([actual[k] for k in expected], list(expected.values()), atol=4e-13, rtol=4e-13)
    assert as_dict(state) == before


@pytest.mark.parametrize('mode', ['local','inverted','select','soa','fingerprint','inverted+local','inverted+fingerprint','inverted+local+fingerprint'])
@pytest.mark.parametrize('gate_label', ['ZI','XX'])
@pytest.mark.parametrize('support', ['closed', 'gradient_only_missing'])
def test_adjoint_layout_and_index_probe(mode, gate_label, support):
    # Probe closed support and missing-partner growth before full backward
    # integration. A zero-primal/nonzero-adjoint row must remain meaningful.
    labels = [''.join(p) for p in itertools.product('IXYZ', repeat=2)]
    rng = np.random.default_rng(94)
    pairs = {p: (float(c),float(g)) for p,c,g in zip(labels,rng.normal(size=16),rng.normal(size=16))}
    pairs['XI'] = (0., 1.)
    if support == 'gradient_only_missing':
        pairs = {'XI' if gate_label == 'ZI' else 'ZI': (0., 1.)}
    original = tb.create_gradient_op(pairs, precision='double')
    expected, _, dtheta, _ = tb.conjugate_pauli_rot_backward(original, gate_label, .43, 0.)
    gate, scalars = tb._gate_data(gate_label, .43, 0., 2, torch.float64, original.c_array.device)
    desc = {gate.data_ptr(): descriptor(gate_label, original.c_array.device)}
    storage = OptionStorage(original, .1, mode, desc)
    adjoint = torch.zeros(storage.capacity, dtype=torch.float64, device='cuda')
    initial_rows = original.get_size()
    adjoint[:initial_rows].copy_(original.grad_c_array)
    active, count = (None, storage.n)
    if 'inverted' in mode or mode == 'select':
        active, count = storage.active_rows(gate)
    storage.counts.zero_()
    blocks = (count+127)//128
    gradient_stats = torch.zeros(blocks, dtype=torch.float64, device='cuda')
    kind, info = desc[gate.data_ptr()]
    update_variant[(blocks,)](storage.keys, storage.coeff, storage.table, gate, scalars,
        storage.counts, storage.n, storage.table_size-1, storage.width, 128,
        **storage.key_args(), fingerprints=storage.fingerprints, FP='fingerprint' in mode,
        info=info, LOCAL=kind if 'local' in mode else 0,
        active=active, active_n=count, ACTIVE=active is not None,
        grad=adjoint, gradient_stats=gradient_stats, BACKWARD=True, enable_fp_fusion=False)
    assert initial_rows + int(storage.counts[1].item()) == expected.get_size()
    n = storage.n + int(storage.counts[0].item())
    keys = storage.keys[:n].cpu().numpy().view(np.uint32)
    c = storage.coeff[:n].cpu().numpy()
    g = adjoint[:n].cpu().numpy()
    actual = {tuple(k):(a,b) for k,a,b in zip(keys,c,g) if a != 0 or b != 0}
    ek, ec, eg = expected.to_host()
    reference = {tuple(k):(a,b) for k,a,b in zip(ek,ec,eg)}
    assert actual.keys() == reference.keys()
    np.testing.assert_allclose([actual[k] for k in reference], list(reference.values()), atol=3e-13, rtol=3e-13)
    assert gradient_stats.sum().item() == pytest.approx(dtheta, abs=3e-13)


@pytest.mark.parametrize('mode', MODES[1:])
def test_single_precision(mode):
    rng = np.random.default_rng(315)
    labels = [''.join(p) for p in itertools.product('IXYZ',repeat=3)]
    state = tb.create_op(dict(zip(labels,rng.normal(size=64))),precision='single')
    ops = [PauliRotation('rotation',p,t) for p,t in [('ZII',.31),('XXI',-.72),('IYI',.19),('IXX',.53)]]
    expected=as_dict(tb.evolve_step(state,ops,.07))
    actual=as_dict(evolve_options(state,ops,.07,mode=mode))
    assert actual.keys()==expected.keys()
    np.testing.assert_allclose([actual[k] for k in expected],list(expected.values()),atol=2e-6,rtol=2e-6)
