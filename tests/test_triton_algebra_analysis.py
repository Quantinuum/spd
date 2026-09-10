"""GPU joins and analysis: independent oracles, edge cases and memory stress."""
import itertools
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires NVIDIA GPU')

import spd
from spd import triton_backend as gpu, numpy_backend as cpu
from spd.core import BaseSparsePauliOp, BaseSparsePauliGradientOp
from tests.test_triton_invariants import collision_observable, labels_from_keys, matrix
from tests.test_triton_runner import term_dict, assert_states


def cpu_state(data, gradient=False):
    cpu.set_precision('double'); cpu.utils.set_packbit(32)
    state = cpu.SparsePauliGradientOp() if gradient else cpu.SparsePauliOp()
    for label, c in data.items():
        state[tuple(cpu.utils.pauli_str_to_uint(label))] = c
    return state


@pytest.mark.parametrize('count', [0, 1, 127, 129, 8193])
@pytest.mark.parametrize('gradient', [False, True])
def test_join_sparse_union_and_dot(count, gradient):
    rng = np.random.default_rng(219)
    keys = rng.integers(0, 2**32, size=(2*count, 4), dtype=np.uint32)
    labels = labels_from_keys(keys, 64)
    values = rng.normal(size=(2*count, 2))
    a = {p: tuple(c) if gradient else c[0] for p, c in zip(labels[:count], values[:count])}
    b = {p: tuple(c) if gradient else c[0] for p, c in zip(labels[count//2:count//2+count], -values[:count])}
    factory = gpu.create_gradient_op if gradient else gpu.create_op
    ga, gb = factory(a, num_qubits=64, precision='double'), factory(b, num_qubits=64, precision='double')
    ca, cb = cpu_state(a, gradient), cpu_state(b, gradient)
    before = term_dict(ga)
    summed = ga + gb
    expected = ca + cb
    assert term_dict(summed) == pytest.approx({k: tuple(v) if gradient else (v,) for k,v in expected.items()})
    assert term_dict(ga) == before
    assert (ga - ga).get_size() == 0
    assert term_dict(sum([ga, gb])) == term_dict(summed)
    assert isinstance(summed, BaseSparsePauliGradientOp if gradient else BaseSparsePauliOp)
    if not gradient:
        assert ga.dot(gb) == pytest.approx(ca.dot(cb), abs=1e-11)
        assert ga.dot(ga) == pytest.approx(ga.get_norm_square(), abs=1e-10)
        assert ga.inner_product(gb) == ga.dot(gb)


def test_join_forced_collision_chain():
    data = collision_observable()
    a = gpu.create_op(data, precision='double')
    b = gpu.create_op(dict(reversed(list(data.items()))), precision='double')
    assert a.dot(b) == pytest.approx(sum(c*c for c in data.values()), abs=1e-11)
    assert (a-b).get_size() == 0
    assert_states(a+b, 2*a)


@pytest.mark.parametrize('precision', ['single','double'])
def test_arithmetic_zero_semantics_and_validation(precision):
    a = gpu.create_op({'X': 1e-9}, precision=precision)
    b = gpu.create_op({'X': -5e-10}, precision=precision)
    assert (a+b).get_size() == 1  # approved exact-zero addition rule
    assert (a*1e-9).get_size() == 0  # reference near-zero scalar rule
    g = gpu.create_gradient_op({'Z': (1., .3)}, precision=precision)
    h = gpu.create_gradient_op({'Z': (-1., .2)}, precision=precision)
    assert (g+h).to_host()[1][0] == 0
    assert (g+h).to_host()[2][0] == pytest.approx(.5)
    for bad in [1j, [2.], np.nan]:
        with pytest.raises(ValueError):
            a * bad
    with pytest.raises(TypeError):
        a + g
    with pytest.raises(ValueError):
        a + gpu.create_op({'I'*33: 1.}, precision=precision)
    with pytest.raises(ValueError):
        a.dot(gpu.create_op({'Z':1.}, precision='single' if precision=='double' else 'double'))


@pytest.mark.parametrize('union', [False, True])
@pytest.mark.parametrize('empty', [False, True])
def test_l2_support_and_regularized_finite_difference(union, empty):
    data = {} if empty else {'X': .7, 'Y': -.3, 'I': 0.}
    a = gpu.create_op(data, num_qubits=1, precision='double')
    target = gpu.create_op({'Z':.4,'Y':.1,'I':0.}, precision='double')
    fn = gpu.init_gradient_from_l2_difference_union if union else gpu.init_gradient_from_l2_difference
    result = fn(a,target)
    expected_labels = set(p for p,c in data.items() if c) | ({'Z','Y'} if union else set())
    keys,c,g = result.to_host()
    for k, ci, gi in zip(keys,c,g):
        label = gpu.utils.uint_to_pauli_str(k,1)
        assert label in expected_labels
        assert ci == data.get(label,0.)
        assert gi == pytest.approx(2*(ci-{'Z':.4,'Y':.1}.get(label,0.)))
    assert result.get_size() == len(expected_labels)
    if not empty:
        regularized = spd.init_gradient_spo(a, loss_type='l2_difference', target_spo=target, lambda_ose=.2, alpha=2.)
        keys,c,g = regularized.to_host()
        labels = [gpu.utils.uint_to_pauli_str(k,1) for k in keys]
        def loss(v):
            state = gpu.create_op(dict(zip(labels,v)), precision='double')
            return state.get_norm_square()+target.get_norm_square()-2*state.dot(target)+.2*state.get_OSE(2.)
        fd = [(loss(c+d)-loss(c-d))/2e-6 for d in np.eye(len(c))*1e-6]
        np.testing.assert_allclose(g,fd,atol=1e-9)


@pytest.mark.parametrize('size', [1,2,5,31,32,33,63,64,65,121])
@pytest.mark.parametrize('shift', [0,1,-1,37,123])
def test_translation_preserves_suffix_and_adjoint(size,shift):
    rng = np.random.default_rng(779)
    capacity = ((size+31)//32)*32
    labels = [''.join(rng.choice(list('IXYZ'),capacity)) for _ in range(129)]
    data = {p:(i+.1,-i-.2) for i,p in enumerate(labels)}
    state = gpu.create_gradient_op(data, precision='double')
    translated = state.translate(shift,size)
    keys,c,g = translated.to_host()
    expected = {}
    for p,v in data.items():
        prefix,suffix=p[:size],p[size:]
        k=shift%size
        prefix=prefix[-k:]+prefix[:-k] if k else prefix
        expected[prefix+suffix]=v
    actual = {gpu.utils.uint_to_pauli_str(k,capacity):(ci,gi) for k,ci,gi in zip(keys,c,g)}
    assert actual == expected
    assert_states(translated.translate(-shift,size),state)
    assert translated.c_array.data_ptr() == state.c_array.data_ptr()
    assert translated.grad_c_array.data_ptr() == state.grad_c_array.data_ptr()


@pytest.mark.parametrize('precision,tol', [('single',1e-6),('double',1e-12)])
def test_weight_histograms_and_readable_strings(precision,tol):
    data={'I'*65:0., 'X'*65:2., 'Y'+'I'*64:-.3, 'I'*32+'ZZ'+'I'*31:.7}
    state=gpu.create_op(data,precision=precision)
    assert state.get_pauli_weight_counts()=={0:1,1:1,2:1,65:1}
    assert state.get_pauli_weight_count()==state.get_pauli_weight_counts()
    assert state.get_pauli_weight_distribution()==pytest.approx({0:0.,1:.09,2:.49,65:4.},abs=tol)
    assert state.get_Pauli_weight_distribution()==state.get_pauli_weight_distribution()
    assert sum(state.get_pauli_weight_distribution().values())==pytest.approx(state.get_norm_square(),abs=tol)
    assert 'SparsePauliOp[' in str(state) and '=> 2.0' in str(state)
    grad=gpu.create_gradient_op({'Z':(0.,.3)},precision=precision)
    assert 'coeff=0.0, grad=' in str(grad)
    empty=gpu.create_op({},num_qubits=65)
    assert empty.get_pauli_weight_counts()==empty.get_pauli_weight_distribution()=={}


@pytest.mark.parametrize('width',[1,33,65,121])
def test_product_multiword_complex_phase(width):
    rng=np.random.default_rng(73)
    paulis=[''.join(rng.choice(list('IXYZ'),width)) for _ in range(129)]
    generator=''.join(rng.choice(list('IXYZ'),width))
    left=gpu.utils.pauli_str_to_uint(generator)
    right=np.array([gpu.utils.pauli_str_to_uint(p) for p in paulis])
    c=rng.normal(size=129)+1j*rng.normal(size=129)
    keys,coeff=gpu.pauli_product_batched_second_uint(left,.3+.2j,right,c)
    assert coeff.dtype==torch.complex128
    cpu.utils.set_packbit(32)
    expected_keys,expected_c=cpu.pauli_product_batched_second_uint(left,.3+.2j,right,c)
    np.testing.assert_array_equal(keys.cpu().numpy().view(np.uint32),expected_keys)
    np.testing.assert_allclose(coeff.cpu().numpy(),expected_c,atol=1e-12)


def test_noise_word_boundaries_and_invalid_inputs():
    state=gpu.create_gradient_op({'I'*31+'XY'+'I'*32:(.4,.3),'I'*64+'Z':(-.2,.5)},precision='double')
    assert gpu.get_depolarizing_susceptibility(state,(31,64))==pytest.approx(-.12+.1)
    for qubits in [(),(1,1),(-1,),(65,), (0,1,2)]:
        with pytest.raises(ValueError):
            gpu.get_depolarizing_susceptibility(state,qubits)


def test_common_module_and_abstract_surface():
    common=set(spd.numpy_backend.__all__) & set(spd.jax_backend.__all__)
    assert all(hasattr(gpu,name) for name in common)
    assert not gpu.SparsePauliOp.__abstractmethods__
    assert not gpu.SparsePauliGradientOp.__abstractmethods__


def test_rebase_forward_backward_against_numpy():
    from pytket.circuit import Circuit
    circ = Circuit(2).U3(.13,.21,-.07,0).CRy(.17,0,1).ZZPhase(.23,0,1)
    results=[]
    for name in ['numpy','triton']:
        backend=spd.BackendAdapter.from_name(name,precision='double')
        state=spd.create_spo({'ZZ':.7,'XI':-.2},backend=backend)
        final,info=spd.evolve(state,circ.copy(),1e-10,1000,rebase=True,backend=backend,progress=False)
        terminal=spd.init_gradient_spo(final,backend=backend)
        back,grads,back_info=spd.backpropagate(terminal,circ.copy(),1e-10,1000,rebase=True,backend=backend,progress=False)
        results.append((final,grads))
    expected=results[0][0]
    actual=term_dict(results[1][0])
    for k,c in expected.items():
        np.testing.assert_allclose(actual.get(k,(0,)),(c,),atol=1e-10)
    np.testing.assert_allclose(results[0][1],results[1][1],atol=1e-10)


def test_noise_quiet_progress_and_complex_product_precision(monkeypatch):
    from spd.circuit_ir import CircuitIR, SingleQubitClifford
    state=gpu.create_gradient_op({'Z':(.2,.3)},precision='double')
    def unexpected(*args,**kwargs):
        raise AssertionError('Unrequested reporting reduction')
    monkeypatch.setattr(gpu.SparsePauliOp,'get_OSE',unexpected)
    monkeypatch.setattr(gpu.SparsePauliOp,'get_norm_square',unexpected)
    _,_,noise,_=spd.backpropagate_noise_analysis(state,CircuitIR(1,(SingleQubitClifford('OpType.X',0),)),0,8,progress=False)
    assert noise['one_qubit_depolarizing']==pytest.approx([-.06])
    key=gpu.utils.pauli_str_to_uint('X')
    _,coeff=gpu.pauli_product_batched_second_uint(key,np.complex128(.1+.2j),key[None,:],np.array([1.],dtype=np.float32))
    assert coeff.dtype==torch.complex128
