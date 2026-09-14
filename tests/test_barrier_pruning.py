"""Barrier refresh, complete execution records, and public backend integration."""
import pickle

import numpy as np
import pytest

import spd
from spd.circuit_ir import CircuitIR, PauliRotation, SingleQubitClifford, SkippedOperation
from tests.test_pruning import adapter, evolve, p, same, terms
from tests.helpers import assert_info_consistent

METHOD = 'light-cone-barrier'


@pytest.mark.parametrize('name', ['barrier', 'OpType.Barrier'])
def test_refresh_discards_geometric_overestimate(all_backend, name, monkeypatch):
    from spd import pruning
    b = adapter(all_backend)
    barrier = SkippedOperation(name)
    circuit = CircuitIR(4, (barrier, PauliRotation('rx', 'IIXI', .4), barrier,
        barrier, PauliRotation('zz', 'IZZI', .3), SkippedOperation('measure'),
        barrier, PauliRotation('zz', 'ZZII', .2), barrier, barrier))
    state = spd.create_spo({'ZIII': 1.}, backend=b)
    whole, wi = evolve(state, circuit, b)
    calls = []
    original = pruning._support
    def measured(state, backend_name):
        calls.append(backend_name)
        return original(state, backend_name)
    monkeypatch.setattr(pruning, '_support', measured)
    result, info = evolve(state, circuit, b, pruning=METHOD)
    assert len(calls) == 3
    assert wi['pruning']['num_gates_retained'] == 3
    assert info['pruning'] == dict(method=METHOD, num_gates_total=3,
                                   num_gates_retained=1, num_gates_pruned=2)
    assert result._pruning_record.retained_indices == (7,)
    same(result, whole)
    same(result, state)
    assert getattr(state, '_pruning_record', None) is None
    assert_info_consistent(info, 3)
    # Backward must consume the composite record, never query the adjoint support.
    monkeypatch.setattr(pruning, '_support', lambda *a: pytest.fail('backward replanned'))
    g = spd.init_gradient_spo(result, backend=b)
    _, grads, bi = spd.backpropagate(g, circuit, 0., 256, backend=b, progress=False)
    np.testing.assert_array_equal(grads, [0., 0., 0.])
    assert bi['pruning'] == info['pruning']
    with pytest.raises(ValueError, match='does not match'):
        spd.backpropagate(g, CircuitIR(4, circuit.operations[1:]), 0., 256,
                          backend=b, progress=False)


@pytest.mark.parametrize('cutoff,cap', [(0., 256), (.08, 256), (0., 2), (.08, 2)])
def test_matches_explicit_blocks_and_reduced_backward(all_backend, cutoff, cap):
    b = adapter(all_backend)
    blocks = [
        tuple(PauliRotation('rx', p(6, [q], 'X'), .37 + .03*q) for q in range(6)),
        tuple(PauliRotation('zz', p(6, [q,q+1], 'Z'), .43) for q in (0,2,4)),
        tuple(PauliRotation('rx', p(6, [q], 'X'), .29) for q in range(6)),
        (SingleQubitClifford('OpType.H', 1), PauliRotation('zz', 'ZZIIII', .33),
         PauliRotation('rx', 'XIIIII', .41)),
    ]
    ops = tuple(op for block in blocks for op in (*block, SkippedOperation('barrier')))
    circuit = CircuitIR(6, ops)
    state = spd.create_spo({'ZIIIII': .8, 'IZIIII': .23}, backend=b)
    expected = state
    history = {k: [] for k in ('num_str_truncated','truncated_l1_norm','truncated_l2_norm')}
    selected = []
    offset = len(ops)
    for block in reversed(blocks):
        offset -= len(block) + 1
        expected, info = evolve(expected, CircuitIR(6, block), b, cutoff=cutoff, cap=cap)
        selected.extend(offset+i for i in expected._pruning_record.retained_indices)
        for key in history:
            history[key].extend(info['history'][key])
    actual, info = evolve(state, circuit, b, pruning=METHOD, cutoff=cutoff, cap=cap)
    same(actual, expected)
    assert actual._pruning_record.retained_indices == tuple(sorted(selected))
    assert_info_consistent(info, sum(map(len, blocks)))
    for key in history:
        np.testing.assert_allclose(info['history'][key], history[key], atol=1e-12)
    if cap == 2:
        assert info['sum_num_str_truncated'] > 0  # Exercise a binding cap.
    reduced = CircuitIR(6, tuple(ops[i] for i in sorted(selected)))
    reference, _ = evolve(state, reduced, b, pruning=None, cutoff=cutoff, cap=cap)
    same(actual, reference)
    ag = spd.init_gradient_spo(actual, basis='+', backend=b)
    rg = spd.init_gradient_spo(reference, basis='+', backend=b)
    aa, av, ai = spd.backpropagate(ag, circuit, cutoff, cap, backend=b, progress=False)
    rr, rv, _ = spd.backpropagate(rg, reduced, cutoff, cap, backend=b, progress=False)
    same(aa, rr)
    rv = iter(rv)
    expected_grad = [next(rv) if i in selected else 0. for i,op in enumerate(ops)
                     if isinstance(op, PauliRotation)]
    np.testing.assert_allclose(av, expected_grad, atol=2e-10)
    assert_info_consistent(ai, sum(map(len, blocks)))


@pytest.mark.parametrize('ops', [(), (SkippedOperation('barrier'),),
    (PauliRotation('rx', 'XIII', .3), SkippedOperation('measure'),
     PauliRotation('zz', 'ZZII', .2))])
def test_without_barriers_matches_whole_plan(all_backend, ops):
    b = adapter(all_backend)
    state = spd.create_spo({'ZIII': 1.}, backend=b)
    whole, wi = evolve(state, CircuitIR(4, ops), b)
    result, info = evolve(state, CircuitIR(4, ops), b, pruning=METHOD)
    same(result, whole)
    assert result._pruning_record.retained_indices == whole._pruning_record.retained_indices
    assert info['history'] == wi['history']
    assert info['pruning']['method'] == METHOD


def test_all_pruned_keeps_subthreshold_input(all_backend):
    b = adapter(all_backend)
    state = spd.create_spo({'ZIII': .001}, backend=b)
    circuit = CircuitIR(4, (PauliRotation('rx', 'IIXI', .3),
        SkippedOperation('barrier'), PauliRotation('rx', 'IIIX', .2)))
    result, info = evolve(state, circuit, b, pruning=METHOD, cutoff=.01)
    same(result, state)
    assert info['pruning']['num_gates_retained'] == 0
    assert info['sum_num_str_truncated'] == 0


def test_invalid_later_block_rejected_before_mutation():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    b = spd.BackendAdapter.from_name('triton', precision='double')
    state = spd.create_spo({'ZIII': 1.}, backend=b)
    before = terms(state)
    ops = (PauliRotation('invalid', 'IIIA', .2), SkippedOperation('barrier'),
           PauliRotation('rx', 'XIII', .3))
    with pytest.raises(ValueError):
        evolve(state, ops, b, pruning=METHOD, in_place=True)
    assert terms(state) == before


@pytest.mark.parametrize('entry', ['public', 'direct'])
def test_triton_copies_once_and_preserves_record(entry, monkeypatch):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from spd import triton_backend as tb
    b = spd.BackendAdapter.from_name('triton', precision='double')
    state = spd.create_spo({'ZIII': 1.}, backend=b)
    circuit = CircuitIR(4, (PauliRotation('rx','IIXI',.4), SkippedOperation('barrier'),
                           PauliRotation('zz','ZZII',.2)))
    calls = []
    copy = tb.SparsePauliOp.copy
    def counted(self):
        calls.append(self)
        return copy(self)
    monkeypatch.setattr(tb.SparsePauliOp, 'copy', counted)
    result = (evolve(state, circuit, b, pruning=METHOD)[0] if entry == 'public' else
              tb.evolve_step(state, circuit, pruning=METHOD))
    assert len(calls) == 1
    assert getattr(state, '_pruning_record', None) is None
    for saved in (result.copy(), pickle.loads(pickle.dumps(result))):
        assert saved._pruning_record.method == METHOD
        g = spd.init_gradient_spo(saved, backend=b)
        _, grads, info = spd.backpropagate(g, circuit, 0., 256, backend=b, progress=False)
        assert grads == [0., 0.]
        assert info['pruning']['num_gates_retained'] == 1


def test_pytket_shared_parameters_and_finite_differences(all_backend):
    from spd.ansatz import tfi_2d_hva
    b = adapter(all_backend)
    params = np.array([.05, .09, .07, -.11])
    state = spd.create_spo({'ZZII': -.7, 'ZIZI': -.4, 'XIII': -1.3}, backend=b)
    def evaluate(values):
        ansatz = tfi_2d_hva(values, system_size_x=2, system_size_y=2)
        result, _ = evolve(state, ansatz.circuit, b, pruning=METHOD)
        return result, ansatz
    result, ansatz = evaluate(params)
    full, _ = evolve(state, ansatz.circuit, b, pruning=None)
    same(result, full)
    terminal = spd.init_gradient_spo(result, basis='+', backend=b)
    _, gates, _ = spd.backpropagate(terminal, ansatz.circuit, 0., 256,
                                   backend=b, progress=False)
    gradients = ansatz.parameter_gradients(gates)
    finite_difference = []
    for i in range(4):
        delta = np.zeros(4); delta[i] = 1e-5
        a, _ = evaluate(params + delta)
        z, _ = evaluate(params - delta)
        finite_difference.append((float(a.get_expectation_value('+')) -
                                  float(z.get_expectation_value('+'))) / 2e-5)
    np.testing.assert_allclose(gradients, finite_difference, atol=2e-7, rtol=2e-7)


def test_openqasm_barriers_and_progress(capsys):
    from spd.openqasm_frontend import parse_openqasm_str
    b = spd.BackendAdapter.from_name('numpy', precision='double')
    circuit = parse_openqasm_str('''OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rx(0.4) q[2];
barrier q;
rzz(0.3) q[1],q[2];
barrier q;
rzz(0.2) q[0],q[1];
''')
    result, info = spd.evolve(spd.create_spo({'ZII': 1.}, backend=b), circuit, 0., 256,
                              backend=b, pruning=METHOD, progress=True)
    assert info['pruning']['num_gates_retained'] == 1
    assert 'Progress: 100.00%' in capsys.readouterr().out
    assert result._pruning_record.method == METHOD
