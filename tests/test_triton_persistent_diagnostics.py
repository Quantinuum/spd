"""SPO-owned persistent diagnostics against the original per-gate implementation."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires NVIDIA GPU')

import spd
from spd import triton_backend as gpu
from spd.circuit_ir import PauliRotation, SingleQubitClifford, TwoQubitClifford, SkippedOperation, CircuitIR
from spd.triton_backend.operations import _rotation
from spd.triton_backend.persistent import _Storage
from tests.test_triton_backend import as_dict


def reference(state, operation, cutoff, cap):
    if isinstance(operation, PauliRotation):
        return _rotation(state, operation.pauli, operation.theta, cutoff, cap, False)
    backend = spd.BackendAdapter.from_name('triton', precision='double')
    return backend.apply_forward(state, operation, cutoff, cap)


def assert_states(actual, expected, tol):
    a, b = as_dict(actual), as_dict(expected)
    assert a.keys() == b.keys()
    np.testing.assert_allclose([a[k] for k in b], list(b.values()), atol=tol, rtol=tol)


def assert_info(actual, expected, tol):
    assert actual['num_str_truncated'] == expected['num_str_truncated']
    for k in ['truncated_l1_norm', 'truncated_l2_norm']:
        assert actual[k] == pytest.approx(expected[k], abs=tol, rel=tol)


@pytest.mark.parametrize('nq', [3, 65, 121])
@pytest.mark.parametrize('precision,tol', [('single', 3e-6), ('double', 5e-13)])
@pytest.mark.parametrize('cutoff,cap', [(0., None), (.2, None), (.2, 17)])
def test_per_gate_diagnostics_and_live_counts(nq, precision, tol, cutoff, cap):
    rng = np.random.default_rng(473)
    labels = [''.join(row) for row in rng.choice(list('IXYZ'), (129, nq))]
    initial = gpu.create_op(dict(zip(labels, rng.normal(size=129))), precision=precision)
    before = tuple(a.copy() for a in initial.to_host())
    work, expected = initial.copy(), initial
    operations = []
    for gate in ['H', 'S', 'Sdg', 'X', 'Y', 'Z', 'CX', 'CY', 'CZ']:
        operations.append(PauliRotation('rotation', ''.join(rng.choice(list('IXYZ'), nq)), float(rng.uniform(-2, 2))))
        operations.append(TwoQubitClifford('OpType.'+gate, 0, nq-1) if gate.startswith('C')
                          else SingleQubitClifford('OpType.'+gate, nq-1))
    for operation in operations:
        expected, count, _, info = reference(expected, operation, cutoff, cap)
        _, actual_count, _, actual_info = work.apply_in_place(operation, cutoff, cap)
        assert actual_count == count == work.get_size()
        assert_info(actual_info, info, tol)
        # Inspection only: filter dead slots from a copy without compacting storage.
        keys, c, _ = work._raw_arrays()
        nonzero = c != 0
        snapshot = gpu.SparsePauliOp(keys[nonzero], c[nonzero], nq)
        assert_states(snapshot, expected, tol)
        assert work.get_norm_square() == pytest.approx(expected.get_norm_square(), abs=tol, rel=tol)
        assert work.get_OSE() == pytest.approx(expected.get_OSE(), abs=tol, rel=tol)
    assert_states(work.compact(), expected, tol)
    for a,b in zip(initial.to_host(), before):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('value', [np.nextafter(.5, 0), .5, np.nextafter(.5, 1)])
def test_diagnostic_cutoff_equality(value):
    initial = gpu.create_op({'Z':value}, precision='double')
    actual, count, info = gpu.conjugate_pauli_rot_forward(initial, 'I', 0., .5)
    expected, expected_count, _, expected_info = _rotation(initial, 'I', 0., .5, None, False)
    assert count == expected_count
    assert_info(info, expected_info, 0.)
    assert_states(actual, expected, 0.)


def test_cap_diagnostics_do_not_subtract_large_retained_norm():
    initial = gpu.create_op({'I':1e20, 'X':1e-10, 'Y':2e-10}, precision='double')
    actual, size, info = gpu.conjugate_pauli_rot_forward(initial, 'I', 0., 0., max_num_str=np.int64(1))
    assert size == 1
    assert info['num_str_truncated'] == 2
    assert info['truncated_l1_norm'] == pytest.approx(3e-10, abs=1e-25)
    assert info['truncated_l2_norm'] == pytest.approx(np.sqrt(5)*1e-10, abs=1e-25)


def test_dead_keys_do_not_inflate_discarded_count():
    initial = gpu.create_op({'X':.09,'Y':.8,'Z':-.7,'I':.05}, precision='double')
    work, expected = initial.copy(), initial
    for p,t in [('Z',.6),('X',-.8),('Y',.9),('I',0.),('I',0.),('X',.2),('Z',.3)]:
        op=PauliRotation('rotation',p,t)
        expected, count, _, info = reference(expected,op,.1,None)
        _, actual_count, _, actual_info = work.apply_in_place(op,.1,None)
        assert actual_count == count
        assert_info(actual_info,info,5e-15)
    assert_states(work.compact(),expected,5e-15)


@pytest.mark.parametrize('progress', [False, True])
def test_public_history_uses_one_storage_and_live_reporting(monkeypatch, progress):
    import spd.run_circuit as runner
    initial = gpu.create_op({'X':.4,'Y':.7,'Z':.01,'I':0.}, precision='double')
    ops=(SingleQubitClifford('OpType.S',0), SkippedOperation('barrier')) + (PauliRotation('rotation','Z',.031),)*20
    expected=initial;history=runner._init_history();sizes=[];norms=[]
    for op in reversed(ops):
        expected,count,_,info=reference(expected,op,.1,32)
        if info is not None:runner._append_step_info(history,info)
        if count is not None:sizes.append(count);norms.append(expected.get_norm_square()/initial.get_norm_square())
    builds=[];reports=[];orig=_Storage._rebuild
    def rebuild(storage):
        builds.append(id(storage));orig(storage)
    monkeypatch.setattr(_Storage,'_rebuild',rebuild)
    monkeypatch.setattr(runner,'_print_progress',lambda *args,**kw: reports.append(args))
    actual, info=spd.evolve(initial,CircuitIR(1,ops),.1,17,progress=progress)
    assert len(set(builds))==1 and len(builds)==2  # initial index and exact S map
    assert type(actual) is gpu.SparsePauliOp
    assert_states(actual,expected,5e-14)
    expected_info=runner._finalize_info(history)
    for k in history:np.testing.assert_allclose(info['history'][k],expected_info['history'][k],atol=5e-14)
    for k in info.keys()-{'history'}:assert info[k]==pytest.approx(expected_info[k],abs=5e-14)
    if progress:
        assert [r[2] for r in reports[:-1]]==sizes
        np.testing.assert_allclose([r[1] for r in reports[:-1]],norms,atol=5e-14)
    else:assert not reports


def test_skipped_sequences_and_export_ownership():
    initial=gpu.create_op({'X':.4,'Y':.7,'I':0.},precision='double')
    result,info=spd.evolve(initial,CircuitIR(1,(SkippedOperation('barrier'),)),0.,8,progress=False)
    assert result is initial and info['num_steps_tracked']==0
    work=initial.copy()
    work.apply_in_place(SingleQubitClifford('OpType.H',0),0.,8)
    first=work.copy().compact();before=tuple(a.copy() for a in first.to_host())
    assert first.get_size()==3
    work.apply_in_place(PauliRotation('rotation','Z',.4),0.,8)
    work.compact()
    for a,b in zip(first.to_host(),before):np.testing.assert_array_equal(a,b)


def test_direct_empty_rotation_preserves_identity_and_validates_generator():
    initial = gpu.create_op({}, num_qubits=65)
    result, size, info = gpu.conjugate_pauli_rot_forward(initial, 'X', .3, .1)
    assert result is initial and size == 0
    assert_info(info, {'num_str_truncated':0, 'truncated_l1_norm':0., 'truncated_l2_norm':0.}, 0.)
    with pytest.raises(ValueError):
        gpu.conjugate_pauli_rot_forward(initial, 'invalid', .3, .1)
