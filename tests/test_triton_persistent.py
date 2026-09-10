"""Semantic checks for the opt-in persistent timestep prototype."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires NVIDIA GPU')

from spd.circuit_ir import PauliRotation
from spd.triton_backend import create_op, evolve_step
from spd.triton_backend.persistent import evolve_step_persistent
from tests.test_triton_backend import as_dict
from tests.test_triton_invariants import collision_observable


def check(state, ops, cutoff, cap=None, dead_fraction=.1):
    before = as_dict(state)
    expected = as_dict(evolve_step(state, ops, cutoff, cap))
    stats = {}
    result = evolve_step_persistent(state, ops, cutoff, cap,
                                    dead_fraction=dead_fraction, stats=stats)
    actual = as_dict(result)
    assert actual.keys() == expected.keys()
    np.testing.assert_allclose([actual[k] for k in expected], list(expected.values()),
                               atol=3e-13, rtol=3e-13)
    assert as_dict(state) == before
    return actual, stats


@pytest.mark.parametrize('nq', [3, 33, 121])
@pytest.mark.parametrize('cutoff', [0., .03, .5])
@pytest.mark.parametrize('cap', [None, 17])
def test_random_sequences(nq, cutoff, cap):
    rng = np.random.default_rng(910)
    labels = [''.join(row) for row in rng.choice(list('IXYZ'), (129, nq))]
    state = create_op(dict(zip(labels, rng.normal(size=len(labels)))), precision='double')
    ops = [PauliRotation('rotation', ''.join(rng.choice(list('IXYZ'), nq)), float(rng.uniform(-3, 3)))
           for _ in range(8)]
    check(state, ops, cutoff, cap)


def test_collision_chains():
    state = create_op(collision_observable(), precision='double')
    ops = [PauliRotation('rotation', p, .43) for p in ['YZ'*8, 'XY'*8, 'ZX'*8]]
    for _ in range(3):
        check(state, ops, .01)


@pytest.mark.parametrize('dead_fraction', [.01, .9])
def test_deletion_and_legitimate_reactivation(dead_fraction):
    # Keep deleted keys physically present with the high compaction threshold.
    # Subsequent rotations must see zero, not the discarded old coefficient.
    state = create_op({'X': .09, 'Y': .8, 'Z': -.7, 'I': .05}, precision='double')
    ops = [PauliRotation('rotation', p, t) for p,t in
           [('Z', .6), ('X', -.8), ('Y', .9), ('X', .2), ('Z', .3)]]
    check(state, ops, .1, dead_fraction=dead_fraction)


@pytest.mark.parametrize('value', [np.nextafter(.5, 0), .5, np.nextafter(.5, 1)])
def test_strict_cutoff(value):
    state = create_op({'Z': value}, precision='double')
    actual, _ = check(state, [PauliRotation('rotation', 'I', 0.)], .5)
    assert len(actual) == (value > .5)


def test_empty_and_no_operations():
    state = create_op({'Z': 0.}, precision='double')
    assert evolve_step_persistent(state, []) is state
    check(create_op({}, num_qubits=3), [PauliRotation('rotation', 'XYZ', .3)], 0.)


@pytest.mark.parametrize('dead_fraction', [.01, .9])
def test_every_gate_with_persistent_dead_keys(dead_fraction):
    from spd.triton_backend import _gate_data, conjugate_pauli_rotation
    from spd.triton_backend.persistent import _Storage
    rng = np.random.default_rng(418)
    labels = [''.join(row) for row in rng.choice(list('IXYZ'), (129, 8))]
    state = create_op(dict(zip(labels, rng.normal(size=129))), precision='double')
    storage = _Storage(state, dead_fraction)
    for index in range(24):
        gate = ['I'] * 8
        gate[index % 8] = 'XYZ'[index % 3]
        label = ''.join(gate)
        theta = float(rng.uniform(-2, 2))
        packed, params = _gate_data(label, theta, .3, 8, state.c_array.dtype, state.c_array.device)
        storage.apply(packed, params, None)
        state = conjugate_pauli_rotation(state, label, theta, .3)
        expected = as_dict(state)
        keys = storage.keys[:storage.n].cpu().numpy().view(np.uint32)
        coefficients = storage.coeff[:storage.n].cpu().numpy()
        actual = {tuple(k): float(c) for k,c in zip(keys, coefficients) if c != 0}
        assert storage.live == len(actual) == len(expected)
        assert len({tuple(k) for k in keys}) == storage.n
        table = storage.table.cpu().numpy()
        assert sorted(table[table >= 0].tolist()) == list(range(storage.n))
        assert actual.keys() == expected.keys()
        np.testing.assert_allclose([actual[k] for k in expected], list(expected.values()), atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize('precision,atol', [('single', 2e-6), ('double', 3e-13)])
def test_two_qubit_dense_oracle(precision, atol):
    import itertools
    import math
    from tests.test_triton_invariants import matrix, labels_from_keys
    labels = [''.join(p) for p in itertools.product('IXYZ', repeat=2)]
    basis = np.array([matrix(p) for p in labels])
    rng = np.random.default_rng(711)
    coefficients = rng.normal(size=16).astype(np.float32 if precision == 'single' else np.float64)
    state = create_op(dict(zip(labels, coefficients)), precision=precision)
    initial = np.einsum('i,ijk->jk', coefficients, basis)
    for gate, generator in zip(labels, basis):
        theta = .731
        u = math.cos(theta/2)*np.eye(4) - 1j*math.sin(theta/2)*generator
        evolved = u.conj().T @ initial @ u
        expected_c = np.einsum('aij,ji->a', basis, evolved).real/4
        expected = np.einsum('i,ijk->jk', np.where(abs(expected_c) > .2, expected_c, 0), basis)
        result = evolve_step_persistent(state, [PauliRotation('rotation', gate, theta)], .2)
        keys, values = result.to_host()
        actual = sum((c*matrix(p) for p,c in zip(labels_from_keys(keys, 2), values)), np.zeros((4,4), complex))
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=atol)
