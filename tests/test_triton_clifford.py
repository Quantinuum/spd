"""All Clifford maps against dense unitaries and multiword reference states."""

import itertools
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA GPU")

from spd import triton_backend as gpu, numpy_backend as cpu
from tests.test_triton_invariants import matrix, labels_from_keys
from tests.test_triton_backward import reference_state, terms

GATES = ["H", "S", "Sdg", "X", "Y", "Z", "CX", "CY", "CZ"]


def unitary(gate):
    if gate.startswith('C'):
        return np.kron(np.diag([1, 0]), np.eye(2)) + np.kron(np.diag([0, 1]), matrix(gate[1]))
    local = {'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
             'S': np.diag([1, 1j]), 'Sdg': np.diag([1, -1j])}
    return np.kron(local[gate] if gate in local else matrix(gate), np.eye(2))


@pytest.mark.parametrize('gate', GATES)
@pytest.mark.parametrize('backward', [False, True])
@pytest.mark.parametrize('precision,tol', [('single', 3e-6), ('double', 1e-13)])
def test_dense_unitary_conjugation(gate, backward, precision, tol):
    labels = [''.join(p) for p in itertools.product('IXYZ', repeat=2)]
    values = np.random.default_rng(289).normal(size=(16, 2))
    initial = (gpu.create_gradient_op(dict(zip(labels, values)), precision=precision) if backward
               else gpu.create_op(dict(zip(labels, values[:, 0])), precision=precision))
    before = tuple(a.copy() for a in initial.to_host())
    args = (0, 1) if gate.startswith('C') else (0,)
    result = getattr(gpu, f'conjugate_{gate}_{"backward" if backward else "forward"}')(initial, *args)
    actual = result.to_host()
    result_labels = labels_from_keys(actual[0], 2)
    assert len(set(result_labels)) == 16
    u = unitary(gate)
    for channel in range(2 if backward else 1):
        op = sum(c * matrix(p) for p, c in zip(labels, values[:, channel]))
        expected = u @ op @ u.conj().T if backward else u.conj().T @ op @ u
        got = sum(c * matrix(p) for p, c in zip(result_labels, actual[channel + 1]))
        np.testing.assert_allclose(got, expected, atol=tol, rtol=tol)
    for a, b in zip(initial.to_host(), before):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('gate', GATES)
@pytest.mark.parametrize('backward', [False, True])
@pytest.mark.parametrize('qubits', [(31, 32), (31, 64), (32, 33)])
def test_word_boundaries_against_numpy(gate, backward, qubits):
    cpu.utils.set_packbit(32); cpu.set_precision('double')
    rng = np.random.default_rng(654)
    labels = [''.join(rng.choice(list('IXYZ'), 65)) for _ in range(129)]
    data = dict(zip(labels, rng.normal(size=(129, 2))))
    a = reference_state(cpu, data) if backward else cpu.create_op({p: pair[0] for p, pair in data.items()})
    b = gpu.create_gradient_op(data) if backward else gpu.create_op({p: pair[0] for p, pair in data.items()})
    args = qubits if gate.startswith('C') else qubits[:1]
    name = f'conjugate_{gate}_{"backward" if backward else "forward"}'
    a, b = getattr(cpu, name)(a, *args), getattr(gpu, name)(b, *args)
    if backward:
        actual = terms(b)
    else:
        keys, c = b.to_host()
        actual = {tuple(k): v for k, v in zip(keys, c)}
    assert actual.keys() == a.keys()
    np.testing.assert_allclose(list(actual.values()), [a[k] for k in actual], atol=1e-14)


def test_clifford_input_validation_and_empty():
    state = gpu.create_op({}, num_qubits=65)
    assert gpu.conjugate_CY_forward(state, 0, 64).get_size() == 0
    grad = gpu.create_gradient_op({}, num_qubits=65)
    assert gpu.conjugate_S_backward(grad, 0).get_size() == 0
    with pytest.raises(ValueError):
        gpu.conjugate_CX_forward(state, 0, 0)
    with pytest.raises(ValueError):
        gpu.conjugate_H_forward(state, 65)
    with pytest.raises(TypeError):
        gpu.conjugate_H_backward(state, 0)
