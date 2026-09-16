"""References use dense state-picture Kraus evolution, independent of SPD."""
from dataclasses import replace
import pickle

import numpy as np
import pytest

import spd
from spd.circuit_ir import PauliRotation as Rot, SingleQubitClifford as One, TwoQubitClifford as Two, SkippedOperation
from spd.checkpoints import CheckpointStore


@pytest.fixture(params=["numpy", "triton"])
def channel_backend_name(request):
    """Backends implementing native channel kernels; no host fallback."""
    if request.param == "triton":
        torch = pytest.importorskip("torch")
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
    return request.param

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])
PAULIS = dict(zip('IXYZ', [I, X, Y, Z]))


def tensor(matrices):
    out = np.ones((1, 1))
    for m in matrices:
        out = np.kron(out, m)
    return out


def embed(m, q, n):
    return tensor([m if i == q else I for i in range(n)])


def dense(circuit, observable, basis='0'):
    n = circuit.system_size
    initial = np.array([1, 0]) if basis == '0' else np.array([1, 1]) / np.sqrt(2)
    ket = tensor([initial.reshape(2, 1) for _ in range(n)]).ravel()
    rho = np.outer(ket, ket.conj())
    for op in circuit.operations:
        if isinstance(op, (spd.CreateZero, spd.ResetZero)):
            kraus = [embed(np.array([[1, 0], [0, 0]]), op.qubit, n),
                     embed(np.array([[0, 1], [0, 0]]), op.qubit, n)]
            rho = sum(k @ rho @ k.conj().T for k in kraus)
            continue
        if isinstance(op, (spd.Discard, SkippedOperation)):
            # Discarded columns are identity in the final observable. Keeping
            # their environment in this reference gives the same partial trace.
            continue
        if isinstance(op, Rot):
            p = tensor([PAULIS[a] for a in op.pauli[:n].ljust(n, 'I')])
            u = np.cos(op.theta / 2) * np.eye(2**n) - 1j * np.sin(op.theta / 2) * p
        elif isinstance(op, One):
            matrix = {'OpType.H': (X + Z) / np.sqrt(2), 'OpType.S': np.diag([1, 1j]),
                      'OpType.X': X, 'OpType.Y': Y, 'OpType.Z': Z}[op.gate_name]
            u = embed(matrix, op.qubit, n)
        elif isinstance(op, Two):
            target = {'OpType.CX': X, 'OpType.CY': Y, 'OpType.CZ': Z}[op.gate_name]
            u = embed((I + Z)/2, op.control_qubit, n) + embed((I - Z)/2, op.control_qubit, n) @ embed(target, op.target_qubit, n)
        rho = u @ rho @ u.conj().T
    o = sum(c * tensor([PAULIS[a] for a in p]) for p, c in observable.items())
    return float(np.trace(rho @ o).real)


def finite_differences(circuit, observable, basis='0'):
    gradients = []
    for i, op in enumerate(circuit.operations):
        if not isinstance(op, Rot):
            continue
        shifted = []
        for delta in [-1e-6, 1e-6]:
            ops = list(circuit.operations)
            ops[i] = replace(op, theta=op.theta + delta)
            shifted.append(dense(spd.CircuitIR(circuit.system_size, ops), observable, basis))
        gradients.append((shifted[1] - shifted[0]) / 2e-6)
    return gradients


def evaluate(circuit, observable, backend='numpy', basis='0', **kwargs):
    spo = spd.create_spo(observable, system_size=circuit.system_size, backend_name=backend, precision='double')
    f, info = spd.evolve(spo, circuit, 0, 100000, progress=False, **kwargs)
    value = float(f.get_expectation_value(basis))
    g = spd.init_gradient_spo(f, basis=basis)
    result, grads, _ = spd.backpropagate(g, circuit, 0, 100000, progress=False)
    return value, np.array(grads), f, info, result


@pytest.mark.parametrize('basis', ['0', '+'])
def test_mixed_inputs_entanglement_and_multiple_boundaries(channel_backend_name, basis):
    backend_name = channel_backend_name
    c = spd.CircuitIR(4, [
        Rot('Ry', 'IYII', .37), spd.CreateZero(3), Two('OpType.CX', 1, 3),
        Rot('RXX', 'IXIX', -.21), spd.ResetZero(1), Rot('Rx', 'IIIX', .61),
        spd.CreateZero(0), Two('OpType.CY', 3, 0), Rot('RYY', 'YIIY', .14),
        spd.ResetZero(3), Rot('Ry', 'YIII', -.72), spd.Discard(2), spd.Discard(1),
    ])
    observable = {'ZIIX': .8, 'XIII': -.17, 'IIIZ': .51, 'IIII': .3}
    value, grads, _, _, _ = evaluate(c, observable, backend_name, basis)
    assert c.initial_active_qubits == [1, 2]
    assert c.final_active_qubits == [0, 3]
    assert value == pytest.approx(dense(c, observable, basis), abs=2e-12)
    np.testing.assert_allclose(grads, finite_differences(c, observable, basis), atol=2e-9)


@pytest.mark.parametrize('observable', [{'I': 1., 'Z': -1.}, {'X': 1.}, {'Y': 1.}, {'I': 1., 'Z': 1.}])
@pytest.mark.parametrize('channel', [spd.CreateZero, spd.ResetZero])
def test_cancellation_and_killed_terms_restore_earlier_gradient(channel_backend_name, observable, channel):
    backend_name = channel_backend_name
    c = spd.CircuitIR(1, [channel(0), Rot('Ry', 'Y', 0.)])
    value, grads, _, _, _ = evaluate(c, observable, backend_name)
    assert value == pytest.approx(dense(c, observable))
    np.testing.assert_allclose(grads, finite_differences(c, observable), atol=1e-9)


def test_exact_cancellation_nonzero_derivative_with_shared_parameter(channel_backend_name):
    backend_name = channel_backend_name
    # At p=0: <X> = 0, while d<X>/dp = 1. Cancellation below yields a zero
    # observable before an ordinary-input rotation, but a nonzero later gradient.
    def circuit(p):
        return spd.CircuitIR(2, [Rot('Rx', 'XI', .3*p), spd.CreateZero(1),
                                  Rot('Ry', 'IY', 1.7*p), spd.ResetZero(0),
                                  Rot('Rx', 'IX', -.2*p)])
    c = circuit(0.)
    obs = {'II': 1., 'IZ': -1., 'IX': 1.}
    value, grads, _, _, _ = evaluate(c, obs, backend_name)
    vc = spd.VariationalCircuit(c, [0, 0, 0], (1,), [.3, 1.7, -.2])
    reference = (dense(circuit(1e-6), obs) - dense(circuit(-1e-6), obs)) / 2e-6
    assert value == 0.
    assert vc.parameter_gradients(grads)[0] == pytest.approx(reference, abs=1e-9)


def test_creation_widths_and_scalar():
    ops = []
    for created, pairs in [([3, 7], [(3, 7)]), ([1, 5], [(1, 3), (5, 7)]),
                           ([0, 2, 4, 6], [(0, 1), (2, 3), (4, 5), (6, 7)])]:
        ops.extend(spd.CreateZero(q) for q in created)
        for a, b in pairs:
            p = ['I'] * 8
            p[a] = 'Y'
            ops.extend([Rot('Ry', ''.join(p), .23), Two('OpType.CX', a, b)])
        ops.append(SkippedOperation('OpType.Barrier'))
    c = spd.CircuitIR(8, ops)
    value, _, f, info, _ = evaluate(c, {'ZZZZZZZZ': 1.})
    assert c.initial_active_qubits == []
    assert f.active_qubits == []
    assert list(f) == [()]
    widths = info['active_widths']
    assert [w for i, w in enumerate(widths) if i == 0 or widths[i-1] != w] == list(range(8, -1, -1))
    assert all(w in widths for w in [8, 4, 2, 0])
    assert value == pytest.approx(dense(c, {'ZZZZZZZZ': 1.}), abs=1e-12)


def test_reset_equivalent_to_discard_and_prepare_new_index():
    a = spd.CircuitIR(2, [Rot('Ry', 'YI', .3), Two('OpType.CX', 0, 1), spd.ResetZero(0),
                         Rot('Ry', 'YI', .7), Two('OpType.CX', 0, 1)])
    b = spd.CircuitIR(3, [Rot('Ry', 'YII', .3), Two('OpType.CX', 0, 1), spd.Discard(0),
                         spd.CreateZero(2), Rot('Ry', 'IIY', .7), Two('OpType.CX', 2, 1)])
    av, ag, *_ = evaluate(a, {'ZZ': 1.})
    bv, bg, *_ = evaluate(b, {'IZZ': 1.})
    assert av == pytest.approx(bv)
    np.testing.assert_allclose(ag, bg, atol=1e-12)


def test_disk_round_trips_isolation_and_cleanup(tmp_path, monkeypatch):
    c = spd.CircuitIR(2, [spd.CreateZero(1), Rot('Ry', 'IY', .4), spd.ResetZero(0),
                         Rot('Rx', 'XI', .2), spd.ResetZero(1)])
    o = spd.create_spo({'ZZ': 1.}, backend_name='numpy', precision='double')
    a, _ = spd.evolve(o, c, 0, 100, progress=False, checkpoint_directory=tmp_path, checkpoint_memory_budget_bytes=0)
    b, _ = spd.evolve(o, c, 0, 100, progress=False, checkpoint_directory=tmp_path, checkpoint_memory_budget_bytes=0)
    ta, tb = a._channel_checkpoints, b._channel_checkpoints
    assert ta.directory != tb.directory
    assert len(list(ta.directory.glob('*.pickle'))) == 3
    loaded = []
    original = CheckpointStore.load
    def load(self, i):
        result = original(self, i)
        loaded.append(i)
        assert result.system_size == 2
        assert result.active_qubits is not None
        return result
    monkeypatch.setattr(CheckpointStore, 'load', load)
    spd.backpropagate(spd.init_gradient_spo(a), c, 0, 100, progress=False)
    assert loaded == [0, 2, 4]
    assert not ta.directory.exists()
    assert tb.directory.exists()
    with pytest.raises(ValueError, match='live checkpoints'):
        spd.backpropagate(spd.init_gradient_spo(a), c, 0, 100, progress=False)
    spd.close_channel_checkpoints(b)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('ops', [
    [One('OpType.H', 1), spd.CreateZero(1)],
    [spd.ResetZero(1), spd.CreateZero(1)],
    [spd.CreateZero(1), spd.CreateZero(1)],
    [spd.Discard(1), spd.Discard(1)],
    [spd.Discard(1), Rot('Rx', 'IX', .1)],
    [spd.CreateZero(1), spd.Discard(1), spd.CreateZero(1)],
    [spd.CreateZero(2)], [spd.ResetZero(-1)], [spd.Discard(3)],
])
def test_invalid_activity(ops):
    with pytest.raises(ValueError):
        spd.CircuitIR(2, ops)


def test_barrier_does_not_use_inactive_indices():
    c = spd.CircuitIR(2, [SkippedOperation('OpType.Barrier'), spd.CreateZero(1), One('OpType.H', 0)])
    assert c.initial_active_qubits == [0]
    assert evaluate(c, {'ZZ': 1.})[0] == pytest.approx(0.)


@pytest.mark.parametrize('active,terms', [([0, 0], {'II': 1}), ([2], {'I': 1}), ([-1], {'I': 1}),
                                         ([0], {'II': 1}), ([], {'I': 1})])
def test_invalid_spo_metadata(active, terms):
    with pytest.raises(ValueError):
        spd.create_spo(terms, system_size=2, active_qubits=active, backend_name='numpy')


def test_observable_output_conventions_and_nonconsecutive_translation(channel_backend_name):
    backend_name = channel_backend_name
    c = spd.CircuitIR(4, [spd.CreateZero(3), spd.Discard(0), spd.Discard(2),
                         Rot('Ry', 'IYII', .4), Two('OpType.CX', 1, 3)])
    full = spd.create_spo({'IZIZ': 1.}, backend_name=backend_name, precision='double')
    # Explicit input ordering is normalized to deterministic sorted columns.
    compact = spd.create_spo({'ZZ': 1.}, system_size=4, active_qubits=[3, 1],
                             backend_name=backend_name, precision='double')
    for o in [full, compact]:
        f, _ = spd.evolve(o, c, 0, 100, progress=False)
        assert float(f.get_expectation_value()) == pytest.approx(dense(c, {'IZIZ': 1.}))
        spd.close_channel_checkpoints(f)
    for o in [spd.create_spo({'XZII': 1.}, backend_name=backend_name),
              spd.create_spo({'Z': 1.}, system_size=4, active_qubits=[1], backend_name=backend_name)]:
        with pytest.raises(ValueError, match='discarded|final active'):
            spd.evolve(o, c, 0, 100, progress=False)


def test_zero_operator_and_scalar_are_distinct(backend_name):
    zero = spd.create_spo({}, system_size=2, active_qubits=[1], backend_name=backend_name)
    scalar = spd.create_spo({'': 2.}, system_size=2, active_qubits=[], backend_name=backend_name)
    assert zero.active_qubits == [1]
    assert scalar.active_qubits == []
    assert float(zero.get_expectation_value()) == 0.
    assert float(scalar.get_expectation_value()) == 2.
    for obj in [zero, scalar, spd.init_gradient_spo(scalar).to_spo()]:
        copy = pickle.loads(pickle.dumps(obj))
        assert copy.active_qubits == obj.active_qubits
        assert copy.system_size == 2


def test_pytket_creation_discard_reset_and_shared_factors(channel_backend_name):
    backend_name = channel_backend_name
    from pytket import Circuit, Qubit
    from spd.pytket_frontend import parse_pytket_circuit
    def make(p):
        c = Circuit(3)
        c.qubit_create(Qubit(2))
        c.Ry(.7*p, 0).CX(0, 2).Reset(0).Rz(.2, 2).Rx(-.4*p, 2)
        c.qubit_discard(Qubit(1))
        c.add_barrier([0, 1, 2])
        return c
    c = make(.31)
    ir = parse_pytket_circuit(c, 32)
    assert ir.initial_active_qubits == [0, 1]
    assert ir.final_active_qubits == [0, 2]
    vc = spd.VariationalCircuit(c, [0, -1, 0], (1,), [.7, 1., -.4])
    spo = spd.create_spo({'IIZ': 1.}, backend_name=backend_name, precision='double')
    f, _ = spd.evolve(spo, c, 0, 1000, progress=False)
    _, grads, _ = spd.backpropagate(spd.init_gradient_spo(f), c, 0, 1000, progress=False)
    expected = (dense(parse_pytket_circuit(make(.310001), 32), {'IIZ': 1.}) -
                dense(parse_pytket_circuit(make(.309999), 32), {'IIZ': 1.})) / 2e-6
    assert vc.parameter_gradients(grads)[0] == pytest.approx(expected, abs=2e-9)
    ordinary = Circuit(1); ordinary.add_qubit(Qubit(1))
    assert parse_pytket_circuit(ordinary, 32).initial_active_qubits == [0, 1]


def test_approximation_keeps_channels_exact_and_reports_unitary_truncation():
    c = spd.CircuitIR(2, [spd.CreateZero(1), spd.ResetZero(0)])
    o = spd.create_spo({'IZ': .001, 'II': -.001, 'XY': .01}, backend_name='numpy', precision='double')
    f, info = spd.evolve(o, c, .1, 1, progress=False)
    assert float(f.get_expectation_value()) == 0.
    assert info['sum_num_str_truncated'] == 0
    _, grads, backinfo = spd.backpropagate(spd.init_gradient_spo(f), c, .1, 1, progress=False)
    assert grads == []
    assert backinfo['sum_num_str_truncated'] == 0


def test_checkpoint_mismatch_rejected_without_consuming_tape():
    c = spd.CircuitIR(1, [spd.CreateZero(0), Rot('Rx', 'X', .2)])
    o = spd.create_spo({'Z': 1.}, backend_name='numpy')
    f, _ = spd.evolve(o, c, 0, 100, progress=False)
    g = spd.init_gradient_spo(f)
    with pytest.raises(ValueError, match='match'):
        spd.backpropagate(g, c, .1, 100, progress=False)
    spd.backpropagate(g, c, 0, 100, progress=False)


def test_packed_storage_really_shrinks_across_word_boundary():
    c = spd.CircuitIR(33, [*[spd.CreateZero(i) for i in range(32)],
                           Rot('phase', 'I' * 33, .1), spd.CreateZero(32)])
    o = spd.create_spo({'Z' * 33: 1.}, backend_name='numpy')
    f, info = spd.evolve(o, c, 0, 100, progress=False)
    tape = f._channel_checkpoints
    a, b = tape.load(33), tape.load(0)
    assert len(next(iter(a))) == 4
    assert len(next(iter(b))) == 2
    assert a.active_qubits == list(range(33))
    assert b.active_qubits == list(range(32))
    assert list(f) == [()]
    assert f.get_expectation_value() == 1.
    spd.backpropagate(spd.init_gradient_spo(f), c, 0, 100, progress=False)


def test_channel_scalars_and_gradients(channel_backend_name):
    name = channel_backend_name
    c = spd.CircuitIR(1, [spd.CreateZero(0), Rot('Ry', 'Y', 0.)])
    value, grads, f, _, result = evaluate(c, {'X': 1.}, name)
    assert value == 0.
    assert f.active_qubits == []
    assert grads[0] == pytest.approx(1.)
    assert result.active_qubits == [0]
    clone = f.copy() if hasattr(f, 'copy') else f * 1
    assert clone.active_qubits == []
    assert clone.system_size == 1


def test_metadata_survives_scalar_algebra_and_jax_pytree(backend_name):
    o = spd.create_spo({'ZI': 1., 'IX': .2}, system_size=4, active_qubits=[1, 3], backend_name=backend_name)
    for changed in [o * 2, o * 0, o + o, spd.init_gradient_spo(o).to_spo()]:
        assert changed.active_qubits == [1, 3]
        assert changed.system_size == 4
    other = spd.create_spo({'ZI': 1.}, system_size=4, active_qubits=[0, 2], backend_name=backend_name)
    with pytest.raises(ValueError, match='mapping'):
        o + other
    if backend_name == 'jax':
        import jax
        cloned = jax.tree_util.tree_map(lambda x: x, o)
        assert cloned.active_qubits == [1, 3]
        assert cloned.system_size == 4


def test_nonzero_angle_exact_cancellation_retains_derivative(channel_backend_name):
    backend_name = channel_backend_name
    angle = .43
    c = spd.CircuitIR(1, [spd.ResetZero(0), Rot('Ry', 'Y', angle)])
    value, grads, *_ = evaluate(c, {'I': -np.cos(angle), 'Z': 1.}, backend_name)
    assert abs(value) < 1e-15
    assert grads[0] == pytest.approx(-np.sin(angle), abs=1e-12)


@pytest.mark.parametrize('seed', range(6))
def test_random_unitary_segments_against_density_matrix(seed, channel_backend_name):
    rng = np.random.default_rng(seed)
    def rotation(active):
        p = ['I'] * 3
        for i in active:
            p[i] = rng.choice(list('IXYZ'))
        return Rot('Pauli', ''.join(p), float(rng.uniform(-1, 1)))
    c = spd.CircuitIR(3, [rotation([0, 1]), spd.CreateZero(2), rotation([0, 1, 2]),
                         Two('OpType.CX', 0, 2), spd.ResetZero(0), rotation([0, 1, 2]),
                         spd.ResetZero(2), rotation([0, 1, 2]), spd.Discard(1)])
    obs = {'ZIZ': .2, 'XIY': .7, 'YIX': -.4, 'III': .1}
    value, grads, *_ = evaluate(c, obs, channel_backend_name)
    assert value == pytest.approx(dense(c, obs), abs=1e-12)
    np.testing.assert_allclose(grads, finite_differences(c, obs), atol=1e-9)


def test_unitary_truncation_remains_a_gradient_approximation():
    c = spd.CircuitIR(2, [spd.CreateZero(0), Rot('Ry', 'YI', .01), spd.ResetZero(1)])
    o = spd.create_spo({'ZI': 1.}, backend_name='numpy', precision='double')
    f, info = spd.evolve(o, c, .1, 100, progress=False)
    assert info['history']['num_str_truncated'] == [0, 1, 0]
    assert f.get_expectation_value() == pytest.approx(np.cos(.01))
    _, grads, _ = spd.backpropagate(spd.init_gradient_spo(f), c, .1, 100, progress=False)
    # The existing reconstruction algorithm lost the small X partner. This is
    # intentionally an approximate gradient, not the derivative of cos(.01).
    assert grads == [0.]
    assert evaluate(c, {'ZI': 1.})[1][0] == pytest.approx(-np.sin(.01))


@pytest.mark.parametrize('budget', [0, 512 * 1024**2])
def test_cleanup_on_execution_errors(tmp_path, monkeypatch, budget):
    c = spd.CircuitIR(1, [Rot('Ry', 'Y', .1), spd.ResetZero(0)])
    o = spd.create_spo({'Z': 1.}, backend_name='numpy')
    from spd.backend_adapter import BackendAdapter
    def fail(*args, **kwargs):
        raise RuntimeError('injected execution failure')
    with monkeypatch.context() as patch:
        patch.setattr(BackendAdapter, 'apply_forward', fail)
        with pytest.raises(RuntimeError, match='injected'):
            spd.evolve(o, c, 0, 100, progress=False, checkpoint_directory=tmp_path, checkpoint_memory_budget_bytes=budget)
    assert list(tmp_path.iterdir()) == []
    f, _ = spd.evolve(o, c, 0, 100, progress=False, checkpoint_directory=tmp_path, checkpoint_memory_budget_bytes=budget)
    with monkeypatch.context() as patch:
        patch.setattr(BackendAdapter, 'apply_backward', fail)
        with pytest.raises(RuntimeError, match='injected'):
            spd.backpropagate(spd.init_gradient_spo(f), c, 0, 100, progress=False)
    assert list(tmp_path.iterdir()) == []


def test_pytket_numeric_pauli_exp_box_across_reset():
    from pytket import Circuit
    from pytket.circuit import PauliExpBox
    from pytket.pauli import Pauli
    c = Circuit(2).Ry(.2, 0).Reset(1)
    c.add_pauliexpbox(PauliExpBox([Pauli.X, Pauli.Y], .31), [0, 1])
    from spd.pytket_frontend import parse_pytket_circuit
    ir = parse_pytket_circuit(c, 32)
    value, gradients, *_ = evaluate(ir, {'ZZ': 1.})
    assert value == pytest.approx(dense(ir, {'ZZ': 1.}))
    np.testing.assert_allclose(gradients, finite_differences(ir, {'ZZ': 1.}), atol=1e-9)


def test_global_phase_on_empty_active_set(channel_backend_name):
    backend_name = channel_backend_name
    c = spd.CircuitIR(1, [Rot('Phase', 'I', .37), SkippedOperation('OpType.Barrier'),
                         spd.CreateZero(0), Rot('Ry', 'Y', .2)])
    value, grads, *_ = evaluate(c, {'Z': 1.}, backend_name)
    assert value == pytest.approx(np.cos(.2))
    np.testing.assert_allclose(grads, [0., -np.sin(.2)], atol=1e-12)


def test_scalar_creation_preserves_requested_precision(all_backend):
    name, _ = all_backend
    from spd.numpy_backend import utils
    utils.set_precision('single')
    value = 1.1234567891234567
    o = spd.create_spo({'': value}, system_size=2, active_qubits=[],
                       backend_name=name, precision='double')
    assert float(o.get_expectation_value()) == value


@pytest.mark.parametrize('angles', [(0., 0.), (.31, -.48)])
def test_barrier_separated_creations_use_one_complete_checkpoint(tmp_path, monkeypatch, angles):
    barriers = [SkippedOperation('OpType.Barrier'), SkippedOperation('OpType.Barrier')]
    c = spd.CircuitIR(2, [*barriers, spd.CreateZero(1), *barriers, spd.CreateZero(0),
                         *barriers, Rot('Ry', 'YI', angles[0]), Rot('Ry', 'IY', angles[1])])
    obs = {'II': 1., 'ZZ': -1., 'XI': .7, 'YX': .2, 'IX': -.3}
    o = spd.create_spo(obs, backend_name='numpy', precision='double')
    f, info = spd.evolve(o, c, 0., 1000, progress=False, checkpoint_directory=tmp_path, checkpoint_memory_budget_bytes=0)
    tape = f._channel_checkpoints
    assert [p.name for p in tape.directory.glob('*.pickle')] == ['2.pickle']
    original = tape.load(2)
    assert original.active_qubits == [0, 1]
    assert original.system_size == 2
    assert original.get_size() >= len(obs)  # Include killed terms and zero rotation partners.
    assert info['active_widths'] == [2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0]
    assert info['num_steps_tracked'] == 4  # Two rotations, two contractions; no barriers.
    assert float(f.get_expectation_value()) == pytest.approx(dense(c, obs), abs=1e-12)
    calls = []
    load = CheckpointStore.load
    def track_load(self, key):
        calls.append(key)
        return load(self, key)
    monkeypatch.setattr(CheckpointStore, 'load', track_load)
    _, gradients, backinfo = spd.backpropagate(spd.init_gradient_spo(f), c, 0., 1000, progress=False)
    assert calls == [2]
    assert backinfo['num_steps_tracked'] == 4
    np.testing.assert_allclose(gradients, finite_differences(c, obs), atol=1e-9)
    assert not tape.directory.exists()


def test_grouping_stops_at_gates_and_repeated_indices(tmp_path):
    from spd.run_circuit import _group_contractions, _ContractionGroup
    barrier = SkippedOperation('OpType.Barrier')
    c = spd.CircuitIR(3, [spd.CreateZero(2), barrier, spd.ResetZero(0), barrier,
                         spd.ResetZero(0), One('OpType.H', 1), spd.ResetZero(2)])
    groups = [op for op in _group_contractions(c.operations) if isinstance(op, _ContractionGroup)]
    assert [len(op.contractions) for op in groups] == [2, 1, 1]
    assert [op.checkpoint_key for op in groups] == [0, 4, 6]
    obs = {'XIZ': 1., 'IZZ': .7, 'IYI': .3}
    value, gradients, *_ = evaluate(c, obs, checkpoint_directory=tmp_path, checkpoint_memory_budget_bytes=0)
    assert value == pytest.approx(dense(c, obs), abs=1e-12)
    assert gradients.size == 0
    assert list(tmp_path.iterdir()) == []


def test_large_creation_batch_has_one_checkpoint(tmp_path, monkeypatch):
    c = spd.CircuitIR(33, [spd.CreateZero(i) for i in range(33)])
    o = spd.create_spo({'Z' * 33: 1.}, backend_name='numpy')
    f, info = spd.evolve(o, c, 0., 100, progress=False, checkpoint_directory=tmp_path, checkpoint_memory_budget_bytes=0)
    tape = f._channel_checkpoints
    assert len(list(tape.directory.glob('*.pickle'))) == 1
    assert len(next(iter(tape.load(0)))) == 4
    assert list(f) == [()]  # Four packed words become a true zero-column scalar.
    assert info['active_widths'] == list(range(33, -1, -1))
    _, gradients, _ = spd.backpropagate(spd.init_gradient_spo(f), c, 0., 100, progress=False)
    assert gradients == []
    assert f.get_expectation_value() == 1.


def test_backend_channel_capability_preflight(all_backend, tmp_path, monkeypatch):
    name, module = all_backend
    adapter = spd.BackendAdapter.from_name(name, precision='double')
    if name in ('numpy', 'triton'):
        adapter.require_channel_support()
        return
    o = spd.create_spo({'II': 1.}, backend=adapter)
    g = spd.init_gradient_spo(o, backend=adapter)
    def unexpected(*args, **kwargs):
        pytest.fail('Unsupported backend started execution or allocated a checkpoint')
    monkeypatch.setattr(CheckpointStore, '__init__', unexpected)
    monkeypatch.setattr(module, 'conjugate_H_forward', unexpected)
    monkeypatch.setattr(module, 'conjugate_H_backward', unexpected)
    monkeypatch.setattr(spd.numpy_backend, 'contract_zero_forward', unexpected)
    for channel in [spd.CreateZero(0), spd.ResetZero(0), spd.Discard(0)]:
        c = spd.CircuitIR(2, [One('OpType.H', 1), channel, One('OpType.H', 1)])
        with pytest.raises(NotImplementedError, match=f"backend '{name}'"):
            spd.evolve(o, c, 0., 100, backend=adapter, checkpoint_directory=tmp_path, checkpoint_memory_budget_bytes=0)
        with pytest.raises(NotImplementedError, match=f"backend '{name}'"):
            spd.backpropagate(g, c, 0., 100, backend=adapter)
    assert list(tmp_path.iterdir()) == []
    assert o.get_expectation_value() == 1.
    # Direct adapter dispatch also rejects unsupported channels rather than
    # reaching a NumPy implementation or an opaque missing-attribute error.
    with pytest.raises(NotImplementedError, match=f"backend '{name}'"):
        adapter.apply_forward(o, spd.ResetZero(0), 0., 100)


def test_runner_dispatches_groups_and_gates_to_selected_adapter(monkeypatch):
    from spd import numpy_backend as module
    adapter = spd.BackendAdapter.from_name('numpy', precision='double')
    calls = []
    for name in ['contract_zero_forward', 'contract_zero_backward',
                 'conjugate_pauli_rot_forward', 'conjugate_pauli_rot_backward',
                 'insert_identity_forward', 'insert_identity_backward']:
        original = getattr(module, name)
        def track(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, name, track)
    c = spd.CircuitIR(3, [spd.CreateZero(2), spd.CreateZero(0),
                         Rot('Ry', 'YII', .3), spd.Discard(1)])
    o = spd.create_spo({'ZII': 1.}, backend=adapter)
    f, _ = spd.evolve(o, c, 0., 100, backend=adapter, progress=False)
    spd.backpropagate(spd.init_gradient_spo(f, backend=adapter), c, 0., 100,
                      backend=adapter, progress=False)
    assert calls == ['insert_identity_forward', 'conjugate_pauli_rot_forward', 'contract_zero_forward',
                     'contract_zero_backward', 'conjugate_pauli_rot_backward', 'insert_identity_backward']


def test_channel_progress_uses_shared_reporting(capsys):
    c = spd.CircuitIR(2, [spd.CreateZero(0), spd.CreateZero(1), Rot('Ry', 'YI', .2)])
    o = spd.create_spo({'ZI': 1.}, backend_name='numpy')
    f, _ = spd.evolve(o, c, 0., 100, progress=True)
    spd.backpropagate(spd.init_gradient_spo(f), c, 0., 100, progress=True)
    assert 'Zero-state contractions (2)' in capsys.readouterr().out


def test_mixed_creation_reset_group_with_shared_parameter(tmp_path):
    barrier = SkippedOperation('OpType.Barrier')
    def circuit(p):
        return spd.CircuitIR(4, [
            Rot('Ry', 'YIII', .7*p), Two('OpType.CX', 0, 1),
            spd.ResetZero(0), barrier, spd.CreateZero(2), spd.CreateZero(3),
            spd.ResetZero(1), barrier,
            Rot('Ry', 'IIYI', -.4*p), Rot('Ry', 'IIIY', 1.2*p),
            Two('OpType.CX', 2, 3), Rot('Ry', 'YIII', .3*p),
        ])
    c = circuit(.41)
    obs = {'ZIZZ': .6, 'IIXI': 1., 'IIIY': .4, 'ZIII': .1}
    o = spd.create_spo(obs, backend_name='numpy', precision='double')
    f, _ = spd.evolve(o, c, 0., 1000, progress=False, checkpoint_directory=tmp_path, checkpoint_memory_budget_bytes=0)
    tape = f._channel_checkpoints
    assert [p.name for p in tape.directory.glob('*.pickle')] == ['2.pickle']
    assert f.active_qubits == [0, 1]  # Resets retain their columns; creations do not.
    assert f.get_expectation_value() == pytest.approx(dense(c, obs), abs=1e-12)
    _, gradients, _ = spd.backpropagate(spd.init_gradient_spo(f), c, 0., 1000, progress=False)
    np.testing.assert_allclose(gradients, finite_differences(c, obs), atol=1e-9)
    variational = spd.VariationalCircuit(c, [0, 0, 0, 0], (1,), [.7, -.4, 1.2, .3])
    reference = (dense(circuit(.410001), obs) - dense(circuit(.409999), obs)) / 2e-6
    assert variational.parameter_gradients(gradients)[0] == pytest.approx(reference, abs=1e-9)
    assert gradients[0] == pytest.approx(0., abs=1e-12)
    assert not tape.directory.exists()


def test_checkpoint_storage_policies_match_dense_gradients(tmp_path, monkeypatch):
    c = spd.CircuitIR(2, [spd.CreateZero(1), Rot('Ry', 'IY', .4), spd.ResetZero(0),
                         Rot('Rx', 'XI', .2), spd.ResetZero(1), Rot('Ry', 'IY', .3)])
    obs = {'ZZ': 1., 'XI': .2, 'IZ': -.3}
    o = spd.create_spo(obs, backend_name='numpy', precision='double')
    expected_gradient = finite_differences(c, obs)
    released = []
    take = CheckpointStore.take

    def record_take(self, key):
        result = take(self, key)
        assert key not in self._snapshots
        released.append(key)
        return result

    monkeypatch.setattr(CheckpointStore, 'take', record_take)
    mixed_budget = None
    for policy in ('memory', 'disk', 'mixed'):
        budget = {'memory': 512 * 1024**2, 'disk': 0, 'mixed': mixed_budget}[policy]
        f, _ = spd.evolve(o, c, 0, 100, progress=False, checkpoint_directory=tmp_path,
                          checkpoint_memory_budget_bytes=budget)
        tape = f._channel_checkpoints
        if policy == 'memory':
            assert tape.directory is None
            assert len(tape._snapshots) == 3
            mixed_budget = max(entry.size for entry in tape._snapshots.values())
        elif policy == 'disk':
            assert tape.memory_bytes == 0
            assert len(list(tape.directory.iterdir())) == 3
        else:
            assert 0 < tape.memory_bytes <= budget
            assert 0 < len(list(tape.directory.iterdir())) < 3
        assert f.get_expectation_value() == pytest.approx(dense(c, obs), abs=1e-12)
        _, gradients, _ = spd.backpropagate(spd.init_gradient_spo(f), c, 0, 100, progress=False)
        np.testing.assert_allclose(gradients, expected_gradient, atol=1e-9)
        assert released[-3:] == [0, 2, 4]
        assert tape.closed and tape.memory_bytes == 0 and not tape._snapshots
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('budget', [0, 4 * 1024**3])
def test_retained_snapshots_are_not_mutated(tmp_path, monkeypatch, budget):
    c = spd.CircuitIR(3, [spd.CreateZero(2), Rot('Ry', 'IIY', .3),
                         spd.ResetZero(0), spd.CreateZero(1), SkippedOperation('OpType.Barrier'),
                         Rot('Ry', 'YII', .4), Two('OpType.CX', 0, 2), spd.ResetZero(2),
                         Rot('Rx', 'IIX', .2)])
    obs = {'ZZZ': 1., 'XYZ': .2, 'III': -.3}
    saved = []
    save = CheckpointStore.save

    def record(self, key, state):
        saved.append((state, dict(state), list(state.qubit_indices), state.system_size))
        save(self, key, state)
        if budget:
            assert self._snapshots[key].value is state

    monkeypatch.setattr(CheckpointStore, 'save', record)
    value, gradients, *_ = evaluate(c, obs, checkpoint_directory=tmp_path,
                                     checkpoint_memory_budget_bytes=budget)
    assert len(saved) == 3  # Independent reset/creation share their boundary.
    for state, coefficients, active, system_size in saved:
        assert dict(state) == coefficients
        assert state.qubit_indices == active and state.system_size == system_size
    assert value == pytest.approx(dense(c, obs), abs=1e-12)
    np.testing.assert_allclose(gradients, finite_differences(c, obs), atol=1e-9)
    assert list(tmp_path.iterdir()) == []


def test_backend_checkpoint_helpers_are_created_once(monkeypatch):
    from spd.checkpoints import CheckpointBackend
    from spd.numpy_backend.sparse_pauli import checkpoint_size
    c = spd.CircuitIR(1, [spd.CreateZero(0), Rot('Ry', 'Y', .4)])
    o = spd.create_spo({'Z': 1.}, backend_name='numpy', precision='double')
    backend = spd.BackendAdapter.from_name('numpy', precision='double')
    calls = []

    def factory(state):
        calls.append(tuple(state.qubit_indices))
        return CheckpointBackend(checkpoint_size)

    monkeypatch.setattr(backend.module, 'create_checkpoint_backend', factory, raising=False)
    f, _ = spd.evolve(o, c, 0, 100, backend=backend, progress=False)
    _, gradients, _ = spd.backpropagate(spd.init_gradient_spo(f), c, 0, 100,
                                        backend=backend, progress=False)
    assert calls == [(0,)]
    np.testing.assert_allclose(gradients, [-np.sin(.4)], atol=1e-12)
