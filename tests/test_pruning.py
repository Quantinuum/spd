"""Observable-level pruning contracts across public runners and backends."""
from dataclasses import replace

import numpy as np
import pytest

import spd
from spd.circuit_ir import (
    CircuitIR, PauliRotation, SingleQubitClifford, TwoQubitClifford, SkippedOperation,
)
from tests.helpers import assert_info_consistent


def adapter(all_backend):
    name, module = all_backend
    return spd.BackendAdapter(name, module, 32, precision="double")


def terms(state):
    if isinstance(state, dict):
        return {tuple(map(int, k)): np.asarray(v) for k, v in state.items()}
    if hasattr(state, 'to_host'):
        arrays = state.to_host()
    else:
        arrays = (np.asarray(state.xz_array), np.asarray(state.c_array))
        if hasattr(state, 'grad_c_array'):
            arrays += (np.asarray(state.grad_c_array),)
    return {tuple(map(int, k)): np.asarray([a[i] for a in arrays[1:]]).squeeze()
            for i, k in enumerate(arrays[0]) if any(a[i] != 0 for a in arrays[1:])}


def same(a, b, atol=2e-10):
    a, b = terms(a), terms(b)
    for key in a.keys() | b.keys():
        np.testing.assert_allclose(a.get(key, 0.), b.get(key, 0.), atol=atol, rtol=atol)


def p(n, sites, axis):
    label = ['I'] * n
    for q in sites:
        label[q] = axis
    return ''.join(label)


def brickwork(n, rounds=2, periodic=False):
    ops = []
    for t in range(rounds):
        ops.extend(PauliRotation('rx', p(n, [q], 'X'), .29 + .03*t) for q in range(n))
        for parity in (0, 1):
            for q in range(parity, n if periodic else n-1, 2):
                ops.append(PauliRotation('rzz', p(n, [q, (q+1)%n], 'Z'), .37 + .017*q))
            ops.append(SkippedOperation('barrier'))
    return CircuitIR(n, tuple(ops))


def evolve(state, circuit, backend, *, pruning='light-cone', cutoff=0., cap=256, **kwargs):
    return spd.evolve(state, circuit, cutoff, cap, backend=backend,
                      progress=False, pruning=pruning, **kwargs)


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('cutoff,cap', [(0., 256), (.03, 16)])
def test_brickwork_coefficients_and_history(all_backend, periodic, cutoff, cap):
    b = adapter(all_backend)
    circuit = brickwork(12, periodic=periodic)
    original = spd.create_spo({p(12, [0], 'Z'): .8, p(12, [1], 'Y'): -.23}, backend=b)
    snapshot = terms(original)
    full, full_info = evolve(original, circuit, b, pruning=None, cutoff=cutoff, cap=cap)
    reduced, info = evolve(original, circuit, b, cutoff=cutoff, cap=cap)
    same(full, reduced)
    assert terms(original).keys() == snapshot.keys()
    for k, v in terms(original).items():
        np.testing.assert_array_equal(v, snapshot[k])
    assert getattr(original, '_pruning_record', None) is None
    assert info['pruning']['num_gates_pruned'] > 0
    assert_info_consistent(info, full_info['num_steps_tracked'])
    for key in info['history']:
        np.testing.assert_allclose(info['history'][key], full_info['history'][key], atol=2e-10)


def test_direction_cliffords_and_padding(all_backend):
    b = adapter(all_backend)
    # Reverse propagation of Z_32 must reach 31, then 30, across a word boundary.
    circuit = CircuitIR(34, (
        TwoQubitClifford('OpType.CX', 30, 31),
        TwoQubitClifford('OpType.CX', 31, 32),
        SingleQubitClifford('OpType.H', 2),
        SingleQubitClifford('OpType.S', 32),
    ))
    original = spd.create_spo({p(34, [32], 'Z'): 1.}, backend=b)
    full, _ = evolve(original, circuit, b, pruning=None)
    reduced, info = evolve(original, circuit, b)
    same(full, reduced)
    assert info['pruning']['num_gates_pruned'] == 1
    assert reduced._pruning_record.retained_indices == (0, 1, 3)


@pytest.mark.parametrize('data,cutoff,cap', [({'ZIII': .001}, .01, 256),
                                          ({'ZIII': .3, 'XIII': .8}, 0., 1),
                                          ({'IIII': 1.}, 0., 256),
                                          ({'ZIII': 0.}, 0., 256)])
def test_all_pruned_completely_skips_cleanup_and_preserves_input(all_backend, data, cutoff, cap):
    b = adapter(all_backend)
    original = spd.create_spo(data, backend=b)
    circuit = CircuitIR(4, (SkippedOperation('barrier'), PauliRotation('rx', 'IIIX', .2)))
    reduced, info = evolve(original, circuit, b, cutoff=cutoff, cap=cap)
    same(original, reduced)
    assert reduced is not original  # Forward metadata must not alter the input.
    assert getattr(original, '_pruning_record', None) is None
    assert info['pruning']['num_gates_retained'] == 0
    assert_info_consistent(info, 1)
    assert info['sum_num_str_truncated'] == 0
    if data == {'ZIII': .001}:
        full, _ = evolve(original, circuit, b, pruning=None, cutoff=cutoff, cap=cap)
        assert full.get_norm_square() == 0
        assert float(reduced.get_norm_square()) > 0


@pytest.mark.parametrize('pruning', ['small-angle', '', True, object()])
def test_unknown_method_rejected(pruning):
    b = spd.BackendAdapter.from_name('numpy')
    original = spd.create_spo({'Z': 1.}, backend=b)
    with pytest.raises(ValueError, match='pruning'):
        evolve(original, (), b, pruning=pruning)


@pytest.mark.parametrize('operation', [PauliRotation('rx', 'IIIX', float('nan')),
                                      PauliRotation('bad', 'IIIA', .2),
                                      SingleQubitClifford('unknown', 3),
                                      TwoQubitClifford('OpType.CX', 3, 3),
                                      SingleQubitClifford('OpType.H', 33)])
def test_pruning_does_not_hide_invalid_operations(all_backend, operation):
    b = adapter(all_backend)
    original = spd.create_spo({'ZIII': 1.}, backend=b)
    with pytest.raises((ValueError, TypeError)):
        evolve(original, (operation,), b)


def test_physical_support_validated(all_backend):
    b = adapter(all_backend)
    original = spd.create_spo({'IIIZ': 1.}, backend=b)
    with pytest.raises(ValueError, match='physical circuit'):
        evolve(original, CircuitIR(3, ()), b)


def test_repeated_calls_recompute_support(all_backend):
    b = adapter(all_backend)
    circuit = brickwork(12, rounds=1)
    full = spd.create_spo({p(12, [5], 'Z'): 1.}, backend=b)
    pruned = spd.create_spo({p(12, [5], 'Z'): 1.}, backend=b)
    counts = []
    for _ in range(3):
        full, _ = evolve(full, circuit, b, pruning=None, cutoff=.01)
        pruned, info = evolve(pruned, circuit, b, cutoff=.01)
        counts.append(info['pruning']['num_gates_retained'])
        same(full, pruned)
    assert counts[-1] > counts[0]
    # An unpruned no-op call clears the old record without changing its input.
    unpruned, _ = evolve(pruned, (), b, pruning=None)
    assert unpruned._pruning_record is None
    assert pruned._pruning_record is not None


@pytest.mark.parametrize('cutoff,cap', [(0., 256), (.03, 16)])
@pytest.mark.parametrize('loss', ['basis', 'l2', 'ose'])
def test_backward_follows_forward_and_original_slots(all_backend, cutoff, cap, loss):
    b = adapter(all_backend)
    circuit = brickwork(8, rounds=1)
    original = spd.create_spo({p(8, [0], 'Z'): .8, p(8, [1], 'Y'): .23}, backend=b)
    target = spd.create_spo({p(8, [0], 'X'): .3, p(8, [7], 'Y'): .4}, backend=b)
    kwargs = {'loss_type': 'l2_difference', 'target_spo': target} if loss == 'l2' else (
        {'lambda_ose': .13} if loss == 'ose' else {})
    full, _ = evolve(original, circuit, b, pruning=None, cutoff=cutoff, cap=cap)
    pruned, _ = evolve(original, circuit, b, cutoff=cutoff, cap=cap)
    a = spd.init_gradient_spo(full, backend=b, **kwargs)
    z = spd.init_gradient_spo(pruned, backend=b, **kwargs)
    a, ag, ai = spd.backpropagate(a, circuit, cutoff, cap, backend=b, progress=False)
    zfinal, zg, zi = spd.backpropagate(z, circuit, cutoff, cap, backend=b, progress=False)
    np.testing.assert_allclose(zg, ag, atol=2e-9, rtol=2e-9)
    same(a.to_spo(), zfinal.to_spo(), atol=2e-9)
    # The adjoint belongs to the reduced circuit, and need not match the full
    # circuit's adjoint. Compare all SPGO channels to that explicit execution.
    manual_terminal = spd.init_gradient_spo(full, backend=b, **kwargs)
    reduced_circuit = CircuitIR(8, pruned._pruning_record.retained_operations)
    manual, _, _ = spd.backpropagate(manual_terminal, reduced_circuit, cutoff, cap,
                                     backend=b, progress=False)
    same(manual, zfinal, atol=2e-9)
    assert_info_consistent(zi, ai['num_steps_tracked'])
    kept = set(pruned._pruning_record.retained_indices)
    assert any(i not in kept for i, o in enumerate(circuit.operations) if isinstance(o, PauliRotation))
    j = 0
    for i, op in enumerate(circuit.operations):
        if isinstance(op, PauliRotation):
            if i not in kept:
                assert zg[j] == 0
            j += 1
    assert z._pruning_record is not None
    assert zfinal._pruning_record is None
    changed = CircuitIR(8, (replace(circuit.operations[0], theta=.99),) + circuit.operations[1:])
    with pytest.raises(ValueError, match='does not match'):
        spd.backpropagate(z, changed, cutoff, cap, backend=b, progress=False)
    with pytest.raises(NotImplementedError, match='Noise analysis'):
        spd.backpropagate_noise_analysis(z, circuit, cutoff, cap, backend=b, progress=False)


def test_shared_parameters_and_finite_difference(all_backend):
    pytest.importorskip('pytket')
    from spd.ansatz import tfi_1d_hva
    b = adapter(all_backend)
    params = np.array([.17, -.23, .11, .07])
    original = spd.create_spo({'Z' + 'I'*11: 1.}, backend=b)
    def forward(values):
        ansatz = tfi_1d_hva(values, system_size=12)
        result, _ = evolve(original, ansatz.circuit, b)
        return ansatz, result
    ansatz, result = forward(params)
    gradient = spd.init_gradient_spo(result, backend=b)
    _, raw, _ = spd.backpropagate(gradient, ansatz.circuit, 0., 256, backend=b, progress=False)
    actual = ansatz.parameter_gradients(raw)
    fd = []
    for i in range(4):
        delta = np.eye(4)[i] * 1e-5
        plus, minus = forward(params+delta)[1], forward(params-delta)[1]
        fd.append((float(plus.get_expectation_value()) - float(minus.get_expectation_value())) / 2e-5)
    np.testing.assert_allclose(actual, fd, atol=2e-7, rtol=2e-7)


def test_zero_angle_gate_is_retained(all_backend):
    b = adapter(all_backend)
    circuit = CircuitIR(4, (PauliRotation('rx', 'XIII', 0.), PauliRotation('rx', 'IIIX', 0.)))
    original = spd.create_spo({'ZIII': 1.}, backend=b)
    result, info = evolve(original, circuit, b)
    assert result._pruning_record.retained_indices == (0,)
    grad = spd.init_gradient_spo(result, backend=b)
    _, values, _ = spd.backpropagate(grad, circuit, 0., 256, backend=b, progress=False)
    assert len(values) == 2 and values[1] == 0


def test_triton_stepwise_and_in_place():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    from spd import triton_backend as gpu
    b = spd.BackendAdapter.from_name('triton', precision='double')
    circuit = brickwork(12, rounds=1)
    source = spd.create_spo({p(12, [5], 'Z'): 1.}, backend=b)
    inplace = source.copy()
    stepwise = source.copy()
    reference = source.copy()
    for _ in range(3):
        reference, _ = evolve(reference, circuit, b, pruning=None, cutoff=.01)
        result, _ = evolve(inplace, circuit, b, in_place=True, cutoff=.01)
        assert result is inplace
        stepwise = gpu.evolve_step(stepwise, circuit, .01, 256, pruning='light-cone')
        same(reference, inplace)
        same(reference, stepwise)
    grad = spd.init_gradient_spo(stepwise, backend=b)
    returned, values, _ = spd.backpropagate(grad, circuit, .01, 256, backend=b,
                                           progress=False, in_place=True)
    assert returned is grad and grad._pruning_record is None
    assert len(values) == sum(isinstance(o, PauliRotation) for o in circuit.operations)
    # Shared tensor storage detaches correctly even with pruning.
    keys, coeff = source.xz_array, source.c_array
    aliased = gpu.SparsePauliOp(keys, coeff, source.num_qubits)
    expected = terms(aliased)
    evolve(source, circuit, b, in_place=True)
    assert terms(aliased).keys() == expected.keys()
    for k, v in terms(aliased).items():
        np.testing.assert_array_equal(v, expected[k])


def test_empty_state_backward(all_backend):
    b = adapter(all_backend)
    if b.name == 'numpy':
        empty = b.module.SparsePauliOp()
    elif b.name == 'jax':
        empty = b.module.SparsePauliOp(np.empty((0, 2), dtype=np.uint32), np.empty(0))
    else:
        empty = b.module.create_op({}, num_qubits=4, precision='double')
    circuit = CircuitIR(4, (PauliRotation('rx', 'IIIX', .2),))
    result, info = evolve(empty, circuit, b)
    assert info['pruning']['num_gates_retained'] == 0
    terminal = spd.init_gradient_spo(result, backend=b)
    back, grads, info = spd.backpropagate(terminal, circuit, 0., 256, backend=b, progress=False)
    assert grads == [0.]
    assert not terms(back)


def test_against_dense_operator(all_backend):
    from tests.test_noise_susceptibility import _matrix, _rotation
    b = adapter(all_backend)
    circuit = CircuitIR(4, (
        PauliRotation('ry', 'YIII', .23),
        PauliRotation('rzz', 'ZZII', .37),
        PauliRotation('rx', 'IXII', -.17),
        PauliRotation('rx', 'IIIX', .43),
    ))
    data = {'XIII': .8, 'IZII': .23}
    original = spd.create_spo(data, backend=b)
    expected = sum(v * _matrix(k) for k, v in data.items())
    for op in reversed(circuit.operations):
        unitary = _rotation(op)
        expected = unitary.conj().T @ expected @ unitary
    result, info = evolve(original, circuit, b)
    actual = np.zeros_like(expected)
    for key, coeff in terms(result).items():
        label = b.utils.uint32_to_pauli_str(np.asarray(key, dtype=np.uint32), 32)[:4]
        actual += coeff * _matrix(label)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert info['pruning']['num_gates_pruned'] == 1


def test_fixed_depth_2d_gate_counts_and_dispatch():
    b = spd.BackendAdapter.from_name('numpy', precision='double')
    counts = []
    for side in (8, 12, 16):
        n = side**2
        ops = [PauliRotation('rx', p(n, [q], 'X'), .29) for q in range(n)]
        for axis in (0, 1):
            for parity in (0, 1):
                for x in range(side):
                    for y in range(side):
                        pos = x if axis == 0 else y
                        if pos % 2 == parity and pos + 1 < side:
                            q = x*side + y
                            ops.append(PauliRotation('rzz', p(n, [q, q+(side if axis == 0 else 1)], 'Z'), .37))
        original = spd.create_spo({p(n, [(side//2)*side+side//2], 'X'): 1.}, backend=b)
        calls = []
        apply = b.apply_forward
        def count(*args, **kwargs):
            calls.append(1)
            return apply(*args, **kwargs)
        b.apply_forward = count
        try:
            result, info = evolve(original, CircuitIR(n, tuple(ops)), b, cutoff=.01, cap=32)
        finally:
            b.apply_forward = apply
        counts.append(info['pruning']['num_gates_retained'])
        assert len(calls) == counts[-1]
        assert counts[-1] < len(ops)
    assert len(set(counts)) == 1


def test_triton_record_survives_copy_pickle_and_clears_on_gate():
    import pickle
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    b = spd.BackendAdapter.from_name('triton', precision='double')
    original = spd.create_spo({'ZIII': 1.}, backend=b)
    circuit = CircuitIR(4, (PauliRotation('rx', 'XIII', .3), PauliRotation('rx', 'IIIX', .2)))
    result, _ = evolve(original, circuit, b)
    for clone in (result.copy(), pickle.loads(pickle.dumps(result))):
        assert clone._pruning_record == result._pruning_record
        grad = spd.init_gradient_spo(clone, backend=b)
        _, values, info = spd.backpropagate(grad, circuit, 0., 256, backend=b, progress=False)
        assert values[1] == 0 and info['pruning']['num_gates_pruned'] == 1
        clone.apply_in_place(PauliRotation('rx', 'XIII', .1))
        assert clone._pruning_record is None
    assert result._pruning_record is not None


@pytest.mark.parametrize('cutoff', [-.1, float('nan'), float('inf')])
def test_invalid_cutoff_rejected_even_when_every_gate_is_pruned(all_backend, cutoff):
    b = adapter(all_backend)
    original = spd.create_spo({'ZIII': 1.}, backend=b)
    with pytest.raises(ValueError, match='trunc_val'):
        evolve(original, (), b, cutoff=cutoff)


def test_openqasm_and_operation_sequence(all_backend):
    from spd.openqasm_frontend import parse_openqasm_str
    b = adapter(all_backend)
    circuit = parse_openqasm_str('''OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.3) q[0];
cx q[0],q[1];
rx(0.4) q[3];
''')
    original = spd.create_spo({'IZII': 1.}, backend=b)
    full, _ = evolve(original, circuit, b, pruning=None)
    for representation in (circuit, circuit.operations):
        reduced, info = evolve(original, representation, b)
        same(full, reduced)
        assert info['pruning']['num_gates_pruned'] == 1
        terminal = spd.init_gradient_spo(reduced, backend=b)
        _, grads, _ = spd.backpropagate(terminal, representation, 0., 256, backend=b, progress=False)
        assert len(grads) == 2 and grads[1] == 0


def test_input_width_checked(all_backend):
    b = adapter(all_backend)
    original = spd.create_spo({'Z': 1.}, backend=b)
    with pytest.raises(ValueError, match='packed width'):
        evolve(original, CircuitIR(64, ()), b)


def test_triton_all_pruned_in_place():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    b = spd.BackendAdapter.from_name('triton', precision='double')
    original = spd.create_spo({'ZIII': .001}, backend=b)
    circuit = CircuitIR(4, (PauliRotation('rx', 'IIIX', .3),))
    storage = original._storage
    result, info = evolve(original, circuit, b, in_place=True, cutoff=.01)
    assert result is original and result._storage is storage
    assert info['pruning']['num_gates_retained'] == 0
    terminal = spd.init_gradient_spo(result, backend=b)
    back, grads, info = spd.backpropagate(terminal, circuit, .01, 256,
                                         backend=b, progress=False, in_place=True)
    assert back is terminal and grads == [0.]
    assert back._pruning_record is None


def test_direct_triton_physical_size_is_not_padding():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    from spd import triton_backend as gpu
    b = spd.BackendAdapter.from_name('triton', precision='double')
    state = gpu.create_op({'ZIII': 1.}, precision='double')
    with pytest.raises(ValueError, match='outside'):
        evolve(state, (SingleQubitClifford('OpType.H', 5),), b)
    with pytest.raises(ValueError, match='physical size'):
        evolve(state, CircuitIR(6, ()), b)


@pytest.mark.parametrize('n', [0, 1, 257, 4097])
@pytest.mark.parametrize('words', [1, 2, 5])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_gpu_support_reduction(n, words, dtype):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from spd.triton_backend.support import support_mask
    rng = np.random.default_rng(39)
    # Limit live support so ignoring zero rows and OR-ing all words both matter.
    keys = rng.integers(0, 2**32, (n, words*2), dtype=np.uint32)
    coeff = rng.normal(size=n).astype(dtype)
    coeff[::2] = 0
    keys[1::2] &= np.uint32(0x80000001)
    expected = np.bitwise_or.reduce(keys[coeff != 0], axis=0, initial=np.uint32(0))
    expected = expected[:words] | expected[words:]
    actual = support_mask(torch.tensor(keys.view(np.int32), device='cuda'),
                          torch.tensor(coeff, device='cuda'))
    np.testing.assert_array_equal(actual[:-1], expected)
    assert actual[-1] == 0


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_gpu_support_rejects_nonfinite_coefficients(value):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from spd import triton_backend as tb
    from spd.pruning import _support
    state = tb.create_op({'Z': 1.}, precision='double')
    state._storage.coeff[0] = value
    with pytest.raises(ValueError, match='finite input coefficients'):
        _support(state, 'triton')


def test_gpu_support_strided_views():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from spd.triton_backend.support import support_mask
    keys = torch.zeros((600, 20), dtype=torch.int32, device='cuda')
    coefficients = torch.zeros(600, dtype=torch.float64, device='cuda')
    keys[598, 18] = -2147483648  # Z, word 4, highest bit, last live row.
    coefficients[598] = 1.
    keys[0, 0] = 1  # Dead row must not contribute.
    mask = support_mask(keys[::2, ::2], coefficients[::2])
    np.testing.assert_array_equal(mask, [0, 0, 0, 0, 2**31, 0])
    assert keys[598, 18].item() == -2147483648
    assert coefficients[598].item() == 1.
