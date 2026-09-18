"""Dense references below implement the specification independently of the builder."""
import numpy as np
import pytest
from functools import partial
from pytket import OpType
from scipy.linalg import expm

import spd
from spd.ansatz.hva_mera import (
    HVATensor, binary_mera_layers, binary_mera_parameter_shape,
    tfi_hva_tensor_layer, tfi_binary_mera, binary_mera_qubit_initializations,
    binary_mera_causal_cone,
)


binary_mera_layers = partial(binary_mera_layers, chi=2)
binary_mera_parameter_shape = partial(binary_mera_parameter_shape, chi=2)
tfi_binary_mera = partial(tfi_binary_mera, chi=2)
binary_mera_qubit_initializations = partial(binary_mera_qubit_initializations, chi=2)
binary_mera_causal_cone = partial(binary_mera_causal_cone, chi=2)


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1., -1.]).astype(complex)


def pauli_matrix(n, sites, matrix):
    result = np.ones((1, 1), complex)
    for i in range(n):
        result = np.kron(result, matrix if i in sites else I)
    return result


def hamiltonian(n, g=.9):
    terms = {}
    for i in range(n):
        for sites, label, weight in [((i, (i+1) % n), 'Z', -1.), ((i,), 'X', -g)]:
            key = ''.join(label if j in sites else 'I' for j in range(n))
            terms[key] = terms.get(key, 0.) + weight
    dense = sum(value * pauli_matrix(n, [i for i, p in enumerate(key) if p != 'I'],
                                      X if 'X' in key else Z)
                for key, value in terms.items())
    return terms, dense


def exact_reference(params, n):
    """Independent chi=2 schedule with shared, position-dependent angles."""
    state=np.zeros(2**n,complex)
    state[0]=1
    tangent=np.zeros((params.size,2**n),complex)
    roles=2*(n.bit_length()-1)-1
    assert params.ndim==1 and params.size%(3*roles)==0
    rounds=params.size//(3*roles)

    def rotate(sites,matrix,angle,index=None):
        nonlocal state,tangent
        p=pauli_matrix(n,sites,matrix)
        theta=np.pi*angle
        u=np.cos(theta/2)*np.eye(2**n)-1j*np.sin(theta/2)*p
        state=u@state
        tangent=tangent@u.T
        if index is not None:
            tangent[index]+=-.5j*np.pi*(p@state)

    rotate([n-1],Y,.5)
    cursor=0
    stride=n//2
    while stride:
        sites=list(range(stride-1,n,stride))
        pairs=list(zip(sites[::2],sites[1::2]))
        for a,_ in pairs:
            rotate([a],Y,.5)
        layer_roles=[pairs]
        if len(sites)>2:
            layer_roles.append([(sites[j],sites[(j+1)%len(sites)])
                                for j in range(1,len(sites),2)])
        for tensors in layer_roles:
            for _ in range(rounds):
                for pair in tensors:
                    rotate(pair,Z,params[cursor],cursor)
                for position in range(2):
                    for pair in tensors:
                        rotate([pair[position]],X,params[cursor+1+position],
                               cursor+1+position)
                cursor+=3
        stride//=2
    assert cursor==params.size
    return state,tangent


def spd_value_gradient(vc, terms, cutoff=0.):
    backend = spd.BackendAdapter.from_name('numpy', packbit=32, precision='double')
    initial = spd.create_spo(terms, backend=backend)
    final, info = spd.evolve(initial, vc.circuit, cutoff, 4**vc.circuit.system_size
                            if isinstance(vc.circuit, spd.CircuitIR) else 4**vc.circuit.n_qubits,
                            backend=backend, progress=False)
    value = final.get_expectation_value(basis='0')
    grad = spd.init_gradient_spo(final, basis='0', backend=backend)
    _, raw, _ = spd.backpropagate(grad, vc.circuit, cutoff,
                                 4**len(next(iter(terms))), backend=backend, progress=False)
    return value, vc.parameter_gradients(raw), info


def test_unitary_and_isometry_with_noncanonical_index_roles():
    params=np.arange(22,dtype=float).reshape(2,11)/101.-.1
    tensor=HVATensor((2,0,3,1))
    u=tfi_hva_tensor_layer(params,[tensor],4).circuit.get_unitary()
    np.testing.assert_allclose(u.conj().T@u,np.eye(16),atol=1e-13)
    iso=HVATensor((2,0,3,1),(0,1),(2,3))
    full=tfi_hva_tensor_layer(params,[iso],4).circuit.get_unitary()
    columns=[0,4,8,12]
    w=full[:,columns]
    np.testing.assert_allclose(w.conj().T@w,np.eye(4),atol=1e-13)
    expected=np.eye(16,dtype=complex)
    for row in params:
        schedule=[((tensor.outputs[0],tensor.outputs[1]),Z,row[0]),
                  ((tensor.outputs[2],tensor.outputs[3]),Z,row[1])]
        schedule += [((site,),X,row[2+j]) for j,site in enumerate(tensor.outputs)]
        schedule += [((tensor.outputs[1],tensor.outputs[2]),Z,row[6])]
        schedule += [((site,),X,row[7+j]) for j,site in enumerate(tensor.outputs)]
        for sites,matrix,angle in schedule:
            expected=expm(-.5j*np.pi*angle*pauli_matrix(4,sites,matrix))@expected
    np.testing.assert_allclose(u,expected,atol=1e-13)


def test_network_indexing_parameter_count_and_initializations():
    layers = binary_mera_layers(8)
    assert [layer.scale for layer in layers] == [2, 1, 0]
    assert [layer.ancillas for layer in layers] == [(3,), (1, 5), (0, 2, 4, 6)]
    assert [[t.outputs for t in l.disentanglers] for l in layers] == [
        [], [(3, 5), (7, 1)], [(1, 2), (3, 4), (5, 6), (7, 0)]]
    assert binary_mera_parameter_shape(8,2)==(30,)
    params=np.arange(30)/101.
    vc = tfi_binary_mera(params, 8, 2)
    initializations = binary_mera_qubit_initializations(8, 2)
    assert [indices for _, indices in initializations] == [(7,), (3,), (1, 5), (0, 2, 4, 6)]
    commands = vc.circuit.get_commands()
    for offset, indices in initializations:
        assert commands[offset].op.type == OpType.Ry
        assert set(q.index[0] for c in commands[offset:offset+len(indices)] for q in c.qubits) == set(indices)
        # Future ancillas never participate in earlier quantum gates.
        assert all(not (set(indices) & {q.index[0] for q in c.qubits})
                   for c in commands[:offset] if c.op.type != OpType.Barrier)
    rotations = [c for c in commands if c.op.type != OpType.Barrier]
    for c, i, f in zip(rotations, vc.gate_parameter_indices, vc.gate_parameter_factors):
        assert float(c.op.params[0]) == pytest.approx(.5 if i == -1 else params.flat[i]*f)
    assert sum(vc.gate_parameter_indices == -1) == 8
    assert set(vc.gate_parameter_factors) == {1.}
    assert set(vc.gate_parameter_indices)==set(range(30))|{-1}
    counts=np.bincount(vc.gate_parameter_indices[vc.gate_parameter_indices>=0])
    np.testing.assert_array_equal(counts,[1]*6+[2]*12+[4]*12)


def test_synchronized_even_odd_layers_across_four_qubit_tensors():
    params=np.arange(22,dtype=float).reshape(2,11)/101.
    vc=tfi_hva_tensor_layer(
        params,[HVATensor((0,1,2,3)),HVATensor((4,5,6,7))],8)
    groups,current=[],[]
    for command in vc.circuit.get_commands():
        if command.op.type==OpType.Barrier:
            groups.append(current)
            current=[]
        else:
            current.append(tuple(q.index[0] for q in command.qubits))
    expected=[[(0,1),(2,3),(4,5),(6,7)],
              [(i,) for i in range(8)],
              [(1,2),(5,6)],
              [(i,) for i in range(8)]]
    assert [set(group) for group in groups]==[set(group) for group in expected*2]
    for group in groups:
        flattened=[i for gate in group for i in gate]
        assert len(flattened)==len(set(flattened))
    counts=np.bincount(vc.gate_parameter_indices)
    np.testing.assert_array_equal(counts,np.full(22,2))


@pytest.mark.parametrize('n', [4,8])
def test_causal_cones(n):
    for i in range(n):
        cone = binary_mera_causal_cone(n, [i, (i+1)%n])
        assert all(len(level) <= 3 for level in cone)
        assert cone[-1] == (n-1,)
    assert binary_mera_causal_cone(n, []) == ((),) * n.bit_length()


@pytest.mark.parametrize('steps', [1,2])
def test_exact_state_energy_gradient_and_spd(steps):
    n = 4
    params = np.random.default_rng(91+steps).normal(0, .2, binary_mera_parameter_shape(n, steps))
    vc = tfi_binary_mera(params, n, steps)
    state, tangent = exact_reference(params, n)
    np.testing.assert_allclose(vc.circuit.get_statevector(), state, atol=1e-13)
    terms, h = hamiltonian(n)
    exact = np.vdot(state, h @ state).real
    exact_grad = (2 * (tangent.conj() @ (h @ state)).real).reshape(params.shape)
    value, grad, info = spd_value_gradient(vc, terms)
    assert value == pytest.approx(exact, abs=1e-11)
    np.testing.assert_allclose(grad, exact_grad, atol=1e-10)
    assert sum(info['history']['num_str_truncated']) == 0
    finite = np.zeros(params.size)
    for i in range(params.size):
        shifted = params.copy()
        shifted.flat[i] += 1e-6
        plus = tfi_binary_mera(shifted,n,steps).circuit.get_statevector()
        shifted.flat[i] -= 2e-6
        minus = tfi_binary_mera(shifted,n,steps).circuit.get_statevector()
        finite[i] = (np.vdot(plus,h@plus).real-np.vdot(minus,h@minus).real)/2e-6
    np.testing.assert_allclose(exact_grad.ravel(), finite, atol=2e-8)


def test_zero_angles_are_plus_product_with_correct_tfi_normalization():
    params = np.zeros(binary_mera_parameter_shape(4))
    state = tfi_binary_mera(params, 4).circuit.get_statevector()
    np.testing.assert_allclose(state, np.ones(16)/4, atol=1e-14)
    _, h = hamiltonian(4, g=.9)
    assert np.vdot(state,h@state).real / 4 == pytest.approx(-.9)


@pytest.mark.parametrize('bad', [0,1,3,6,4.,True])
def test_invalid_sizes(bad):
    with pytest.raises(ValueError):
        binary_mera_layers(bad)


def test_invalid_tensors_and_parameters():
    for args in [((0,0),), ((0,1),(0,),(0,)), ((0,1),(0,),()), ((0,1.2),)]:
        with pytest.raises(ValueError):
            HVATensor(*args)
    with pytest.raises(ValueError):
        tfi_hva_tensor_layer([[.1,.2]], [HVATensor((0,1)),HVATensor((1,2))],3)
    with pytest.raises(ValueError):
        tfi_binary_mera(np.zeros(6),4)
    with pytest.raises(ValueError):
        tfi_binary_mera(np.full((3,1,2),np.nan),4)
