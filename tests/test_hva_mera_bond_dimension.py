"""Bond dimensions, mixed bottom widths, and independent exact references."""
import numpy as np
import pytest
from pytket import OpType

from spd.ansatz import (
    HVATensor, binary_mera_sites, binary_mera_layers, binary_mera_parameter_shape,
    binary_mera_qubit_initializations, binary_mera_causal_cone,
    tfi_hva_tensor_layer, tfi_binary_mera,
)
from examples.gradient.run_1d_tfi_hva_tmera import (
    exact_state_and_tangents, exact_value_gradient, hamiltonian_action,
)


def test_chi4_site_counts_and_binary_placement():
    assert [binary_mera_sites(8,s) for s in range(4)] == [
        tuple((i,) for i in range(8)),
        ((0,1),(2,3),(4,5),(6,7)),
        ((2,3),(6,7)),
        ((6,7),),
    ]
    layers = binary_mera_layers(8)
    assert [l.ancillas for l in layers] == [(2,3),(0,1,4,5),()]
    assert [[t.outputs for t in l.isometries] for l in layers] == [
        [(2,3,6,7)], [(0,1,2,3),(4,5,6,7)], [(0,1),(2,3),(4,5),(6,7)]]
    assert [[t.retained for t in l.isometries] for l in layers] == [
        [(6,7)], [(2,3),(6,7)], [(0,1),(2,3),(4,5),(6,7)]]
    assert [[t.outputs for t in l.disentanglers] for l in layers] == [
        [], [(2,3,4,5),(6,7,0,1)], [(1,2),(3,4),(5,6),(7,0)]]


def test_chi8_dimension_ramp_and_partial_ancilla_transition():
    assert [len(binary_mera_sites(8,s,chi=8)[0]) for s in range(4)] == [1,2,3,3]
    layers = binary_mera_layers(8,chi=8)
    assert [l.ancillas for l in layers] == [(1,2,3),(0,4),()]
    assert [(len(l.isometries[0].retained),len(l.isometries[0].outputs)) for l in layers] == [(3,6),(3,4),(2,2)]
    # No fictitious third qubit at the first renormalized scale.
    assert binary_mera_sites(8,1,chi=8) == ((0,1),(2,3),(4,5),(6,7))


@pytest.mark.parametrize('chi', [2,4,8])
@pytest.mark.parametrize('n', [4,8,16])
def test_nested_index_partition_and_tensor_site_cones(n,chi):
    layers = binary_mera_layers(n,chi=chi)
    created = list(binary_mera_sites(n,len(layers),chi=chi)[0])
    for layer in layers:
        outputs = [i for t in layer.isometries for i in t.outputs]
        assert len(outputs) == len(set(outputs))
        assert [t.retained for t in layer.isometries] == list(binary_mera_sites(n,layer.scale+1,chi=chi))
        assert set(created) | set(layer.ancillas) == set(outputs)
        created.extend(layer.ancillas)
    assert sorted(created) == list(range(n))
    for i in range(n):
        cone = binary_mera_causal_cone(n,[i,(i+1)%n],chi=chi)
        for scale, support in enumerate(cone):
            sites = binary_mera_sites(n,scale,chi=chi)
            assert sum(bool(set(site) & set(support)) for site in sites) <= 3
            assert all(not set(site) & set(support) or set(site) <= set(support)
                       for site in sites)


def test_chi4_synchronized_brickwork_and_parameter_counts():
    params=np.arange(39)/137.
    vc=tfi_binary_mera(params,8)
    matchings=[]
    current=[]
    for command in vc.circuit.get_commands():
        if command.op.type==OpType.Barrier:
            if current and current[0][0]==OpType.ZZPhase:
                matchings.append({qubits for _,qubits in current})
            flattened=[i for _,qubits in current for i in qubits]
            assert len(flattened)==len(set(flattened))
            current=[]
        else:
            current.append((command.op.type,tuple(q.index[0] for q in command.qubits)))
    assert matchings==[
        {(2,3),(6,7)}, {(3,6)},
        {(0,1),(2,3),(4,5),(6,7)}, {(1,2),(5,6)},
        {(2,3),(4,5),(6,7),(0,1)}, {(3,4),(7,0)},
        {(0,1),(2,3),(4,5),(6,7)},
        {(1,2),(3,4),(5,6),(7,0)},
    ]
    assert sum(c.op.type==OpType.Barrier for c in vc.circuit.get_commands())==19
    np.testing.assert_array_equal(
        np.bincount(vc.gate_parameter_indices[vc.gate_parameter_indices>=0]),
        [1]*11+[2]*22+[4]*6)
    assert sum(vc.gate_parameter_indices==-1)==8
    rotations=[c for c in vc.circuit.get_commands() if c.op.type!=OpType.Barrier]
    for command,index,factor in zip(
            rotations,vc.gate_parameter_indices,vc.gate_parameter_factors):
        assert float(command.op.params[0])==pytest.approx(
            .5 if index==-1 else params[index]*factor)
        assert factor==1.


def test_chi4_parameter_shapes_and_optional_physical_layer():
    assert binary_mera_parameter_shape(8)==(39,)
    assert binary_mera_parameter_shape(8,physical_bottom=False)==(33,)
    assert binary_mera_parameter_shape(16)==(61,)
    assert binary_mera_parameter_shape(16,physical_bottom=False)==(55,)
    common=np.random.default_rng(9).normal(0,.1,33)
    full=np.concatenate((common,np.zeros(6)))
    a=tfi_binary_mera(common,8,physical_bottom=False).circuit.get_statevector()
    b=tfi_binary_mera(full,8,physical_bottom=True).circuit.get_statevector()
    np.testing.assert_allclose(a,b,atol=1e-13)


@pytest.mark.parametrize('chi,groups', [(4,[(6,7),(2,3),(0,1,4,5)]),
                                      (8,[(5,6,7),(1,2,3),(0,4)])])
def test_qubit_initializations_exclude_square_bottom_and_preserve_first_use(chi,groups):
    shape = binary_mera_parameter_shape(8,2,chi=chi)
    vc = tfi_binary_mera(np.random.default_rng(3).normal(0,.1,shape),8,2,chi=chi)
    initializations = binary_mera_qubit_initializations(8,2,chi=chi)
    commands = vc.circuit.get_commands()
    assert [group for _,group in initializations] == groups
    for offset,indices in initializations:
        assert all(c.op.type == OpType.Ry for c in commands[offset:offset+len(indices)])
        assert {q.index[0] for c in commands[offset:offset+len(indices)] for q in c.qubits} == set(indices)
        assert all(not {q.index[0] for q in c.qubits} & set(indices)
                   for c in commands[:offset] if c.op.type != OpType.Barrier)


@pytest.mark.parametrize('outputs,retained,ancillas', [
    ((0,1,2,3),(2,3),(0,1)),  # chi4 saturated W: 2 -> 4
    ((0,1),(0,1),()),        # physical bottom W: 2 -> 2
    ((0,1,2,3),(1,2,3),(0,)), # chi8 transition W: 3 -> 4
])
def test_actual_square_and_expanding_isometries(outputs,retained,ancillas):
    n = len(outputs)
    tensor = HVATensor(outputs,retained,ancillas)
    count=3 if n==2 else 3*n-1
    params=np.random.default_rng(17+n).normal(0,.2,(2,count))
    circuit=tfi_hva_tensor_layer(params,[tensor],n).circuit
    u = circuit.get_unitary()
    columns = [i for i in range(2**n) if all(not (i & (1<<(n-1-q))) for q in ancillas)]
    w = u[:,columns]
    np.testing.assert_allclose(u.conj().T@u,np.eye(2**n),atol=1e-13)
    np.testing.assert_allclose(w.conj().T@w,np.eye(2**len(retained)),atol=1e-13)
    assert sum(c.op.type == OpType.Ry for c in circuit.get_commands()) == len(ancillas)


@pytest.mark.parametrize('n,chi,steps', [(4,4,2),(8,4,1),(8,4,2),(8,8,1)])
def test_independent_state_tangents_and_all_parameter_finite_differences(n,chi,steps):
    params = np.random.default_rng(11+chi+steps).normal(0,.15,binary_mera_parameter_shape(n,steps,chi=chi))
    state,_ = exact_state_and_tangents(params,n,chi)
    vc = tfi_binary_mera(params,n,steps,chi=chi)
    np.testing.assert_allclose(state,vc.circuit.get_statevector(),atol=1e-13)
    value,gradient = exact_value_gradient(params,n,chi,.9)
    finite = np.zeros(params.size)
    for i in range(params.size):
        plus,minus = params.copy(),params.copy()
        plus.flat[i] += 1e-6
        minus.flat[i] -= 1e-6
        a = tfi_binary_mera(plus,n,steps,chi=chi).circuit.get_statevector()
        b = tfi_binary_mera(minus,n,steps,chi=chi).circuit.get_statevector()
        finite[i] = (np.vdot(a,hamiltonian_action(a,n,.9)).real
                     - np.vdot(b,hamiltonian_action(b,n,.9)).real) / 2e-6
    np.testing.assert_allclose(gradient.ravel(),finite,atol=2e-8)
    assert value == pytest.approx(np.vdot(state,hamiltonian_action(state,n,.9)).real)


@pytest.mark.parametrize('chi', [0,3,16,4.,True])
def test_invalid_chi(chi):
    with pytest.raises(ValueError):
        binary_mera_layers(8,chi=chi)


@pytest.mark.parametrize('scale', [-1,4,1.,True])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):
        binary_mera_sites(8,scale)


def test_top_dimension_cannot_exceed_available_qubits():
    with pytest.raises(ValueError,match='top'):
        binary_mera_layers(2,chi=8)
