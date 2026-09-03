import math

import numpy as np
import pytest
from pytket import Circuit, OpType

import spd
from spd.ansatz import tfi_1d_hva, tfi_2d_hva, tfi_3d_hva


def test_parameter_gradients_sum_shared_gates_and_restore_shape():
    circuit = Circuit(2)
    circuit.Rx(0.1, 0)
    circuit.Rz(0.2, 0)
    circuit.Rx(0.1, 1)

    vcircuit = spd.VariationalCircuit(
        circuit,
        gate_parameter_indices=[0, -1, 0],
        gate_parameter_factors=[1.0, 1.0, 0.5],
        parameter_shape=(1, 1),
    )

    gradients = vcircuit.parameter_gradients([2.0, 100.0, 4.0])
    np.testing.assert_allclose(gradients, [[4.0 * math.pi]])


def test_variational_circuit_validates_metadata_and_gradient_count():
    circuit = Circuit(1).Rx(0.1, 0)

    with pytest.raises(ValueError, match="metadata for 1 rotation gates"):
        spd.VariationalCircuit(circuit, [], (1,))

    vcircuit = spd.VariationalCircuit(circuit, [0], (1,))
    with pytest.raises(ValueError, match="Expected 1 gate gradients"):
        vcircuit.parameter_gradients([])


def test_parameter_gradient_matches_pytket_phase_derivative():
    parameter = 0.2
    circuit = Circuit(1).Rx(parameter, 0)
    vcircuit = spd.VariationalCircuit(circuit, [0], (1,))

    initial_spo = spd.create_spo({"Z": 1.0})
    final_spo, _ = spd.evolve(initial_spo, circuit, 1e-12, 100)
    initial_spgo = spd.init_gradient_spo(final_spo, basis="0")
    _, gate_gradients, _ = spd.backpropagate(initial_spgo, circuit, 1e-12, 100)

    gradients = vcircuit.parameter_gradients(gate_gradients)
    np.testing.assert_allclose(
        gradients,
        [-math.pi * math.sin(math.pi * parameter)],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("generator", "kwargs", "expected_counts"),
    [
        (tfi_1d_hva, {"system_size": 4}, (4, 4)),
        (tfi_2d_hva, {"system_size_x": 2, "system_size_y": 4}, (16, 8)),
        (
            tfi_3d_hva,
            {"system_size_x": 2, "system_size_y": 2, "system_size_z": 2},
            (24, 8),
        ),
    ],
)
def test_tfi_generators_map_rotation_gates_to_layer_parameters(
    generator, kwargs, expected_counts
):
    vcircuit = generator(np.asarray([0.1, 0.2]), **kwargs)

    assert vcircuit.parameter_shape == (2,)
    assert tuple(np.bincount(vcircuit.gate_parameter_indices)) == expected_counts
    gradients = vcircuit.parameter_gradients(
        np.ones(vcircuit.gate_parameter_indices.size)
    )
    np.testing.assert_allclose(gradients, np.asarray(expected_counts) * math.pi)


def test_tfi_1d_zero_basis_preserves_parameter_order():
    vcircuit = tfi_1d_hva(np.asarray([0.1, 0.2]), system_size=4, basis="0")
    np.testing.assert_array_equal(
        vcircuit.gate_parameter_indices,
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
    )


def _zz_groups(circuit):
    groups = []
    current_group = []
    for command in circuit.get_commands():
        if command.op.type == OpType.ZZPhase:
            current_group.append(tuple(qubit.index[0] for qubit in command.qubits))
        elif command.op.type == OpType.Barrier and current_group:
            groups.append(current_group)
            current_group = []
    return groups


@pytest.mark.parametrize(
    ("generator", "kwargs", "expected_groups"),
    [
        (tfi_1d_hva, {"system_size": 4}, 2),
        (tfi_2d_hva, {"system_size_x": 4, "system_size_y": 4}, 4),
        (
            tfi_3d_hva,
            {"system_size_x": 4, "system_size_y": 4, "system_size_z": 4},
            6,
        ),
    ],
)
def test_tfi_hva_uses_disjoint_brickwork_groups(generator, kwargs, expected_groups):
    circuit = generator(np.asarray([0.1, 0.2]), **kwargs).circuit
    groups = _zz_groups(circuit)

    assert len(groups) == expected_groups
    for group in groups:
        qubits = [qubit for edge in group for qubit in edge]
        assert len(qubits) == len(set(qubits))


@pytest.mark.parametrize(
    ("generator", "kwargs"),
    [
        (tfi_1d_hva, {"system_size": 3}),
        (tfi_2d_hva, {"system_size_x": 3, "system_size_y": 4}),
        (
            tfi_3d_hva,
            {"system_size_x": 4, "system_size_y": 3, "system_size_z": 4},
        ),
    ],
)
def test_periodic_tfi_hva_rejects_odd_dimensions(generator, kwargs):
    with pytest.raises(ValueError, match="must be even"):
        generator(np.asarray([0.1, 0.2]), **kwargs)
