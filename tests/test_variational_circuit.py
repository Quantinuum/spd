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


def _legacy_tfi_circuit(params, dimension, linear_system_size, basis):
    system_size = linear_system_size**dimension
    circuit = Circuit(system_size, system_size)

    def add_x_layer(parameter):
        for qubit in range(system_size):
            circuit.Rx(parameter, qubit)
        circuit.add_barrier(list(range(system_size)))

    def add_zz_layer(parameter):
        for qubit in range(system_size):
            for axis in range(dimension):
                stride = linear_system_size**axis
                coordinate = (qubit // stride) % linear_system_size
                neighbor = qubit + stride
                if coordinate == linear_system_size - 1:
                    neighbor -= linear_system_size * stride
                circuit.ZZPhase(parameter, qubit, neighbor)
        circuit.add_barrier(list(range(system_size)))

    for layer in range(len(params) // 2):
        first = 2 * layer
        if dimension == 1 and basis == "0":
            add_x_layer(params[first])
            add_zz_layer(params[first + 1])
        else:
            add_zz_layer(params[first])
            add_x_layer(params[first + 1])
    circuit.measure_all()
    return circuit


def _local_tfi_hamiltonian(dimension, linear_system_size, g=3.1):
    system_size = linear_system_size**dimension
    hamiltonian = {}
    for axis in range(dimension):
        paulis = ["I"] * system_size
        paulis[0] = "Z"
        paulis[linear_system_size**axis] = "Z"
        hamiltonian["".join(paulis)] = -1.0
    paulis = ["I"] * system_size
    paulis[0] = "X"
    hamiltonian["".join(paulis)] = -g
    return hamiltonian


def _energy_and_gate_gradients(circuit, hamiltonian, basis):
    backend = spd.BackendAdapter.from_name("numpy", packbit=32, precision="double")
    initial_spo = spd.create_spo(hamiltonian, backend=backend)
    final_spo, _ = spd.evolve(
        initial_spo,
        circuit,
        trunc_val=1e-14,
        max_num_str=100_000,
        backend=backend,
    )
    energy = final_spo.get_expectation_value(basis=basis)
    initial_spgo = spd.init_gradient_spo(final_spo, basis=basis, backend=backend)
    _, gate_gradients, _ = spd.backpropagate(
        initial_spgo,
        circuit,
        trunc_val=1e-14,
        max_num_str=100_000,
        backend=backend,
    )
    return energy, np.asarray(gate_gradients)


def _legacy_parameter_gradients(gate_gradients, params, dimension, system_size):
    per_layer = (dimension + 1) * system_size
    result = []
    for layer in range(len(params) // 2):
        start = layer * per_layer
        split = start + dimension * system_size
        end = start + per_layer
        result.extend(
            [
                np.sum(gate_gradients[start:split]) * math.pi,
                np.sum(gate_gradients[split:end]) * math.pi,
            ]
        )
    return np.asarray(result)


@pytest.mark.parametrize(
    ("dimension", "linear_system_size", "basis"),
    [(1, 4, "+"), (1, 4, "0"), (2, 2, "+"), (3, 2, "+")],
)
def test_tfi_hva_matches_archived_energy_gradients_and_three_steps(
    dimension, linear_system_size, basis
):
    legacy_params = np.asarray([0.13, -0.21])
    new_params = legacy_params.copy()
    hamiltonian = _local_tfi_hamiltonian(dimension, linear_system_size)
    system_size = linear_system_size**dimension

    for _ in range(3):
        legacy_circuit = _legacy_tfi_circuit(
            legacy_params,
            dimension,
            linear_system_size,
            basis,
        )
        legacy_energy, legacy_gate_gradients = _energy_and_gate_gradients(
            legacy_circuit,
            hamiltonian,
            basis,
        )
        legacy_gradients = _legacy_parameter_gradients(
            legacy_gate_gradients,
            legacy_params,
            dimension,
            system_size,
        )

        if dimension == 1:
            ansatz = tfi_1d_hva(
                new_params,
                system_size=linear_system_size,
                basis=basis,
            )
        elif dimension == 2:
            ansatz = tfi_2d_hva(
                new_params,
                system_size_x=linear_system_size,
                system_size_y=linear_system_size,
            )
        else:
            ansatz = tfi_3d_hva(
                new_params,
                system_size_x=linear_system_size,
                system_size_y=linear_system_size,
                system_size_z=linear_system_size,
            )
        new_energy, new_gate_gradients = _energy_and_gate_gradients(
            ansatz.circuit,
            hamiltonian,
            basis,
        )
        new_gradients = ansatz.parameter_gradients(new_gate_gradients)

        assert new_energy == pytest.approx(legacy_energy, abs=1e-10)
        np.testing.assert_allclose(new_gradients, legacy_gradients, atol=1e-10)

        legacy_params -= 0.01 * legacy_gradients
        new_params -= 0.01 * new_gradients

    np.testing.assert_allclose(new_params, legacy_params, atol=1e-10)
