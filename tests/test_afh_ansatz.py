import math

import numpy as np
import pytest
from pytket import Circuit, OpType

import spd
from spd.ansatz import afh_1d_hva, afh_2d_hva, afh_3d_hva


@pytest.mark.parametrize(
    ("generator", "kwargs", "expected_counts"),
    [
        (afh_1d_hva, {"system_size": 4}, (4, 4, 4, 4)),
        (afh_2d_hva, {"system_size_x": 2, "system_size_y": 4}, (16, 16, 16, 8)),
        (
            afh_3d_hva,
            {"system_size_x": 2, "system_size_y": 2, "system_size_z": 2},
            (24, 24, 24, 8),
        ),
    ],
)
def test_afh_generators_map_rotation_gates_to_layer_parameters(
    generator, kwargs, expected_counts
):
    ansatz = generator(np.asarray([0.1, 0.2, 0.3, 0.4]), **kwargs)

    assert ansatz.parameter_shape == (4,)
    assert tuple(np.bincount(ansatz.gate_parameter_indices)) == expected_counts
    gradients = ansatz.parameter_gradients(
        np.ones(ansatz.gate_parameter_indices.size)
    )
    np.testing.assert_allclose(
        gradients,
        np.asarray([*expected_counts[:3], 0]) * math.pi,
        atol=1e-12,
    )


def test_afh_staggered_rotation_factors_and_neel_state():
    ansatz = afh_2d_hva(
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        system_size_x=2,
        system_size_y=2,
    )
    commands = ansatz.circuit.get_commands()
    x_qubits = [
        command.qubits[0].index[0]
        for command in commands
        if command.op.type == OpType.X
    ]
    rz_factors = ansatz.gate_parameter_factors[
        ansatz.gate_parameter_indices == 3
    ]

    assert x_qubits == [1, 2]
    np.testing.assert_array_equal(rz_factors, [1.0, -1.0, -1.0, 1.0])


def _interaction_groups(circuit):
    groups = []
    current = []
    interaction_types = {OpType.XXPhase, OpType.YYPhase, OpType.ZZPhase}
    for command in circuit.get_commands():
        if command.op.type in interaction_types:
            current.append(tuple(qubit.index[0] for qubit in command.qubits))
        elif command.op.type == OpType.Barrier and current:
            groups.append(current)
            current = []
    return groups


@pytest.mark.parametrize(
    ("generator", "kwargs", "expected_groups"),
    [
        (afh_1d_hva, {"system_size": 4}, 6),
        (afh_2d_hva, {"system_size_x": 4, "system_size_y": 4}, 12),
        (
            afh_3d_hva,
            {"system_size_x": 4, "system_size_y": 4, "system_size_z": 4},
            18,
        ),
    ],
)
def test_afh_hva_uses_disjoint_brickwork_groups(generator, kwargs, expected_groups):
    circuit = generator(np.asarray([0.1, 0.2, 0.3, 0.4]), **kwargs).circuit
    groups = _interaction_groups(circuit)

    assert len(groups) == expected_groups
    for group in groups:
        qubits = [qubit for edge in group for qubit in edge]
        assert len(qubits) == len(set(qubits))


@pytest.mark.parametrize(
    ("generator", "kwargs"),
    [
        (afh_1d_hva, {"system_size": 3}),
        (afh_2d_hva, {"system_size_x": 3, "system_size_y": 4}),
        (
            afh_3d_hva,
            {"system_size_x": 4, "system_size_y": 3, "system_size_z": 4},
        ),
    ],
)
def test_periodic_afh_hva_rejects_odd_dimensions(generator, kwargs):
    with pytest.raises(ValueError, match="must be even"):
        generator(np.asarray([0.1, 0.2, 0.3, 0.4]), **kwargs)


def test_afh_hva_rejects_invalid_parameter_shape():
    with pytest.raises(ValueError, match="nonempty flat array"):
        afh_1d_hva(np.ones((1, 4)), system_size=4)
    with pytest.raises(ValueError, match="XX, YY, ZZ"):
        afh_1d_hva(np.ones(3), system_size=4)


def _geometry(dimension, linear_system_size):
    dimensions = (linear_system_size,) * dimension
    strides = tuple(
        math.prod(dimensions[axis + 1 :]) for axis in range(dimension)
    )

    def qubit_index(coords):
        return sum(coord * stride for coord, stride in zip(coords, strides))

    return dimensions, strides, qubit_index


def _legacy_afh_circuit(params, dimension, linear_system_size):
    dimensions, _, qubit_index = _geometry(dimension, linear_system_size)
    system_size = linear_system_size**dimension
    circuit = Circuit(system_size, system_size)

    for coords in np.ndindex(dimensions):
        if sum(coords) % 2:
            circuit.X(qubit_index(coords))

    for layer in range(len(params) // 4):
        for offset, gate_name in enumerate(("XXPhase", "YYPhase", "ZZPhase")):
            for coords in np.ndindex(dimensions):
                for axis, size in enumerate(dimensions):
                    neighbor = list(coords)
                    neighbor[axis] = (neighbor[axis] + 1) % size
                    getattr(circuit, gate_name)(
                        params[4 * layer + offset],
                        qubit_index(coords),
                        qubit_index(neighbor),
                    )
            circuit.add_barrier(list(range(system_size)))

        for coords in np.ndindex(dimensions):
            sign = 1 if sum(coords) % 2 == 0 else -1
            circuit.Rz(sign * params[4 * layer + 3], qubit_index(coords))
        circuit.add_barrier(list(range(system_size)))

    circuit.measure_all()
    return circuit


def _local_afh_hamiltonian(dimension, linear_system_size):
    _, strides, _ = _geometry(dimension, linear_system_size)
    system_size = linear_system_size**dimension
    hamiltonian = {}
    for neighbor in strides:
        for pauli in "XYZ":
            paulis = ["I"] * system_size
            paulis[0] = pauli
            paulis[neighbor] = pauli
            hamiltonian["".join(paulis)] = 1.0
    return hamiltonian


def _energy_and_gate_gradients(circuit, hamiltonian):
    backend = spd.BackendAdapter.from_name("numpy", packbit=32, precision="double")
    initial_spo = spd.create_spo(hamiltonian, backend=backend)
    final_spo, _ = spd.evolve(
        initial_spo,
        circuit,
        trunc_val=1e-14,
        max_num_str=100_000,
        backend=backend,
    )
    energy = final_spo.get_expectation_value(basis="0")
    initial_spgo = spd.init_gradient_spo(final_spo, basis="0", backend=backend)
    _, gate_gradients, _ = spd.backpropagate(
        initial_spgo,
        circuit,
        trunc_val=1e-14,
        max_num_str=100_000,
        backend=backend,
    )
    return energy, np.asarray(gate_gradients)


def _legacy_parameter_gradients(
    gate_gradients, params, dimension, linear_system_size
):
    system_size = linear_system_size**dimension
    stagger_signs = np.asarray(
        [
            1 if sum(coords) % 2 == 0 else -1
            for coords in np.ndindex((linear_system_size,) * dimension)
        ]
    )
    gradients = []
    start = 0
    for parameter_index in range(len(params)):
        multiplicity = dimension if parameter_index % 4 != 3 else 1
        stop = start + multiplicity * system_size
        window = gate_gradients[start:stop]
        if parameter_index % 4 == 3:
            gradients.append(np.dot(window, stagger_signs) * math.pi)
        else:
            gradients.append(np.sum(window) * math.pi)
        start = stop
    return np.asarray(gradients)


def _new_afh_ansatz(params, dimension, linear_system_size):
    if dimension == 1:
        return afh_1d_hva(params, system_size=linear_system_size)
    if dimension == 2:
        return afh_2d_hva(
            params,
            system_size_x=linear_system_size,
            system_size_y=linear_system_size,
        )
    return afh_3d_hva(
        params,
        system_size_x=linear_system_size,
        system_size_y=linear_system_size,
        system_size_z=linear_system_size,
    )


@pytest.mark.parametrize(
    ("dimension", "linear_system_size", "num_steps"),
    [(1, 4, 3), (2, 2, 3)],
)
def test_afh_hva_matches_legacy_energy_gradients_and_updates(
    dimension, linear_system_size, num_steps
):
    legacy_params = np.asarray([0.07, -0.11, 0.13, -0.17])
    new_params = legacy_params.copy()
    hamiltonian = _local_afh_hamiltonian(dimension, linear_system_size)

    for _ in range(num_steps):
        legacy_circuit = _legacy_afh_circuit(
            legacy_params,
            dimension,
            linear_system_size,
        )
        legacy_energy, legacy_gate_gradients = _energy_and_gate_gradients(
            legacy_circuit,
            hamiltonian,
        )
        legacy_gradients = _legacy_parameter_gradients(
            legacy_gate_gradients,
            legacy_params,
            dimension,
            linear_system_size,
        )

        ansatz = _new_afh_ansatz(new_params, dimension, linear_system_size)
        new_energy, new_gate_gradients = _energy_and_gate_gradients(
            ansatz.circuit,
            hamiltonian,
        )
        new_gradients = ansatz.parameter_gradients(new_gate_gradients)

        assert new_energy == pytest.approx(legacy_energy, abs=1e-10)
        np.testing.assert_allclose(new_gradients, legacy_gradients, atol=1e-10)

        legacy_params -= 0.01 * legacy_gradients
        new_params -= 0.01 * new_gradients

    np.testing.assert_allclose(new_params, legacy_params, atol=1e-10)


def _without_measurements(circuit):
    result = Circuit(circuit.n_qubits)
    for command in circuit.get_commands():
        if command.op.type not in (OpType.Barrier, OpType.Measure):
            result.add_gate(
                command.op,
                [qubit.index[0] for qubit in command.qubits],
            )
    return result


def _dense_hamiltonian(hamiltonian):
    matrices = {
        "I": np.eye(2),
        "X": np.asarray([[0, 1], [1, 0]]),
        "Y": np.asarray([[0, -1j], [1j, 0]]),
        "Z": np.asarray([[1, 0], [0, -1]]),
    }
    result = None
    for pauli_string, coefficient in hamiltonian.items():
        term = np.asarray([[1.0]])
        for pauli in pauli_string:
            term = np.kron(term, matrices[pauli])
        result = coefficient * term if result is None else result + coefficient * term
    return result


def _dense_energy(circuit, hamiltonian):
    state = _without_measurements(circuit).get_statevector()
    return float(np.real(np.vdot(state, hamiltonian @ state)))


def _finite_difference_gradients(circuit_builder, params, hamiltonian):
    gradients = []
    step = 1e-6
    for index in range(len(params)):
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        gradients.append(
            (
                _dense_energy(circuit_builder(plus), hamiltonian)
                - _dense_energy(circuit_builder(minus), hamiltonian)
            )
            / (2 * step)
        )
    return np.asarray(gradients)


def test_afh_3d_matches_legacy_dense_energy_and_parameter_gradients():
    legacy_params = np.asarray([0.07, -0.11, 0.13, -0.17])
    new_params = legacy_params.copy()
    hamiltonian = _dense_hamiltonian(_local_afh_hamiltonian(3, 2))
    legacy_builder = lambda values: _legacy_afh_circuit(values, 3, 2)
    new_builder = lambda values: _new_afh_ansatz(values, 3, 2).circuit

    for _ in range(3):
        assert _dense_energy(new_builder(new_params), hamiltonian) == pytest.approx(
            _dense_energy(legacy_builder(legacy_params), hamiltonian),
            abs=1e-9,
        )
        legacy_gradients = _finite_difference_gradients(
            legacy_builder,
            legacy_params,
            hamiltonian,
        )
        new_gradients = _finite_difference_gradients(
            new_builder,
            new_params,
            hamiltonian,
        )
        np.testing.assert_allclose(new_gradients, legacy_gradients, atol=1e-8)
        legacy_params -= 0.01 * legacy_gradients
        new_params -= 0.01 * new_gradients

    np.testing.assert_allclose(new_params, legacy_params, atol=1e-9)


def test_afh_parameter_gradients_include_ose_and_staggered_factors():
    params = np.asarray([0.07, -0.11, 0.13, -0.17])
    hamiltonian = _local_afh_hamiltonian(1, 4)
    backend = spd.BackendAdapter.from_name("numpy", packbit=32, precision="double")
    lambda_ose = 0.1
    alpha = 1.0

    def cost(values):
        ansatz = afh_1d_hva(values, system_size=4)
        initial_spo = spd.create_spo(hamiltonian, backend=backend)
        final_spo, _ = spd.evolve(
            initial_spo,
            ansatz.circuit,
            trunc_val=1e-14,
            max_num_str=100_000,
            backend=backend,
        )
        return (
            final_spo.get_expectation_value(basis="0")
            + lambda_ose * final_spo.get_OSE(alpha=alpha)
        )

    ansatz = afh_1d_hva(params, system_size=4)
    initial_spo = spd.create_spo(hamiltonian, backend=backend)
    final_spo, _ = spd.evolve(
        initial_spo,
        ansatz.circuit,
        trunc_val=1e-14,
        max_num_str=100_000,
        backend=backend,
    )
    initial_spgo = spd.init_gradient_spo(
        final_spo,
        basis="0",
        lambda_ose=lambda_ose,
        alpha=alpha,
        backend=backend,
    )
    _, gate_gradients, _ = spd.backpropagate(
        initial_spgo,
        ansatz.circuit,
        trunc_val=1e-14,
        max_num_str=100_000,
        backend=backend,
    )

    finite_difference = []
    step = 1e-6
    for index in range(len(params)):
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference.append((cost(plus) - cost(minus)) / (2 * step))

    np.testing.assert_allclose(
        ansatz.parameter_gradients(gate_gradients),
        finite_difference,
        rtol=1e-5,
        atol=1e-6,
    )
