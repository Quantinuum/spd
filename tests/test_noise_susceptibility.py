import numpy as np
import pytest

import spd
from spd.circuit_ir import (
    CircuitIR,
    PauliRotation,
    SingleQubitClifford,
    SkippedOperation,
    TwoQubitClifford,
)
from tests.helpers import assert_info_consistent, to_grad_term_dict


PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _matrix(pauli):
    result = np.array([[1.0]], dtype=complex)
    for letter in pauli:
        result = np.kron(result, PAULI[letter])
    return result


def _rotation(operation):
    generator = _matrix(operation.pauli)
    identity = np.eye(generator.shape[0], dtype=complex)
    return np.cos(operation.theta / 2) * identity - 1j * np.sin(operation.theta / 2) * generator


def _depolarize_pair(rho, qubits, probability, num_qubits):
    twirled = np.zeros_like(rho)
    for first in "IXYZ":
        for second in "IXYZ":
            pauli = ["I"] * num_qubits
            pauli[qubits[0]] = first
            pauli[qubits[1]] = second
            matrix = _matrix("".join(pauli))
            twirled += matrix @ rho @ matrix
    return (1.0 - probability) * rho + (probability / 16.0) * twirled


def _depolarize_qubit(rho, qubit, probability, num_qubits):
    twirled = np.zeros_like(rho)
    for letter in "IXYZ":
        pauli = ["I"] * num_qubits
        pauli[qubit] = letter
        matrix = _matrix("".join(pauli))
        twirled += matrix @ rho @ matrix
    return (1.0 - probability) * rho + (probability / 4.0) * twirled


def _dense_expectation(
    operations,
    observable,
    *,
    one_qubit_probabilities=None,
    two_qubit_probabilities=None,
):
    num_qubits = len(operations[0].pauli)
    one_qubit_probabilities = one_qubit_probabilities or [0.0] * len(operations)
    two_qubit_probabilities = two_qubit_probabilities or [0.0] * len(operations)
    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1.0
    rho = np.outer(state, state.conj())

    for index, operation in enumerate(operations):
        unitary = _rotation(operation)
        rho = unitary @ rho @ unitary.conj().T
        qubits = tuple(i for i, letter in enumerate(operation.pauli) if letter != "I")
        if len(qubits) == 1:
            rho = _depolarize_qubit(
                rho,
                qubits[0],
                one_qubit_probabilities[index],
                num_qubits,
            )
        elif len(qubits) == 2:
            rho = _depolarize_pair(
                rho,
                qubits,
                two_qubit_probabilities[index],
                num_qubits,
            )

    return float(np.real(np.trace(_matrix(observable) @ rho)))


def _backend(backend_name):
    backend = spd.BackendAdapter.from_name(backend_name, packbit=32, precision="double")
    if backend_name == "jax":
        backend.module.set_algorithm("stack_sort_merge")
    return backend


def _make_spgo(backend_name, terms):
    module = getattr(spd, f"{backend_name}_backend")
    if backend_name == "numpy":
        spgo = module.SparsePauliGradientOp()
        for pauli, (coeff, grad) in terms.items():
            spgo[tuple(module.utils.pauli_str_to_uint(pauli))] = (coeff, grad)
        return spgo

    rows = np.asarray([module.utils.pauli_str_to_uint(pauli) for pauli in terms])
    coeffs = np.asarray([value[0] for value in terms.values()])
    grads = np.asarray([value[1] for value in terms.values()])
    return module.SparsePauliGradientOp(rows, coeffs, grads)


@pytest.mark.parametrize("backend_name", ["numpy", "jax"])
def test_depolarizing_susceptibility_kernel_uses_x_y_and_z_support(backend_name):
    backend = _backend(backend_name)
    spgo = _make_spgo(
        backend_name,
        {
            "II": (2.0, 3.0),
            "XI": (1.0, 2.0),
            "IZ": (3.0, 4.0),
            "YY": (5.0, 6.0),
        },
    )

    susceptibility = backend.get_two_qubit_depolarizing_susceptibility(spgo, (0, 1))

    assert np.isclose(float(np.asarray(susceptibility)), -44.0)

    one_qubit = backend.get_one_qubit_depolarizing_susceptibility(spgo, 0)
    assert np.isclose(float(np.asarray(one_qubit)), -32.0)


@pytest.mark.parametrize("backend_name", ["numpy", "jax"])
def test_noise_analysis_aligns_with_rx_rzz_operations_and_matches_backpropagate(backend_name):
    backend = _backend(backend_name)
    operations = (
        PauliRotation("RX", "XII", 0.31),
        PauliRotation("RZZ", "ZZI", -0.27),
        PauliRotation("RX", "IXI", 0.19),
        PauliRotation("RZZ", "IZZ", 0.23),
    )
    circuit_ir = CircuitIR(system_size=3, operations=operations)
    observable = spd.create_spo({"ZII": 0.7, "IZI": -0.2, "IIZ": 0.4}, backend=backend)
    final_spo, _ = spd.evolve(observable, operations, 1e-12, 1_000_000, backend=backend)
    initial_spgo = spd.init_gradient_spo(final_spo, basis="0", backend=backend)

    ordinary_spgo, ordinary_grads, ordinary_info = spd.backpropagate(
        initial_spgo,
        circuit_ir,
        1e-12,
        1_000_000,
        backend=backend,
    )
    analyzed_spgo, parameter_grads, noise_grads, info = spd.backpropagate_noise_analysis(
        initial_spgo,
        circuit_ir,
        1e-12,
        1_000_000,
        backend=backend,
    )

    assert len(parameter_grads) == len(operations)
    assert set(noise_grads) == {
        "one_qubit_depolarizing",
        "two_qubit_depolarizing",
    }
    assert all(len(values) == len(operations) for values in noise_grads.values())
    assert noise_grads["two_qubit_depolarizing"][0] == 0
    assert noise_grads["two_qubit_depolarizing"][2] == 0
    assert noise_grads["one_qubit_depolarizing"][1] == 0
    assert noise_grads["one_qubit_depolarizing"][3] == 0
    assert np.allclose(np.asarray(parameter_grads), np.asarray(ordinary_grads), atol=1e-9)
    assert to_grad_term_dict(backend_name, backend.module, analyzed_spgo, 3) == pytest.approx(
        to_grad_term_dict(backend_name, backend.module, ordinary_spgo, 3),
        abs=1e-9,
    )
    assert info == ordinary_info
    assert_info_consistent(info, expected_steps=len(operations))


@pytest.mark.parametrize("backend_name", ["numpy", "jax"])
def test_per_gate_and_total_susceptibility_match_dense_finite_difference(backend_name):
    backend = _backend(backend_name)
    operations = (
        PauliRotation("RX", "XII", 0.31),
        PauliRotation("RZZ", "ZZI", -0.27),
        PauliRotation("RX", "IXI", 0.19),
        PauliRotation("RZZ", "IZZ", 0.23),
    )
    circuit_ir = CircuitIR(system_size=3, operations=operations)
    observable_pauli = "ZIZ"
    observable = spd.create_spo({observable_pauli: 1.0}, backend=backend)
    final_spo, _ = spd.evolve(observable, operations, 1e-12, 1_000_000, backend=backend)
    initial_spgo = spd.init_gradient_spo(final_spo, basis="0", backend=backend)
    _, _, noise_grads, _ = spd.backpropagate_noise_analysis(
        initial_spgo,
        circuit_ir,
        1e-12,
        1_000_000,
        backend=backend,
    )

    eps = 1e-6
    finite_difference_grads = []
    for noisy_index in (1, 3):
        plus = [0.0] * len(operations)
        minus = [0.0] * len(operations)
        plus[noisy_index] = eps
        minus[noisy_index] = -eps
        finite_difference_grads.append(
            (
                _dense_expectation(
                    operations,
                    observable_pauli,
                    two_qubit_probabilities=plus,
                )
                - _dense_expectation(
                    operations,
                    observable_pauli,
                    two_qubit_probabilities=minus,
                )
            )
            / (2 * eps)
        )

    two_qubit_grads = noise_grads["two_qubit_depolarizing"]
    actual_per_gate = [float(np.asarray(two_qubit_grads[index])) for index in (1, 3)]
    assert np.allclose(actual_per_gate, finite_difference_grads, atol=1e-7)

    plus = [eps if index in (1, 3) else 0.0 for index in range(len(operations))]
    minus = [-eps if index in (1, 3) else 0.0 for index in range(len(operations))]
    common_p_grad = (
        _dense_expectation(operations, observable_pauli, two_qubit_probabilities=plus)
        - _dense_expectation(operations, observable_pauli, two_qubit_probabilities=minus)
    ) / (2 * eps)
    assert np.isclose(sum(actual_per_gate), common_p_grad, atol=1e-7)


@pytest.mark.parametrize("backend_name", ["numpy", "jax"])
def test_one_qubit_susceptibilities_match_dense_finite_difference(backend_name):
    backend = _backend(backend_name)
    operations = (
        PauliRotation("RX", "XII", 0.31),
        PauliRotation("RZZ", "ZZI", -0.27),
        PauliRotation("RX", "IXI", 0.19),
        PauliRotation("RZZ", "IZZ", 0.23),
    )
    circuit_ir = CircuitIR(system_size=3, operations=operations)
    observable_pauli = "ZIZ"
    observable = spd.create_spo({observable_pauli: 1.0}, backend=backend)
    final_spo, _ = spd.evolve(observable, circuit_ir, 1e-12, 1_000_000, backend=backend)
    initial_spgo = spd.init_gradient_spo(final_spo, basis="0", backend=backend)
    _, _, noise_grads, _ = spd.backpropagate_noise_analysis(
        initial_spgo,
        circuit_ir,
        1e-12,
        1_000_000,
        backend=backend,
    )

    eps = 1e-6
    expected = []
    active_indices = (0, 2)
    for noisy_index in active_indices:
        plus = [0.0] * len(operations)
        minus = [0.0] * len(operations)
        plus[noisy_index] = eps
        minus[noisy_index] = -eps
        expected.append(
            (
                _dense_expectation(
                    operations,
                    observable_pauli,
                    one_qubit_probabilities=plus,
                )
                - _dense_expectation(
                    operations,
                    observable_pauli,
                    one_qubit_probabilities=minus,
                )
            )
            / (2 * eps)
        )

    actual = [
        float(np.asarray(noise_grads["one_qubit_depolarizing"][index]))
        for index in active_indices
    ]
    assert np.allclose(actual, expected, atol=1e-7)

    plus = [eps if index in active_indices else 0.0 for index in range(len(operations))]
    minus = [-eps if index in active_indices else 0.0 for index in range(len(operations))]
    expected_total = (
        _dense_expectation(
            operations,
            observable_pauli,
            one_qubit_probabilities=plus,
        )
        - _dense_expectation(
            operations,
            observable_pauli,
            one_qubit_probabilities=minus,
        )
    ) / (2 * eps)
    assert np.isclose(sum(actual), expected_total, atol=1e-7)


@pytest.mark.parametrize("backend_name", ["numpy", "jax"])
def test_noise_analysis_aligns_cliffords_and_skipped_operations(backend_name):
    backend = _backend(backend_name)
    circuit_ir = CircuitIR(
        system_size=3,
        operations=(
            SingleQubitClifford("OpType.H", 0),
            TwoQubitClifford("OpType.CX", 0, 1),
            SkippedOperation("barrier"),
        ),
    )
    observable = spd.create_spo({"ZII": 1.0}, backend=backend)
    final_spo, _ = spd.evolve(observable, circuit_ir, 1e-12, 1_000_000, backend=backend)
    initial_spgo = spd.init_gradient_spo(final_spo, backend=backend)

    _, parameter_grads, noise_grads, _ = spd.backpropagate_noise_analysis(
        initial_spgo,
        circuit_ir,
        1e-12,
        1_000_000,
        backend=backend,
    )

    assert parameter_grads == []
    assert all(len(values) == 3 for values in noise_grads.values())
    assert noise_grads["two_qubit_depolarizing"][0] == 0
    assert noise_grads["one_qubit_depolarizing"][1] == 0
    assert all(values[2] == 0 for values in noise_grads.values())


@pytest.mark.parametrize("backend_name", ["numpy", "jax"])
def test_noise_analysis_rejects_rotation_with_more_than_two_qubits(backend_name):
    backend = _backend(backend_name)
    operations = (PauliRotation("three_qubit_rotation", "XXX", 0.2),)
    circuit_ir = CircuitIR(system_size=3, operations=operations)
    observable = spd.create_spo({"ZII": 1.0}, backend=backend)
    final_spo, _ = spd.evolve(observable, operations, 1e-12, 1_000_000, backend=backend)
    initial_spgo = spd.init_gradient_spo(final_spo, backend=backend)

    with pytest.raises(ValueError, match="Compile the circuit"):
        spd.backpropagate_noise_analysis(
            initial_spgo,
            circuit_ir,
            1e-12,
            1_000_000,
            backend=backend,
        )


@pytest.mark.parametrize("backend_name", ["numpy", "jax"])
def test_noise_analysis_rejects_raw_operation_sequences(backend_name):
    backend = _backend(backend_name)
    operations = [PauliRotation("RX", "X", 0.2)]
    observable = spd.create_spo({"Z": 1.0}, backend=backend)
    final_spo, _ = spd.evolve(observable, operations, 1e-12, 1000, backend=backend)
    initial_spgo = spd.init_gradient_spo(final_spo, backend=backend)

    with pytest.raises(TypeError, match="CircuitIR"):
        spd.backpropagate_noise_analysis(
            initial_spgo,
            operations,
            1e-12,
            1000,
            backend=backend,
        )
