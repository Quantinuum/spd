import numpy as np
import pytest

import spd
from spd.circuit_ir import PauliRotation
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


def _dense_expectation(operations, observable, noise_probabilities):
    num_qubits = len(operations[0].pauli)
    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1.0
    rho = np.outer(state, state.conj())

    for index, operation in enumerate(operations):
        unitary = _rotation(operation)
        rho = unitary @ rho @ unitary.conj().T
        qubits = tuple(i for i, letter in enumerate(operation.pauli) if letter != "I")
        if len(qubits) == 2:
            rho = _depolarize_pair(
                rho,
                qubits,
                noise_probabilities[index],
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


@pytest.mark.parametrize("backend_name", ["numpy", "jax"])
def test_noise_analysis_aligns_with_rx_rzz_operations_and_matches_backpropagate(backend_name):
    backend = _backend(backend_name)
    operations = [
        PauliRotation("RX", "XII", 0.31),
        PauliRotation("RZZ", "ZZI", -0.27),
        PauliRotation("RX", "IXI", 0.19),
        PauliRotation("RZZ", "IZZ", 0.23),
    ]
    observable = spd.create_spo({"ZII": 0.7, "IZI": -0.2, "IIZ": 0.4}, backend=backend)
    final_spo, _ = spd.evolve(observable, operations, 1e-12, 1_000_000, backend=backend)
    initial_spgo = spd.init_gradient_spo(final_spo, basis="0", backend=backend)

    ordinary_spgo, ordinary_grads, ordinary_info = spd.backpropagate(
        initial_spgo,
        operations,
        1e-12,
        1_000_000,
        backend=backend,
    )
    analyzed_spgo, parameter_grads, noise_grads, info = spd.backpropagate_noise_analysis(
        initial_spgo,
        operations,
        1e-12,
        1_000_000,
        backend=backend,
    )

    assert len(parameter_grads) == len(operations)
    assert len(noise_grads) == len(operations)
    assert noise_grads[0] == 0
    assert noise_grads[2] == 0
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
    operations = [
        PauliRotation("RX", "XII", 0.31),
        PauliRotation("RZZ", "ZZI", -0.27),
        PauliRotation("RX", "IXI", 0.19),
        PauliRotation("RZZ", "IZZ", 0.23),
    ]
    observable_pauli = "ZIZ"
    observable = spd.create_spo({observable_pauli: 1.0}, backend=backend)
    final_spo, _ = spd.evolve(observable, operations, 1e-12, 1_000_000, backend=backend)
    initial_spgo = spd.init_gradient_spo(final_spo, basis="0", backend=backend)
    _, _, noise_grads, _ = spd.backpropagate_noise_analysis(
        initial_spgo,
        operations,
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
                _dense_expectation(operations, observable_pauli, plus)
                - _dense_expectation(operations, observable_pauli, minus)
            )
            / (2 * eps)
        )

    actual_per_gate = [float(np.asarray(noise_grads[index])) for index in (1, 3)]
    assert np.allclose(actual_per_gate, finite_difference_grads, atol=1e-7)

    plus = [eps if index in (1, 3) else 0.0 for index in range(len(operations))]
    minus = [-eps if index in (1, 3) else 0.0 for index in range(len(operations))]
    common_p_grad = (
        _dense_expectation(operations, observable_pauli, plus)
        - _dense_expectation(operations, observable_pauli, minus)
    ) / (2 * eps)
    assert np.isclose(sum(actual_per_gate), common_p_grad, atol=1e-7)


@pytest.mark.parametrize("backend_name", ["numpy", "jax"])
def test_noise_analysis_rejects_rotation_with_more_than_two_qubits(backend_name):
    backend = _backend(backend_name)
    operations = [PauliRotation("three_qubit_rotation", "XXX", 0.2)]
    observable = spd.create_spo({"ZII": 1.0}, backend=backend)
    final_spo, _ = spd.evolve(observable, operations, 1e-12, 1_000_000, backend=backend)
    initial_spgo = spd.init_gradient_spo(final_spo, backend=backend)

    with pytest.raises(ValueError, match="Compile the circuit"):
        spd.backpropagate_noise_analysis(
            initial_spgo,
            operations,
            1e-12,
            1_000_000,
            backend=backend,
        )
