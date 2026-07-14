"""Per-gate noise susceptibility for a short TFI Trotter circuit."""

import spd
from spd.circuit_ir import PauliRotation


def pauli_on_qubits(system_size, axis, qubits):
    pauli = ["I"] * system_size
    for qubit in qubits:
        pauli[qubit] = axis
    return "".join(pauli)


def tfi_trotter_circuit(system_size, total_time, num_steps, coupling, field):
    dt = total_time / num_steps
    operations = []
    for _ in range(num_steps):
        for qubit in range(system_size):
            operations.append(
                PauliRotation(
                    gate_name="RX",
                    pauli=pauli_on_qubits(system_size, "X", [qubit]),
                    theta=-2.0 * dt * field,
                )
            )
        for qubit in range(system_size):
            operations.append(
                PauliRotation(
                    gate_name="RZZ",
                    pauli=pauli_on_qubits(system_size, "Z", [qubit, (qubit + 1) % system_size]),
                    theta=-2.0 * dt * coupling,
                )
            )
    return operations


if __name__ == "__main__":
    system_size = 4
    operations = tfi_trotter_circuit(
        system_size,
        total_time=0.4,
        num_steps=2,
        coupling=1.0,
        field=0.7,
    )
    backend = spd.BackendAdapter.from_name("numpy", packbit=32, precision="double")
    observable = spd.create_spo(
        {pauli_on_qubits(system_size, "X", [system_size // 2]): 1.0},
        backend=backend,
    )
    final_spo, _ = spd.evolve(
        observable,
        operations,
        trunc_val=1e-12,
        max_num_str=1_000_000,
        backend=backend,
    )
    initial_spgo = spd.init_gradient_spo(final_spo, basis="0", backend=backend)
    _, _, noise_grads, _ = spd.backpropagate_noise_analysis(
        initial_spgo,
        operations,
        trunc_val=1e-12,
        max_num_str=1_000_000,
        backend=backend,
    )

    for index, (operation, susceptibility) in enumerate(zip(operations, noise_grads)):
        qubits = tuple(i for i, letter in enumerate(operation.pauli) if letter != "I")
        if len(qubits) == 2:
            print(index, qubits, float(susceptibility))

    print("total susceptibility:", float(sum(noise_grads)))

