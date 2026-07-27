"""Per-gate noise susceptibility for a short TFI Trotter circuit."""

import spd
from spd.circuit_ir import PauliRotation, TwoQubitClifford


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

def tfi_trotter_circuit_cx(system_size, total_time, num_steps, coupling, field):
    dt = total_time / num_steps
    operations = []

    for _ in range(num_steps):
        # Transverse-field term:
        # exp(+i dt * field * X)
        for qubit in range(system_size):
            operations.append(
                PauliRotation(
                    gate_name="RX",
                    pauli=pauli_on_qubits(
                        system_size,
                        "X",
                        [qubit],
                    ),
                    theta=-2.0 * dt * field,
                )
            )

        # Nearest-neighbor ZZ interaction:
        # RZZ(theta) = CX(control, target) RZ(target, theta) CX(control, target)
        for qubit in range(system_size):
            control = qubit
            target = (qubit + 1) % system_size
            theta = -2.0 * dt * coupling

            operations.extend(
                [
                    TwoQubitClifford(
                        gate_name="OpType.CX",
                        control_qubit=control,
                        target_qubit=target,
                    ),
                    PauliRotation(
                        gate_name="RZ",
                        pauli=pauli_on_qubits(
                            system_size,
                            "Z",
                            [target],
                        ),
                        theta=theta,
                    ),
                    TwoQubitClifford(
                        gate_name="OpType.CX",
                        control_qubit=control,
                        target_qubit=target,
                    ),
                ]
            )

    return operations


if __name__ == "__main__":
    system_size = 40
    operations = tfi_trotter_circuit(
        system_size,
        total_time=1.0,
        num_steps=2,
        coupling=1.0,
        field=0.7,
        )
    circuit_ir = spd.CircuitIR(system_size=system_size, operations=tuple(operations))

    # system_size = 4
    # operations = tfi_trotter_circuit_cx(
    #     system_size,
    #     total_time=0.4,
    #     num_steps=2,
    #     coupling=1.0,
    #     field=0.7,
    # )
    backend = spd.BackendAdapter.from_name("numpy", packbit=32, precision="double")
    observable = spd.create_spo(
        # {pauli_on_qubits(system_size, "X", [system_size // 2]): 1.0},
        {pauli_on_qubits(system_size, "Z", [system_size // 2, system_size // 2 + 1]): 1.0},
        backend=backend,
    )
    final_spo, _ = spd.evolve(
        observable,
        circuit_ir,
        trunc_val=1e-12,
        max_num_str=1_000_000,
        backend=backend,
    )
    exp_val = final_spo.get_expectation_value(basis="0")
    print("final expectation value:", float(exp_val))

    initial_spgo = spd.init_gradient_spo(final_spo, basis="0", backend=backend)
    _, _, noise_grads, _ = spd.backpropagate_noise_analysis(
        initial_spgo,
        circuit_ir,
        trunc_val=1e-12,
        max_num_str=1_000_000,
        backend=backend,
    )

    for noise_name, susceptibilities in noise_grads.items():
        print(noise_name)
        for index, susceptibility in enumerate(susceptibilities):
            if susceptibility != 0:
                print(index, float(susceptibility))
        print("total:", float(sum(susceptibilities)))
