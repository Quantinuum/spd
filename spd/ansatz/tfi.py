"""Translationally invariant transverse-field Ising ansatz circuits."""

import numpy as np
from pytket import Circuit

from ..variational_circuit import VariationalCircuit


def _parameters(params):
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 1 or params.size == 0 or params.size % 2 != 0:
        raise ValueError(
            "params must be a nonempty flat array of alternating TFI layer angles."
        )
    return params


def _finish(circuit, parameter_indices, parameter_shape):
    for qubit in range(circuit.n_qubits):
        circuit.Measure(qubit, qubit)
    return VariationalCircuit(circuit, parameter_indices, parameter_shape)


def _periodic_bond_layers(length):
    """Return disjoint nearest-neighbor bond starts for a periodic axis."""
    if length < 2 or length % 2 != 0:
        raise ValueError("Periodic TFI dimensions must be even and at least 2.")
    return (range(0, length, 2), range(1, length, 2))


def tfi_1d_hva(params, system_size=12, basis="+"):
    """Build a periodic 1D TFI HVA with brickwork interaction layers."""
    params = _parameters(params)
    if basis not in ("+", "0"):
        raise ValueError("basis must be '+' or '0'.")

    circuit = Circuit(system_size, system_size)
    parameter_indices = []

    def add_zz_layer(parameter_index):
        for bond_starts in _periodic_bond_layers(system_size):
            for qubit in bond_starts:
                circuit.ZZPhase(
                    params[parameter_index], qubit, (qubit + 1) % system_size
                )
                parameter_indices.append(parameter_index)
            circuit.add_barrier(list(range(system_size)))

    def add_x_layer(parameter_index):
        for qubit in range(system_size):
            circuit.Rx(params[parameter_index], qubit)
            parameter_indices.append(parameter_index)
        circuit.add_barrier(list(range(system_size)))

    for layer in range(params.size // 2):
        first = 2 * layer
        if basis == "+":
            add_zz_layer(first)
            add_x_layer(first + 1)
        else:
            add_x_layer(first)
            add_zz_layer(first + 1)

    return _finish(circuit, parameter_indices, params.shape)


def tfi_2d_hva(params, system_size_x=4, system_size_y=4):
    """Build a periodic 2D TFI HVA with brickwork interaction layers."""
    params = _parameters(params)
    system_size = system_size_x * system_size_y
    circuit = Circuit(system_size, system_size)
    parameter_indices = []

    for layer in range(params.size // 2):
        zz_index = 2 * layer
        x_index = zz_index + 1
        for x_starts in _periodic_bond_layers(system_size_x):
            for x in x_starts:
                for y in range(system_size_y):
                    qubit = x * system_size_y + y
                    x_neighbor = ((x + 1) % system_size_x) * system_size_y + y
                    circuit.ZZPhase(params[zz_index], qubit, x_neighbor)
                    parameter_indices.append(zz_index)
            circuit.add_barrier(list(range(system_size)))

        for y_starts in _periodic_bond_layers(system_size_y):
            for x in range(system_size_x):
                for y in y_starts:
                    qubit = x * system_size_y + y
                    y_neighbor = x * system_size_y + (y + 1) % system_size_y
                    circuit.ZZPhase(params[zz_index], qubit, y_neighbor)
                    parameter_indices.append(zz_index)
            circuit.add_barrier(list(range(system_size)))

        for qubit in range(system_size):
            circuit.Rx(params[x_index], qubit)
            parameter_indices.append(x_index)
        circuit.add_barrier(list(range(system_size)))

    return _finish(circuit, parameter_indices, params.shape)


def tfi_3d_hva(params, system_size_x=4, system_size_y=4, system_size_z=4):
    """Build a periodic 3D TFI HVA with brickwork interaction layers."""
    params = _parameters(params)
    plane_size = system_size_y * system_size_z
    system_size = system_size_x * plane_size
    circuit = Circuit(system_size, system_size)
    parameter_indices = []

    for layer in range(params.size // 2):
        zz_index = 2 * layer
        x_index = zz_index + 1
        for x_starts in _periodic_bond_layers(system_size_x):
            for x in x_starts:
                for y in range(system_size_y):
                    for z in range(system_size_z):
                        qubit = x * plane_size + y * system_size_z + z
                        x_neighbor = (
                            ((x + 1) % system_size_x) * plane_size
                            + y * system_size_z
                            + z
                        )
                        circuit.ZZPhase(params[zz_index], qubit, x_neighbor)
                        parameter_indices.append(zz_index)
            circuit.add_barrier(list(range(system_size)))

        for y_starts in _periodic_bond_layers(system_size_y):
            for x in range(system_size_x):
                for y in y_starts:
                    for z in range(system_size_z):
                        qubit = x * plane_size + y * system_size_z + z
                        y_neighbor = (
                            x * plane_size
                            + ((y + 1) % system_size_y) * system_size_z
                            + z
                        )
                        circuit.ZZPhase(params[zz_index], qubit, y_neighbor)
                        parameter_indices.append(zz_index)
            circuit.add_barrier(list(range(system_size)))

        for z_starts in _periodic_bond_layers(system_size_z):
            for x in range(system_size_x):
                for y in range(system_size_y):
                    for z in z_starts:
                        qubit = x * plane_size + y * system_size_z + z
                        z_neighbor = (
                            x * plane_size
                            + y * system_size_z
                            + (z + 1) % system_size_z
                        )
                        circuit.ZZPhase(params[zz_index], qubit, z_neighbor)
                        parameter_indices.append(zz_index)
            circuit.add_barrier(list(range(system_size)))

        for qubit in range(system_size):
            circuit.Rx(params[x_index], qubit)
            parameter_indices.append(x_index)
        circuit.add_barrier(list(range(system_size)))

    return _finish(circuit, parameter_indices, params.shape)
