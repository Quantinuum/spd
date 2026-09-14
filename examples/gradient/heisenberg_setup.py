"""Compatibility helpers for existing AFH benchmarks."""

import math

import numpy as np

from spd.ansatz import afh_1d_hva, afh_2d_hva, afh_3d_hva


def gen_1d_AFH_ansatz_circuit(thetas, system_size=12):
    return afh_1d_hva(thetas, system_size=system_size).circuit


def gen_2d_AFH_ansatz_circuit(thetas, system_size_x=4, system_size_y=4):
    return afh_2d_hva(
        thetas,
        system_size_x=system_size_x,
        system_size_y=system_size_y,
    ).circuit


def gen_3d_AFH_ansatz_circuit(
    thetas,
    system_size_x=4,
    system_size_y=4,
    system_size_z=4,
):
    return afh_3d_hva(
        thetas,
        system_size_x=system_size_x,
        system_size_y=system_size_y,
        system_size_z=system_size_z,
    ).circuit


def _hamiltonian(dimensions, full):
    system_size = math.prod(dimensions)
    strides = tuple(math.prod(dimensions[axis + 1 :]) for axis in range(len(dimensions)))
    if full:
        bonds = []
        for coords in np.ndindex(dimensions):
            qubit = sum(coord * stride for coord, stride in zip(coords, strides))
            for axis, dimension in enumerate(dimensions):
                neighbor = list(coords)
                neighbor[axis] = (neighbor[axis] + 1) % dimension
                neighbor_qubit = sum(
                    coord * stride for coord, stride in zip(neighbor, strides)
                )
                bonds.append((qubit, neighbor_qubit))
    else:
        bonds = [(0, stride) for stride in strides]

    hamiltonian = {}
    for qubit, neighbor in bonds:
        for pauli in "XYZ":
            paulis = ["I"] * system_size
            paulis[qubit] = pauli
            paulis[neighbor] = pauli
            hamiltonian["".join(paulis)] = 1.0
    return hamiltonian


def gen_1d_Hamiltonian_dict(system_size, full=True):
    return _hamiltonian((system_size,), full)


def gen_2d_Hamiltonian_dict(system_size_x, system_size_y, full=True):
    return _hamiltonian((system_size_x, system_size_y), full)


def gen_3d_Hamiltonian_dict(
    system_size_x,
    system_size_y,
    system_size_z,
    full=True,
):
    return _hamiltonian((system_size_x, system_size_y, system_size_z), full)
