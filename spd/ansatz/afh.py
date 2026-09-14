"""Antiferromagnetic Heisenberg variational circuits."""

import math

import numpy as np
from pytket import Circuit

from ..variational_circuit import VariationalCircuit


def _parameters(params):
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 1 or params.size == 0 or params.size % 4 != 0:
        raise ValueError(
            "params must be a nonempty flat array of XX, YY, ZZ, and staggered Rz angles."
        )
    return params


def _dimensions(*dimensions):
    dimensions = tuple(int(size) for size in dimensions)
    if any(size < 2 or size % 2 for size in dimensions):
        raise ValueError("Periodic AFH dimensions must be even and at least 2.")
    return dimensions


def _afh_hva(params, dimensions):
    params = _parameters(params)
    dimensions = _dimensions(*dimensions)
    system_size = math.prod(dimensions)
    strides = tuple(math.prod(dimensions[axis + 1 :]) for axis in range(len(dimensions)))
    circuit = Circuit(system_size, system_size)
    parameter_indices = []
    parameter_factors = []

    def qubit_index(coords):
        return sum(coord * stride for coord, stride in zip(coords, strides))

    for coords in np.ndindex(dimensions):
        if sum(coords) % 2:
            circuit.X(qubit_index(coords))

    for layer in range(params.size // 4):
        for offset, gate_name in enumerate(("XXPhase", "YYPhase", "ZZPhase")):
            parameter_index = 4 * layer + offset
            for axis, dimension in enumerate(dimensions):
                for parity in (0, 1):
                    for coords in np.ndindex(dimensions):
                        if coords[axis] % 2 != parity:
                            continue
                        neighbor = list(coords)
                        neighbor[axis] = (neighbor[axis] + 1) % dimension
                        getattr(circuit, gate_name)(
                            params[parameter_index],
                            qubit_index(coords),
                            qubit_index(neighbor),
                        )
                        parameter_indices.append(parameter_index)
                        parameter_factors.append(1.0)
                    circuit.add_barrier(list(range(system_size)))

        parameter_index = 4 * layer + 3
        for coords in np.ndindex(dimensions):
            factor = 1.0 if sum(coords) % 2 == 0 else -1.0
            circuit.Rz(factor * params[parameter_index], qubit_index(coords))
            parameter_indices.append(parameter_index)
            parameter_factors.append(factor)
        circuit.add_barrier(list(range(system_size)))

    circuit.measure_all()
    return VariationalCircuit(
        circuit,
        parameter_indices,
        params.shape,
        gate_parameter_factors=parameter_factors,
    )


def afh_1d_hva(params, system_size=12):
    """Build a periodic 1D AFH HVA initialized in the Neel state."""
    return _afh_hva(params, (system_size,))


def afh_2d_hva(params, system_size_x=4, system_size_y=4):
    """Build a periodic 2D AFH HVA initialized in the Neel state."""
    return _afh_hva(params, (system_size_x, system_size_y))


def afh_3d_hva(params, system_size_x=4, system_size_y=4, system_size_z=4):
    """Build a periodic 3D AFH HVA initialized in the Neel state."""
    return _afh_hva(params, (system_size_x, system_size_y, system_size_z))
