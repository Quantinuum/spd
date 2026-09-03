"""A pytket circuit together with its variational-parameter metadata."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class VariationalCircuit:
    """Map pytket rotation-gate gradients back to optimizer parameters.

    The metadata follows the parameterized rotation commands in
    ``circuit.get_commands()`` order. Repeated indices denote shared
    parameters. Use index ``-1`` for fixed rotations.
    """

    circuit: object
    gate_parameter_indices: np.ndarray
    parameter_shape: tuple
    gate_parameter_factors: np.ndarray = None

    def __post_init__(self):
        from pytket.circuit import Circuit

        from .pytket_frontend import _ROTATION_DISPATCH

        if not isinstance(self.circuit, Circuit):
            raise TypeError("circuit must be a pytket Circuit.")

        indices = np.asarray(self.gate_parameter_indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError("gate_parameter_indices must be one-dimensional.")

        shape = tuple(int(size) for size in self.parameter_shape)
        if not shape or any(size < 1 for size in shape):
            raise ValueError("parameter_shape must contain positive dimensions.")

        if self.gate_parameter_factors is None:
            factors = np.ones(indices.size, dtype=np.float64)
        else:
            factors = np.asarray(self.gate_parameter_factors, dtype=np.float64)
        if factors.shape != indices.shape:
            raise ValueError("gate_parameter_factors must match gate_parameter_indices.")

        num_parameters = int(np.prod(shape))
        if np.any(indices < -1) or np.any(indices >= num_parameters):
            raise ValueError(
                "gate parameter indices must be -1 or valid indices into parameter_shape."
            )

        num_rotation_gates = sum(
            command.op.type in _ROTATION_DISPATCH
            for command in self.circuit.get_commands()
        )
        if indices.size != num_rotation_gates:
            raise ValueError(
                f"Expected metadata for {num_rotation_gates} rotation gates, "
                f"got {indices.size}."
            )

        self.gate_parameter_indices = indices
        self.gate_parameter_factors = factors
        self.parameter_shape = shape

    def parameter_gradients(self, gate_gradients):
        """Sum SPD rotation-angle gradients into the optimizer parameter shape."""
        gate_gradients = np.asarray(gate_gradients)
        if gate_gradients.shape != self.gate_parameter_indices.shape:
            raise ValueError(
                f"Expected {self.gate_parameter_indices.size} gate gradients, "
                f"got shape {gate_gradients.shape}."
            )

        result = np.zeros(
            int(np.prod(self.parameter_shape)),
            dtype=np.result_type(gate_gradients.dtype, np.float64),
        )
        active = self.gate_parameter_indices >= 0
        np.add.at(
            result,
            self.gate_parameter_indices[active],
            math.pi * self.gate_parameter_factors[active] * gate_gradients[active],
        )
        return result.reshape(self.parameter_shape)
