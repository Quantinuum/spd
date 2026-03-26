"""Thin execution-facing wrapper around a backend module.

The adapter keeps `run_circuit.py` from depending directly on backend
selection, packbit setup, Clifford dispatch tables, and rotation packing
details.

Boundary:
- backend modules still own kernels and factories
- `SPO` / `SPGO` still own intrinsic object behavior
- the adapter translates IR operations into backend calls
"""

from dataclasses import dataclass
from typing import Protocol

from .circuit_ir import PauliRotation, SingleQubitClifford, SkippedOperation, TwoQubitClifford


class BackendModule(Protocol):
    utils: object

    def create_measurement_op(self, measurement_dict, padded_system_size): ...
    def create_op(self, pauli_dict): ...
    def create_gradient_spo(self, spo, basis="0"): ...
    def get_norm_square(self, obj): ...
    def get_size(self, obj): ...
    def get_expectation_value(self, spo, basis="0"): ...
    def conjugated_pauli_forward(self, spo, xzk, theta, trunc_val): ...
    def conjugated_pauli_backward(self, spgo, xzk, theta, trunc_val): ...
    def conjugated_pauli_batched_uint32_H(self, spo, qubit): ...
    def conjugated_pauli_batched_uint32_S(self, spo, qubit): ...
    def conjugated_pauli_batched_uint32_Sdg(self, spo, qubit): ...
    def conjugated_pauli_batched_uint32_CX(self, spo, control_qubit, target_qubit): ...
    def conjugated_pauli_batched_uint32_CY(self, spo, control_qubit, target_qubit): ...
    def conjugated_pauli_batched_uint32_CZ(self, spo, control_qubit, target_qubit): ...
    def conjugated_pauli_batched_uint32_X(self, spo, qubit): ...
    def conjugated_pauli_batched_uint32_Y(self, spo, qubit): ...
    def conjugated_pauli_batched_uint32_Z(self, spo, qubit): ...


@dataclass
class BackendAdapter:
    name: str
    module: BackendModule
    packbit: int

    def __post_init__(self):
        self.module.utils.set_packbit(self.packbit)
        self.utils = self.module.utils
        self._clifford_dispatch = {
            "OpType.H": self.module.conjugated_pauli_batched_uint32_H,
            "OpType.S": self.module.conjugated_pauli_batched_uint32_S,
            "OpType.Sdg": self.module.conjugated_pauli_batched_uint32_Sdg,
            "OpType.CX": self.module.conjugated_pauli_batched_uint32_CX,
            "OpType.CY": self.module.conjugated_pauli_batched_uint32_CY,
            "OpType.CZ": self.module.conjugated_pauli_batched_uint32_CZ,
            "OpType.X": self.module.conjugated_pauli_batched_uint32_X,
            "OpType.Y": self.module.conjugated_pauli_batched_uint32_Y,
            "OpType.Z": self.module.conjugated_pauli_batched_uint32_Z,
        }

    @classmethod
    def from_name(cls, backend_name, packbit=32):
        if backend_name == "numpy":
            from . import numpy_backend as backend_module
        elif backend_name == "jax":
            from . import jax_backend as backend_module
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")

        return cls(name=backend_name, module=backend_module, packbit=packbit)

    def create_initial_spo(self, measure_qubits_data, padded_system_size):
        if isinstance(measure_qubits_data, dict):
            key = next(iter(measure_qubits_data))
            if isinstance(key, tuple):
                return self.module.create_measurement_op(measure_qubits_data, padded_system_size)
            if isinstance(key, str):
                return self.module.create_op(measure_qubits_data)
            raise ValueError("measure_qubits_data dict key must be tuple or str")

        if isinstance(measure_qubits_data, list):
            measurement_dict = {tuple(measure_qubits_data): 1.0}
            return self.module.create_measurement_op(measurement_dict, padded_system_size)

        raise ValueError("measure_qubits_data must be a dict or list")

    def create_gradient_spo(self, spo, basis="0"):
        return self.module.create_gradient_spo(spo, basis=basis)

    def apply_forward(self, spo, operation, trunc_val):
        if isinstance(operation, PauliRotation):
            xzk = self.utils.pauli_str_to_uint(operation.pauli)
            return self.module.conjugated_pauli_forward(spo, xzk, operation.theta, trunc_val)

        if isinstance(operation, SingleQubitClifford):
            return self._clifford_dispatch[operation.gate_name](spo, operation.qubit), None

        if isinstance(operation, TwoQubitClifford):
            return (
                self._clifford_dispatch[operation.gate_name](
                    spo,
                    operation.control_qubit,
                    operation.target_qubit,
                ),
                None,
            )

        if isinstance(operation, SkippedOperation):
            return spo, None

        raise ValueError(f"Unsupported operation: {operation}")

    def apply_backward(self, spgo, operation, trunc_val):
        if isinstance(operation, PauliRotation):
            xzk = self.utils.pauli_str_to_uint(operation.pauli)
            return self.module.conjugated_pauli_backward(spgo, xzk, operation.theta, trunc_val)

        if isinstance(operation, SkippedOperation):
            return spgo, None, None

        raise ValueError(f"Unsupported operation in backward pass: {operation}")
