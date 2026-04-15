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
    SparsePauliOp: type
    SparsePauliGradientOp: type

    def set_precision(self, precision: str): ...
    def create_measurement_op(self, measurement_dict, padded_system_size): ...
    def create_op(self, pauli_dict): ...
    def init_gradient_spo(
        self,
        spo,
        *,
        loss_type="basis_expectation",
        basis="0",
        target_spo=None,
        lambda_ose=0.0,
        alpha=1.0,
    ): ...
    def conjugated_pauli_forward(self, spo, xzk, theta, trunc_val, max_num_str): ...
    def conjugated_pauli_backward(self, spgo, xzk, theta, trunc_val, max_num_str): ...
    def conjugated_pauli_batched_uint32_H(self, spo, qubit): ...
    def conjugated_pauli_batched_uint32_S(self, spo, qubit): ...
    def conjugated_pauli_batched_uint32_Sdg(self, spo, qubit): ...
    def conjugated_pauli_batched_uint32_CX(self, spo, control_qubit, target_qubit): ...
    def conjugated_pauli_batched_uint32_CY(self, spo, control_qubit, target_qubit): ...
    def conjugated_pauli_batched_uint32_CZ(self, spo, control_qubit, target_qubit): ...
    def conjugated_pauli_batched_uint32_X(self, spo, qubit): ...
    def conjugated_pauli_batched_uint32_Y(self, spo, qubit): ...
    def conjugated_pauli_batched_uint32_Z(self, spo, qubit): ...
    def conjugated_pauli_batched_uint32_X_backward(self, spgo, qubit): ...


@dataclass
class BackendAdapter:
    name: str
    module: BackendModule
    packbit: int
    precision: str = "single"

    def __post_init__(self):
        self.module.utils.set_packbit(self.packbit)
        self.module.set_precision(self.precision)
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
        self._clifford_backward_dispatch = {
            "OpType.X": self.module.conjugated_pauli_batched_uint32_X_backward,
        }

    @classmethod
    def from_name(cls, backend_name, packbit=32, precision="single"):
        if backend_name == "numpy":
            from . import numpy_backend as backend_module
        elif backend_name == "jax":
            from . import jax_backend as backend_module
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")

        return cls(name=backend_name, module=backend_module, packbit=packbit, precision=precision)

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

    def init_gradient_spo(
        self,
        spo,
        *,
        loss_type="basis_expectation",
        basis="0",
        target_spo=None,
        lambda_ose=0.0,
        alpha=1.0,
    ):
        return self.module.init_gradient_spo(
            spo,
            loss_type=loss_type,
            basis=basis,
            target_spo=target_spo,
            lambda_ose=lambda_ose,
            alpha=alpha,
        )

    def apply_forward(self, spo, operation, trunc_val, max_num_str):
        if isinstance(operation, PauliRotation):
            xzk = self.utils.pauli_str_to_uint(operation.pauli)
            return self.module.conjugated_pauli_forward(
                spo, xzk, operation.theta, trunc_val, max_num_str=max_num_str
            )

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

    def apply_backward(self, spgo, operation, trunc_val, max_num_str):
        if isinstance(operation, PauliRotation):
            xzk = self.utils.pauli_str_to_uint(operation.pauli)
            return self.module.conjugated_pauli_backward(
                spgo, xzk, operation.theta, trunc_val, max_num_str=max_num_str
            )

        if isinstance(operation, SingleQubitClifford):
            # add Warning this is a temporary hack until we implement proper SPGO support for cliffords
            import warnings
            warnings.warn(
                "Applying a single-qubit Clifford in the backward pass is not fully supported. "
                "The gradient will be incorrect if the Clifford changes the Pauli type of any rotation generator."
            )
            if operation.gate_name not in self._clifford_backward_dispatch:
                raise NotImplementedError(
                    f"Backward support for {operation.gate_name} is not implemented."
                )
            # Keep num_string as None so non-parameterized Clifford steps are not
            # appended to the parameter-gradient list in run_circuit.py.
            return (
                self._clifford_backward_dispatch[operation.gate_name](
                    spgo, operation.qubit
                ),
                None,
                None,
            )

        if isinstance(operation, SkippedOperation):
            return spgo, None, None

        raise ValueError(f"Unsupported operation in backward pass: {operation}")

    def is_spo_instance(self, obj) -> bool:
        return isinstance(obj, self.module.SparsePauliOp)

    def is_spgo_instance(self, obj) -> bool:
        return isinstance(obj, self.module.SparsePauliGradientOp)
