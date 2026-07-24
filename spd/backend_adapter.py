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


def _zero_step_info():
    return {
        "num_str_truncated": 0,
        "truncated_l1_norm": 0.0,
        "truncated_l2_norm": 0.0,
    }


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
    def get_two_qubit_depolarizing_susceptibility(self, spgo, qubits): ...
    def conjugate_pauli_rot_forward(self, spo, xzk, theta, trunc_val, max_num_str): ...
    def conjugate_pauli_rot_backward(self, spgo, xzk, theta, trunc_val, max_num_str): ...
    def conjugate_H_forward(self, spo, qubit): ...
    def conjugate_S_forward(self, spo, qubit): ...
    def conjugate_Sdg_forward(self, spo, qubit): ...
    def conjugate_CX_forward(self, spo, control_qubit, target_qubit): ...
    def conjugate_CY_forward(self, spo, control_qubit, target_qubit): ...
    def conjugate_CZ_forward(self, spo, control_qubit, target_qubit): ...
    def conjugate_X_forward(self, spo, qubit): ...
    def conjugate_Y_forward(self, spo, qubit): ...
    def conjugate_Z_forward(self, spo, qubit): ...
    def conjugate_H_backward(self, spgo, qubit): ...
    def conjugate_S_backward(self, spgo, qubit): ...
    def conjugate_Sdg_backward(self, spgo, qubit): ...
    def conjugate_CX_backward(self, spgo, control_qubit, target_qubit): ...
    def conjugate_CY_backward(self, spgo, control_qubit, target_qubit): ...
    def conjugate_CZ_backward(self, spgo, control_qubit, target_qubit): ...
    def conjugate_X_backward(self, spgo, qubit): ...
    def conjugate_Y_backward(self, spgo, qubit): ...
    def conjugate_Z_backward(self, spgo, qubit): ...


@dataclass
class BackendAdapter:
    name: str
    module: BackendModule
    packbit: int
    precision: str = "single"

    def __post_init__(self):
        if self.packbit != 32:
            raise ValueError(
                f"BackendAdapter currently requires packbit=32, got {self.packbit}."
            )
        self.module.utils.set_packbit(self.packbit)
        self.module.set_precision(self.precision)
        self.utils = self.module.utils
        self._clifford_forward_dispatch = {
            "OpType.H": self.module.conjugate_H_forward,
            "OpType.S": self.module.conjugate_S_forward,
            "OpType.Sdg": self.module.conjugate_Sdg_forward,
            "OpType.CX": self.module.conjugate_CX_forward,
            "OpType.CY": self.module.conjugate_CY_forward,
            "OpType.CZ": self.module.conjugate_CZ_forward,
            "OpType.X": self.module.conjugate_X_forward,
            "OpType.Y": self.module.conjugate_Y_forward,
            "OpType.Z": self.module.conjugate_Z_forward,
        }
        self._clifford_backward_dispatch = {
            "OpType.H": self.module.conjugate_H_backward,
            "OpType.S": self.module.conjugate_S_backward,
            "OpType.Sdg": self.module.conjugate_Sdg_backward,
            "OpType.CX": self.module.conjugate_CX_backward,
            "OpType.CY": self.module.conjugate_CY_backward,
            "OpType.CZ": self.module.conjugate_CZ_backward,
            "OpType.X": self.module.conjugate_X_backward,
            "OpType.Y": self.module.conjugate_Y_backward,
            "OpType.Z": self.module.conjugate_Z_backward,
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

    def create_initial_spo(self, measure_qubits_data, padded_system_size=None):
        if isinstance(measure_qubits_data, dict):
            key = next(iter(measure_qubits_data))
            if isinstance(key, str):
                return self.module.create_op(measure_qubits_data)
            if isinstance(key, tuple):
                raise ValueError(
                    "Tuple-key measurement dicts are no longer supported. "
                    "Use a list of qubits for Z-basis measurements or a string-key dict for general Paulis."
                )
            raise ValueError("measure_qubits_data dict key must be str")

        if isinstance(measure_qubits_data, list):
            if padded_system_size is None:
                raise ValueError(
                    "padded_system_size is required when measure_qubits_data is a list of qubits."
                )
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

    def get_two_qubit_depolarizing_susceptibility(self, spgo, qubits):
        return self.module.get_two_qubit_depolarizing_susceptibility(spgo, qubits)

    def apply_forward(self, spo, operation, trunc_val, max_num_str):
        if isinstance(operation, PauliRotation):
            xzk = self.utils.pauli_str_to_uint(operation.pauli)
            next_state, num_string, step_info = self.module.conjugate_pauli_rot_forward(
                spo, xzk, operation.theta, trunc_val, max_num_str=max_num_str
            )
            return next_state, num_string, None, step_info

        if isinstance(operation, SingleQubitClifford):
            next_state = self._clifford_forward_dispatch[operation.gate_name](spo, operation.qubit)
            return next_state, next_state.get_size(), None, _zero_step_info()

        if isinstance(operation, TwoQubitClifford):
            next_state = self._clifford_forward_dispatch[operation.gate_name](
                spo,
                operation.control_qubit,
                operation.target_qubit,
            )
            return next_state, next_state.get_size(), None, _zero_step_info()

        if isinstance(operation, SkippedOperation):
            return spo, None, None, None

        raise ValueError(f"Unsupported operation: {operation}")

    def apply_backward(self, spgo, operation, trunc_val, max_num_str):
        if isinstance(operation, PauliRotation):
            xzk = self.utils.pauli_str_to_uint(operation.pauli)
            return self.module.conjugate_pauli_rot_backward(
                spgo, xzk, operation.theta, trunc_val, max_num_str=max_num_str
            )

        if isinstance(operation, SingleQubitClifford):
            if operation.gate_name not in self._clifford_backward_dispatch:
                raise NotImplementedError(
                    f"Backward support for {operation.gate_name} is not implemented."
                )
            # Keep grad_i as None because Clifford gates have no parameter gradient.
            next_state = self._clifford_backward_dispatch[operation.gate_name](spgo, operation.qubit)
            return next_state, next_state.get_size(), None, _zero_step_info()

        if isinstance(operation, TwoQubitClifford):
            if operation.gate_name not in self._clifford_backward_dispatch:
                raise NotImplementedError(
                    f"Backward support for {operation.gate_name} is not implemented."
                )
            # Keep grad_i as None because Clifford gates have no parameter gradient.
            next_state = self._clifford_backward_dispatch[operation.gate_name](
                spgo,
                operation.control_qubit,
                operation.target_qubit,
            )
            return next_state, next_state.get_size(), None, _zero_step_info()

        if isinstance(operation, SkippedOperation):
            return spgo, None, None, None

        raise ValueError(f"Unsupported operation in backward pass: {operation}")

    def is_spo_instance(self, obj) -> bool:
        return isinstance(obj, self.module.SparsePauliOp)

    def is_spgo_instance(self, obj) -> bool:
        return isinstance(obj, self.module.SparsePauliGradientOp)
