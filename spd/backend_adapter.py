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

from .circuit_ir import (PauliRotation, SingleQubitClifford, SkippedOperation, TwoQubitClifford,
                         CreateZero, ResetZero, Discard)


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
    def create_op(self, pauli_dict, *, num_qubits=None): ...
    # Optional channel capability: all five entry points must be implemented.
    def reindex_spo(self, spo, num_qubits, columns): ...
    def contract_zero_forward(self, spo, num_qubits, columns, remove_columns): ...
    def contract_zero_backward(self, spgo, checkpoint, num_qubits, columns, remove_columns): ...
    def insert_identity_forward(self, spo, num_qubits, column): ...
    def insert_identity_backward(self, spgo, num_qubits, column): ...
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
    def get_one_qubit_depolarizing_susceptibility(self, spgo, qubit): ...
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
        elif backend_name == "triton":
            from . import triton_backend as backend_module
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")

        return cls(name=backend_name, module=backend_module, packbit=packbit, precision=precision)

    def create_initial_spo(self, measure_qubits_data, padded_system_size=None):
        if isinstance(measure_qubits_data, dict):
            if not measure_qubits_data:
                if padded_system_size is None:
                    raise ValueError("system_size is required for an empty observable.")
                return self.module.create_op({}, num_qubits=padded_system_size)
            key = next(iter(measure_qubits_data))
            if isinstance(key, str):
                if self.name == "triton":
                    width = (padded_system_size if padded_system_size is not None else
                             self.packbit * ((max(map(len, measure_qubits_data)) + self.packbit - 1) // self.packbit))
                    return self.module.create_op(measure_qubits_data, num_qubits=width, precision=self.precision)
                return self.module.create_op(measure_qubits_data, num_qubits=padded_system_size)
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
            if self.name == "triton":
                return self.module.create_measurement_op(measurement_dict, padded_system_size, precision=self.precision)
            return self.module.create_measurement_op(measurement_dict, padded_system_size)

        raise ValueError("measure_qubits_data must be a dict or list")

    def require_channel_support(self):
        """Fail before execution when this backend lacks the channel interface."""
        required = ("reindex_spo", "contract_zero_forward", "contract_zero_backward",
                    "insert_identity_forward", "insert_identity_backward")
        if any(not callable(getattr(self.module, name, None)) for name in required):
            raise NotImplementedError(
                f"Static channel execution is not implemented for backend '{self.name}'. "
                "Choose backend_name='numpy' explicitly; execution never falls back to another backend."
            )

    def create_checkpoint_backend(self, state):
        """Capture backend size/transfer helpers once, without retaining state."""
        factory = getattr(self.module, "create_checkpoint_backend", None)
        if factory is not None:
            return factory(state)
        if self.name == "numpy":
            from .checkpoints import CheckpointBackend
            return CheckpointBackend(size=self.module.checkpoint_size)
        raise NotImplementedError(
            f"Backend '{self.name}' must implement create_checkpoint_backend for channel snapshots."
        )

    def reindex_spo(self, spo, num_qubits, columns):
        self.require_channel_support()
        return self.module.reindex_spo(spo, num_qubits, columns)

    def apply_zero_contractions_forward(self, spo, columns, remove_columns):
        self.require_channel_support()
        result = self.module.contract_zero_forward(spo, len(spo.qubit_indices), columns, remove_columns)
        return result, result.get_size(), None, _zero_step_info()

    def apply_zero_contractions_backward(self, spgo, checkpoint, columns, remove_columns):
        self.require_channel_support()
        result = self.module.contract_zero_backward(
            spgo, checkpoint, len(checkpoint.qubit_indices), columns, remove_columns)
        return result, result.get_size(), None, _zero_step_info()

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

    def get_one_qubit_depolarizing_susceptibility(self, spgo, qubit):
        return self.module.get_one_qubit_depolarizing_susceptibility(spgo, qubit)

    def apply_forward(self, spo, operation, trunc_val, max_num_str):
        if isinstance(operation, (CreateZero, ResetZero)):
            remove = [operation.qubit] if isinstance(operation, CreateZero) else []
            return self.apply_zero_contractions_forward(spo, [operation.qubit], remove)
        if isinstance(operation, Discard):
            self.require_channel_support()
            result = self.module.insert_identity_forward(spo, len(spo.qubit_indices), operation.qubit)
            return result, result.get_size(), None, _zero_step_info()
        if isinstance(operation, PauliRotation):
            # Reuse Triton's cached string packing, ignoring frontend identity padding.
            xzk = (operation.pauli.rstrip("I") if self.name == "triton"
                   else self.utils.pauli_str_to_uint(operation.pauli))
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

    def apply_backward(self, spgo, operation, trunc_val, max_num_str, *, checkpoint=None):
        if isinstance(operation, (CreateZero, ResetZero)):
            self.require_channel_support()
            if checkpoint is None:
                raise ValueError("Creation/reset backward requires the pre-contraction checkpoint.")
            remove = [operation.qubit] if isinstance(operation, CreateZero) else []
            return self.apply_zero_contractions_backward(spgo, checkpoint, [operation.qubit], remove)
        if isinstance(operation, Discard):
            self.require_channel_support()
            result = self.module.insert_identity_backward(spgo, len(spgo.qubit_indices), operation.qubit)
            return result, result.get_size(), None, _zero_step_info()
        if isinstance(operation, PauliRotation):
            # Reuse Triton's cached string packing, ignoring frontend identity padding.
            xzk = (operation.pauli.rstrip("I") if self.name == "triton"
                   else self.utils.pauli_str_to_uint(operation.pauli))
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
        return isinstance(obj, self.module.SparsePauliOp) and not self.is_spgo_instance(obj)

    def is_spgo_instance(self, obj) -> bool:
        return isinstance(obj, self.module.SparsePauliGradientOp)
