"""Backend-agnostic execution IR for supported static circuit operations.

This module intentionally models only the subset of operations that the current
SPD executor can evaluate cleanly. It is not meant to be a universal circuit
IR; it is the lowered operation layer consumed by the backend adapter.
"""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class PauliRotation:
    gate_name: str
    pauli: str
    # SPD uses exp(-i * theta * P / 2) as the rotation convention.
    # Frontends should convert their native parameterization into this theta.
    theta: float


@dataclass(frozen=True)
class SingleQubitClifford:
    gate_name: str
    qubit: int


@dataclass(frozen=True)
class TwoQubitClifford:
    gate_name: str
    control_qubit: int
    target_qubit: int


@dataclass(frozen=True)
class SkippedOperation:
    gate_name: str


CircuitOperation = Union[
    PauliRotation,
    SingleQubitClifford,
    TwoQubitClifford,
    SkippedOperation,
]


def get_operation_qubits(operation: CircuitOperation) -> tuple[int, ...]:
    """Return the physical qubits acted on by an IR operation."""
    if isinstance(operation, PauliRotation):
        return tuple(index for index, pauli in enumerate(operation.pauli) if pauli != "I")
    if isinstance(operation, SingleQubitClifford):
        return (operation.qubit,)
    if isinstance(operation, TwoQubitClifford):
        return (operation.control_qubit, operation.target_qubit)
    if isinstance(operation, SkippedOperation):
        return ()
    raise TypeError(f"Unsupported circuit operation: {type(operation)!r}")


@dataclass(frozen=True)
class CircuitIR:
    """A lowered circuit together with its physical system size."""

    system_size: int
    operations: tuple[CircuitOperation, ...]

    def __post_init__(self):
        if not isinstance(self.system_size, int) or isinstance(self.system_size, bool):
            raise TypeError("system_size must be an integer.")
        if self.system_size < 1:
            raise ValueError("system_size must be at least 1.")

        operations = tuple(self.operations)
        object.__setattr__(self, "operations", operations)
        for index, operation in enumerate(operations):
            if not isinstance(
                operation,
                (PauliRotation, SingleQubitClifford, TwoQubitClifford, SkippedOperation),
            ):
                raise TypeError(
                    "operations must contain CircuitOperation instances; "
                    f"got {type(operation)!r} at index {index}."
                )
            invalid_qubits = [
                qubit
                for qubit in get_operation_qubits(operation)
                if qubit < 0 or qubit >= self.system_size
            ]
            if invalid_qubits:
                raise ValueError(
                    f"Operation at index {index} acts outside system_size={self.system_size}: "
                    f"{invalid_qubits}."
                )
