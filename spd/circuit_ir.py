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


@dataclass(frozen=True)
class CreateZero:
    """Prepare an inactive fixed index in |0>; permitted once per index."""
    qubit: int
    gate_name: str = "CreateZero"


@dataclass(frozen=True)
class ResetZero:
    """Trace out an active index and replace its state by |0>."""
    qubit: int
    gate_name: str = "ResetZero"


@dataclass(frozen=True)
class Discard:
    """Trace out an active index, removing it from the outputs."""
    qubit: int
    gate_name: str = "Discard"


CHANNEL_TYPES = (CreateZero, ResetZero, Discard)


CircuitOperation = Union[
    PauliRotation,
    SingleQubitClifford,
    TwoQubitClifford,
    SkippedOperation,
    CreateZero, ResetZero, Discard,
]


def get_operation_qubits(operation: CircuitOperation) -> tuple[int, ...]:
    """Return the physical qubits acted on by an IR operation."""
    if isinstance(operation, PauliRotation):
        return tuple(index for index, pauli in enumerate(operation.pauli) if pauli != "I")
    if isinstance(operation, (SingleQubitClifford, *CHANNEL_TYPES)):
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
                (PauliRotation, SingleQubitClifford, TwoQubitClifford, SkippedOperation, *CHANNEL_TYPES),
            ):
                raise TypeError(
                    "operations must contain CircuitOperation instances; "
                    f"got {type(operation)!r} at index {index}."
                )
            if isinstance(operation, PauliRotation) and set(operation.pauli) - set("IXYZ"):
                raise ValueError("Rotation Paulis must contain only I/X/Y/Z.")
            if any(not isinstance(q, int) or isinstance(q, bool)
                   for q in get_operation_qubits(operation)):
                raise TypeError("Qubit indices must be integers.")
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

        created = [op.qubit for op in operations if isinstance(op, CreateZero)]
        if len(set(created)) != len(created):
            raise ValueError("At most one CreateZero is allowed per index; recreation is unsupported.")
        active = set(range(self.system_size)) - set(created)
        for index, op in enumerate(operations):
            used = set(get_operation_qubits(op))
            if isinstance(op, CreateZero):
                if op.qubit in active:
                    raise ValueError(f"CreateZero at {index} requires an inactive index.")
                active.add(op.qubit)
            else:
                if not used <= active:
                    raise ValueError(f"Operation at {index} requires active indices: {sorted(used - active)}.")
                if isinstance(op, Discard):
                    active.remove(op.qubit)

    @property
    def initial_active_qubits(self):
        created = {op.qubit for op in self.operations if isinstance(op, CreateZero)}
        return sorted(set(range(self.system_size)) - created)

    @property
    def final_active_qubits(self):
        discarded = {op.qubit for op in self.operations if isinstance(op, Discard)}
        return sorted(set(range(self.system_size)) - discarded)
