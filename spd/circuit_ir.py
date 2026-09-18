"""Backend-agnostic execution IR for supported static circuit operations.

This module intentionally models only the subset of operations that the current
SPD executor can evaluate cleanly. It is not meant to be a universal circuit
IR; it is the lowered operation layer consumed by the backend adapter.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional, TextIO, Union


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
GATE_TYPES = (PauliRotation, SingleQubitClifford, TwoQubitClifford)


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

    def statistics(self) -> Dict[str, Any]:
        """Return gate, channel, and depth statistics for this circuit.

        Gate arity is determined from the qubits on which an operation acts, so
        a :class:`PauliRotation` can contribute to any of the arity counts. The
        two-qubit depth is the longest dependency-path depth when exactly
        two-qubit gates have cost one and all other operations have cost zero.
        """
        gate_operations = [
            operation for operation in self.operations
            if isinstance(operation, GATE_TYPES)
        ]
        arities = Counter(len(get_operation_qubits(operation)) for operation in gate_operations)
        operation_types = Counter(type(operation).__name__ for operation in self.operations)
        gate_names = Counter(operation.gate_name for operation in gate_operations)

        qubit_depths = [0] * self.system_size
        for operation in self.operations:
            qubits = get_operation_qubits(operation)
            if not qubits:
                continue
            input_depth = max(qubit_depths[qubit] for qubit in qubits)
            output_depth = input_depth + int(
                isinstance(operation, GATE_TYPES) and len(qubits) == 2
            )
            for qubit in qubits:
                qubit_depths[qubit] = output_depth

        return {
            "system_size": self.system_size,
            "total_operations": len(self.operations),
            "gate_operations": len(gate_operations),
            "single_qubit_gates": arities[1],
            "two_qubit_gates": arities[2],
            "multi_qubit_gates": sum(count for arity, count in arities.items() if arity > 2),
            "zero_qubit_gates": arities[0],
            "two_qubit_depth": max(qubit_depths, default=0),
            "create_zero": operation_types["CreateZero"],
            "reset_zero": operation_types["ResetZero"],
            "discard": operation_types["Discard"],
            "skipped_operations": operation_types["SkippedOperation"],
            "operation_types": dict(sorted(operation_types.items())),
            "gate_names": dict(sorted(gate_names.items())),
        }

    def draw(self, output: str = "text", filename: Optional[str] = None,
             fold: int = 40) -> str:
        """Render the circuit as dependency-free text or SVG.

        ``fold`` limits the dependency layers per panel, keeping moderate and
        large circuits readable without creating an unbounded canvas.
        """
        from .circuit_visualization import draw_circuit
        return draw_circuit(self, output=output, filename=filename, fold=fold)

    def print_statistics(self, file: Optional[TextIO] = None) -> None:
        """Print a human-readable summary of :meth:`statistics`."""
        statistics = self.statistics()
        labels = (
            ("System size", "system_size"),
            ("Total operations", "total_operations"),
            ("Gate operations", "gate_operations"),
            ("Single-qubit gates", "single_qubit_gates"),
            ("Two-qubit gates", "two_qubit_gates"),
            ("Multi-qubit gates", "multi_qubit_gates"),
            ("Zero-qubit gates", "zero_qubit_gates"),
            ("Two-qubit depth", "two_qubit_depth"),
            ("CreateZero", "create_zero"),
            ("ResetZero", "reset_zero"),
            ("Discard", "discard"),
            ("Skipped operations", "skipped_operations"),
        )
        print("CircuitIR statistics:", file=file)
        for label, key in labels:
            print("  {}: {}".format(label, statistics[key]), file=file)
        print("  Operation types: {}".format(statistics["operation_types"]), file=file)
        print("  Gate names: {}".format(statistics["gate_names"]), file=file)
