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
