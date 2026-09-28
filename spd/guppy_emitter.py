"""Render SPD circuits as standalone Guppy v1 source modules."""

from __future__ import annotations

import keyword
import math
from pathlib import Path

from .circuit_ir import (
    CircuitIR,
    CreateZero,
    Discard,
    PauliRotation,
    ResetZero,
    SingleQubitClifford,
    SkippedOperation,
    TwoQubitClifford,
)


_SINGLE_QUBIT_GATES = {"H": "h", "S": "s", "Sdg": "sdg", "X": "x", "Y": "y", "Z": "z"}
_TWO_QUBIT_GATES = {"CX": "cx", "CY": "cy", "CZ": "cz"}
_BARRIERS = {"barrier", "Barrier", "OpType.Barrier"}


def _normalize_gate_name(gate_name: str) -> str:
    return gate_name.removeprefix("OpType.")


def _render_pauli_rotation(operation: PauliRotation) -> list[str]:
    support = [
        (axis, qubit)
        for qubit, axis in enumerate(operation.pauli)
        if axis != "I"
    ]
    if not support:
        raise ValueError(
            "All-identity Pauli rotations are unsupported because they add a global phase."
        )

    lines: list[str] = []
    for axis, qubit in support:
        if axis == "X":
            lines.append(f"h(qs[{qubit}])")
        elif axis == "Y":
            lines.extend((f"sdg(qs[{qubit}])", f"h(qs[{qubit}])"))

    for (_, control), (_, target) in zip(support, support[1:]):
        lines.append(f"cx(qs[{control}], qs[{target}])")
    half_turns = float(operation.theta) / math.pi
    lines.append(f"rz(qs[{support[-1][1]}], angle({half_turns!r}))")
    for (_, control), (_, target) in reversed(tuple(zip(support, support[1:]))):
        lines.append(f"cx(qs[{control}], qs[{target}])")

    for axis, qubit in reversed(support):
        if axis == "X":
            lines.append(f"h(qs[{qubit}])")
        elif axis == "Y":
            lines.extend((f"h(qs[{qubit}])", f"s(qs[{qubit}])"))
    return lines


def _render_single_qubit_clifford(operation: SingleQubitClifford) -> list[str]:
    name = _normalize_gate_name(operation.gate_name)
    try:
        gate = _SINGLE_QUBIT_GATES[name]
    except KeyError as error:
        raise ValueError(f"Unsupported single-qubit Clifford: {operation.gate_name!r}.") from error
    return [f"{gate}(qs[{operation.qubit}])"]


def _render_two_qubit_clifford(operation: TwoQubitClifford) -> list[str]:
    name = _normalize_gate_name(operation.gate_name)
    try:
        gate = _TWO_QUBIT_GATES[name]
    except KeyError as error:
        raise ValueError(f"Unsupported two-qubit Clifford: {operation.gate_name!r}.") from error
    return [f"{gate}(qs[{operation.control_qubit}], qs[{operation.target_qubit}])"]


def _render_operation(operation: object) -> list[str]:
    if isinstance(operation, PauliRotation):
        return _render_pauli_rotation(operation)
    if isinstance(operation, SingleQubitClifford):
        return _render_single_qubit_clifford(operation)
    if isinstance(operation, TwoQubitClifford):
        return _render_two_qubit_clifford(operation)
    if isinstance(operation, ResetZero):
        return [f"reset(qs[{operation.qubit}])"]
    if isinstance(operation, SkippedOperation):
        if operation.gate_name in _BARRIERS:
            return ["barrier(qs)"]
        if "measure" in operation.gate_name.lower():
            raise ValueError(
                f"Measurement operation {operation.gate_name!r} is unsupported "
                "in reusable Guppy functions."
            )
        raise ValueError(f"Unsupported skipped operation: {operation.gate_name!r}.")
    if isinstance(operation, (CreateZero, Discard)):
        raise ValueError(
            f"{type(operation).__name__} is unsupported by the fixed-width Guppy emitter."
        )
    raise TypeError(f"Unsupported circuit operation: {type(operation)!r}.")


def _indent(lines: list[str]) -> str:
    return "\n".join(f"    {line}" for line in lines)


def render_guppy_source(
    circuit_ir: CircuitIR,
    *,
    function_name: str = "apply_circuit",
    include_main: bool = True,
) -> str:
    """Render a standalone Guppy v1 Python module."""
    if not isinstance(circuit_ir, CircuitIR):
        raise TypeError("circuit_ir must be a CircuitIR instance.")
    if (
        not isinstance(function_name, str)
        or not function_name.isidentifier()
        or keyword.iskeyword(function_name)
    ):
        raise ValueError("function_name must be a Python identifier and not a keyword.")

    operation_lines = [
        line
        for operation in circuit_ir.operations
        for line in _render_operation(operation)
    ]
    quantum_names = {"qubit"}
    for operation in circuit_ir.operations:
        if isinstance(operation, PauliRotation):
            quantum_names.update(("cx", "h", "rz", "s", "sdg"))
        elif isinstance(operation, SingleQubitClifford):
            quantum_names.add(_SINGLE_QUBIT_GATES[_normalize_gate_name(operation.gate_name)])
        elif isinstance(operation, TwoQubitClifford):
            quantum_names.add(_TWO_QUBIT_GATES[_normalize_gate_name(operation.gate_name)])
        elif isinstance(operation, ResetZero):
            quantum_names.add("reset")
    if include_main:
        quantum_names.update(("collect_measurements", "measure_array"))

    imports = [
        "from guppylang import guppy",
        "from guppylang.std.builtins import array, output, owned",
        f"from guppylang.std.quantum import {', '.join(sorted(quantum_names))}",
    ]
    if any(isinstance(operation, PauliRotation) for operation in circuit_ir.operations):
        imports.append("from guppylang.std.angles import angle")
    if any(isinstance(operation, SkippedOperation) for operation in circuit_ir.operations):
        imports.append("from guppylang.std.platform import barrier")

    size = circuit_ir.system_size
    body = operation_lines + ["return qs"]
    sections = [
        "\n".join(imports),
        (
            "@guppy\n"
            f"def {function_name}(qs: array[qubit, {size}] @owned) -> array[qubit, {size}]:\n"
            f"{_indent(body)}"
        ),
    ]
    if include_main:
        sections.append(
            "@guppy\n"
            "def main() -> None:\n"
            f"    qs = array(qubit() for _ in range({size}))\n"
            f"    qs = {function_name}(qs)\n"
            '    output("q", collect_measurements(measure_array(qs)))'
        )
    return "\n\n".join(sections) + "\n"


def write_guppy_source(
    path: str | Path,
    circuit_ir: CircuitIR,
    *,
    function_name: str = "apply_circuit",
    include_main: bool = True,
) -> None:
    """Write the deterministic source returned by :func:`render_guppy_source`."""
    source = render_guppy_source(
        circuit_ir,
        function_name=function_name,
        include_main=include_main,
    )
    Path(path).write_text(source, encoding="utf-8")
