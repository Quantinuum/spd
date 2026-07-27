"""Built-in OpenQASM 2 parser for the SPD-supported circuit subset.

This frontend lowers supported OpenQASM statements directly into SPD's
backend-agnostic IR without requiring pytket. It intentionally supports only
the static gate subset that SPD can execute today.
"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Tuple

from .circuit_ir import (
    CircuitIR,
    PauliRotation,
    SingleQubitClifford,
    SkippedOperation,
    TwoQubitClifford,
)


_COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
_REGISTER_DECL_RE = re.compile(r"^(qreg|creg)\s+([A-Za-z_]\w*)\[(\d+)\]$", re.IGNORECASE)
_QUBIT_RE = re.compile(r"^([A-Za-z_]\w*)\[(\d+)\]$")
_GATE_CALL_RE = re.compile(r"^([A-Za-z_]\w*)(?:\((.*)\))?\s+(.+)$")

_SINGLE_QUBIT_ROTATIONS = {"rx": "X", "ry": "Y", "rz": "Z"}
_TWO_QUBIT_ROTATIONS = {"rzz": "Z", "rxx": "X", "ryy": "Y"}
_SINGLE_QUBIT_CLIFFORDS = {
    "h": "OpType.H",
    "s": "OpType.S",
    "sdg": "OpType.Sdg",
    "x": "OpType.X",
    "y": "OpType.Y",
    "z": "OpType.Z",
}
_TWO_QUBIT_CLIFFORDS = {
    "cx": "OpType.CX",
    "cy": "OpType.CY",
    "cz": "OpType.CZ",
}
_SUPPORTED_INCLUDES = {"qelib1.inc", "stdgates.inc"}


def parse_openqasm_str(source: str, padded_system_size: int | None = None):
    """Parse an OpenQASM 2 program string into SPD IR operations.

    Returns a `CircuitIR`.
    """
    statements = _split_statements(source)
    if not statements:
        raise ValueError("OpenQASM source is empty.")

    registers: Dict[str, Tuple[int, int]] = {}
    classical_registers: Dict[str, int] = {}
    operations = []
    system_size = 0
    version_seen = False

    for statement in statements:
        if statement.upper().startswith("OPENQASM"):
            parts = statement.split()
            if len(parts) != 2 or parts[1] != "2.0":
                raise ValueError("Only OPENQASM 2.0 is currently supported.")
            version_seen = True
            continue

        if statement.lower().startswith("include"):
            include_path = _parse_include(statement)
            if include_path not in _SUPPORTED_INCLUDES:
                raise ValueError(f"Unsupported include file: {include_path}")
            continue

        decl_match = _REGISTER_DECL_RE.match(statement)
        if decl_match:
            kind, name, width_str = decl_match.groups()
            width = int(width_str)
            if width < 1:
                raise ValueError(f"Register '{name}' must have positive width.")
            if kind.lower() == "qreg":
                if name in registers:
                    raise ValueError(f"Duplicate quantum register declaration: {name}")
                registers[name] = (system_size, width)
                system_size += width
            else:
                classical_registers[name] = width
            continue

        if not version_seen:
            raise ValueError("OPENQASM 2.0 version declaration must appear before operations.")
        if system_size == 0:
            raise ValueError("At least one qreg declaration is required before gate statements.")

        effective_padded_system_size = _resolve_padded_system_size(system_size, padded_system_size)
        operations.append(
            _parse_operation(
                statement,
                registers=registers,
                classical_registers=classical_registers,
                padded_system_size=effective_padded_system_size,
            )
        )

    if not version_seen:
        raise ValueError("Missing OPENQASM version declaration.")
    if system_size == 0:
        raise ValueError("Missing qreg declaration.")

    return CircuitIR(system_size=system_size, operations=tuple(operations))


def parse_openqasm_file(path: str, padded_system_size: int | None = None):
    """Parse an OpenQASM file into SPD IR operations.

    Returns a `CircuitIR`.
    """
    with open(path, "r", encoding="utf-8") as f:
        return parse_openqasm_str(f.read(), padded_system_size)


def _split_statements(source: str) -> List[str]:
    stripped = _COMMENT_RE.sub("", source)
    return [statement.strip() for statement in stripped.split(";") if statement.strip()]


def _parse_include(statement: str) -> str:
    match = re.match(r'^include\s+"([^"]+)"$', statement, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Malformed include statement: {statement}")
    return match.group(1)


def _parse_operation(statement, *, registers, classical_registers, padded_system_size):
    if statement.lower().startswith("barrier"):
        return SkippedOperation(gate_name="barrier")

    if statement.lower().startswith("measure"):
        _validate_measure(statement, registers, classical_registers)
        return SkippedOperation(gate_name="measure")

    match = _GATE_CALL_RE.match(statement)
    if match is None:
        raise ValueError(f"Unsupported or malformed statement: {statement}")

    gate_name_raw, params_raw, args_raw = match.groups()
    gate_name = gate_name_raw.lower()
    args = [arg.strip() for arg in args_raw.split(",") if arg.strip()]

    if gate_name in _SINGLE_QUBIT_ROTATIONS:
        if params_raw is None:
            raise ValueError(f"Gate '{gate_name_raw}' requires one angle parameter.")
        if len(args) != 1:
            raise ValueError(f"Gate '{gate_name_raw}' requires one qubit argument.")
        theta = _evaluate_angle(params_raw)
        qubit = _resolve_qubit(args[0], registers)
        pauli = _build_pauli_string(padded_system_size, ((qubit, _SINGLE_QUBIT_ROTATIONS[gate_name]),))
        return PauliRotation(gate_name=f"OpenQASM.{gate_name_raw}", pauli=pauli, theta=theta)

    if gate_name in _TWO_QUBIT_ROTATIONS:
        if params_raw is None:
            raise ValueError(f"Gate '{gate_name_raw}' requires one angle parameter.")
        if len(args) != 2:
            raise ValueError(f"Gate '{gate_name_raw}' requires two qubit arguments.")
        theta = _evaluate_angle(params_raw)
        qubits = [_resolve_qubit(arg, registers) for arg in args]
        axis = _TWO_QUBIT_ROTATIONS[gate_name]
        pauli = _build_pauli_string(padded_system_size, ((qubits[0], axis), (qubits[1], axis)))
        return PauliRotation(gate_name=f"OpenQASM.{gate_name_raw}", pauli=pauli, theta=theta)

    if gate_name in _SINGLE_QUBIT_CLIFFORDS:
        if params_raw is not None:
            raise ValueError(f"Gate '{gate_name_raw}' does not take parameters.")
        if len(args) != 1:
            raise ValueError(f"Gate '{gate_name_raw}' requires one qubit argument.")
        qubit = _resolve_qubit(args[0], registers)
        return SingleQubitClifford(gate_name=_SINGLE_QUBIT_CLIFFORDS[gate_name], qubit=qubit)

    if gate_name in _TWO_QUBIT_CLIFFORDS:
        if params_raw is not None:
            raise ValueError(f"Gate '{gate_name_raw}' does not take parameters.")
        if len(args) != 2:
            raise ValueError(f"Gate '{gate_name_raw}' requires two qubit arguments.")
        control_qubit = _resolve_qubit(args[0], registers)
        target_qubit = _resolve_qubit(args[1], registers)
        return TwoQubitClifford(
            gate_name=_TWO_QUBIT_CLIFFORDS[gate_name],
            control_qubit=control_qubit,
            target_qubit=target_qubit,
        )

    raise ValueError(f"Unsupported OpenQASM gate: {gate_name_raw}")


def _validate_measure(statement, registers, classical_registers):
    match = re.match(
        r"^measure\s+(.+?)\s*->\s*([A-Za-z_]\w*\[\d+\])$",
        statement,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Malformed measure statement: {statement}")
    qubit_ref, bit_ref = match.groups()
    _resolve_qubit(qubit_ref.strip(), registers)
    bit_match = _QUBIT_RE.match(bit_ref)
    if bit_match is None:
        raise ValueError(f"Malformed classical bit reference: {bit_ref}")
    register_name, bit_idx_str = bit_match.groups()
    if register_name not in classical_registers:
        raise ValueError(f"Unknown classical register: {register_name}")
    bit_idx = int(bit_idx_str)
    if bit_idx >= classical_registers[register_name]:
        raise ValueError(f"Classical bit out of range: {bit_ref}")


def _resolve_qubit(reference: str, registers: Dict[str, Tuple[int, int]]) -> int:
    match = _QUBIT_RE.match(reference)
    if match is None:
        raise ValueError(f"Malformed qubit reference: {reference}")
    register_name, local_index_str = match.groups()
    if register_name not in registers:
        raise ValueError(f"Unknown quantum register: {register_name}")
    base_index, width = registers[register_name]
    local_index = int(local_index_str)
    if local_index >= width:
        raise ValueError(f"Qubit index out of range: {reference}")
    return base_index + local_index


def _build_pauli_string(padded_system_size: int, assignments) -> str:
    pauli = ["I"] * padded_system_size
    for qubit, axis in assignments:
        pauli[qubit] = axis
    return "".join(pauli)


def _resolve_padded_system_size(system_size: int, padded_system_size: int | None) -> int:
    if padded_system_size is None:
        return 32 * ((system_size + 31) // 32)
    return padded_system_size


def _evaluate_angle(expr: str) -> float:
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Malformed angle expression: {expr}") from exc
    return float(_eval_ast(parsed.body))


def _eval_ast(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id == "pi":
        return 3.141592653589793
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div),
    ):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    raise ValueError("Only numeric angle expressions with +, -, *, /, and pi are supported.")
