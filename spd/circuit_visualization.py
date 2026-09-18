"""Dependency-free, scalable text and SVG rendering for CircuitIR."""

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .circuit_ir import (
    CircuitIR, CircuitOperation, CreateZero, Discard, PauliRotation,
    ResetZero, SkippedOperation, get_operation_qubits,
)


@dataclass(frozen=True)
class _PlacedOperation:
    operation: CircuitOperation
    qubits: Tuple[int, ...]
    layer: int
    label: str


def _label(operation: CircuitOperation) -> str:
    if isinstance(operation, PauliRotation):
        return "{}({:.6g})".format(operation.gate_name, operation.theta)
    if isinstance(operation, CreateZero):
        return "|0>"
    if isinstance(operation, ResetZero):
        return "reset |0>"
    if isinstance(operation, Discard):
        return "discard"
    return operation.gate_name.rsplit(".", 1)[-1]


def _layout(circuit: CircuitIR) -> Tuple[Tuple[_PlacedOperation, ...], int]:
    """Place each operation in its earliest dependency-safe layer."""
    next_layer = [0] * circuit.system_size
    placed = []
    for operation in circuit.operations:
        qubits = get_operation_qubits(operation)
        if isinstance(operation, SkippedOperation):
            # Skipped operations normally represent barriers. Making them global
            # prevents visual reordering across a barrier.
            qubits = tuple(range(circuit.system_size))
        layer = max((next_layer[q] for q in qubits), default=max(next_layer, default=0))
        for qubit in qubits:
            next_layer[qubit] = layer + 1
        placed.append(_PlacedOperation(operation, tuple(qubits), layer, _label(operation)))
    return tuple(placed), max(next_layer, default=0)


def _chunks(layer_count: int, fold: int) -> Sequence[Tuple[int, int]]:
    if layer_count == 0:
        return ((0, 0),)
    return tuple((start, min(start + fold, layer_count))
                 for start in range(0, layer_count, fold))


def _text(circuit: CircuitIR, placed: Sequence[_PlacedOperation],
          layer_count: int, fold: int) -> str:
    if not placed:
        return "CircuitIR({} qubits, empty)".format(circuit.system_size)
    lines = []
    for panel_index, (start, stop) in enumerate(_chunks(layer_count, fold)):
        panel = tuple(item for item in placed if start <= item.layer < stop)
        widths = [max([5] + [len(item.label) + 2 for item in panel if item.layer == layer])
                  for layer in range(start, stop)]
        if panel_index:
            lines.append("")
        if layer_count > fold:
            lines.append("layers {}-{}".format(start, stop - 1))
        for qubit in range(circuit.system_size):
            cells = []
            for offset, layer in enumerate(range(start, stop)):
                item = next((candidate for candidate in panel
                             if candidate.layer == layer and qubit in candidate.qubits), None)
                value = "[{}]".format(item.label) if item else "-"
                cells.append(value.center(widths[offset], "-"))
            lines.append("q{}: ".format(qubit) + "".join(cells))
    return "\n".join(lines)


def _svg(circuit: CircuitIR, placed: Sequence[_PlacedOperation],
         layer_count: int, fold: int) -> str:
    left, column_width, row_height = 58, 112, 48
    panel_gap, top = 34, 24
    chunks = _chunks(layer_count, fold)
    visible_layers = max((stop - start for start, stop in chunks), default=0)
    width = max(220, left + visible_layers * column_width + 30)
    panel_height = circuit.system_size * row_height + panel_gap
    height = top + len(chunks) * panel_height
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
        'viewBox="0 0 {} {}" role="img" aria-labelledby="title desc">'.format(
            width, height, width, height),
        '<title id="title">CircuitIR diagram</title>',
        '<desc id="desc">{} qubits, {} operations, folded every {} layers.</desc>'.format(
            circuit.system_size, len(circuit.operations), fold),
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px}'
        '.wire{stroke:#667085;stroke-width:1.5}.link{stroke:#344054;stroke-width:1.5}'
        '.gate{fill:#eef4ff;stroke:#475467;stroke-width:1.2}'
        '.channel{fill:#ecfdf3;stroke:#027a48;stroke-width:1.2}'
        '.skipped{fill:#f2f4f7;stroke:#667085;stroke-dasharray:4 3}'
        '.label{fill:#101828;text-anchor:middle;dominant-baseline:middle}'
        '.qubit{fill:#344054;text-anchor:end;dominant-baseline:middle}'
        '.panel{fill:#667085}</style>',
    ]
    for panel_index, (start, stop) in enumerate(chunks):
        panel_top = top + panel_index * panel_height
        if layer_count > fold:
            out.append('<text class="panel" x="8" y="{}">layers {}–{}</text>'.format(
                panel_top, start, stop - 1))
        wire_top = panel_top + 18
        panel_width = max(1, stop - start) * column_width
        for qubit in range(circuit.system_size):
            y = wire_top + qubit * row_height
            out.append('<text class="qubit" x="44" y="{}">q{}</text>'.format(y, qubit))
            out.append('<line class="wire" x1="{}" y1="{}" x2="{}" y2="{}"/>'.format(
                left, y, left + panel_width, y))
        for item in placed:
            if not start <= item.layer < stop or not item.qubits:
                continue
            x = left + (item.layer - start) * column_width + column_width / 2
            ys = [wire_top + qubit * row_height for qubit in item.qubits]
            if len(ys) > 1:
                out.append('<line class="link" x1="{}" y1="{}" x2="{}" y2="{}"/>'.format(
                    x, min(ys), x, max(ys)))
            css = "channel" if isinstance(item.operation, (CreateZero, ResetZero, Discard)) else "gate"
            if isinstance(item.operation, SkippedOperation):
                css = "skipped"
            box_width = min(100, max(42, len(item.label) * 7 + 14))
            for index, y in enumerate(ys):
                text = item.label if index == 0 else "•"
                out.append('<rect class="{}" x="{}" y="{}" width="{}" height="28" rx="4"/>'.format(
                    css, x - box_width / 2, y - 14, box_width))
                out.append('<text class="label" x="{}" y="{}">{}</text>'.format(
                    x, y, escape(text)))
    out.append("</svg>")
    return "\n".join(out)


def draw_circuit(circuit: CircuitIR, output: str = "text",
                 filename: Optional[str] = None, fold: int = 40) -> str:
    """Render a circuit as text or standalone SVG and return the result."""
    if output not in ("text", "svg"):
        raise ValueError("output must be 'text' or 'svg'.")
    if not isinstance(fold, int) or isinstance(fold, bool) or fold < 1:
        raise ValueError("fold must be a positive integer.")
    placed, layer_count = _layout(circuit)
    rendered = (_text if output == "text" else _svg)(
        circuit, placed, layer_count, fold)
    if filename is not None:
        Path(filename).write_text(rendered, encoding="utf-8")
    return rendered
