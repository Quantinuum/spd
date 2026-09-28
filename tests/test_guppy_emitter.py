import importlib.util
import math
import subprocess
import sys

import numpy as np
import pytest

import spd
from spd import CircuitIR, CreateZero, Discard, ResetZero
from spd.circuit_ir import (
    PauliRotation,
    SingleQubitClifford,
    SkippedOperation,
    TwoQubitClifford,
)
from spd.guppy_emitter import render_guppy_source, write_guppy_source


def _representative_circuit():
    return CircuitIR(4, (
        PauliRotation("descriptive-name", "XIYZ", -0.3),
        SingleQubitClifford("OpType.Sdg", 1),
        TwoQubitClifford("OpType.CZ", 0, 3),
        SkippedOperation("barrier"),
        ResetZero(2),
    ))


def _import_generated(tmp_path, source, name="generated_guppy"):
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _kron_all(operators):
    result = np.asarray([[1]], dtype=complex)
    for operator in operators:
        result = np.kron(result, operator)
    return result


def _two_qubit_matrix(gate, control, target, size):
    result = np.zeros((2**size, 2**size), dtype=complex)
    for column in range(2**size):
        bits = [(column >> (size - qubit - 1)) & 1 for qubit in range(size)]
        local_column = 2 * bits[control] + bits[target]
        for local_row in range(4):
            output_bits = bits.copy()
            output_bits[control], output_bits[target] = divmod(local_row, 2)
            row = sum(bit << (size - qubit - 1) for qubit, bit in enumerate(output_bits))
            result[row, column] = gate[local_row, local_column]
    return result


def _direct_unitary(circuit):
    identity = np.eye(2, dtype=complex)
    gates = {
        "H": np.asarray([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        "S": np.diag([1, 1j]),
        "Sdg": np.diag([1, -1j]),
        "X": np.asarray([[0, 1], [1, 0]], dtype=complex),
        "Y": np.asarray([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.diag([1, -1]),
    }
    controlled = {
        "CX": np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        "CY": np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]),
        "CZ": np.diag([1, 1, 1, -1]),
    }
    result = np.eye(2**circuit.system_size, dtype=complex)
    for operation in circuit.operations:
        if isinstance(operation, PauliRotation):
            pauli = _kron_all(gates[axis] if axis != "I" else identity for axis in operation.pauli)
            full_gate = (
                np.cos(operation.theta / 2) * np.eye(2**circuit.system_size)
                - 1j * np.sin(operation.theta / 2) * pauli
            )
        elif isinstance(operation, SingleQubitClifford):
            name = operation.gate_name.removeprefix("OpType.")
            full_gate = _kron_all(
                gates[name] if qubit == operation.qubit else identity
                for qubit in range(circuit.system_size)
            )
        elif isinstance(operation, TwoQubitClifford):
            name = operation.gate_name.removeprefix("OpType.")
            full_gate = _two_qubit_matrix(
                controlled[name], operation.control_qubit, operation.target_qubit,
                circuit.system_size,
            )
        else:
            continue
        result = full_gate @ result
    return result


def _assert_equal_up_to_global_phase(actual, expected):
    pivot = int(np.argmax(np.abs(expected)))
    phase = actual[pivot] / expected[pivot]
    np.testing.assert_allclose(actual, phase * expected, atol=1e-9, rtol=1e-9)


def test_render_is_deterministic_and_default_contains_main():
    circuit = _representative_circuit()

    first = render_guppy_source(circuit)
    second = render_guppy_source(circuit)

    assert first == second
    assert "def apply_circuit(" in first
    assert "def main() -> None:" in first
    assert 'output("q", collect_measurements(measure_array(qs)))' in first
    assert "pytket" not in first


def test_render_without_main_keeps_reusable_function():
    source = render_guppy_source(CircuitIR(1, ()), include_main=False)

    assert "def apply_circuit(" in source
    assert "def main(" not in source
    assert "measure_array" not in source


def test_write_matches_render(tmp_path):
    circuit = _representative_circuit()
    destination = tmp_path / "circuit.py"

    write_guppy_source(destination, circuit, function_name="cooling_cycle")

    assert destination.read_text(encoding="utf-8") == render_guppy_source(
        circuit, function_name="cooling_cycle"
    )


def test_operation_mapping_and_order():
    circuit = CircuitIR(4, (
        PauliRotation("ignored", "XIYI", math.pi / 2),
        SkippedOperation("OpType.Barrier"),
        TwoQubitClifford("CY", 3, 0),
        ResetZero(1),
    ))

    source = render_guppy_source(circuit, include_main=False)
    statements = [
        "h(qs[0])",
        "sdg(qs[2])",
        "h(qs[2])",
        "cx(qs[0], qs[2])",
        "rz(qs[2], angle(0.5))",
        "cx(qs[0], qs[2])",
        "h(qs[2])",
        "s(qs[2])",
        "h(qs[0])",
        "barrier(qs)",
        "cy(qs[3], qs[0])",
        "reset(qs[1])",
    ]
    positions = []
    start = 0
    for statement in statements:
        position = source.index(statement, start)
        positions.append(position)
        start = position + len(statement)
    assert positions == sorted(positions)


def test_no_barrier_is_added():
    source = render_guppy_source(
        CircuitIR(1, (SingleQubitClifford("H", 0),)),
        include_main=False,
    )

    assert "barrier" not in source


@pytest.mark.parametrize("name, emitted", [
    ("H", "h"), ("S", "s"), ("Sdg", "sdg"),
    ("X", "x"), ("Y", "y"), ("Z", "z"),
])
def test_single_qubit_cliffords(name, emitted):
    source = render_guppy_source(
        CircuitIR(1, (SingleQubitClifford(f"OpType.{name}", 0),)),
        include_main=False,
    )
    assert f"{emitted}(qs[0])" in source


@pytest.mark.parametrize("name, emitted", [("CX", "cx"), ("CY", "cy"), ("CZ", "cz")])
def test_two_qubit_cliffords(name, emitted):
    source = render_guppy_source(
        CircuitIR(2, (TwoQubitClifford(f"OpType.{name}", 1, 0),)),
        include_main=False,
    )
    assert f"{emitted}(qs[1], qs[0])" in source


@pytest.mark.parametrize("theta", [0.0, -0.25, 100 * math.pi])
def test_rotation_angles_are_converted_to_half_turns(theta):
    source = render_guppy_source(
        CircuitIR(1, (PauliRotation("ignored", "Z", theta),)),
        include_main=False,
    )
    assert f"angle({theta / math.pi!r})" in source


@pytest.mark.parametrize("operation, message", [
    (CreateZero(0), "CreateZero"),
    (Discard(0), "Discard"),
    (SkippedOperation("Measure"), "Measurement"),
    (SkippedOperation("noop"), "skipped"),
    (SingleQubitClifford("T", 0), "single-qubit"),
    (TwoQubitClifford("SWAP", 0, 1), "two-qubit"),
    (PauliRotation("phase", "II", 0.1), "All-identity"),
])
def test_unsupported_operations_raise_clear_errors(operation, message):
    size = 2 if isinstance(operation, (TwoQubitClifford, PauliRotation)) else 1
    with pytest.raises(ValueError, match=message):
        render_guppy_source(CircuitIR(size, (operation,)))


@pytest.mark.parametrize("name", ["", "two words", "2cool", "class", None])
def test_invalid_function_names(name):
    with pytest.raises(ValueError, match="function_name"):
        render_guppy_source(CircuitIR(1, ()), function_name=name)


def test_non_circuit_input_is_rejected():
    with pytest.raises(TypeError, match="CircuitIR"):
        render_guppy_source(object())


def test_public_api_does_not_import_guppy():
    assert spd.render_guppy_source is render_guppy_source
    result = subprocess.run(
        [sys.executable, "-c", "import spd, sys; print('guppylang' in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


@pytest.mark.skipif(
    importlib.util.find_spec("guppylang") is None, reason="Guppy extra not installed"
)
def test_generated_definitions_check_and_main_compiles(tmp_path):
    module = _import_generated(tmp_path, render_guppy_source(_representative_circuit()))

    module.apply_circuit.check()
    module.main.check()
    module.main.compile()


@pytest.mark.skipif(
    importlib.util.find_spec("guppylang") is None, reason="Guppy extra not installed"
)
def test_emitted_unitary_matches_direct_circuit_matrix(tmp_path, monkeypatch):
    circuit = CircuitIR(3, (
        SingleQubitClifford("H", 0),
        SingleQubitClifford("S", 1),
        SingleQubitClifford("Sdg", 2),
        SingleQubitClifford("X", 0),
        SingleQubitClifford("Y", 1),
        SingleQubitClifford("Z", 2),
        TwoQubitClifford("CX", 0, 1),
        TwoQubitClifford("CY", 1, 2),
        TwoQubitClifford("CZ", 2, 0),
        PauliRotation("ignored", "XII", 0.0),
        PauliRotation("ignored", "IYI", -0.7),
        PauliRotation("ignored", "IIZ", 9 * math.pi),
        PauliRotation("ignored", "XXI", 0.2),
        PauliRotation("ignored", "IYY", -0.3),
        PauliRotation("ignored", "ZZI", 0.4),
        PauliRotation("ignored", "XYI", 0.5),
        PauliRotation("ignored", "IYZ", -0.6),
        PauliRotation("ignored", "XIZ", 0.8),
        PauliRotation("ignored", "XYZ", -0.9),
        SkippedOperation("barrier"),
    ))
    source = render_guppy_source(circuit, include_main=False) + """
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array

@guppy
def unitary_main(bits: array[bool, 3]) -> None:
    qs = array(qubit() for _ in range(3))
    if bits[0]:
        x(qs[0])
    if bits[1]:
        x(qs[1])
    if bits[2]:
        x(qs[2])
    qs = apply_circuit(qs)
    barrier(qs)
    state_output("state", qs[0], qs[1], qs[2])
    discard_array(qs)

@guppy
def superposition_main() -> None:
    qs = array(qubit() for _ in range(3))
    h(qs[0])
    h(qs[1])
    h(qs[2])
    qs = apply_circuit(qs)
    barrier(qs)
    state_output("state", qs[0], qs[1], qs[2])
    discard_array(qs)
"""
    module = _import_generated(tmp_path, source, "generated_unitary_guppy")
    cache = tmp_path / "zig-cache"
    monkeypatch.setenv("ZIG_GLOBAL_CACHE_DIR", str(cache / "global"))
    monkeypatch.setenv("ZIG_LOCAL_CACHE_DIR", str(cache / "local"))

    try:
        emulator = module.unitary_main.emulator(n_qubits=3)
        columns = []
        for basis in range(8):
            bits = [bool(basis & (1 << (2 - qubit))) for qubit in range(3)]
            result = emulator.run(bits=bits)
            distribution = result.partial_state_dicts()[0]["state"].state_distribution()
            assert len(distribution) == 1
            assert distribution[0].probability == pytest.approx(1.0, abs=1e-12)
            columns.append(distribution[0].state)
    except PermissionError:
        pytest.skip("Selene cannot open its loopback result socket in this sandbox")

    emitted = np.column_stack(columns)
    expected = _direct_unitary(circuit)
    for column in range(8):
        _assert_equal_up_to_global_phase(emitted[:, column], expected[:, column])

    result = module.superposition_main.emulator(n_qubits=3).run()
    actual_superposition = (
        result.partial_state_dicts()[0]["state"].state_distribution()[0].state
    )
    expected_superposition = expected @ (np.ones(8, dtype=complex) / np.sqrt(8))
    _assert_equal_up_to_global_phase(actual_superposition, expected_superposition)


@pytest.mark.skipif(
    importlib.util.find_spec("guppylang") is None, reason="Guppy extra not installed"
)
def test_reset_cycle_can_be_called_twice(tmp_path):
    source = render_guppy_source(
        CircuitIR(1, (SingleQubitClifford("X", 0), ResetZero(0))),
    )
    source += """
@guppy
def repeated_main() -> None:
    qs = array(qubit() for _ in range(1))
    qs = apply_circuit(qs)
    qs = apply_circuit(qs)
    output("q", collect_measurements(measure_array(qs)))
"""
    module = _import_generated(tmp_path, source, "generated_repeated_guppy")

    module.repeated_main.check()
    module.repeated_main.compile()


@pytest.mark.skipif(
    importlib.util.find_spec("guppylang") is None, reason="Guppy extra not installed"
)
def test_reset_zero_matches_entangled_reset_channel(tmp_path, monkeypatch):
    circuit = CircuitIR(3, (
        SingleQubitClifford("X", 0),
        ResetZero(0),
        SingleQubitClifford("H", 1),
        TwoQubitClifford("CX", 1, 2),
        ResetZero(1),
    ))
    module = _import_generated(
        tmp_path,
        render_guppy_source(circuit),
        "generated_reset_channel_guppy",
    )
    cache = tmp_path / "zig-cache-reset"
    monkeypatch.setenv("ZIG_GLOBAL_CACHE_DIR", str(cache / "global"))
    monkeypatch.setenv("ZIG_LOCAL_CACHE_DIR", str(cache / "local"))

    try:
        result = (
            module.main.emulator(n_qubits=3)
            .with_seed(7)
            .with_shots(256)
            .run()
        )
    except PermissionError:
        pytest.skip("Selene cannot open its loopback result socket in this sandbox")

    bits = [shot.entries[0][1] for shot in result.results]
    assert all(value[0] == 0 for value in bits)
    assert all(value[1] == 0 for value in bits)
    one_frequency = sum(value[2] for value in bits) / len(bits)
    assert one_frequency == pytest.approx(0.5, abs=0.1)
