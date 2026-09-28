"""Render a small CircuitIR as a standalone Guppy v1 module."""

import importlib.util
from pathlib import Path

from spd import CircuitIR, ResetZero, write_guppy_source
from spd.circuit_ir import (
    PauliRotation,
    SingleQubitClifford,
    SkippedOperation,
)


circuit_ir = CircuitIR(2, (
    PauliRotation("RXX", "XX", 0.3),
    SingleQubitClifford("H", 0),
    SkippedOperation("barrier"),
    ResetZero(1),
))
destination = Path("generated_guppy_circuit.py")
write_guppy_source(destination, circuit_ir)

spec = importlib.util.spec_from_file_location("generated_guppy_circuit", destination)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

assert hasattr(module, "apply_circuit")
assert hasattr(module, "main")
module.apply_circuit.check()
module.main.check()
print(f"Wrote and checked {destination}")
