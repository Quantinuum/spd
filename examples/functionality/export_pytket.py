"""Export CircuitIR to pytket and parse it back."""

from spd import CircuitIR, ResetZero, export_pytket_circuit, parse_pytket_circuit
from spd.circuit_ir import PauliRotation, SkippedOperation, TwoQubitClifford


barrier = SkippedOperation("barrier")
circuit_ir = CircuitIR(2, (
    PauliRotation("Ry", "YI", .3), barrier,
    TwoQubitClifford("CX", 0, 1), barrier,
    ResetZero(1),
))

pytket_circuit = export_pytket_circuit(circuit_ir)
round_trip = parse_pytket_circuit(pytket_circuit)

print(pytket_circuit)
print(round_trip.draw())
