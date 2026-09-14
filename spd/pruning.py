"""Internal circuit pruning plans. Currently only exact light-cone pruning.

The public entry point is evolve(..., pruning=...). Plans describe
one forward execution, not a user-managed cache. No angle-based pruning occurs.
"""

from copy import copy
from dataclasses import dataclass
import math

import numpy as np

from .circuit_ir import (
    PauliRotation, SingleQubitClifford, TwoQubitClifford, SkippedOperation,
    get_operation_qubits,
)


def _validate_method(pruning):
    if pruning is not None and (not isinstance(pruning, str) or pruning not in ("light-cone", "light-cone-barrier")):
        raise ValueError('pruning must be None, "light-cone", or "light-cone-barrier".')


def _support(state, backend_name):
    """Read nonzero primal support once, using existing packed representations.

    Do not export/compact Triton's mutable storage just to inspect it. Dead
    slots and JAX padding have zero primal coefficients and do not seed a cone.
    """
    if backend_name == "numpy":
        if not state:
            return set(), 0
        keys = np.asarray(list(state), dtype=np.uint32)
        coefficients = np.asarray(list(state.values()))
    elif backend_name == "triton":
        keys, coefficients, _ = state._raw_arrays()
        from .triton_backend.support import support_mask
        reduced = support_mask(keys, coefficients)
        if reduced[-1]:
            raise ValueError("Pruning requires finite input coefficients.")
        return _unpack_support(reduced[:-1])
    else:
        keys = np.asarray(state.xz_array, dtype=np.uint32)
        coefficients = np.asarray(state.c_array)
    words = keys.shape[1] // 2
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Pruning requires finite input coefficients.")
    mask = np.bitwise_or.reduce(keys[coefficients != 0], axis=0, initial=np.uint32(0))
    mask = mask[:words] | mask[words:]
    return _unpack_support(mask)


def _unpack_support(mask):
    words = len(mask)
    support = {q for q in range(words * 32) if int(mask[q // 32]) & (1 << (31 - q % 32))}
    return support, words * 32


def _validate_operations(operations, system_size, storage_size, backend_name):
    """Validate even omitted gates, before any state mutation."""
    limit = system_size if system_size is not None else storage_size
    for op in operations:
        if isinstance(op, SkippedOperation):
            continue
        if isinstance(op, PauliRotation):
            if not isinstance(op.pauli, str) or set(op.pauli) - set("IXYZ"):
                raise ValueError("Rotation generators must be I/X/Y/Z strings.")
            if not math.isfinite(op.theta):
                raise ValueError("Rotation angles must be finite.")
            if storage_size and backend_name != "triton" and 32 * ((len(op.pauli) + 31) // 32) != storage_size:
                raise ValueError("Rotation generator packed width does not match the state.")
        elif isinstance(op, SingleQubitClifford):
            if op.gate_name not in {"OpType.H", "OpType.S", "OpType.Sdg", "OpType.X", "OpType.Y", "OpType.Z"}:
                raise ValueError(f"Unsupported single-qubit Clifford: {op.gate_name}")
        elif isinstance(op, TwoQubitClifford):
            if op.gate_name not in {"OpType.CX", "OpType.CY", "OpType.CZ"}:
                raise ValueError(f"Unsupported two-qubit Clifford: {op.gate_name}")
            if op.control_qubit == op.target_qubit:
                raise ValueError("Two-qubit Clifford sites must be distinct.")
        else:
            raise TypeError(f"Unsupported circuit operation: {type(op)!r}")
        for q in get_operation_qubits(op):
            if (not isinstance(q, (int, np.integer)) or isinstance(q, (bool, np.bool_))
                    or q < 0 or (limit and q >= limit) or (storage_size and q >= storage_size)):
                raise ValueError(f"Operation acts outside the circuit/state: qubit {q}.")


@dataclass(frozen=True)
class _LightConePlan:
    operations: tuple
    system_size: int | None
    retained_indices: tuple
    method: str = "light-cone"

    @property
    def retained_operations(self):
        return tuple(self.operations[i] for i in self.retained_indices)

    def check_circuit(self, operations, system_size):
        # Angles must match too: this is the record of one forward call.
        if tuple(operations) != self.operations or system_size != self.system_size:
            raise ValueError("Backward circuit does not match the pruned forward execution.")

    def align_info(self, info, *, reverse):
        kept = set(self.retained_indices)
        order = range(len(self.operations) - 1, -1, -1) if reverse else range(len(self.operations))
        order = [i for i in order if not isinstance(self.operations[i], SkippedOperation)]
        history = {}
        for key, values in info["history"].items():
            values = iter(values)
            history[key] = [next(values) if i in kept else 0 for i in order]
        info["history"] = history
        info["num_steps_tracked"] = len(order)
        info["pruning"] = {"method": self.method, "num_gates_total": len(order),
                           "num_gates_retained": len(kept), "num_gates_pruned": len(order) - len(kept)}
        return info

    def align_gradients(self, gradients):
        kept = set(self.retained_indices)
        gradients = iter(gradients)
        return [next(gradients) if i in kept else 0.0
                for i, op in enumerate(self.operations) if isinstance(op, PauliRotation)]


def _validated_support(state, operations, system_size, backend_name, trunc_val):
    if not math.isfinite(trunc_val) or trunc_val < 0:
        raise ValueError("trunc_val must be finite and nonnegative.")
    if backend_name == "triton" and state._storage.grad is not None:
        raise TypeError("Pruning requires an SPO, not a gradient operator.")
    support, storage_size = _support(state, backend_name)
    if backend_name == "triton":
        storage_size = state.num_qubits  # Direct GPU states can be narrower than their packed word.
    if storage_size and any(q >= storage_size for q in support):
        raise ValueError("Input observable acts outside the state size.")
    if system_size is not None and any(q >= system_size for q in support):
        raise ValueError("Input observable acts outside the physical circuit.")
    if system_size is not None and storage_size and system_size > storage_size:
        raise ValueError("Physical circuit exceeds the input state's packed width or physical size.")
    _validate_operations(operations, system_size, storage_size, backend_name)
    return support


def _retained_indices(operations, support, start=0, stop=None):
    retained = []
    stop = len(operations) if stop is None else stop
    for i in range(stop - 1, start - 1, -1):
        qubits = get_operation_qubits(operations[i])
        if not support.isdisjoint(qubits):
            retained.append(i)
            support.update(qubits)
    return tuple(reversed(retained))


def _plan_forward(state, operations, system_size, backend_name, trunc_val):
    operations = tuple(operations)
    support = _validated_support(state, operations, system_size, backend_name, trunc_val)
    return _LightConePlan(operations, system_size, _retained_indices(operations, support))


class _BarrierPruning:
    """One forward traversal, planning each block from its current primal state.

    Validate the full circuit before execution. Empty blocks and other skipped
    operations do not trigger scans. Original indices form one backward record.
    """

    def __init__(self, state, operations, system_size, backend_name, trunc_val):
        self.operations = tuple(operations)
        self.system_size = system_size
        self.backend_name = backend_name
        self.initial_support = _validated_support(
            state, self.operations, system_size, backend_name, trunc_val,
        )
        self.blocks = []
        start = 0
        has_gate = False
        for i, op in enumerate(self.operations):
            if isinstance(op, SkippedOperation) and op.gate_name in ("barrier", "OpType.Barrier"):
                if has_gate:
                    self.blocks.append((start, i))
                start, has_gate = i + 1, False
            elif not isinstance(op, SkippedOperation):
                has_gate = True
        if has_gate:
            self.blocks.append((start, len(self.operations)))
        self.retained = []

    def iter_operations(self, get_state):
        # The callback observes the runner's latest state on functional backends
        # as well as Triton's persistent in-place storage.
        for block_index, (start, stop) in enumerate(reversed(self.blocks)):
            support = (self.initial_support.copy() if block_index == 0 else
                       _support(get_state(), self.backend_name)[0])
            indices = _retained_indices(self.operations, support, start, stop)
            self.retained.extend(reversed(indices))
            for i in reversed(indices):
                yield i, self.operations[i]

    def record(self):
        return _LightConePlan(self.operations, self.system_size,
                              tuple(reversed(self.retained)), "light-cone-barrier")


def _finish_record(result, source, record, backend_name, *, in_place=False):
    """Attach/clear per-call metadata without modifying a functional input."""
    if result is source and record is None and getattr(source, "_pruning_record", None) is None:
        return result
    if result is source and not in_place and (record is not None or getattr(source, "_pruning_record", None) is not None):
        result = source.copy() if backend_name == "triton" else copy(source)
    result._pruning_record = record
    return result
