"""Execution helpers for SPD state propagation.

This module separates state evolution from expectation evaluation. Public entry
points accept an existing backend-specific SPO or SPGO together with either a
`pytket` circuit or a lowered SPD IR operation sequence.
"""

from collections.abc import Sequence
from bisect import bisect_left
from dataclasses import dataclass, replace
from copy import copy
import math
import sys
import time

import numpy as np
import psutil

from .backend_adapter import BackendAdapter
from .checkpoints import (CheckpointStore, DEFAULT_CHECKPOINT_MEMORY_BUDGET_BYTES,
                          DEFAULT_CHECKPOINT_DEVICE_MEMORY_BUDGET_BYTES)
from .pruning import _validate_method, _plan_forward, _finish_record, _BarrierPruning
from .circuit_ir import (
    CircuitIR, CHANNEL_TYPES, CreateZero, ResetZero, Discard,
    PauliRotation,
    SingleQubitClifford,
    SkippedOperation,
    TwoQubitClifford,
    get_operation_qubits,
)

_PACKBIT = 32
_IR_OPERATION_TYPES = (
    PauliRotation,
    SingleQubitClifford,
    TwoQubitClifford,
    SkippedOperation,
    *CHANNEL_TYPES,
)


@dataclass(frozen=True)
class _ContractionGroup:
    """Execution-only batch of independent contractions and intervening barriers."""
    checkpoint_key: int
    operations: tuple

    @property
    def contractions(self):
        return tuple(op for op in self.operations if isinstance(op, (CreateZero, ResetZero)))

    @property
    def gate_name(self):
        return "Zero-state contractions (" + str(len(self.contractions)) + ")"


def _is_barrier(operation):
    return isinstance(operation, SkippedOperation) and operation.gate_name in ("Barrier", "barrier", "OpType.Barrier")


def _group_contractions(operations):
    """Group distinct indices across barriers, stopping at any other operation.

    Repeated resets are separate groups. No gates are reordered, including
    gates on disjoint indices: a checkpoint must describe one exact boundary.
    """
    grouped = []
    index = 0
    while index < len(operations):
        operation = operations[index]
        if not isinstance(operation, (CreateZero, ResetZero)):
            grouped.append(operation)
            index += 1
            continue
        start, used = index, set()
        while index < len(operations):
            operation = operations[index]
            if isinstance(operation, (CreateZero, ResetZero)) and operation.qubit not in used:
                used.add(operation.qubit)
            elif not _is_barrier(operation):
                break
            index += 1
        grouped.append(_ContractionGroup(start, tuple(operations[start:index])))
    return tuple(grouped)


def _channel_circuit(circuit, state):
    operations = circuit.operations if isinstance(circuit, CircuitIR) else circuit
    if (not any(isinstance(op, CHANNEL_TYPES) for op in operations)
            and state.active_qubits is None and getattr(state, "_channel_checkpoints", None) is None):
        return None
    if isinstance(circuit, CircuitIR):
        return circuit
    if state.system_size is None:
        raise ValueError("Channel operation sequences require an SPO system_size or CircuitIR.")
    return CircuitIR(state.system_size, tuple(operations))


def _checkpoint_signature(circuit, backend, trunc_val, max_num_str):
    return (circuit, backend.name, backend.precision, trunc_val, max_num_str)


def _prepare_channel_observable(spo, circuit, backend):
    if spo.system_size is not None and spo.system_size != circuit.system_size:
        raise ValueError("Observable system_size does not match circuit.")
    source = copy(spo)
    source.set_active_qubits(circuit.system_size, spo.active_qubits)
    active, final = source.qubit_indices, circuit.final_active_qubits
    if set(active) != set(final) and spo.active_qubits is not None:
        raise ValueError("Observable active_qubits must match the circuit's final active outputs.")
    # The backend verifies width and rejects nonidentity on omitted columns.
    result = backend.reindex_spo(source, len(active), [active.index(q) for q in final])
    return result.set_active_qubits(circuit.system_size, final)


def _operation_in_columns(operation, active, *, backward=False):
    """Translate original integer indices to the current SPO column positions."""
    if isinstance(operation, PauliRotation):
        pauli = operation.pauli
        return replace(operation, pauli="".join(pauli[q] if q < len(pauli) else "I" for q in active))
    if isinstance(operation, SingleQubitClifford):
        return replace(operation, qubit=active.index(operation.qubit))
    if isinstance(operation, TwoQubitClifford):
        return replace(operation, control_qubit=active.index(operation.control_qubit),
                       target_qubit=active.index(operation.target_qubit))
    if isinstance(operation, Discard):
        column = active.index(operation.qubit) if backward else bisect_left(active, operation.qubit)
        return replace(operation, qubit=column)
    return operation


def _contraction_columns(group, active):
    columns = [active.index(op.qubit) for op in group.contractions]
    removed = [active.index(op.qubit) for op in group.contractions if isinstance(op, CreateZero)]
    return columns, removed


def _apply_channel_forward(state, operation, backend, checkpoints, trunc_val, max_num_str):
    active = state.qubit_indices
    if isinstance(operation, _ContractionGroup):
        checkpoints.save(operation.checkpoint_key, state)
        columns, removed = _contraction_columns(operation, active)
        result = backend.apply_zero_contractions_forward(
            state, columns, removed, preserve_zero_support=trunc_val == 0)
        active = [q for column, q in enumerate(active) if column not in removed]
    else:
        translated = _operation_in_columns(operation, active)
        result = backend.apply_forward(state, translated, trunc_val, max_num_str)
        if isinstance(operation, Discard):
            active.insert(translated.qubit, operation.qubit)
    result[0].set_active_qubits(state.system_size, active)
    return result


def _apply_channel_backward(state, operation, backend, checkpoints, trunc_val, max_num_str):
    if isinstance(operation, _ContractionGroup):
        original = checkpoints.take(operation.checkpoint_key)
        active = original.qubit_indices
        columns, removed = _contraction_columns(operation, active)
        result = backend.apply_zero_contractions_backward(state, original, columns, removed)
        # Only this boundary's complete SPO is held while applying its transpose.
        del original
    else:
        active = state.qubit_indices
        translated = _operation_in_columns(operation, active, backward=True)
        result = backend.apply_backward(state, translated, trunc_val, max_num_str)
        if isinstance(operation, Discard):
            active.remove(operation.qubit)
    result[0].set_active_qubits(state.system_size, active)
    return result


def close_channel_checkpoints(state):
    """Release an evaluation's checkpoints when backward is not needed."""
    checkpoints = getattr(state, "_channel_checkpoints", None)
    if checkpoints is not None:
        checkpoints.close()


def _zero_step_info():
    return {
        "num_str_truncated": 0,
        "truncated_l1_norm": 0.0,
        "truncated_l2_norm": 0.0,
    }


def _init_history():
    return {
        "num_str_truncated": [],
        "truncated_l1_norm": [],
        "truncated_l2_norm": [],
    }


def _append_step_info(history, step_info):
    history["num_str_truncated"].append(int(step_info["num_str_truncated"]))
    history["truncated_l1_norm"].append(float(step_info["truncated_l1_norm"]))
    history["truncated_l2_norm"].append(float(step_info["truncated_l2_norm"]))


def _finalize_info(history):
    total_l2_sq = sum(value * value for value in history["truncated_l2_norm"])
    return {
        "history": history,
        "num_steps_tracked": len(history["num_str_truncated"]),
        "sum_num_str_truncated": sum(history["num_str_truncated"]),
        "sum_truncated_l1_norm": sum(history["truncated_l1_norm"]),
        "sum_truncated_l2_norm": sum(history["truncated_l2_norm"]),
        "total_truncated_l2_norm": math.sqrt(total_l2_sq),
    }


def _normalize_max_num_str(backend_name, max_num_str):
    max_num_str = int(max_num_str)
    if max_num_str < 1:
        raise ValueError("max_num_str must be a positive integer.")
    if backend_name in ("jax", "triton"):
        return 1 if max_num_str == 1 else 1 << math.ceil(math.log2(max_num_str))
    return max_num_str


def _compute_padded_system_size(system_size, packbit):
    return packbit * ((system_size + packbit - 1) // packbit)


def _make_backend(backend_name, *, packbit=_PACKBIT, precision="single"):
    return BackendAdapter.from_name(backend_name, packbit=packbit, precision=precision)


def _default_backend_name():
    """Prefer the optional NVIDIA GPU backend; use NumPy on CPU."""
    try:
        import torch
    except ImportError:
        return "numpy"
    if torch.version.cuda is None or not torch.cuda.is_available():
        return "numpy"
    try:
        import triton  # noqa: F401 -- verify the optional runtime is importable
    except ImportError:
        return "numpy"
    return "triton"


def _resolve_backend_for_creation(backend_name, backend, *, precision="single"):
    if backend is None:
        return _make_backend(
            _default_backend_name() if backend_name is None else backend_name,
            packbit=_PACKBIT, precision=precision,
        )
    if not isinstance(backend, BackendAdapter):
        raise TypeError("backend must be a BackendAdapter when provided.")
    return backend


def _precision_from_dtype(dtype):
    if dtype is None:
        return None
    return "double" if np.dtype(dtype) == np.dtype(np.float64) else "single"


def _infer_backend_name_and_precision(state):
    # Recognize an already-loaded Triton state before importing JAX utilities.
    triton_module = sys.modules.get("spd.triton_backend")
    if triton_module is not None and isinstance(state, triton_module.SparsePauliOp):
        return "triton", state.precision

    from . import numpy_backend

    if isinstance(state, numpy_backend.SparsePauliOp):
        if len(state) == 0:
            return "numpy", numpy_backend.utils.get_precision()
        coeff = next(iter(state.values()))
        return "numpy", _precision_from_dtype(np.asarray(coeff).dtype)

    if isinstance(state, numpy_backend.SparsePauliGradientOp):
        if len(state) == 0:
            return "numpy", numpy_backend.utils.get_precision()
        coeff, _ = next(iter(state.values()))
        return "numpy", _precision_from_dtype(np.asarray(coeff).dtype)

    from . import jax_backend

    if isinstance(state, jax_backend.SparsePauliOp):
        return "jax", _precision_from_dtype(np.asarray(state.c_array).dtype)

    if isinstance(state, jax_backend.SparsePauliGradientOp):
        return "jax", _precision_from_dtype(np.asarray(state.c_array).dtype)

    raise TypeError(
        "state must be a backend-specific SparsePauliOp or SparsePauliGradientOp."
    )


def _resolve_backend_from_state(state, backend, *, state_name="state"):
    if backend is None:
        inferred_backend_name, inferred_precision = _infer_backend_name_and_precision(state)
        return _make_backend(
            inferred_backend_name,
            packbit=_PACKBIT,
            precision=inferred_precision or "single",
        )

    if not isinstance(backend, BackendAdapter):
        raise TypeError("backend must be a BackendAdapter when provided.")

    if not (backend.is_spo_instance(state) or backend.is_spgo_instance(state)):
        expected_kind = (
            "SparsePauliGradientOp" if "Gradient" in type(state).__name__ else "SparsePauliOp"
        )
        raise TypeError(
            f"{state_name} must be a {backend.name} {expected_kind} when backend='{backend.name}'."
        )

    return backend


def _setup_pytket_operations(circ, backend, rebase):
    """Prepare lowered operations for a pytket circuit."""
    from .pytket_frontend import maybe_rebase_pytket_circuit, parse_pytket_circuit

    if rebase:
        maybe_rebase_pytket_circuit(circ)

    padded_system_size = _compute_padded_system_size(circ.n_qubits, backend.packbit)
    return parse_pytket_circuit(circ, padded_system_size)


def _validate_ir_operations(input_circuit):
    if isinstance(input_circuit, (str, bytes)):
        raise TypeError(
            "input_circuit must be a pytket Circuit or a sequence of CircuitOperation objects."
        )
    if not isinstance(input_circuit, Sequence):
        raise TypeError(
            "input_circuit must be a pytket Circuit or a sequence of CircuitOperation objects."
        )

    operations = list(input_circuit)
    for index, operation in enumerate(operations):
        if not isinstance(operation, _IR_OPERATION_TYPES):
            raise TypeError(
                "input_circuit sequence elements must be CircuitOperation instances; "
                f"got {type(operation)!r} at index {index}."
            )
    return operations


def _normalize_input_circuit(input_circuit, backend, rebase):
    try:
        from pytket.circuit import Circuit as PytketCircuit
    except ImportError:
        PytketCircuit = None

    if PytketCircuit is not None and isinstance(input_circuit, PytketCircuit):
        return _setup_pytket_operations(input_circuit, backend, rebase)

    if rebase:
        raise ValueError("rebase=True is only supported when input_circuit is a pytket Circuit.")

    if isinstance(input_circuit, CircuitIR):
        return input_circuit

    return _validate_ir_operations(input_circuit)


def _print_progress(
    gate_name,
    weight_left,
    current_row_size,
    ose,
    command_idx,
    total_num_gate,
    step_time,
    total_start_time,
    end,
):
    """Print one progress line for a forward or backward execution step."""
    process = psutil.Process()
    print(
        gate_name,
        weight_left,
        process.memory_info().rss / 1e6,
        "MB",
        f"Size: {current_row_size}",
        f"OSE: {float(ose):.6f}" if ose is not None else "OSE: N/A",
        "Progress: {:.2f}%".format(100 * (command_idx + 1) / total_num_gate),
        "gates:",
        (command_idx + 1),
        "/",
        total_num_gate - (command_idx + 1),
        "M rows/s:",
        current_row_size / step_time / 1e6,
        "Time:",
        "{:.2f}".format(step_time),
        "Total Time:",
        "{:.2f}s".format(time.time() - total_start_time),
        "--------",
        end=end,
    )


def _run_operation_loop(operations, state, apply_fn, total_start_time, *, progress=True,
                        pruning_schedule=None):
    """Run operations with shared timing, size, weight, and progress reporting."""
    total_num_gate = len(operations)
    initial_weight = state.get_norm_square() if progress else None
    max_num_string = 0
    last_stats = None
    history = _init_history()

    execution = (enumerate(operations) if pruning_schedule is None else
                 ((total_num_gate - 1 - i, op)
                  for i, op in pruning_schedule.iter_operations(lambda: state)))
    for command_idx, operation in execution:
        t0 = time.time()
        state, num_string, extra, step_info = apply_fn(state, operation)
        if step_info is not None:
            # Grouping does not change operation-aligned truncation diagnostics.
            count = len(operation.contractions) if isinstance(operation, _ContractionGroup) else 1
            for _ in range(count):
                _append_step_info(history, step_info)
        if num_string is None:
            continue

        max_num_string = max(max_num_string, num_string)
        t1 = time.time()

        if not progress:
            continue

        current_weight = state.get_norm_square()
        weight_left = current_weight / initial_weight if initial_weight else 0.0
        current_row_size = state.get_size()
        ose = state.get_OSE() if current_weight else 0.0
        step_time = t1 - t0

        _print_progress(
            operation.gate_name,
            weight_left,
            current_row_size,
            ose,
            command_idx,
            total_num_gate,
            step_time,
            total_start_time,
            end="\r",
        )
        last_stats = {
            "gate_name": operation.gate_name,
            "weight_left": weight_left,
            "current_row_size": current_row_size,
            "ose": ose,
            "command_idx": command_idx,
            "step_time": step_time,
        }

    if last_stats is not None:
        _print_progress(
            last_stats["gate_name"],
            last_stats["weight_left"],
            last_stats["current_row_size"],
            last_stats["ose"],
            last_stats["command_idx"] if pruning_schedule is None else total_num_gate - 1,
            total_num_gate,
            last_stats["step_time"],
            total_start_time,
            end="\n",
        )

    return state, max_num_string, last_stats, _finalize_info(history)


def _save_state_pickle(state, prefix, trunc_val):
    import pickle

    with open(f"{prefix}_{trunc_val}.pickle", "wb") as f:
        pickle.dump(state, f)


def create_spo(
    data,
    *,
    system_size=None,
    active_qubits=None,
    backend_name=None,
    precision="single",
    backend=None,
):
    """Construct a backend-specific SparsePauliOp from simple user-facing data.

    With backend_name=None, prefer Triton when its optional runtime and NVIDIA
    CUDA are available, otherwise use NumPy. An explicit name or adapter takes
    precedence. Subsequent evolution infers the backend from the state.

    system_size is the fixed circuit index-space size. active_qubits[k] gives
    the index represented by Pauli column k; None means full canonical order,
    and [] permits scalar data such as {"": 1.0}. Compact string keys must have
    exactly len(active_qubits) columns. Measurement lists use original indices.
    """
    backend = _resolve_backend_for_creation(
        backend_name,
        backend,
        precision=precision,
    )

    if active_qubits is not None:
        if system_size is None:
            raise ValueError("system_size is required with active_qubits.")
        active_qubits = list(active_qubits)
        width = len(active_qubits)
    else:
        width = system_size
    if isinstance(data, list):
        if system_size is None:
            raise ValueError("system_size is required when data is a list of qubits.")
        active = list(range(system_size)) if active_qubits is None else active_qubits
        if any(q not in active for q in data):
            raise ValueError("Measurement indices must be active.")
        data = {''.join('Z' if q in data else 'I' for q in active): 1.0}
    if not isinstance(data, dict):
        raise ValueError("data must be a list of qubits or a string-key dict of Pauli coefficients.")
    if any(isinstance(p, tuple) for p in data):
        raise ValueError("Tuple-key measurement dicts are no longer supported.")
    if any(not isinstance(p, str) or set(p) - set('IXYZ') for p in data):
        raise ValueError("Observable keys must be I/X/Y/Z strings.")
    if width is None:
        width = max(map(len, data), default=0)
        if width == 0:
            raise ValueError("system_size is required for an empty or scalar observable.")
        system_size = width
    if active_qubits is None:
        if any(len(p) > width for p in data):
            raise ValueError("Pauli string width exceeds system_size.")
        # Preserve the legacy convention that omitted trailing indices are I.
        padded = {}
        for p, coeff in data.items():
            key = p.ljust(width, 'I')
            padded[key] = padded.get(key, 0.) + coeff
        data = padded
    elif any(len(p) != width for p in data):
        raise ValueError("Pauli string width must agree with the active register width.")
    result = backend.create_initial_spo(data, _compute_padded_system_size(width, backend.packbit))
    return result.set_active_qubits(system_size, active_qubits)


def evolve(
    spo,
    input_circuit,
    trunc_val,
    max_num_str,
    *,
    rebase=False,
    save_strings=False,
    backend=None,
    progress=True,
    in_place=False,
    pruning=None,
    checkpoint_directory=None,
    checkpoint_memory_budget_bytes=DEFAULT_CHECKPOINT_MEMORY_BUDGET_BYTES,
    checkpoint_device_memory_budget_bytes=DEFAULT_CHECKPOINT_DEVICE_MEMORY_BUDGET_BYTES,
):
    """Propagate an observable in reverse circuit order.

    All operations dispatch to the selected backend. Unsupported channel
    backends fail before execution, without CPU fallback. Independent creation/
    reset operations separated only by barriers share one complete checkpoint.
    Execution retains native snapshots within checkpoint_memory_budget_bytes
    (CPU, default 4 GiB) and checkpoint_device_memory_budget_bytes (per device,
    default 1 GiB). Oldest snapshots move device -> host -> disk as needed;
    serialization happens only on disk spill under checkpoint_directory. Zero
    bypasses retention in that tier. Budgets exclude live states and buffers.
    Backward consumes snapshots; expectation-only callers close them explicitly.
    Channels are exact; unitary segments retain existing SPD approximation rules.
    info["active_widths"] reports the initial width and each original reverse step.

    For legacy circuits, progress=False skips reporting reductions.

    pruning="light-cone" builds a cone from this input SPO once per call.
    pruning="light-cone-barrier" refreshes from the current SPO before each
    nonempty barrier-delimited block, in reverse circuit order. With no barriers
    it uses one plan. All blocks share one forward record for backward.
    One complete-circuit call scans only the initial SPO. Repeated calls scan
    each evolving SPO but can find tighter cones after truncation. Triton reduces
    support on GPU and transfers only the compact mask to the host.
    Omitted gates perform no truncation/capping; history retains original slots.
    The result carries its forward record through init_gradient_spo to backward.

    Truncation diagnostics/history are returned regardless of progress.
    in_place=True is supported only for Triton: mutate the supplied SPO and
    retain its storage across calls. Shared tensor buffers detach before mutation.
    An error after execution starts can leave earlier gates applied.
    """
    _validate_method(pruning)
    total_start_time = time.time()
    backend = _resolve_backend_from_state(spo, backend, state_name="spo")
    if not backend.is_spo_instance(spo):
        raise TypeError(
            f"spo must be a {backend.name} SparsePauliOp when backend='{backend.name}'."
        )

    if in_place and backend.name != "triton":
        raise NotImplementedError("in_place evolution is supported only for Triton")

    max_num_str = _normalize_max_num_str(backend.name, max_num_str)
    normalized_circuit = _normalize_input_circuit(input_circuit, backend, rebase)
    operations = (
        normalized_circuit.operations
        if isinstance(normalized_circuit, CircuitIR)
        else normalized_circuit
    )

    channel_circuit = _channel_circuit(normalized_circuit, spo)
    if channel_circuit is not None:
        backend.require_channel_support()
        if save_strings or pruning is not None or in_place:
            raise NotImplementedError("Channel/compact execution does not support save_strings, pruning, or in_place.")
        if not math.isfinite(trunc_val) or trunc_val < 0:
            raise ValueError("trunc_val must be finite and nonnegative.")

    plan = None
    schedule = None
    system_size = normalized_circuit.system_size if isinstance(normalized_circuit, CircuitIR) else None
    if pruning == "light-cone-barrier":
        schedule = _BarrierPruning(spo, operations, system_size, backend.name, trunc_val)
    elif pruning is not None:
        plan = _plan_forward(spo, operations, system_size, backend.name, trunc_val)
        operations = plan.retained_operations

    checkpoints = None
    active_widths = None
    if channel_circuit is not None:
        loop_state = _prepare_channel_observable(spo, channel_circuit, backend)
        operations = _group_contractions(channel_circuit.operations)
        checkpoints = CheckpointStore(
            _checkpoint_signature(channel_circuit, backend, trunc_val, max_num_str),
            checkpoint_directory, memory_budget_bytes=checkpoint_memory_budget_bytes,
            device_memory_budget_bytes=checkpoint_device_memory_budget_bytes,
            backend=backend.create_checkpoint_backend(loop_state))
        active_widths = [len(loop_state.qubit_indices)]

        def apply_forward(state, operation):
            result = _apply_channel_forward(state, operation, backend, checkpoints, trunc_val, max_num_str)
            members = operation.operations[::-1] if isinstance(operation, _ContractionGroup) else (operation,)
            for member in members:
                change = -1 if isinstance(member, CreateZero) else 1 if isinstance(member, Discard) else 0
                active_widths.append(active_widths[-1] + change)
            return result
    elif backend.name == "triton":
        loop_state = spo if in_place or all(isinstance(op, SkippedOperation) for op in operations) else spo.copy()
        apply_forward = lambda state, operation: state.apply_in_place(operation, trunc_val, max_num_str)
    else:
        loop_state = spo
        apply_forward = lambda state, operation: backend.apply_forward(
            state, operation, trunc_val=trunc_val, max_num_str=max_num_str,
        )

    try:
        final_spo, _, _, info = _run_operation_loop(
            operations[::-1], loop_state, apply_forward, total_start_time, progress=progress,
            pruning_schedule=schedule,
        )
    except BaseException:
        if checkpoints is not None:
            checkpoints.close()
        raise
    if backend.name == "triton" and not in_place and final_spo is not spo:
        final_spo = final_spo.compact()

    if schedule is not None:
        plan = schedule.record()
    if plan is not None:
        info = plan.align_info(info, reverse=True)
    if channel_circuit is None:
        spo._copy_metadata_to(final_spo)
    final_spo = _finish_record(final_spo, spo, plan, backend.name, in_place=in_place)
    if checkpoints is not None:
        final_spo._channel_checkpoints = checkpoints
        info["active_widths"] = active_widths

    if save_strings:
        _save_state_pickle(final_spo, "strings", trunc_val)

    return final_spo, info


def init_gradient_spo(
    final_spo,
    *,
    loss_type="basis_expectation",
    basis="0",
    target_spo=None,
    lambda_ose=0.0,
    alpha=1.0,
    backend=None,
):
    """Construct the initial backward SPGO for the requested terminal loss."""
    backend = _resolve_backend_from_state(final_spo, backend, state_name="final_spo")
    if not backend.is_spo_instance(final_spo):
        raise TypeError(
            f"final_spo must be a {backend.name} SparsePauliOp when backend='{backend.name}'."
        )
    if target_spo is not None and not backend.is_spo_instance(target_spo):
        raise TypeError(
            f"target_spo must be a {backend.name} SparsePauliOp when backend='{backend.name}'."
        )
    checkpoints = getattr(final_spo, "_channel_checkpoints", None)
    if checkpoints is not None and (loss_type != "basis_expectation" or lambda_ose != 0):
        raise NotImplementedError("Channel gradients currently support basis_expectation loss without OSE only.")
    result = backend.init_gradient_spo(
        final_spo,
        loss_type=loss_type,
        basis=basis,
        target_spo=target_spo,
        lambda_ose=lambda_ose,
        alpha=alpha,
    )

    final_spo._copy_metadata_to(result)
    result._pruning_record = getattr(final_spo, "_pruning_record", None)
    result._channel_checkpoints = checkpoints
    return result


def backpropagate(
    spgo,
    input_circuit,
    trunc_val,
    max_num_str,
    *,
    rebase=False,
    save_strings=False,
    backend=None,
    progress=True,
    in_place=False,
):
    """Propagate an SPGO backward; progress=False skips reporting reductions.

    Automatically follows the record carried from a pruned forward call by
    init_gradient_spo. The circuit must match that forward call, including angles.
    Omitted rotation gradients are zero in their original slots.

    Angle gradients and truncation diagnostics are always returned.
    in_place=True mutates the Triton SPGO and retains storage across calls.
    Shared buffers detach first; errors may leave earlier gates applied.
    """
    total_start_time = time.time()
    backend = _resolve_backend_from_state(spgo, backend, state_name="spgo")
    if not backend.is_spgo_instance(spgo):
        raise TypeError(
            f"spgo must be a {backend.name} SparsePauliGradientOp when backend='{backend.name}'."
        )

    if in_place and backend.name != "triton":
        raise NotImplementedError("in_place backward evolution is supported only for Triton")

    max_num_str = _normalize_max_num_str(backend.name, max_num_str)
    normalized_circuit = _normalize_input_circuit(input_circuit, backend, rebase)
    operations = (
        normalized_circuit.operations
        if isinstance(normalized_circuit, CircuitIR)
        else normalized_circuit
    )
    channel_circuit = _channel_circuit(normalized_circuit, spgo)
    checkpoints = getattr(spgo, "_channel_checkpoints", None)
    if channel_circuit is not None:
        backend.require_channel_support()
        if in_place or save_strings:
            raise NotImplementedError("Channel backward does not support in_place or save_strings.")
        if checkpoints is None or checkpoints.closed:
            raise ValueError("Channel backward requires live checkpoints from evolve/init_gradient_spo.")
        if checkpoints.signature != _checkpoint_signature(channel_circuit, backend, trunc_val, max_num_str):
            raise ValueError("Channel circuit, backend, precision, and approximation settings must match forward.")
        if spgo.system_size != channel_circuit.system_size or spgo.qubit_indices != channel_circuit.initial_active_qubits:
            raise ValueError("Gradient active_qubits must match the forward circuit inputs.")
        operations = _group_contractions(channel_circuit.operations)
    plan = getattr(spgo, "_pruning_record", None)
    if plan is not None:
        plan.check_circuit(operations, normalized_circuit.system_size if isinstance(normalized_circuit, CircuitIR) else None)
        if not math.isfinite(trunc_val) or trunc_val < 0:
            raise ValueError("trunc_val must be finite and nonnegative.")
        operations = plan.retained_operations
    grads = []

    if channel_circuit is not None:
        loop_state = spgo
        apply_backward = lambda state, operation: _apply_channel_backward(
            state, operation, backend, checkpoints, trunc_val, max_num_str)
    elif backend.name == "triton":
        loop_state = spgo if in_place or all(isinstance(op, SkippedOperation) for op in operations) else spgo.copy()
        apply_backward = lambda state, operation: state.apply_in_place(operation, trunc_val, max_num_str)
    else:
        loop_state = spgo
        apply_backward = lambda state, operation: backend.apply_backward(
            state, operation, trunc_val=trunc_val, max_num_str=max_num_str,
        )

    def _apply_backward(state, operation):
        next_state, num_string, grad_i, step_info = apply_backward(state, operation)
        if grad_i is not None:
            grads.append(grad_i)
        return next_state, num_string, grad_i, step_info

    try:
        final_spgo, _, _, info = _run_operation_loop(
            operations, loop_state, _apply_backward, total_start_time, progress=progress,
        )
    finally:
        if checkpoints is not None:
            checkpoints.close()

    if backend.name == "triton" and not in_place and final_spgo is not spgo:
        final_spgo.compact()

    if plan is not None:
        info = plan.align_info(info, reverse=False)
        grads = plan.align_gradients(grads)
    if channel_circuit is None:
        spgo._copy_metadata_to(final_spgo)
    final_spgo = _finish_record(final_spgo, spgo, None, backend.name, in_place=in_place)
    final_spgo._channel_checkpoints = None

    if save_strings:
        _save_state_pickle(final_spgo, "grad_strings", trunc_val)

    return final_spgo, grads, info


def backpropagate_noise_analysis(
    spgo,
    input_circuit,
    trunc_val,
    max_num_str,
    *,
    rebase=False,
    save_strings=False,
    backend=None,
    progress=True,
    in_place=False,
):
    """Backpropagate an SPGO and measure operation-aligned noise susceptibilities.

    in_place=True is Triton-only and retains mutable storage across calls.
    Shared buffers detach first; errors may leave earlier gates applied.
    """
    if getattr(spgo, "_pruning_record", None) is not None:
        raise NotImplementedError("Noise analysis of a pruned forward execution is not supported.")
    total_start_time = time.time()
    backend = _resolve_backend_from_state(spgo, backend, state_name="spgo")
    if not backend.is_spgo_instance(spgo):
        raise TypeError(
            f"spgo must be a {backend.name} SparsePauliGradientOp when backend='{backend.name}'."
        )

    if in_place and backend.name != "triton":
        raise NotImplementedError("in_place backward evolution is supported only for Triton")

    max_num_str = _normalize_max_num_str(backend.name, max_num_str)
    circuit_ir = _normalize_input_circuit(input_circuit, backend, rebase)
    if not isinstance(circuit_ir, CircuitIR):
        raise TypeError(
            "backpropagate_noise_analysis requires a pytket Circuit or CircuitIR "
            "with a physical system_size."
        )
    if _channel_circuit(circuit_ir, spgo) is not None:
        raise NotImplementedError("Noise analysis of channel/compact circuits is unsupported.")
    operations = circuit_ir.operations
    parameter_grads = []
    noise_grads = {
        "one_qubit_depolarizing": [],
        "two_qubit_depolarizing": [],
    }

    if backend.name == "triton":
        loop_state = spgo if in_place or all(isinstance(op, SkippedOperation) for op in operations) else spgo.copy()
        apply_backward = lambda state, operation: state.apply_in_place(operation, trunc_val, max_num_str)
    else:
        loop_state = spgo
        apply_backward = lambda state, operation: backend.apply_backward(
            state, operation, trunc_val=trunc_val, max_num_str=max_num_str,
        )

    def _apply_backward(state, operation):
        next_state, num_string, grad_i, step_info = apply_backward(state, operation)
        if grad_i is not None:
            parameter_grads.append(grad_i)

        active_qubits = get_operation_qubits(operation)
        if isinstance(operation, SkippedOperation):
            noise_grads["one_qubit_depolarizing"].append(0.0)
            noise_grads["two_qubit_depolarizing"].append(0.0)
            return next_state, num_string, grad_i, step_info

        if len(active_qubits) not in (1, 2):
            raise ValueError(
                "Noise analysis requires gates acting on one or two qubits. "
                "Compile the circuit to single- and two-qubit gates first."
            )

        if len(active_qubits) == 1:
            noise_grads["one_qubit_depolarizing"].append(
                backend.get_one_qubit_depolarizing_susceptibility(
                    next_state,
                    active_qubits[0],
                )
            )
            noise_grads["two_qubit_depolarizing"].append(0.0)
        else:
            noise_grads["one_qubit_depolarizing"].append(0.0)
            noise_grads["two_qubit_depolarizing"].append(
                backend.get_two_qubit_depolarizing_susceptibility(
                    next_state,
                    active_qubits,
                )
            )

        return next_state, num_string, grad_i, step_info

    final_spgo, _, _, info = _run_operation_loop(
        operations,
        loop_state,
        _apply_backward,
        total_start_time,
        progress=progress,
    )

    if backend.name == "triton" and not in_place and final_spgo is not spgo:
        final_spgo.compact()

    if save_strings:
        _save_state_pickle(final_spgo, "grad_strings", trunc_val)

    return final_spgo, parameter_grads, noise_grads, info
