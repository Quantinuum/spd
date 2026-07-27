"""Execution helpers for SPD state propagation.

This module separates state evolution from expectation evaluation. Public entry
points accept an existing backend-specific SPO or SPGO together with either a
`pytket` circuit or a lowered SPD IR operation sequence.
"""

from collections.abc import Sequence
import math
import time

import numpy as np
import psutil

from .backend_adapter import BackendAdapter
from .circuit_ir import (
    CircuitIR,
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
)


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
    if backend_name == "jax":
        return 1 if max_num_str == 1 else 1 << math.ceil(math.log2(max_num_str))
    return max_num_str


def _compute_padded_system_size(system_size, packbit):
    return packbit * ((system_size + packbit - 1) // packbit)


def _make_backend(backend_name, *, packbit=_PACKBIT, precision="single"):
    return BackendAdapter.from_name(backend_name, packbit=packbit, precision=precision)


def _resolve_backend_for_creation(backend_name, backend, *, precision="single"):
    if backend is None:
        return _make_backend(backend_name, packbit=_PACKBIT, precision=precision)
    if not isinstance(backend, BackendAdapter):
        raise TypeError("backend must be a BackendAdapter when provided.")
    return backend


def _precision_from_dtype(dtype):
    if dtype is None:
        return None
    return "double" if np.dtype(dtype) == np.dtype(np.float64) else "single"


def _infer_backend_name_and_precision(state):
    from . import jax_backend, numpy_backend

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

    if isinstance(state, jax_backend.SparsePauliOp):
        return "jax", _precision_from_dtype(np.asarray(state.c_array).dtype)

    if isinstance(state, jax_backend.SparsePauliGradientOp):
        return "jax", _precision_from_dtype(np.asarray(state.c_array).dtype)

    raise TypeError(
        "state must be a backend-specific SparsePauliOp or SparsePauliGradientOp."
    )


def _resolve_backend_from_state(state, backend, *, state_name="state"):
    inferred_backend_name, inferred_precision = _infer_backend_name_and_precision(state)

    if backend is None:
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


def _run_operation_loop(operations, state, apply_fn, total_start_time):
    """Run operations with shared timing, size, weight, and progress reporting."""
    total_num_gate = len(operations)
    initial_weight = state.get_norm_square()
    max_num_string = 0
    last_stats = None
    history = _init_history()

    for command_idx, operation in enumerate(operations):
        t0 = time.time()
        state, num_string, extra, step_info = apply_fn(state, operation)
        if step_info is not None:
            _append_step_info(history, step_info)
        if num_string is None:
            continue

        max_num_string = max(max_num_string, num_string)
        t1 = time.time()

        current_weight = state.get_norm_square()
        weight_left = current_weight / initial_weight
        current_row_size = state.get_size()
        ose = state.get_OSE()
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
            last_stats["command_idx"],
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
    backend_name="numpy",
    precision="single",
    backend=None,
):
    """Construct a backend-specific SparsePauliOp from simple user-facing data."""
    backend = _resolve_backend_for_creation(
        backend_name,
        backend,
        precision=precision,
    )

    if isinstance(data, list):
        if system_size is None:
            raise ValueError("system_size is required when data is a list of qubits.")
        padded_system_size = _compute_padded_system_size(system_size, backend.packbit)
        return backend.create_initial_spo(data, padded_system_size)

    if isinstance(data, dict):
        return backend.create_initial_spo(data)

    raise ValueError("data must be a list of qubits or a string-key dict of Pauli coefficients.")


def evolve(
    spo,
    input_circuit,
    trunc_val,
    max_num_str,
    *,
    rebase=False,
    save_strings=False,
    backend=None,
):
    """Propagate a backend-specific SPO forward through a circuit."""
    total_start_time = time.time()
    backend = _resolve_backend_from_state(spo, backend, state_name="spo")
    if not backend.is_spo_instance(spo):
        raise TypeError(
            f"spo must be a {backend.name} SparsePauliOp when backend='{backend.name}'."
        )

    max_num_str = _normalize_max_num_str(backend.name, max_num_str)
    normalized_circuit = _normalize_input_circuit(input_circuit, backend, rebase)
    operations = (
        normalized_circuit.operations
        if isinstance(normalized_circuit, CircuitIR)
        else normalized_circuit
    )

    final_spo, _, _, info = _run_operation_loop(
        operations[::-1],
        spo,
        lambda state, operation: backend.apply_forward(
            state,
            operation,
            trunc_val=trunc_val,
            max_num_str=max_num_str,
        ),
        total_start_time,
    )

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
    return backend.init_gradient_spo(
        final_spo,
        loss_type=loss_type,
        basis=basis,
        target_spo=target_spo,
        lambda_ose=lambda_ose,
        alpha=alpha,
    )


def backpropagate(
    spgo,
    input_circuit,
    trunc_val,
    max_num_str,
    *,
    rebase=False,
    save_strings=False,
    backend=None,
):
    """Propagate a backend-specific SPGO backward through a circuit."""
    total_start_time = time.time()
    backend = _resolve_backend_from_state(spgo, backend, state_name="spgo")
    if not backend.is_spgo_instance(spgo):
        raise TypeError(
            f"spgo must be a {backend.name} SparsePauliGradientOp when backend='{backend.name}'."
        )

    max_num_str = _normalize_max_num_str(backend.name, max_num_str)
    normalized_circuit = _normalize_input_circuit(input_circuit, backend, rebase)
    operations = (
        normalized_circuit.operations
        if isinstance(normalized_circuit, CircuitIR)
        else normalized_circuit
    )
    grads = []

    def _apply_backward(state, operation):
        next_state, num_string, grad_i, step_info = backend.apply_backward(
            state,
            operation,
            trunc_val=trunc_val,
            max_num_str=max_num_str,
        )
        if grad_i is not None:
            grads.append(grad_i)
        return next_state, num_string, grad_i, step_info

    final_spgo, _, _, info = _run_operation_loop(
        operations,
        spgo,
        _apply_backward,
        total_start_time,
    )

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
):
    """Backpropagate an SPGO and measure operation-aligned noise susceptibilities."""
    total_start_time = time.time()
    backend = _resolve_backend_from_state(spgo, backend, state_name="spgo")
    if not backend.is_spgo_instance(spgo):
        raise TypeError(
            f"spgo must be a {backend.name} SparsePauliGradientOp when backend='{backend.name}'."
        )

    max_num_str = _normalize_max_num_str(backend.name, max_num_str)
    circuit_ir = _normalize_input_circuit(input_circuit, backend, rebase)
    if not isinstance(circuit_ir, CircuitIR):
        raise TypeError(
            "backpropagate_noise_analysis requires a pytket Circuit or CircuitIR "
            "with a physical system_size."
        )
    operations = circuit_ir.operations
    parameter_grads = []
    noise_grads = {
        "one_qubit_depolarizing": [],
        "two_qubit_depolarizing": [],
    }

    def _apply_backward(state, operation):
        next_state, num_string, grad_i, step_info = backend.apply_backward(
            state,
            operation,
            trunc_val=trunc_val,
            max_num_str=max_num_str,
        )
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
        spgo,
        _apply_backward,
        total_start_time,
    )

    if save_strings:
        _save_state_pickle(final_spgo, "grad_strings", trunc_val)

    return final_spgo, parameter_grads, noise_grads, info
