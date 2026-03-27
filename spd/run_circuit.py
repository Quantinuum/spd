"""Forward and backward runners for SPD circuit execution.

This module is the thin orchestration layer above the frontend parser and the
backend adapter. It prepares the run, delegates operation execution, and handles
progress/logging for forward and backward circuit evaluation.
"""

import math
import psutil, time, sys

from .backend_adapter import BackendAdapter
from .pytket_frontend import maybe_rebase_pytket_circuit, parse_pytket_circuit


# [TODO]: Merging the conj, merge into a single function


def _normalize_max_num_str(backend_name, max_num_str):
    max_num_str = int(max_num_str)
    if max_num_str < 1:
        raise ValueError("max_num_str must be a positive integer.")
    if backend_name == "jax":
        return 1 if max_num_str == 1 else 1 << math.ceil(math.log2(max_num_str))
    return max_num_str


def _setup_run(circ, backend_name, rebase, packbit=32, precision="single"):
    """Prepare backend, padded system size, and parsed operations."""
    backend = BackendAdapter.from_name(backend_name, packbit=packbit, precision=precision)

    if rebase:
        maybe_rebase_pytket_circuit(circ)

    original_system_size = circ.n_qubits
    padded_system_size = packbit * ((original_system_size + packbit - 1) // packbit)
    operations = parse_pytket_circuit(circ, padded_system_size)
    return backend, padded_system_size, operations


def _print_progress(
    gate_name,
    weight_left,
    current_row_size,
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

    for command_idx, operation in enumerate(operations):
        t0 = time.time()
        state, num_string, extra = apply_fn(state, operation)
        if num_string is None:
            continue

        max_num_string = max(max_num_string, num_string)
        t1 = time.time()

        current_weight = state.get_norm_square()
        weight_left = current_weight / initial_weight
        current_row_size = state.get_size()
        step_time = t1 - t0

        _print_progress(
            operation.gate_name,
            weight_left,
            current_row_size,
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
            "command_idx": command_idx,
            "step_time": step_time,
        }

    if last_stats is not None:
        _print_progress(
            last_stats["gate_name"],
            last_stats["weight_left"],
            last_stats["current_row_size"],
            last_stats["command_idx"],
            total_num_gate,
            last_stats["step_time"],
            total_start_time,
            end="\n",
        )

    return state, max_num_string, last_stats

def run_pytket_circuit(circ,
                       measure_qubits_data,
                       trunc_val,
                       max_num_str,
                       basis='0',
                       backend_name='numpy',
                       precision='single',
                       rebase=False,
                       log_filename=None,
                       save_strings=False,
                       ):
    """Run a static pytket circuit forward on the selected backend."""
    total_start_time = time.time()
    _PACKBIT = 32
    max_num_str = _normalize_max_num_str(backend_name, max_num_str)
    backend, padded_system_size, operations = _setup_run(
        circ, backend_name, rebase, packbit=_PACKBIT, precision=precision
    )

    sparse_pauli_op = backend.create_initial_spo(measure_qubits_data, padded_system_size)
    sparse_pauli_op, max_num_string, last_stats = _run_operation_loop(
        operations[::-1],
        sparse_pauli_op,
        lambda state, operation: (
            *backend.apply_forward(
                state,
                operation,
                trunc_val=trunc_val,
                max_num_str=max_num_str,
            ),
            None,
        ),
        total_start_time,
    )


    exp_val = sparse_pauli_op.get_expectation_value(basis=basis)

    if log_filename is not None:
        # We log trunc_val, final number of terms, max number string, final weight, total time, exp_val
        assert type(log_filename) == str
        current_row_size = last_stats["current_row_size"] if last_stats is not None else sparse_pauli_op.get_size()
        weight_left = last_stats["weight_left"] if last_stats is not None else 1.0
        with open(log_filename, "a") as f:
            f.write(f"{trunc_val}, {current_row_size}, {max_num_string}, {weight_left}, {time.time() - total_start_time}, {exp_val}\n")
    else:
        pass

    if save_strings:
        import pickle
        pickle.dump(sparse_pauli_op, open(f'strings_{trunc_val}.pickle','wb'))

    return exp_val, sparse_pauli_op

def run_pytket_circuit_backward(circ,
                                final_spo,
                                trunc_val,
                                max_num_str,
                                basis='0',
                                backend_name='numpy',
                                precision='single',
                                rebase=False,
                                log_filename=None,
                                save_strings=False,
                                ):
    """Run the backward pass for a static pytket circuit on the selected backend."""
    total_start_time = time.time()
    _PACKBIT = 32
    max_num_str = _normalize_max_num_str(backend_name, max_num_str)
    backend, padded_system_size, operations = _setup_run(
        circ, backend_name, rebase, packbit=_PACKBIT, precision=precision
    )

    spo_val_grad = backend.create_gradient_spo(final_spo, basis=basis)
    grads = []
    def _apply_backward(state, operation):
        next_state, num_string, grad_i = backend.apply_backward(
            state,
            operation,
            trunc_val=trunc_val,
            max_num_str=max_num_str,
        )
        if num_string is not None:
            grads.append(grad_i)
        return next_state, num_string, grad_i

    spo_val_grad, _, _ = _run_operation_loop(
        operations,
        spo_val_grad,
        _apply_backward,
        total_start_time,
    )

    return grads, spo_val_grad
