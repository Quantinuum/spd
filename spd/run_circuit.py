"""Forward and backward runners for SPD circuit execution.

This module is the thin orchestration layer above the frontend parser and the
backend adapter. It prepares the run, delegates operation execution, and handles
progress/logging for forward and backward circuit evaluation.
"""

import math
import psutil, time, sys

from .backend_adapter import BackendAdapter
from .openqasm_frontend import parse_openqasm_file, parse_openqasm_str


# [TODO]: Merging the conj, merge into a single function


def _normalize_max_num_str(backend_name, max_num_str):
    max_num_str = int(max_num_str)
    if max_num_str < 1:
        raise ValueError("max_num_str must be a positive integer.")
    if backend_name == "jax":
        return 1 if max_num_str == 1 else 1 << math.ceil(math.log2(max_num_str))
    return max_num_str


def _compute_padded_system_size(system_size, packbit):
    return packbit * ((system_size + packbit - 1) // packbit)


def _make_backend(backend_name, *, packbit=32, precision="single"):
    return BackendAdapter.from_name(backend_name, packbit=packbit, precision=precision)


def _resolve_backend(backend_name, backend, *, packbit=32, precision="single"):
    if backend is None:
        return _make_backend(backend_name, packbit=packbit, precision=precision)
    if not isinstance(backend, BackendAdapter):
        raise TypeError("backend must be a BackendAdapter when provided.")
    return backend


def _setup_pytket_run(circ, backend, rebase):
    """Prepare backend, padded system size, and parsed operations for pytket input."""
    from .pytket_frontend import maybe_rebase_pytket_circuit, parse_pytket_circuit

    if rebase:
        maybe_rebase_pytket_circuit(circ)

    original_system_size = circ.n_qubits
    padded_system_size = _compute_padded_system_size(original_system_size, backend.packbit)
    operations = parse_pytket_circuit(circ, padded_system_size)
    return backend, original_system_size, padded_system_size, operations


def _setup_openqasm_file_run(path, backend):
    """Prepare backend, padded system size, and parsed operations for an OpenQASM file."""
    with open(path, "r", encoding="utf-8") as f:
        system_size = _peek_openqasm_system_size(f.read())
    padded_system_size = _compute_padded_system_size(system_size, backend.packbit)
    _, operations = parse_openqasm_file(path, padded_system_size)
    return backend, system_size, padded_system_size, operations


def _setup_openqasm_str_run(source, backend):
    """Prepare backend, padded system size, and parsed operations for an OpenQASM string."""
    system_size = _peek_openqasm_system_size(source)
    padded_system_size = _compute_padded_system_size(system_size, backend.packbit)
    _, operations = parse_openqasm_str(source, padded_system_size)
    return backend, system_size, padded_system_size, operations


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

    return state, max_num_string, last_stats


def _run_forward_operations(
    backend,
    operations,
    *,
    measure_qubits_data,
    padded_system_size,
    trunc_val,
    max_num_str,
    basis,
    total_start_time,
    log_filename=None,
    save_strings=False,
):
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
        assert type(log_filename) == str
        current_row_size = last_stats["current_row_size"] if last_stats is not None else sparse_pauli_op.get_size()
        weight_left = last_stats["weight_left"] if last_stats is not None else 1.0
        with open(log_filename, "a") as f:
            f.write(f"{trunc_val}, {current_row_size}, {max_num_string}, {weight_left}, {time.time() - total_start_time}, {exp_val}\n")

    if save_strings:
        import pickle

        pickle.dump(sparse_pauli_op, open(f"strings_{trunc_val}.pickle", "wb"))

    return exp_val, sparse_pauli_op


def _run_backward_operations(
    backend,
    operations,
    *,
    initial_spgo,
    trunc_val,
    max_num_str,
    total_start_time,
    save_strings=False,
):
    if not backend.is_spgo_instance(initial_spgo):
        raise TypeError(
            f"initial_spgo must be a {backend.name} SparsePauliGradientOp when backend_name='{backend.name}'."
        )

    spo_val_grad = initial_spgo
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

    if save_strings:
        import pickle

        pickle.dump(spo_val_grad, open(f"grad_strings_{trunc_val}.pickle", "wb"))

    return grads, spo_val_grad


def _peek_openqasm_system_size(source):
    import re

    matches = re.findall(r"\bqreg\s+[A-Za-z_]\w*\[(\d+)\]\s*;", source)
    if not matches:
        raise ValueError("OpenQASM source must declare at least one qreg.")
    return sum(int(match) for match in matches)

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
                       backend=None,
                       ):
    """Run a static pytket circuit forward on the selected backend."""
    total_start_time = time.time()
    _PACKBIT = 32
    backend = _resolve_backend(
        backend_name,
        backend,
        packbit=_PACKBIT,
        precision=precision,
    )
    max_num_str = _normalize_max_num_str(backend.name, max_num_str)
    backend, _, padded_system_size, operations = _setup_pytket_run(
        circ, backend, rebase
    )
    return _run_forward_operations(
        backend,
        operations,
        measure_qubits_data=measure_qubits_data,
        padded_system_size=padded_system_size,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        basis=basis,
        total_start_time=total_start_time,
        log_filename=log_filename,
        save_strings=save_strings,
    )

def init_gradient_spo(final_spo,
                      *,
                      loss_type='basis_expectation',
                      basis='0',
                      target_spo=None,
                      lambda_ose=0.0,
                      alpha=1.0,
                      backend_name='numpy',
                      precision='single',
                      backend=None,
                      ):
    """Construct the initial backward SPGO for the requested terminal loss.

    This is the canonical public initializer for backward propagation.
    """
    backend = _resolve_backend(
        backend_name,
        backend,
        packbit=32,
        precision=precision,
    )
    if not backend.is_spo_instance(final_spo):
        raise TypeError(
            f"final_spo must be a {backend.name} SparsePauliOp when backend_name='{backend.name}'."
        )
    if target_spo is not None and not backend.is_spo_instance(target_spo):
        raise TypeError(
            f"target_spo must be a {backend.name} SparsePauliOp when backend_name='{backend.name}'."
        )
    return backend.init_gradient_spo(
        final_spo,
        loss_type=loss_type,
        basis=basis,
        target_spo=target_spo,
        lambda_ose=lambda_ose,
        alpha=alpha,
    )

def run_pytket_backward_from_spgo(circ,
                                  initial_spgo,
                                  trunc_val,
                                  max_num_str,
                                  backend_name='numpy',
                                  precision='single',
                                  rebase=False,
                                  save_strings=False,
                                  backend=None,
                                  ):
    """Propagate a pre-built gradient SPGO backward through a pytket circuit."""
    total_start_time = time.time()
    _PACKBIT = 32
    backend = _resolve_backend(
        backend_name,
        backend,
        packbit=_PACKBIT,
        precision=precision,
    )
    max_num_str = _normalize_max_num_str(backend.name, max_num_str)
    backend, _, _, operations = _setup_pytket_run(
        circ, backend, rebase
    )
    return _run_backward_operations(
        backend,
        operations,
        initial_spgo=initial_spgo,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        total_start_time=total_start_time,
        save_strings=save_strings,
    )

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
                                loss_type='basis_expectation',
                                target_spo=None,
                                lambda_ose=0.0,
                                alpha=1.0,
                                backend=None,
                                ):
    """Run backward propagation by initializing a terminal loss then propagating it."""
    initial_spgo = init_gradient_spo(
        final_spo,
        loss_type=loss_type,
        basis=basis,
        target_spo=target_spo,
        lambda_ose=lambda_ose,
        alpha=alpha,
        backend_name=backend_name,
        precision=precision,
        backend=backend,
    )
    return run_pytket_backward_from_spgo(
        circ,
        initial_spgo,
        trunc_val,
        max_num_str,
        backend_name=backend_name,
        precision=precision,
        rebase=rebase,
        save_strings=save_strings,
        backend=backend,
    )


def run_openqasm_file(path,
                      measure_qubits_data,
                      trunc_val,
                      max_num_str,
                      basis='0',
                      backend_name='numpy',
                      precision='single',
                      log_filename=None,
                      save_strings=False,
                      backend=None,
                      ):
    """Run a static OpenQASM 2 circuit from file on the selected backend."""
    total_start_time = time.time()
    _PACKBIT = 32
    backend = _resolve_backend(
        backend_name,
        backend,
        packbit=_PACKBIT,
        precision=precision,
    )
    max_num_str = _normalize_max_num_str(backend.name, max_num_str)
    backend, _, padded_system_size, operations = _setup_openqasm_file_run(
        path, backend
    )
    return _run_forward_operations(
        backend,
        operations,
        measure_qubits_data=measure_qubits_data,
        padded_system_size=padded_system_size,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        basis=basis,
        total_start_time=total_start_time,
        log_filename=log_filename,
        save_strings=save_strings,
    )


def run_openqasm_str(source,
                     measure_qubits_data,
                     trunc_val,
                     max_num_str,
                     basis='0',
                     backend_name='numpy',
                     precision='single',
                     log_filename=None,
                     save_strings=False,
                     backend=None,
                     ):
    """Run a static OpenQASM 2 circuit from source text on the selected backend."""
    total_start_time = time.time()
    _PACKBIT = 32
    backend = _resolve_backend(
        backend_name,
        backend,
        packbit=_PACKBIT,
        precision=precision,
    )
    max_num_str = _normalize_max_num_str(backend.name, max_num_str)
    backend, _, padded_system_size, operations = _setup_openqasm_str_run(
        source, backend
    )
    return _run_forward_operations(
        backend,
        operations,
        measure_qubits_data=measure_qubits_data,
        padded_system_size=padded_system_size,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        basis=basis,
        total_start_time=total_start_time,
        log_filename=log_filename,
        save_strings=save_strings,
    )


def run_openqasm_backward_from_spgo(path_or_source,
                                    initial_spgo,
                                    trunc_val,
                                    max_num_str,
                                    *,
                                    backend_name='numpy',
                                    precision='single',
                                    save_strings=False,
                                    from_file=True,
                                    backend=None,
                                    ):
    """Propagate a pre-built gradient SPGO backward through an OpenQASM 2 circuit."""
    total_start_time = time.time()
    _PACKBIT = 32
    backend = _resolve_backend(
        backend_name,
        backend,
        packbit=_PACKBIT,
        precision=precision,
    )
    max_num_str = _normalize_max_num_str(backend.name, max_num_str)
    if from_file:
        backend, _, _, operations = _setup_openqasm_file_run(
            path_or_source, backend
        )
    else:
        backend, _, _, operations = _setup_openqasm_str_run(
            path_or_source, backend
        )
    return _run_backward_operations(
        backend,
        operations,
        initial_spgo=initial_spgo,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        total_start_time=total_start_time,
        save_strings=save_strings,
    )


def run_openqasm_file_backward(path,
                               final_spo,
                               trunc_val,
                               max_num_str,
                               basis='0',
                               backend_name='numpy',
                               precision='single',
                               save_strings=False,
                               loss_type='basis_expectation',
                               target_spo=None,
                               lambda_ose=0.0,
                               alpha=1.0,
                               backend=None,
                               ):
    """Run backward propagation through an OpenQASM 2 file."""
    initial_spgo = init_gradient_spo(
        final_spo,
        loss_type=loss_type,
        basis=basis,
        target_spo=target_spo,
        lambda_ose=lambda_ose,
        alpha=alpha,
        backend_name=backend_name,
        precision=precision,
        backend=backend,
    )
    return run_openqasm_backward_from_spgo(
        path,
        initial_spgo,
        trunc_val,
        max_num_str,
        backend_name=backend_name,
        precision=precision,
        save_strings=save_strings,
        from_file=True,
        backend=backend,
    )


def run_openqasm_str_backward(source,
                              final_spo,
                              trunc_val,
                              max_num_str,
                              basis='0',
                              backend_name='numpy',
                              precision='single',
                              save_strings=False,
                              loss_type='basis_expectation',
                              target_spo=None,
                              lambda_ose=0.0,
                              alpha=1.0,
                              backend=None,
                              ):
    """Run backward propagation through an OpenQASM 2 source string."""
    initial_spgo = init_gradient_spo(
        final_spo,
        loss_type=loss_type,
        basis=basis,
        target_spo=target_spo,
        lambda_ose=lambda_ose,
        alpha=alpha,
        backend_name=backend_name,
        precision=precision,
        backend=backend,
    )
    return run_openqasm_backward_from_spgo(
        source,
        initial_spgo,
        trunc_val,
        max_num_str,
        backend_name=backend_name,
        precision=precision,
        save_strings=save_strings,
        from_file=False,
        backend=backend,
    )
