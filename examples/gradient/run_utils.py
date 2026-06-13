import csv
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np


JAX_ALGORITHMS = ("stack_sort_merge", "search_update_merge")

EVAL_FIELDS = [
    "eval",
    "elapsed_s",
    "cost",
    "energy",
    "energy_error",
    "ose",
    "theta_norm",
    "grad_norm",
    "lambda_ose",
]
HISTORY_FIELDS = [
    "step",
    "elapsed_s",
    "cost",
    "energy",
    "energy_error",
    "ose",
    "theta_norm",
    "grad_norm",
    "lambda_ose",
]


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def _format_value(value):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}".replace(".", "p")
    return str(value).replace(".", "p")


def _format_size(size):
    if isinstance(size, (tuple, list)):
        return "x".join(str(int(v)) for v in size)
    return str(int(size))


def _format_init_mode(init_mode):
    if init_mode == "from_file":
        return "file"
    return str(init_mode)


def format_run_name(
    *,
    model,
    dim,
    size,
    num_params,
    method,
    init_mode,
    trunc_val,
    max_num_str,
    lambda_ose,
    g=None,
):
    parts = [
        f"{model}_{dim}d",
        f"L{_format_size(size)}",
        f"np{int(num_params)}",
    ]
    if g is not None:
        parts.append(f"g{_format_value(g)}")
    parts.extend(
        [
            method,
            f"init{_format_init_mode(init_mode)}",
            f"tv{_format_value(trunc_val)}",
            f"mns{_format_value(max_num_str)}",
            f"lo{_format_value(lambda_ose)}",
        ]
    )
    return "_".join(parts)


def add_common_args(
    parser,
    *,
    method="lbfgs",
    methods=("eval_only", "adam", "lbfgs", "basinhopping"),
    trunc_val,
    max_num_str,
    lambda_ose,
):
    parser.add_argument("--method", choices=methods, default=method)
    parser.add_argument("--algorithm", choices=JAX_ALGORITHMS, default="stack_sort_merge")
    parser.add_argument("--init-params-path", default=None)
    parser.add_argument("--random-scale", type=float, default=0.1)
    parser.add_argument("--trunc-val", type=float, default=trunc_val)
    parser.add_argument("--max-num-str", type=int, default=max_num_str)
    parser.add_argument("--lambda-ose", type=float, default=lambda_ose)


def infer_init_mode(init_params_path):
    if init_params_path is None:
        return "random"
    return "from_file"


def make_run_dir(run_name):
    run_dir = Path(run_name)
    run_dir.mkdir()
    return run_dir


def validate_method(method, niter=None):
    valid_methods = {"eval_only", "adam", "lbfgs", "basinhopping"}
    if method not in valid_methods:
        raise ValueError(f"Unsupported method={method}. Expected one of {sorted(valid_methods)}.")
    if method != "eval_only" and niter is not None and niter <= 0:
        raise ValueError("niter must be positive unless method='eval_only'.")


def start_timer():
    return time.perf_counter()


def elapsed_seconds(start_time):
    return float(time.perf_counter() - start_time)


def estimate_jax_memory_usage(system_size, max_num_str, *, packbit=32, precision="double"):
    if system_size <= 0:
        raise ValueError("system_size must be positive.")
    if max_num_str <= 0:
        raise ValueError("max_num_str must be positive.")
    if packbit <= 0:
        raise ValueError("packbit must be positive.")

    rounded_rows = 1 << (int(max_num_str) - 1).bit_length()
    words_per_row = math.ceil(system_size / packbit)
    pauli_bytes = 2 * words_per_row * (packbit // 8)
    coeff_bytes = 8 if precision == "double" else 4
    spo_bytes = rounded_rows * (pauli_bytes + coeff_bytes)
    spgo_bytes = rounded_rows * (pauli_bytes + 2 * coeff_bytes)
    return {
        "rounded_rows": rounded_rows,
        "spo_gib": spo_bytes / 1024**3,
        "spgo_gib": spgo_bytes / 1024**3,
    }


def print_jax_memory_estimate(system_size, max_num_str, *, packbit=32, precision="double"):
    estimate = estimate_jax_memory_usage(
        system_size,
        max_num_str,
        packbit=packbit,
        precision=precision,
    )
    print(
        "Estimated JAX storage at rounded max rows "
        f"{estimate['rounded_rows']}: "
        f"SPO ~{estimate['spo_gib']:.2f} GiB, "
        f"SPGO ~{estimate['spgo_gib']:.2f} GiB."
    )
    if estimate["spgo_gib"] >= 4.0:
        print(
            "Warning: large gradient runs can use much more memory than this "
            "stored-array estimate because of intermediates and allocator behavior."
        )
    return estimate


def init_thetas(
    *,
    num_params,
    init_mode,
    init_params_path=None,
    random_scale=0.1,
):
    random_init = (np.random.rand(num_params) - 0.5) * random_scale
    metadata = {
        "mode": init_mode,
        "path": init_params_path,
        "target_num_params": int(num_params),
        "random_scale": float(random_scale),
    }

    if init_mode == "random":
        metadata.update(
            {
                "source_num_params": None,
                "resize_rule": None,
                "num_copied": 0,
            }
        )
        return random_init, metadata

    if init_mode == "from_file":
        if init_params_path is None:
            raise ValueError("init_params_path is required when init_mode='from_file'.")
        loaded_thetas = np.atleast_1d(np.loadtxt(init_params_path))
        thetas = random_init.copy()
        num_copied = min(len(loaded_thetas), num_params)
        thetas[:num_copied] = loaded_thetas[:num_copied]
        metadata.update(
            {
                "source_num_params": int(len(loaded_thetas)),
                "resize_rule": "copy_prefix_random_tail",
                "num_copied": int(num_copied),
            }
        )
        return thetas, metadata

    raise ValueError("init_mode must be 'random' or 'from_file'.")


def record_eval(
    evals,
    last_eval,
    thetas,
    *,
    cost,
    energy,
    energy_error,
    ose,
    grad_norm,
    lambda_ose,
    run_dir=None,
    start_time=None,
):
    record = {
        "eval": len(evals),
        "elapsed_s": elapsed_seconds(start_time) if start_time is not None else None,
        "cost": float(cost),
        "energy": float(energy),
        "energy_error": float(energy_error),
        "ose": float(ose),
        "theta_norm": float(np.linalg.norm(thetas)),
        "grad_norm": float(grad_norm),
        "lambda_ose": float(lambda_ose),
    }
    evals.append(record)
    last_eval["thetas"] = np.array(thetas, copy=True)
    last_eval["record"] = record
    if run_dir is not None:
        _append_csv_row(Path(run_dir) / "evals.csv", record, EVAL_FIELDS)
    return record


def record_step(history, params_history, last_eval, thetas, run_dir=None, start_time=None):
    matched_record = {}
    if "thetas" in last_eval and np.allclose(thetas, last_eval["thetas"]):
        matched_record = last_eval.get("record", {})

    record = {
        "step": len(history),
        "elapsed_s": elapsed_seconds(start_time) if start_time is not None else None,
        "cost": matched_record.get("cost"),
        "energy": matched_record.get("energy"),
        "energy_error": matched_record.get("energy_error"),
        "ose": matched_record.get("ose"),
        "theta_norm": float(np.linalg.norm(thetas)),
        "grad_norm": matched_record.get("grad_norm"),
        "lambda_ose": matched_record.get("lambda_ose"),
    }
    params = np.array(thetas, copy=True)
    history.append(record)
    params_history.append(params)
    if run_dir is not None:
        run_dir = Path(run_dir)
        _append_csv_row(run_dir / "history.csv", record, HISTORY_FIELDS)
        _append_params_history_row(run_dir / "params_history.csv", record["step"], params)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_header(path, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()


def _append_csv_row(path, row, fields):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow(row)


def _params_history_fields(num_params):
    return ["step"] + [f"theta_{i}" for i in range(num_params)]


def _params_history_row(step, params):
    row = {"step": step}
    row.update({f"theta_{i}": float(value) for i, value in enumerate(params)})
    return row


def _append_params_history_row(path, step, params):
    fields = _params_history_fields(len(params))
    row = _params_history_row(step, params)
    _append_csv_row(path, row, fields)


def _write_params_history(path, params_history):
    max_params = max((len(params) for params in params_history), default=0)
    fields = _params_history_fields(max_params)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step, params in enumerate(params_history):
            writer.writerow(_params_history_row(step, params))


def init_run_outputs(run_dir, *, metadata, initial_params):
    run_dir = Path(run_dir)
    _write_json(run_dir / "metadata.json", metadata)
    np.savetxt(run_dir / "initial_params.txt", np.asarray(initial_params))
    _write_csv_header(run_dir / "evals.csv", EVAL_FIELDS)
    _write_csv_header(run_dir / "history.csv", HISTORY_FIELDS)
    _write_csv_header(
        run_dir / "params_history.csv",
        _params_history_fields(len(initial_params)),
    )


def make_final_summary(result, evals, final_params):
    last_record = evals[-1] if evals else {}
    return {
        "success": getattr(result, "success", None),
        "message": getattr(result, "message", None),
        "num_function_evals": getattr(result, "nfev", len(evals)),
        "num_iterations": getattr(result, "nit", None),
        "final_cost": getattr(result, "fun", last_record.get("cost")),
        "final_energy": last_record.get("energy"),
        "final_energy_error": last_record.get("energy_error"),
        "final_ose": last_record.get("ose"),
        "final_grad_norm": last_record.get("grad_norm"),
        "final_theta_norm": float(np.linalg.norm(final_params)),
    }


def save_run_outputs(
    run_dir,
    *,
    metadata,
    final,
    evals,
    history,
    params_history,
    initial_params,
    final_params,
    result=None,
):
    run_dir = Path(run_dir)
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "final.json", final)
    _write_csv(run_dir / "evals.csv", evals, EVAL_FIELDS)
    _write_csv(run_dir / "history.csv", history, HISTORY_FIELDS)
    _write_params_history(run_dir / "params_history.csv", params_history)
    np.savetxt(run_dir / "initial_params.txt", np.asarray(initial_params))
    np.savetxt(run_dir / "final_params.txt", np.asarray(final_params))
    if result is not None:
        with open(run_dir / "result.pkl", "wb") as f:
            pickle.dump(result, f)
