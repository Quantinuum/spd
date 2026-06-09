import csv
import json
import pickle
from pathlib import Path

import numpy as np


EVAL_FIELDS = [
    "eval",
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
):
    record = {
        "eval": len(evals),
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
    return record


def record_step(history, params_history, last_eval, thetas):
    matched_record = {}
    if "thetas" in last_eval and np.allclose(thetas, last_eval["thetas"]):
        matched_record = last_eval.get("record", {})

    history.append(
        {
            "step": len(history),
            "cost": matched_record.get("cost"),
            "energy": matched_record.get("energy"),
            "energy_error": matched_record.get("energy_error"),
            "ose": matched_record.get("ose"),
            "theta_norm": float(np.linalg.norm(thetas)),
            "grad_norm": matched_record.get("grad_norm"),
            "lambda_ose": matched_record.get("lambda_ose"),
        }
    )
    params_history.append(np.array(thetas, copy=True))


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_params_history(path, params_history):
    max_params = max((len(params) for params in params_history), default=0)
    fields = ["step"] + [f"theta_{i}" for i in range(max_params)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step, params in enumerate(params_history):
            row = {"step": step}
            row.update({f"theta_{i}": float(value) for i, value in enumerate(params)})
            writer.writerow(row)


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
