"""Shared helpers for exploratory time-evolution compression examples."""

from __future__ import annotations

import json
import math
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from pytket import Circuit

import spd


EXAMPLE_DIR = Path(__file__).resolve().parent

CONSTANT_L_X = 10
CONSTANT_L_Y = 10
CONSTANT_SYSTEM_SIZE = CONSTANT_L_X * CONSTANT_L_Y

RAMP_L_X = 14
RAMP_L_Y = 14
RAMP_SYSTEM_SIZE = RAMP_L_X * RAMP_L_Y

TOTAL_TIME = 0.3
J_COUPLING = -1.0
G_FIELD = -3.1

TARGET_STEPS = 100
ANSATZ_STEPS = 5

BACKEND_NAME = "jax"
BACKEND_ALGORITHM = "search_update_merge"
PRECISION = "single"
TARGET_TROTTER_ORDER = 1
TARGET_TRUNC_VAL = 5e-4
TARGET_MAX_NUM_STR = int(1e7)
ANSATZ_TROTTER_ORDER = 1
TRUNC_VAL = 5e-4
MAX_NUM_STR = int(1e7)

OPTIMIZER_METHOD = "L-BFGS-B"
OPTIMIZER_OPTIONS = {
    "maxiter": 10,
    "gtol": 1e-5,
    "ftol": 1e-9,
    "maxls": 20,
}

GRAD_CHECK_EPS = 1e-4
GRAD_CHECK_INDICES = (0, 1)
GRAD_CHECK_RTOL = 1.5e-1
GRAD_CHECK_ATOL = 2e-1

TARGET_X_PATH = EXAMPLE_DIR / "target_spo_x.pkl"
TARGET_Z_PATH = EXAMPLE_DIR / "target_spo_z.pkl"
TARGET_METADATA_PATH = EXAMPLE_DIR / "target_metadata.json"
TARGET_DATA_DIR = EXAMPLE_DIR / f"target_tfi_2d_{CONSTANT_L_X}x{CONSTANT_L_Y}"

OPTIMIZATION_RESULT_PATH = EXAMPLE_DIR / "optimization_result.pkl"
OPTIMIZATION_HISTORY_NPY_PATH = EXAMPLE_DIR / "optimization_history.npy"
OPTIMIZATION_HISTORY_TXT_PATH = EXAMPLE_DIR / "optimization_history.txt"

RAMP_TOTAL_TIME = 0.06
RAMP_TARGET_STEPS = 6
RAMP_ANSATZ_STEPS = 2
RAMP_G_START = 0.0
RAMP_G_END = 3.2

RAMP_TARGET_X_PATH = EXAMPLE_DIR / "ramp_target_spo_x.pkl"
RAMP_TARGET_Z_PATH = EXAMPLE_DIR / "ramp_target_spo_z.pkl"
RAMP_TARGET_METADATA_PATH = EXAMPLE_DIR / "ramp_target_metadata.json"

RAMP_OPTIMIZATION_RESULT_PATH = EXAMPLE_DIR / "ramp_optimization_result.pkl"
RAMP_OPTIMIZATION_HISTORY_NPY_PATH = EXAMPLE_DIR / "ramp_optimization_history.npy"
RAMP_OPTIMIZATION_HISTORY_TXT_PATH = EXAMPLE_DIR / "ramp_optimization_history.txt"


def build_backend():
    """Construct a reusable JAX backend for the exploratory workflow."""
    backend = spd.BackendAdapter.from_name(
        BACKEND_NAME,
        packbit=32,
        precision=PRECISION,
    )
    backend.module.set_algorithm(BACKEND_ALGORITHM)
    return backend


def _rotation_parameter(strength: float, dt: float) -> float:
    """Convert exp(-i * dt * strength * P) into pytket's rotation parameter."""
    return dt * strength / (math.pi / 2.0)


def trotter_layer_parameters(
    num_steps: int,
    total_time: float = TOTAL_TIME,
    *,
    j_coupling: float = J_COUPLING,
    g_field: float = G_FIELD,
) -> np.ndarray:
    dt = total_time / num_steps
    theta_zz = _rotation_parameter(j_coupling, dt)
    theta_x = _rotation_parameter(g_field, dt)
    return np.tile(np.asarray([theta_zz, theta_x], dtype=float), num_steps)


def step_count_from_dt(total_time: float, dt: float) -> int:
    """Return the integer step count implied by dt."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    num_steps_float = total_time / dt
    num_steps = round(num_steps_float)
    if not math.isclose(num_steps_float, num_steps, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"dt={dt} does not divide total_time={total_time}.")
    if num_steps < 1:
        raise ValueError("dt is too large for the requested total_time.")
    return num_steps


def _path_float(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def target_data_dir(
    *,
    total_time: float,
    dt: float,
    trotter_order: int,
    trunc_val: float,
    root: Path = TARGET_DATA_DIR,
) -> Path:
    """Build a stable directory for a target precompute run."""
    return (
        root
        / f"T_{_path_float(total_time)}"
        / f"dt_{_path_float(dt)}_order_{trotter_order}_trunc_{_path_float(trunc_val)}"
    )


def target_file_paths(target_dir: Path) -> dict[str, Path]:
    return {
        "target_x": target_dir / "target_spo_x.pkl",
        "target_z": target_dir / "target_spo_z.pkl",
        "metadata": target_dir / "target_metadata.json",
        "x_info": target_dir / "x_info.json",
        "z_info": target_dir / "z_info.json",
    }


def optimization_output_paths(
    *,
    num_steps: int,
    trotter_order: int,
    root: Path = EXAMPLE_DIR,
) -> dict[str, Path]:
    stem = f"optimization_order_{trotter_order}_steps_{num_steps}"
    return {
        "result": root / f"{stem}.pkl",
        "history_npy": root / f"{stem}.npy",
        "history_txt": root / f"{stem}.txt",
    }


def initial_ansatz_parameters(
    *,
    num_steps: int = ANSATZ_STEPS,
    total_time: float = TOTAL_TIME,
    j_coupling: float = J_COUPLING,
    g_field: float = G_FIELD,
    trotter_order: int = ANSATZ_TROTTER_ORDER,
) -> np.ndarray:
    """Use coarse Trotter parameters as the deterministic initialization."""
    trotter_params = trotter_layer_parameters(
        num_steps,
        total_time=total_time,
        j_coupling=j_coupling,
        g_field=g_field,
    )
    if trotter_order == 1:
        return trotter_params
    if trotter_order == 2:
        theta_zz = float(trotter_params[0])
        theta_x = float(trotter_params[1])
        x_layers = [theta_x / 2.0]
        x_layers.extend([theta_x] * max(0, num_steps - 1))
        x_layers.append(theta_x / 2.0)

        params = []
        for layer in range(num_steps):
            params.extend([x_layers[layer], theta_zz])
        params.append(x_layers[-1])
        return np.asarray(params, dtype=float)
    raise NotImplementedError(f"Ansatz initialization is not implemented for trotter_order={trotter_order}.")


def ansatz_parameter_count(
    num_steps: int,
    *,
    trotter_order: int = ANSATZ_TROTTER_ORDER,
) -> int:
    if trotter_order == 1:
        return 2 * num_steps
    if trotter_order == 2:
        return 2 * num_steps + 1
    raise NotImplementedError(f"Ansatz parameter count is not implemented for trotter_order={trotter_order}.")


def linear_ramp_first_order_parameters(
    num_steps: int,
    total_time: float,
    *,
    g_start: float,
    g_end: float,
    j_coupling: float = J_COUPLING,
) -> np.ndarray:
    """Build a coarse first-order Trotter parameter vector for a linear g ramp."""
    dt = total_time / num_steps
    params = []
    theta_zz = _rotation_parameter(j_coupling, dt)
    for step in range(num_steps):
        midpoint_fraction = (step + 0.5) / num_steps
        g_mid = g_start + (g_end - g_start) * midpoint_fraction
        theta_x = _rotation_parameter(g_mid, dt)
        params.extend([theta_zz, theta_x])
    return np.asarray(params, dtype=float)


def build_2d_tfi_circuit(
    thetas: np.ndarray | list[float],
    *,
    system_size_x: int = CONSTANT_L_X,
    system_size_y: int = CONSTANT_L_Y,
    trotter_order: int = 1,
) -> Circuit:
    """Build a translationally invariant 2D TFI Trotter circuit."""
    thetas = np.asarray(thetas, dtype=float)
    if thetas.ndim != 1 or thetas.size % 2 != 0:
        raise ValueError("thetas must be a flat array with alternating ZZ and X parameters.")
    if trotter_order not in (1, 2, 4):
        raise ValueError("trotter_order must be 1, 2, or 4.")

    system_size = system_size_x * system_size_y
    circ = Circuit(system_size, system_size)
    depth = thetas.size // 2

    def add_zz_layer(theta_zz: float) -> None:
        for x in range(system_size_x):
            for y in range(system_size_y):
                i = x * system_size_y + y
                j = ((x + 1) % system_size_x) * system_size_y + y
                circ.ZZPhase(theta_zz, i, j)

        for x in range(system_size_x):
            for y in range(system_size_y):
                i = x * system_size_y + y
                j = x * system_size_y + (y + 1) % system_size_y
                circ.ZZPhase(theta_zz, i, j)

        circ.add_barrier(list(range(system_size)))

    def add_x_layer(theta_x: float) -> None:
        for qubit in range(system_size):
            circ.Rx(theta_x, qubit)
        circ.add_barrier(list(range(system_size)))

    def emit_merged_x_zz_terms(terms: list[tuple[str, float]]) -> None:
        pending_x = 0.0
        for term_type, theta in terms:
            if term_type == "x":
                pending_x += theta
            else:
                if pending_x != 0.0:
                    add_x_layer(pending_x)
                    pending_x = 0.0
                add_zz_layer(theta)
        if pending_x != 0.0:
            add_x_layer(pending_x)

    def append_second_order_terms(
        terms: list[tuple[str, float]],
        theta_zz: float,
        theta_x: float,
        coefficient: float = 1.0,
    ) -> None:
        terms.extend(
            [
                ("x", coefficient * theta_x / 2.0),
                ("zz", coefficient * theta_zz),
                ("x", coefficient * theta_x / 2.0),
            ]
        )

    if trotter_order == 1:
        for layer in range(depth):
            theta_zz = float(thetas[2 * layer])
            theta_x = float(thetas[2 * layer + 1])
            add_zz_layer(theta_zz)
            add_x_layer(theta_x)
    elif trotter_order == 2:
        terms = []
        for layer in range(depth):
            theta_zz = float(thetas[2 * layer])
            theta_x = float(thetas[2 * layer + 1])
            append_second_order_terms(terms, theta_zz, theta_x)
        emit_merged_x_zz_terms(terms)
    else:
        p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        coefficients = (p, p, 1.0 - 4.0 * p, p, p)
        terms = []
        for layer in range(depth):
            theta_zz = float(thetas[2 * layer])
            theta_x = float(thetas[2 * layer + 1])
            for coefficient in coefficients:
                append_second_order_terms(terms, theta_zz, theta_x, coefficient)
        emit_merged_x_zz_terms(terms)

    for qubit in range(system_size):
        circ.Measure(qubit, qubit)

    return circ


def build_2d_tfi_ansatz_circuit(
    thetas: np.ndarray | list[float],
    *,
    system_size_x: int = CONSTANT_L_X,
    system_size_y: int = CONSTANT_L_Y,
    trotter_order: int = ANSATZ_TROTTER_ORDER,
) -> Circuit:
    """Build the variational 2D TFI ansatz circuit used by the optimizer."""
    thetas = np.asarray(thetas, dtype=float)
    system_size = system_size_x * system_size_y
    circ = Circuit(system_size, system_size)

    def add_zz_layer(theta_zz: float) -> None:
        for x in range(system_size_x):
            for y in range(system_size_y):
                i = x * system_size_y + y
                j = ((x + 1) % system_size_x) * system_size_y + y
                circ.ZZPhase(theta_zz, i, j)

        for x in range(system_size_x):
            for y in range(system_size_y):
                i = x * system_size_y + y
                j = x * system_size_y + (y + 1) % system_size_y
                circ.ZZPhase(theta_zz, i, j)

        circ.add_barrier(list(range(system_size)))

    def add_x_layer(theta_x: float) -> None:
        for qubit in range(system_size):
            circ.Rx(theta_x, qubit)
        circ.add_barrier(list(range(system_size)))

    if trotter_order == 1:
        if thetas.ndim != 1 or thetas.size % 2 != 0:
            raise ValueError("First-order ansatz expects alternating ZZ/X parameters.")
        depth = thetas.size // 2
        for layer in range(depth):
            add_zz_layer(float(thetas[2 * layer]))
            add_x_layer(float(thetas[2 * layer + 1]))
    elif trotter_order == 2:
        if thetas.ndim != 1 or thetas.size < 3 or thetas.size % 2 == 0:
            raise ValueError("Second-order ansatz expects X/ZZ/.../X parameters of odd length.")
        num_steps = (thetas.size - 1) // 2
        x_layers = thetas[0::2]
        zz_layers = thetas[1::2]
        if x_layers.size != num_steps + 1 or zz_layers.size != num_steps:
            raise ValueError("Invalid second-order ansatz parameter layout.")
        add_x_layer(float(x_layers[0]))
        for layer in range(num_steps):
            add_zz_layer(float(zz_layers[layer]))
            add_x_layer(float(x_layers[layer + 1]))
    else:
        raise NotImplementedError(f"Ansatz circuit is not implemented for trotter_order={trotter_order}.")

    for qubit in range(system_size):
        circ.Measure(qubit, qubit)

    return circ


def build_second_order_linear_ramp_tfi_circuit(
    *,
    num_steps: int = RAMP_TARGET_STEPS,
    total_time: float = RAMP_TOTAL_TIME,
    g_start: float = RAMP_G_START,
    g_end: float = RAMP_G_END,
    j_coupling: float = J_COUPLING,
    system_size_x: int = RAMP_L_X,
    system_size_y: int = RAMP_L_Y,
) -> Circuit:
    """Build a second-order 2D TFI Trotter circuit with a linear ramp in g."""
    system_size = system_size_x * system_size_y
    circ = Circuit(system_size, system_size)
    dt = total_time / num_steps
    theta_zz_half = _rotation_parameter(j_coupling, dt / 2.0)

    for step in range(num_steps):
        midpoint_fraction = (step + 0.5) / num_steps
        g_mid = g_start + (g_end - g_start) * midpoint_fraction
        theta_x = _rotation_parameter(g_mid, dt)

        for _ in range(2):
            for x in range(system_size_x):
                for y in range(system_size_y):
                    i = x * system_size_y + y
                    j = ((x + 1) % system_size_x) * system_size_y + y
                    circ.ZZPhase(theta_zz_half, i, j)

            for x in range(system_size_x):
                for y in range(system_size_y):
                    i = x * system_size_y + y
                    j = x * system_size_y + (y + 1) % system_size_y
                    circ.ZZPhase(theta_zz_half, i, j)

            circ.add_barrier(list(range(system_size)))
            if _ == 0:
                for qubit in range(system_size):
                    circ.Rx(theta_x, qubit)
                circ.add_barrier(list(range(system_size)))

    for qubit in range(system_size):
        circ.Measure(qubit, qubit)

    return circ


def representative_x_dict(system_size: int, site: int = 0) -> dict[str, float]:
    pauli = ["I"] * system_size
    pauli[site] = "X"
    return {"".join(pauli): 1.0}


def representative_z_dict(system_size: int, site: int = 0) -> dict[str, float]:
    pauli = ["I"] * system_size
    pauli[site] = "Z"
    return {"".join(pauli): 1.0}


def target_metadata(
    *,
    total_time: float = TOTAL_TIME,
    target_steps: int = TARGET_STEPS,
    trotter_order: int = TARGET_TROTTER_ORDER,
    trunc_val: float = TARGET_TRUNC_VAL,
    max_num_str: int = TARGET_MAX_NUM_STR,
) -> dict[str, object]:
    return {
        "system_size_x": CONSTANT_L_X,
        "system_size_y": CONSTANT_L_Y,
        "system_size": CONSTANT_SYSTEM_SIZE,
        "total_time": total_time,
        "dt": total_time / target_steps,
        "target_steps": target_steps,
        "ansatz_steps": ANSATZ_STEPS,
        "j_coupling": J_COUPLING,
        "g_field": G_FIELD,
        "backend_name": BACKEND_NAME,
        "backend_algorithm": BACKEND_ALGORITHM,
        "precision": PRECISION,
        "trunc_val": trunc_val,
        "max_num_str": max_num_str,
        "trotter_order": trotter_order,
        "target_style": f"{trotter_order}_order_trotter",
    }


def ramp_target_metadata() -> dict[str, object]:
    return {
        "system_size_x": RAMP_L_X,
        "system_size_y": RAMP_L_Y,
        "system_size": RAMP_SYSTEM_SIZE,
        "total_time": RAMP_TOTAL_TIME,
        "target_steps": RAMP_TARGET_STEPS,
        "ansatz_steps": RAMP_ANSATZ_STEPS,
        "j_coupling": J_COUPLING,
        "g_start": RAMP_G_START,
        "g_end": RAMP_G_END,
        "backend_name": BACKEND_NAME,
        "backend_algorithm": BACKEND_ALGORITHM,
        "precision": PRECISION,
        "trunc_val": TRUNC_VAL,
        "max_num_str": MAX_NUM_STR,
        "target_style": "linear_ramp_second_order_trotter",
        "ansatz_style": "first_order_trotter",
    }


def save_metadata(path: Path, metadata: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def load_metadata(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_metadata(metadata: dict[str, object]) -> None:
    expected = target_metadata()
    mismatches = []
    for key in ("system_size_x", "system_size_y", "total_time", "target_steps", "backend_name", "trunc_val", "max_num_str"):
        if metadata.get(key) != expected[key]:
            mismatches.append(f"{key}: expected {expected[key]!r}, got {metadata.get(key)!r}")

    if mismatches:
        mismatch_str = "\n".join(mismatches)
        raise ValueError(f"Target metadata mismatch:\n{mismatch_str}")


def validate_metadata_against_expected(metadata: dict[str, object], expected: dict[str, object]) -> None:
    mismatches = []
    for key in (
        "system_size_x",
        "system_size_y",
        "total_time",
        "target_steps",
        "backend_name",
        "trunc_val",
        "max_num_str",
    ):
        if metadata.get(key) != expected[key]:
            mismatches.append(f"{key}: expected {expected[key]!r}, got {metadata.get(key)!r}")

    if mismatches:
        mismatch_str = "\n".join(mismatches)
        raise ValueError(f"Target metadata mismatch:\n{mismatch_str}")


def validate_target_metadata_for_optimization(
    metadata: dict[str, object],
    *,
    expected_total_time: float | None = None,
    expected_system_size_x: int | None = None,
    expected_system_size_y: int | None = None,
) -> None:
    required_keys = (
        "system_size_x",
        "system_size_y",
        "system_size",
        "total_time",
        "backend_name",
    )
    missing = [key for key in required_keys if key not in metadata]
    if missing:
        raise ValueError(f"Target metadata is missing required keys: {missing}")

    mismatches = []
    if expected_total_time is not None and metadata.get("total_time") != expected_total_time:
        mismatches.append(
            f"total_time: expected {expected_total_time!r}, got {metadata.get('total_time')!r}"
        )
    if expected_system_size_x is not None and metadata.get("system_size_x") != expected_system_size_x:
        mismatches.append(
            f"system_size_x: expected {expected_system_size_x!r}, got {metadata.get('system_size_x')!r}"
        )
    if expected_system_size_y is not None and metadata.get("system_size_y") != expected_system_size_y:
        mismatches.append(
            f"system_size_y: expected {expected_system_size_y!r}, got {metadata.get('system_size_y')!r}"
        )

    if mismatches:
        raise ValueError("Target metadata mismatch:\n" + "\n".join(mismatches))


def expected_raw_gradient_count(
    system_size: int,
    num_steps: int = ANSATZ_STEPS,
    trotter_order: int = ANSATZ_TROTTER_ORDER,
) -> int:
    if trotter_order == 1:
        return num_steps * (3 * system_size)
    if trotter_order == 2:
        return num_steps * (3 * system_size) + system_size
    raise NotImplementedError(f"Gradient compression is not implemented for trotter_order={trotter_order}.")


def compress_tfi_gradients(
    raw_grads,
    *,
    system_size: int,
    num_steps: int = ANSATZ_STEPS,
    trotter_order: int = ANSATZ_TROTTER_ORDER,
) -> np.ndarray:
    """Reduce sitewise SPD gradients into shared translationally invariant layer parameters."""
    raw = np.asarray(raw_grads, dtype=float)
    expected_size = expected_raw_gradient_count(
        system_size,
        num_steps,
        trotter_order=trotter_order,
    )
    if raw.shape != (expected_size,):
        raise ValueError(
            f"Expected {expected_size} raw gradients for {num_steps} layers, got shape {raw.shape}."
        )

    layer_gradients = np.zeros(2 * num_steps, dtype=float)
    zz_terms_per_layer = 2 * system_size

    if trotter_order == 1:
        layer_span = 3 * system_size
        for layer in range(num_steps):
            start = layer * layer_span
            zz_slice = raw[start : start + zz_terms_per_layer]
            x_slice = raw[start + zz_terms_per_layer : start + layer_span]
            layer_gradients[2 * layer] = float(np.sum(zz_slice) * math.pi)
            layer_gradients[2 * layer + 1] = float(np.sum(x_slice) * math.pi)
        return layer_gradients

    if trotter_order == 2:
        cursor = 0
        x_layer_sums = [float(np.sum(raw[cursor : cursor + system_size]))]
        cursor += system_size
        zz_layer_sums = []
        for _ in range(num_steps):
            zz_layer_sums.append(float(np.sum(raw[cursor : cursor + zz_terms_per_layer])))
            cursor += zz_terms_per_layer
            x_layer_sums.append(float(np.sum(raw[cursor : cursor + system_size])))
            cursor += system_size

        grad_params = np.zeros(2 * num_steps + 1, dtype=float)
        for layer in range(num_steps):
            grad_params[2 * layer] = x_layer_sums[layer] * math.pi
            grad_params[2 * layer + 1] = zz_layer_sums[layer] * math.pi
        grad_params[-1] = x_layer_sums[-1] * math.pi
        return grad_params

    raise NotImplementedError(f"Gradient compression is not implemented for trotter_order={trotter_order}.")


def l2_cost(current_spo, target_spo) -> float:
    return float(np.asarray((current_spo - target_spo).get_norm_square()))


def finite_difference(objective_fn, params: np.ndarray, index: int, eps: float = GRAD_CHECK_EPS) -> float:
    plus = np.array(params, dtype=float, copy=True)
    minus = np.array(params, dtype=float, copy=True)
    plus[index] += eps
    minus[index] -= eps
    f_plus = objective_fn(plus)
    f_minus = objective_fn(minus)
    return float((f_plus - f_minus) / (2.0 * eps))


def quiet_call(func, *args, **kwargs):
    """Run an SPD helper without streaming its internal progress output."""
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)
