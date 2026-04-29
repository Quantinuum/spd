"""Optimize a shallow 2D TFI circuit against cached target evolved X/Z operators."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import spd

from common import (
    ANSATZ_STEPS,
    ANSATZ_TROTTER_ORDER,
    BACKEND_NAME,
    CONSTANT_SYSTEM_SIZE,
    GRAD_CHECK_ATOL,
    GRAD_CHECK_INDICES,
    GRAD_CHECK_RTOL,
    MAX_NUM_STR,
    OPTIMIZER_METHOD,
    OPTIMIZER_OPTIONS,
    TARGET_METADATA_PATH,
    TARGET_X_PATH,
    TARGET_Z_PATH,
    TRUNC_VAL,
    ansatz_parameter_count,
    build_2d_tfi_ansatz_circuit,
    build_backend,
    compress_tfi_gradients,
    finite_difference,
    initial_ansatz_parameters,
    l2_cost,
    load_metadata,
    optimization_output_paths,
    quiet_call,
    representative_x_dict,
    representative_z_dict,
    target_file_paths,
    validate_target_metadata_for_optimization,
)


def _load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


class CompressionObjective:
    def __init__(
        self,
        target_spo_x,
        target_spo_z,
        backend,
        *,
        num_steps: int = ANSATZ_STEPS,
        system_size: int = CONSTANT_SYSTEM_SIZE,
        trunc_val: float = TRUNC_VAL,
        max_num_str: int = MAX_NUM_STR,
        trotter_order: int = ANSATZ_TROTTER_ORDER,
        circuit_builder=None,
    ):
        self.target_spo_x = target_spo_x
        self.target_spo_z = target_spo_z
        self.backend = backend
        self.num_steps = num_steps
        self.system_size = system_size
        self.trunc_val = trunc_val
        self.max_num_str = max_num_str
        self.trotter_order = trotter_order
        self.circuit_builder = (
            (lambda thetas: build_2d_tfi_ansatz_circuit(thetas, trotter_order=self.trotter_order))
            if circuit_builder is None
            else circuit_builder
        )
        self.history = []
        self.eval_count = 0

    def _run_forward(self, thetas: np.ndarray):
        circuit = self.circuit_builder(thetas)
        spo_x, _ = quiet_call(
            spd.evolve,
            self.backend.create_initial_spo(representative_x_dict(self.system_size)),
            circuit,
            self.trunc_val,
            self.max_num_str,
            backend=self.backend,
        )
        spo_z, _ = quiet_call(
            spd.evolve,
            self.backend.create_initial_spo(representative_z_dict(self.system_size)),
            circuit,
            self.trunc_val,
            self.max_num_str,
            backend=self.backend,
        )
        return circuit, spo_x, spo_z

    def cost_only(self, thetas: np.ndarray) -> float:
        _, spo_x, spo_z = self._run_forward(thetas)
        return l2_cost(spo_x, self.target_spo_x) + l2_cost(spo_z, self.target_spo_z)

    def __call__(self, thetas):
        thetas = np.asarray(thetas, dtype=float)
        expected_shape = (ansatz_parameter_count(self.num_steps, trotter_order=self.trotter_order),)
        if thetas.shape != expected_shape:
            raise ValueError(f"Expected parameter vector of shape {expected_shape}, got {thetas.shape}.")

        circuit, spo_x, spo_z = self._run_forward(thetas)

        cost_x = l2_cost(spo_x, self.target_spo_x)
        cost_z = l2_cost(spo_z, self.target_spo_z)

        initial_spgo_x = spd.init_gradient_spo(
            spo_x,
            loss_type="l2_difference",
            target_spo=self.target_spo_x,
            backend=self.backend,
        )
        _, raw_grads_x, _ = quiet_call(
            spd.backpropagate,
            initial_spgo_x,
            circuit,
            self.trunc_val,
            self.max_num_str,
            backend=self.backend,
        )

        initial_spgo_z = spd.init_gradient_spo(
            spo_z,
            loss_type="l2_difference",
            target_spo=self.target_spo_z,
            backend=self.backend,
        )
        _, raw_grads_z, _ = quiet_call(
            spd.backpropagate,
            initial_spgo_z,
            circuit,
            self.trunc_val,
            self.max_num_str,
            backend=self.backend,
        )

        grad_x = compress_tfi_gradients(
            raw_grads_x,
            system_size=self.system_size,
            num_steps=self.num_steps,
            trotter_order=self.trotter_order,
        )
        grad_z = compress_tfi_gradients(
            raw_grads_z,
            system_size=self.system_size,
            num_steps=self.num_steps,
            trotter_order=self.trotter_order,
        )
        total_grad = grad_x + grad_z
        total_cost = cost_x + cost_z

        self.eval_count += 1
        record = {
            "eval": self.eval_count,
            "cost": float(total_cost),
            "cost_x": float(cost_x),
            "cost_z": float(cost_z),
            "grad_norm": float(np.linalg.norm(total_grad)),
            "theta_norm": float(np.linalg.norm(thetas)),
            "parameters": np.array(thetas, copy=True),
        }
        self.history.append(record)

        print(
            f"[eval {self.eval_count:03d}] cost={total_cost:.8e} "
            f"(x={cost_x:.8e}, z={cost_z:.8e}) "
            f"|grad|={record['grad_norm']:.8e} |theta|={record['theta_norm']:.8e}"
        )

        return float(total_cost), total_grad


def run_gradient_sanity_checks(objective: CompressionObjective, initial_params: np.ndarray) -> None:
    zero_cost_x = l2_cost(objective.target_spo_x, objective.target_spo_x)
    zero_cost_z = l2_cost(objective.target_spo_z, objective.target_spo_z)
    if zero_cost_x > 1e-12 or zero_cost_z > 1e-12:
        raise AssertionError(
            f"Target self-cost is not zero: X={zero_cost_x}, Z={zero_cost_z}."
        )

    base_cost, analytical_grad = objective(initial_params)
    expected_shape = (ansatz_parameter_count(objective.num_steps, trotter_order=objective.trotter_order),)
    if analytical_grad.shape != expected_shape:
        raise AssertionError(
            f"Expected analytical gradient shape {expected_shape}, got {analytical_grad.shape}."
        )
    if not np.all(np.isfinite(analytical_grad)):
        raise AssertionError("Analytical gradient contains non-finite values.")

    print("Running finite-difference sanity checks.")
    for index in GRAD_CHECK_INDICES:
        fd_grad = finite_difference(objective.cost_only, initial_params, index)
        if not np.isclose(analytical_grad[index], fd_grad, rtol=GRAD_CHECK_RTOL, atol=GRAD_CHECK_ATOL):
            raise AssertionError(
                f"Finite-difference mismatch at index {index}: "
                f"analytical={analytical_grad[index]}, finite_difference={fd_grad}."
            )
        print(
            f"Gradient check index {index}: analytical={analytical_grad[index]:.8e}, "
            f"finite_difference={fd_grad:.8e}"
        )

    print(f"Initial cost for optimization: {base_cost:.8e}")


def save_history(history, npy_path, txt_path) -> None:
    history_array = np.asarray(
        [
            [
                entry["eval"],
                entry["cost"],
                entry["cost_x"],
                entry["cost_z"],
                entry["grad_norm"],
                entry["theta_norm"],
            ]
            for entry in history
        ],
        dtype=float,
    )
    np.save(npy_path, history_array)
    np.savetxt(
        txt_path,
        history_array,
        header="eval cost cost_x cost_z grad_norm theta_norm",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Directory containing target_spo_x.pkl, target_spo_z.pkl, and target_metadata.json.",
    )
    parser.add_argument("--target-x-path", type=Path, default=TARGET_X_PATH)
    parser.add_argument("--target-z-path", type=Path, default=TARGET_Z_PATH)
    parser.add_argument("--target-metadata-path", type=Path, default=TARGET_METADATA_PATH)
    parser.add_argument("--ansatz-steps", type=int, default=ANSATZ_STEPS)
    parser.add_argument("--ansatz-order", type=int, choices=(1, 2), default=ANSATZ_TROTTER_ORDER)
    parser.add_argument("--ansatz-total-t", type=float, default=None)
    parser.add_argument("--trunc-val", type=float, default=TRUNC_VAL)
    parser.add_argument("--max-num-str", type=int, default=MAX_NUM_STR)
    parser.add_argument("--skip-grad-checks", action="store_true")
    parser.add_argument("--result-path", type=Path, default=None)
    parser.add_argument("--history-npy-path", type=Path, default=None)
    parser.add_argument("--history-txt-path", type=Path, default=None)
    return parser.parse_args()


def resolve_target_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.target_dir is not None:
        paths = target_file_paths(args.target_dir)
        return paths["target_x"], paths["target_z"], paths["metadata"]
    return args.target_x_path, args.target_z_path, args.target_metadata_path


if __name__ == "__main__":
    try:
        import scipy.optimize
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for this example. Install it in the active environment to run L-BFGS-B."
        ) from exc

    args = parse_args()
    target_x_path, target_z_path, target_metadata_path = resolve_target_paths(args)
    metadata = load_metadata(target_metadata_path)
    validate_target_metadata_for_optimization(metadata)

    target_spo_x = _load_pickle(target_x_path)
    target_spo_z = _load_pickle(target_z_path)
    backend = build_backend()
    ansatz_total_time = metadata["total_time"] if args.ansatz_total_t is None else args.ansatz_total_t
    system_size_x = int(metadata["system_size_x"])
    system_size_y = int(metadata["system_size_y"])
    system_size = int(metadata["system_size"])
    output_paths = optimization_output_paths(
        num_steps=args.ansatz_steps,
        trotter_order=args.ansatz_order,
    )
    result_path = output_paths["result"] if args.result_path is None else args.result_path
    history_npy_path = output_paths["history_npy"] if args.history_npy_path is None else args.history_npy_path
    history_txt_path = output_paths["history_txt"] if args.history_txt_path is None else args.history_txt_path

    objective = CompressionObjective(
        target_spo_x,
        target_spo_z,
        backend,
        num_steps=args.ansatz_steps,
        system_size=system_size,
        trunc_val=args.trunc_val,
        max_num_str=args.max_num_str,
        trotter_order=args.ansatz_order,
        circuit_builder=lambda thetas: build_2d_tfi_ansatz_circuit(
            thetas,
            system_size_x=system_size_x,
            system_size_y=system_size_y,
            trotter_order=args.ansatz_order,
        ),
    )
    initial_params = initial_ansatz_parameters(
        num_steps=args.ansatz_steps,
        total_time=ansatz_total_time,
        trotter_order=args.ansatz_order,
    )

    print("Loaded cached target data for time-evolution compression.")
    print(f"Target metadata backend: {metadata['backend_name']} ({metadata.get('backend_algorithm', 'default')})")
    print(f"Target metadata path: {target_metadata_path}")
    print(f"Ansatz order: {args.ansatz_order}")
    print(f"Ansatz steps: {args.ansatz_steps}")
    print(f"Ansatz total time: {ansatz_total_time}")
    print(f"Optimization truncation: {args.trunc_val} | max_num_str: {args.max_num_str}")
    print(f"Initial parameter vector shape: {initial_params.shape}")

    if args.skip_grad_checks:
        print("Skipping gradient sanity checks.")
    else:
        run_gradient_sanity_checks(objective, initial_params)

    result = scipy.optimize.minimize(
        objective,
        initial_params,
        method=OPTIMIZER_METHOD,
        jac=True,
        options=OPTIMIZER_OPTIONS,
    )

    if objective.history:
        initial_cost = objective.history[0]["cost"]
        final_cost = objective.history[-1]["cost"]
        if final_cost > initial_cost + 1e-10:
            raise AssertionError(
                f"Optimization did not decrease the cost: initial={initial_cost}, final={final_cost}."
            )

    save_history(objective.history, history_npy_path, history_txt_path)
    with result_path.open("wb") as handle:
        pickle.dump(result, handle)

    print(f"Optimization backend: {BACKEND_NAME}")
    print(f"Optimizer success: {result.success}")
    print(f"Optimizer message: {result.message}")
    print(f"Final cost: {result.fun}")
    print(f"Final parameters: {result.x}")
    print(f"Saved result: {result_path}")
    print(f"Saved history (.npy): {history_npy_path}")
    print(f"Saved history (.txt): {history_txt_path}")
