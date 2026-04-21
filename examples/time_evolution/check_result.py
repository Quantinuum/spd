"""Load saved optimization results and recompute channel-wise target differences."""

from __future__ import annotations

import argparse
import json
import pickle

from common import (
    ANSATZ_STEPS,
    CONSTANT_L_X,
    CONSTANT_L_Y,
    CONSTANT_SYSTEM_SIZE,
    G_FIELD,
    OPTIMIZATION_RESULT_PATH,
    RAMP_ANSATZ_STEPS,
    RAMP_G_END,
    RAMP_G_START,
    RAMP_L_X,
    RAMP_L_Y,
    RAMP_OPTIMIZATION_RESULT_PATH,
    RAMP_SYSTEM_SIZE,
    RAMP_TARGET_METADATA_PATH,
    RAMP_TARGET_X_PATH,
    RAMP_TARGET_Z_PATH,
    RAMP_TOTAL_TIME,
    TARGET_METADATA_PATH,
    TARGET_X_PATH,
    TARGET_Z_PATH,
    TOTAL_TIME,
    build_2d_tfi_circuit,
    build_backend,
    linear_ramp_first_order_parameters,
    l2_cost,
    quiet_call,
    trotter_layer_parameters,
)
from optimize_tfi_2d_compression import CompressionObjective, _load_pickle


SCENARIO_PATHS = {
    "constant": {
        "metadata": TARGET_METADATA_PATH,
        "target_x": TARGET_X_PATH,
        "target_z": TARGET_Z_PATH,
        "result": OPTIMIZATION_RESULT_PATH,
        "system_size": CONSTANT_SYSTEM_SIZE,
        "num_steps": ANSATZ_STEPS,
        "baseline_params": lambda: trotter_layer_parameters(ANSATZ_STEPS, TOTAL_TIME, g_field=G_FIELD),
        "circuit_builder": lambda thetas: build_2d_tfi_circuit(
            thetas,
            system_size_x=CONSTANT_L_X,
            system_size_y=CONSTANT_L_Y,
        ),
    },
    "ramp": {
        "metadata": RAMP_TARGET_METADATA_PATH,
        "target_x": RAMP_TARGET_X_PATH,
        "target_z": RAMP_TARGET_Z_PATH,
        "result": RAMP_OPTIMIZATION_RESULT_PATH,
        "system_size": RAMP_SYSTEM_SIZE,
        "num_steps": RAMP_ANSATZ_STEPS,
        "baseline_params": lambda: linear_ramp_first_order_parameters(
            RAMP_ANSATZ_STEPS,
            RAMP_TOTAL_TIME,
            g_start=RAMP_G_START,
            g_end=RAMP_G_END,
        ),
        "circuit_builder": lambda thetas: build_2d_tfi_circuit(
            thetas,
            system_size_x=RAMP_L_X,
            system_size_y=RAMP_L_Y,
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_PATHS),
        default="constant",
        help="Named benchmark to inspect. Defaults to the constant-TFI compression example.",
    )
    parser.add_argument(
        "--result-path",
        default=None,
        help="Optional override for the optimization result pickle.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scenario_paths = SCENARIO_PATHS[args.scenario]
    result_path = args.result_path or scenario_paths["result"]
    if result_path is None:
        raise ValueError(
            "No default result path is configured for this scenario. "
            "Pass --result-path explicitly."
        )

    with open(scenario_paths["metadata"], "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    target_spo_x = _load_pickle(scenario_paths["target_x"])
    target_spo_z = _load_pickle(scenario_paths["target_z"])
    with open(result_path, "rb") as handle:
        result = pickle.load(handle)

    objective = CompressionObjective(
        target_spo_x,
        target_spo_z,
        build_backend(),
        num_steps=scenario_paths["num_steps"],
        system_size=scenario_paths["system_size"],
        circuit_builder=scenario_paths["circuit_builder"],
    )
    _, spo_x, spo_z = quiet_call(objective._run_forward, result.x)

    baseline_params = scenario_paths["baseline_params"]()
    _, baseline_spo_x, baseline_spo_z = quiet_call(objective._run_forward, baseline_params)

    diff_x = spo_x - target_spo_x
    diff_z = spo_z - target_spo_z
    cost_x = l2_cost(spo_x, target_spo_x)
    cost_z = l2_cost(spo_z, target_spo_z)
    total_cost = cost_x + cost_z
    baseline_cost_x = l2_cost(baseline_spo_x, target_spo_x)
    baseline_cost_z = l2_cost(baseline_spo_z, target_spo_z)
    baseline_total_cost = baseline_cost_x + baseline_cost_z

    print(f"Scenario: {args.scenario}")
    print(f"Result path: {result_path}")
    print(f"Optimizer success: {result.success}")
    print(f"Optimizer message: {result.message}")
    print(f"Saved objective value: {result.fun}")
    print(f"Recomputed total cost: {total_cost}")
    print(f"X-channel cost: {cost_x}")
    print(f"Z-channel cost: {cost_z}")
    print(f"Trotter baseline total cost: {baseline_total_cost}")
    print(f"Trotter baseline X cost: {baseline_cost_x}")
    print(f"Trotter baseline Z cost: {baseline_cost_z}")
    print(f"Optimized better than baseline: {total_cost < baseline_total_cost}")
    print(f"Cost improvement over baseline: {baseline_total_cost - total_cost}")
    print(f"Diff X size: {diff_x.get_size()} | diff X norm^2: {diff_x.get_norm_square()}")
    print(f"Diff Z size: {diff_z.get_size()} | diff Z norm^2: {diff_z.get_norm_square()}")
    print(f"Target metadata: {json.dumps(metadata, indent=2, sort_keys=True)}")
    print(f"Trotter baseline parameters: {baseline_params}")
    print(f"Optimized parameters: {result.x}")
