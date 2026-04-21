"""Optimize a 2-layer first-order 2D TFI circuit against a ramped target circuit."""

from __future__ import annotations

import pickle

import numpy as np
import spd

from common import (
    GRAD_CHECK_ATOL,
    GRAD_CHECK_INDICES,
    GRAD_CHECK_RTOL,
    MAX_NUM_STR,
    OPTIMIZER_METHOD,
    OPTIMIZER_OPTIONS,
    RAMP_ANSATZ_STEPS,
    RAMP_G_END,
    RAMP_G_START,
    RAMP_L_X,
    RAMP_L_Y,
    RAMP_OPTIMIZATION_HISTORY_NPY_PATH,
    RAMP_OPTIMIZATION_HISTORY_TXT_PATH,
    RAMP_OPTIMIZATION_RESULT_PATH,
    RAMP_TARGET_METADATA_PATH,
    RAMP_TARGET_X_PATH,
    RAMP_TARGET_Z_PATH,
    RAMP_TOTAL_TIME,
    RAMP_SYSTEM_SIZE,
    TRUNC_VAL,
    build_backend,
    build_2d_tfi_circuit,
    linear_ramp_first_order_parameters,
    load_metadata,
    validate_metadata_against_expected,
    ramp_target_metadata,
)
from optimize_tfi_2d_compression import (
    CompressionObjective,
    _load_pickle,
    run_gradient_sanity_checks,
    save_history,
)


if __name__ == "__main__":
    try:
        import scipy.optimize
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for this example. Install it in the active environment to run L-BFGS-B."
        ) from exc

    metadata = load_metadata(RAMP_TARGET_METADATA_PATH)
    validate_metadata_against_expected(metadata, ramp_target_metadata())

    target_spo_x = _load_pickle(RAMP_TARGET_X_PATH)
    target_spo_z = _load_pickle(RAMP_TARGET_Z_PATH)
    backend = build_backend()

    objective = CompressionObjective(
        target_spo_x,
        target_spo_z,
        backend,
        num_steps=RAMP_ANSATZ_STEPS,
        system_size=RAMP_SYSTEM_SIZE,
        circuit_builder=lambda thetas: build_2d_tfi_circuit(
            thetas,
            system_size_x=RAMP_L_X,
            system_size_y=RAMP_L_Y,
        ),
    )
    initial_params = linear_ramp_first_order_parameters(
        RAMP_ANSATZ_STEPS,
        RAMP_TOTAL_TIME,
        g_start=RAMP_G_START,
        g_end=RAMP_G_END,
    )

    print("Loaded cached ramp target data for time-evolution compression.")
    print(f"Target metadata backend: {metadata['backend_name']} ({metadata.get('backend_algorithm', 'default')})")
    print(f"Initial parameter vector shape: {initial_params.shape}")
    print(f"Gradient-check indices: {GRAD_CHECK_INDICES} | rtol={GRAD_CHECK_RTOL} | atol={GRAD_CHECK_ATOL}")

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

    save_history(
        objective.history,
        RAMP_OPTIMIZATION_HISTORY_NPY_PATH,
        RAMP_OPTIMIZATION_HISTORY_TXT_PATH,
    )
    with RAMP_OPTIMIZATION_RESULT_PATH.open("wb") as handle:
        pickle.dump(result, handle)

    print("Ramp compression optimization completed.")
    print(f"Optimizer success: {result.success}")
    print(f"Optimizer message: {result.message}")
    print(f"Final cost: {result.fun}")
    print(f"Final parameters: {result.x}")
    print(f"Saved result: {RAMP_OPTIMIZATION_RESULT_PATH}")
    print(f"Saved history (.npy): {RAMP_OPTIMIZATION_HISTORY_NPY_PATH}")
    print(f"Saved history (.txt): {RAMP_OPTIMIZATION_HISTORY_TXT_PATH}")
