"""Precompute target evolved X/Z operators for the linear-ramp 2D TFI benchmark."""

from __future__ import annotations

import pickle

import spd

from common import (
    MAX_NUM_STR,
    RAMP_G_END,
    RAMP_G_START,
    RAMP_SYSTEM_SIZE,
    RAMP_TARGET_METADATA_PATH,
    RAMP_TARGET_STEPS,
    RAMP_TARGET_X_PATH,
    RAMP_TARGET_Z_PATH,
    RAMP_TOTAL_TIME,
    TRUNC_VAL,
    build_backend,
    build_second_order_linear_ramp_tfi_circuit,
    quiet_call,
    ramp_target_metadata,
    representative_x_dict,
    representative_z_dict,
    save_metadata,
)


def _save_pickle(path, obj) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


if __name__ == "__main__":
    backend = build_backend()
    target_circuit = build_second_order_linear_ramp_tfi_circuit()

    _, target_spo_x = quiet_call(
        spd.run_pytket_circuit,
        target_circuit,
        representative_x_dict(RAMP_SYSTEM_SIZE),
        TRUNC_VAL,
        MAX_NUM_STR,
        backend=backend,
    )
    _, target_spo_z = quiet_call(
        spd.run_pytket_circuit,
        target_circuit,
        representative_z_dict(RAMP_SYSTEM_SIZE),
        TRUNC_VAL,
        MAX_NUM_STR,
        backend=backend,
    )

    _save_pickle(RAMP_TARGET_X_PATH, target_spo_x)
    _save_pickle(RAMP_TARGET_Z_PATH, target_spo_z)
    save_metadata(RAMP_TARGET_METADATA_PATH, ramp_target_metadata())

    print("Saved ramp target data for 2D TFI time-evolution compression.")
    print(
        f"Linear ramp g: {RAMP_G_START} -> {RAMP_G_END} | total time: {RAMP_TOTAL_TIME} | "
        f"target steps: {RAMP_TARGET_STEPS}"
    )
    print(f"Target X SPO size: {target_spo_x.get_size()} | norm^2: {target_spo_x.get_norm_square()}")
    print(f"Target Z SPO size: {target_spo_z.get_size()} | norm^2: {target_spo_z.get_norm_square()}")
    print(f"Target X path: {RAMP_TARGET_X_PATH}")
    print(f"Target Z path: {RAMP_TARGET_Z_PATH}")
    print(f"Metadata path: {RAMP_TARGET_METADATA_PATH}")
