"""Precompute target evolved X/Z operators for 2D TFI compression."""

from __future__ import annotations

import pickle

import spd

from common import (
    CONSTANT_SYSTEM_SIZE,
    MAX_NUM_STR,
    TARGET_METADATA_PATH,
    TARGET_STEPS,
    TARGET_X_PATH,
    TARGET_Z_PATH,
    TRUNC_VAL,
    build_2d_tfi_circuit,
    build_backend,
    quiet_call,
    representative_x_dict,
    representative_z_dict,
    save_metadata,
    target_metadata,
    trotter_layer_parameters,
)


def _save_pickle(path, obj) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


if __name__ == "__main__":
    backend = build_backend()
    target_thetas = trotter_layer_parameters(TARGET_STEPS)
    target_circuit = build_2d_tfi_circuit(target_thetas)

    _, target_spo_x = quiet_call(
        spd.run_pytket_circuit,
        target_circuit,
        representative_x_dict(CONSTANT_SYSTEM_SIZE),
        TRUNC_VAL,
        MAX_NUM_STR,
        backend=backend,
    )
    _, target_spo_z = quiet_call(
        spd.run_pytket_circuit,
        target_circuit,
        representative_z_dict(CONSTANT_SYSTEM_SIZE),
        TRUNC_VAL,
        MAX_NUM_STR,
        backend=backend,
    )

    _save_pickle(TARGET_X_PATH, target_spo_x)
    _save_pickle(TARGET_Z_PATH, target_spo_z)
    save_metadata(TARGET_METADATA_PATH, target_metadata())

    print("Saved target data for 2D TFI time-evolution compression.")
    print(f"Target circuit steps: {TARGET_STEPS}")
    print(f"Target X SPO size: {target_spo_x.get_size()} | norm^2: {target_spo_x.get_norm_square()}")
    print(f"Target Z SPO size: {target_spo_z.get_size()} | norm^2: {target_spo_z.get_norm_square()}")
    print(f"Target X path: {TARGET_X_PATH}")
    print(f"Target Z path: {TARGET_Z_PATH}")
    print(f"Metadata path: {TARGET_METADATA_PATH}")
