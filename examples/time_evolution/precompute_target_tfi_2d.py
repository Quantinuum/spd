"""Precompute target evolved X/Z operators for 2D TFI compression."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import spd

from common import (
    CONSTANT_SYSTEM_SIZE,
    MAX_NUM_STR,
    TARGET_STEPS,
    TOTAL_TIME,
    TRUNC_VAL,
    build_2d_tfi_circuit,
    build_backend,
    quiet_call,
    representative_x_dict,
    representative_z_dict,
    save_metadata,
    step_count_from_dt,
    target_data_dir,
    target_metadata,
    trotter_layer_parameters,
)


def _save_pickle(path, obj) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute normal 2D TFI evolved X/Z target SPOs."
    )
    parser.add_argument("--dt", type=float, default=None, help="Trotter time step.")
    parser.add_argument("--total-t", type=float, default=TOTAL_TIME, help="Total evolution time.")
    parser.add_argument(
        "--target-steps",
        type=int,
        default=TARGET_STEPS,
        help="Number of Trotter steps used when --dt is not set.",
    )
    parser.add_argument(
        "--trunc-val",
        type=float,
        default=TRUNC_VAL,
        help="Forward evolution truncation value.",
    )
    parser.add_argument(
        "--trotter-order",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="Use first-order, second-order, or fourth-order Trotter layers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to a parameterized target_tfi_2d subdirectory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dt is None:
        target_steps = args.target_steps
        if target_steps < 1:
            raise ValueError("--target-steps must be positive.")
        dt = args.total_t / target_steps
    else:
        dt = args.dt
        target_steps = step_count_from_dt(args.total_t, dt)

    output_dir = args.output_dir or target_data_dir(
        total_time=args.total_t,
        dt=dt,
        trotter_order=args.trotter_order,
        trunc_val=args.trunc_val,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    target_x_path = output_dir / "target_spo_x.pkl"
    target_z_path = output_dir / "target_spo_z.pkl"
    x_info_path = output_dir / "x_info.json"
    z_info_path = output_dir / "z_info.json"
    metadata_path = output_dir / "target_metadata.json"

    backend = build_backend()
    target_thetas = trotter_layer_parameters(target_steps, total_time=args.total_t)
    target_circuit = build_2d_tfi_circuit(target_thetas, trotter_order=args.trotter_order)

    initial_spo_x = backend.create_initial_spo(representative_x_dict(CONSTANT_SYSTEM_SIZE))
    initial_spo_z = backend.create_initial_spo(representative_z_dict(CONSTANT_SYSTEM_SIZE))
    target_spo_x, x_info = quiet_call(
        spd.evolve,
        initial_spo_x,
        target_circuit,
        args.trunc_val,
        MAX_NUM_STR,
        backend=backend,
    )
    target_spo_z, z_info = quiet_call(
        spd.evolve,
        initial_spo_z,
        target_circuit,
        args.trunc_val,
        MAX_NUM_STR,
        backend=backend,
    )

    _save_pickle(target_x_path, target_spo_x)
    _save_pickle(target_z_path, target_spo_z)
    save_metadata(x_info_path, x_info)
    save_metadata(z_info_path, z_info)
    save_metadata(
        metadata_path,
        target_metadata(
            total_time=args.total_t,
            target_steps=target_steps,
            trotter_order=args.trotter_order,
            trunc_val=args.trunc_val,
        ),
    )

    print("Saved target data for 2D TFI time-evolution compression.")
    print(f"Target circuit steps: {target_steps}")
    print(f"dt: {dt}")
    print(f"Trotter order: {args.trotter_order}")
    print(f"Target X SPO size: {target_spo_x.get_size()} | norm^2: {target_spo_x.get_norm_square()}")
    print(f"Target Z SPO size: {target_spo_z.get_size()} | norm^2: {target_spo_z.get_norm_square()}")
    print(f"Target X total truncated l2 norm: {x_info['total_truncated_l2_norm']}")
    print(f"Target Z total truncated l2 norm: {z_info['total_truncated_l2_norm']}")
    print(f"Target X path: {target_x_path}")
    print(f"Target Z path: {target_z_path}")
    print(f"X info path: {x_info_path}")
    print(f"Z info path: {z_info_path}")
    print(f"Metadata path: {metadata_path}")
