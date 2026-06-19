import csv
import sys
from pathlib import Path

import numpy as np
import pytest


GRADIENT_DIR = Path(__file__).resolve().parents[1] / "examples" / "gradient"
sys.path.insert(0, str(GRADIENT_DIR))

import run_tfi_gs_3d
import run_utils
import tfi_setup


def test_memory_estimate_matches_large_64_qubit_note():
    estimate = run_utils.estimate_jax_memory_usage(
        64,
        int(3e7),
        packbit=32,
        precision="double",
    )

    assert estimate["rounded_rows"] == 33554432
    assert estimate["spo_gib"] == pytest.approx(0.75)
    assert estimate["spgo_gib"] == pytest.approx(1.0)


def test_record_outputs_include_elapsed_seconds(tmp_path):
    run_utils.init_run_outputs(
        tmp_path,
        metadata={"script": "test"},
        initial_params=np.array([0.1, 0.2]),
    )
    evals = []
    history = []
    params_history = []
    last_eval = {}
    start_time = run_utils.start_timer()
    thetas = np.array([0.1, 0.2])

    run_utils.record_eval(
        evals,
        last_eval,
        thetas,
        cost=1.0,
        energy=1.0,
        energy_error=0.0,
        ose=0.0,
        grad_norm=0.5,
        lambda_ose=0.0,
        run_dir=tmp_path,
        start_time=start_time,
    )
    run_utils.record_step(
        history,
        params_history,
        last_eval,
        thetas,
        run_dir=tmp_path,
        start_time=start_time,
    )

    with open(tmp_path / "evals.csv", newline="") as f:
        eval_row = next(csv.DictReader(f))
    with open(tmp_path / "history.csv", newline="") as f:
        history_row = next(csv.DictReader(f))

    assert float(eval_row["elapsed_s"]) >= 0.0
    assert float(history_row["elapsed_s"]) >= 0.0
    assert history_row["cost"] == "1.0"


def test_3d_tfi_circuit_and_gradient_compression():
    system_size_x = 2
    system_size_y = 2
    system_size_z = 2
    system_size = system_size_x * system_size_y * system_size_z
    thetas = np.array([0.1, 0.2])

    circ = tfi_setup.gen_3d_TFI_ansatz_circuit(
        thetas,
        system_size_x,
        system_size_y,
        system_size_z,
    )
    ham = tfi_setup.gen_3d_Hamiltonian_dict(
        system_size_x,
        system_size_y,
        system_size_z,
        g=3.1,
    )

    assert circ.n_qubits == system_size
    assert len(ham) == 4

    raw_grads = np.arange(4 * system_size, dtype=float)
    combined = run_tfi_gs_3d.combine_grads(raw_grads, 2, system_size)
    assert combined[0] == pytest.approx(np.sum(raw_grads[:3 * system_size]) * np.pi)
    assert combined[1] == pytest.approx(np.sum(raw_grads[3 * system_size:]) * np.pi)
