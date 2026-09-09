import csv
import sys
from pathlib import Path

import numpy as np
import pytest


GRADIENT_DIR = Path(__file__).resolve().parents[1] / "examples" / "gradient"
sys.path.insert(0, str(GRADIENT_DIR))

import run_tfi_gs
import run_utils


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


def test_unified_tfi_setup_builds_3d_ansatz_and_local_hamiltonian():
    ansatz = run_tfi_gs.make_tfi_ansatz(
        np.array([0.1, 0.2]),
        dimension=3,
        linear_system_size=2,
    )
    hamiltonian = run_tfi_gs.make_local_tfi_hamiltonian(
        dimension=3,
        linear_system_size=2,
        g=3.1,
    )

    assert ansatz.circuit.n_qubits == 8
    assert len(hamiltonian) == 4
    np.testing.assert_allclose(
        ansatz.parameter_gradients(np.ones(32)),
        np.array([24.0, 8.0]) * np.pi,
    )
