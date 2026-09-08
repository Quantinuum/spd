import pickle
from types import SimpleNamespace

import numpy as np

from examples.benchmark_rspd_2d_obc_xx_z_stepwise import (
    _progress_path,
    load_completed_indices,
    run_step_seeds,
    save_run,
)
from examples.benchmark_residual_corrected_2d_obc_xx_z import (
    load_runs as load_residual_corrected_runs,
    run_seed as residual_corrected_run_seed,
    save_run as save_residual_corrected_run,
    write_statistics as write_residual_corrected_statistics,
)
from results.get_rspd_2d_obc_xx_z_statistics import (
    aggregate_prefix,
    default_run_counts,
)
from results.get_residual_corrected_2d_obc_xx_z_statistics import (
    aggregate_ensemble as aggregate_residual_corrected_ensemble,
    aggregate_steps as aggregate_residual_corrected_steps,
)


def test_run_seeds_are_stable_when_ensemble_or_step_count_grows():
    first = run_step_seeds(1234, run_index=7, num_steps=3)
    extended = run_step_seeds(1234, run_index=7, num_steps=8)
    other = run_step_seeds(1234, run_index=8, num_steps=3)

    assert first == extended[:3]
    assert first != other


def test_completed_run_is_independent_and_removes_progress(tmp_path):
    metadata = {"population_size": 10, "dt": 0.04}
    run = {
        "trajectory_index": 3,
        "estimates": [1.0, 0.9],
        "compression_records": [("diagnostic",)],
    }
    progress_path = _progress_path(tmp_path, 3)
    progress_path.parent.mkdir()
    progress_path.write_bytes(b"checkpoint")

    output_path = save_run(tmp_path, metadata, run)

    assert output_path.name == "trajectory_000003.pkl"
    assert not progress_path.exists()
    with output_path.open("rb") as handle:
        saved = pickle.load(handle)
    assert saved["metadata"] == metadata
    assert saved["run"] == run
    assert load_completed_indices(tmp_path, metadata) == {3}


def test_statistics_use_prefix_ensembles_and_sample_variance():
    estimates = np.asarray(
        [
            [1.0, 0.8],
            [1.0, 1.0],
            [1.0, 0.9],
        ]
    )
    result = aggregate_prefix(
        estimates,
        count=2,
        deterministic=np.asarray([1.0, 0.85]),
    )

    assert np.allclose(result["mean"], [1.0, 0.9])
    assert np.allclose(result["sample_variance"], [0.0, 0.02])
    assert np.allclose(result["standard_error"], [0.0, 0.1])
    assert np.allclose(result["delta_from_deterministic"], [0.0, 0.05])
    assert default_run_counts(9) == [1, 2, 4, 8, 9]


def test_residual_corrected_run_seeds_are_stable_by_run_index():
    first = tuple(residual_corrected_run_seed(1234, index) for index in range(3))
    extended = tuple(residual_corrected_run_seed(1234, index) for index in range(8))

    assert first == extended[:3]
    assert len(set(extended)) == len(extended)


def test_residual_corrected_runs_are_appendable_and_write_prefix_statistics(tmp_path):
    metadata = {"backbone_budget": 10, "correction_budget": 20}
    runs = [
        {
            "run_index": 1,
            "estimate": 0.8,
            "backbone_estimate": 0.7,
            "correction_estimate": 0.1,
            "elapsed_seconds": 2.0,
        },
        {
            "run_index": 0,
            "estimate": 1.0,
            "backbone_estimate": 0.7,
            "correction_estimate": 0.3,
            "elapsed_seconds": 1.0,
        },
    ]
    for run in runs:
        save_residual_corrected_run(tmp_path, metadata, run)

    loaded = load_residual_corrected_runs(tmp_path, metadata)
    rows = write_residual_corrected_statistics(tmp_path, metadata, loaded)

    assert [run["run_index"] for run in loaded] == [0, 1]
    assert rows[-1]["runs"] == 2
    assert np.isclose(rows[-1]["mean"], 0.9)
    assert np.isclose(rows[-1]["sample_variance"], 0.02)
    assert np.isclose(rows[-1]["standard_error"], 0.1)
    assert (tmp_path / "statistics.pkl").exists()
    assert (tmp_path / "statistics.csv").exists()


def test_phase3_residual_corrected_statistics_include_tail_and_step_diagnostics():
    def record(norm_sq, support_before, heavy_count, predicted_mse, candidate):
        stats = SimpleNamespace(
            support_before=support_before,
            heavy_count=heavy_count,
            predicted_mse=predicted_mse,
        )
        return SimpleNamespace(
            sampled_correction_l2_norm_sq=norm_sq,
            correction_candidate_support=candidate,
            randomized_truncation_stats=stats,
        )

    initial = record(0.0, 0, 0, 0.0, 0)
    runs = [
        {
            "estimate": 0.8,
            "backbone_estimate": 0.7,
            "correction_estimate": 0.1,
            "correction_norm_sq": 4.0,
            "elapsed_seconds": 2.0,
            "diagnostics": SimpleNamespace(
                records=(
                    initial,
                    record(1.0, 2, 2, 0.0, 2),
                    record(2.0, 3, 1, 0.5, 3),
                    record(3.0, 4, 0, 1.0, 4),
                    record(4.0, 2, 2, 0.0, 2),
                )
            ),
        },
        {
            "estimate": 1.0,
            "backbone_estimate": 0.7,
            "correction_estimate": 0.3,
            "correction_norm_sq": 6.0,
            "elapsed_seconds": 4.0,
            "diagnostics": SimpleNamespace(
                records=(
                    initial,
                    record(1.5, 2, 2, 0.0, 2),
                    record(2.5, 4, 2, 0.25, 4),
                    record(3.5, 3, 1, 0.75, 3),
                    record(6.0, 2, 2, 0.0, 2),
                )
            ),
        },
    ]
    ensemble = aggregate_residual_corrected_ensemble(runs, 2, reference=0.9)
    steps = aggregate_residual_corrected_steps(
        runs,
        {"num_gates_per_step": 2, "num_steps": 2, "correction_budget": 2},
    )

    assert np.isclose(ensemble["sample_variance"], 0.02)
    assert np.isclose(ensemble["terminal_nonzero_hit_fraction"], 1.0)
    assert np.isclose(ensemble["correction_norm_sq_median"], 5.0)
    assert np.isclose(ensemble["variance_times_seconds"], 0.06)
    assert np.isclose(ensemble["rmse"], 0.1)
    assert np.isclose(steps[0]["correction_norm_sq_median"], 2.25)
    assert np.isclose(steps[0]["heavy_count_median"], 1.5)
    assert np.isclose(steps[1]["predicted_mse_median"], 0.875)
