import numpy as np
import pytest

import spd
from spd.circuit_ir import CircuitIR, PauliRotation
from spd.residual_corrected import (
    _merge_spo,
    _residual_corrected_run_seed,
    _top_k_split,
)


def _rz_circuit(*angles):
    return CircuitIR(
        1,
        tuple(PauliRotation("RZ", "Z", angle) for angle in angles),
    )


def _sum_spo(left, right):
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, 0.0) + value
        if result[key] == 0.0:
            result.pop(key)
    return result


def _assert_spo_close(actual, expected, atol=1.0e-12):
    assert set(actual) == set(expected)
    for key in actual:
        assert np.isclose(actual[key], expected[key], atol=atol)


def test_exact_backbone_plus_correction_telescopes_without_correction_truncation():
    observable = spd.create_spo({"X": 1.0}, precision="double")
    circuit = _rz_circuit(0.37, -0.61, 0.23)

    result = spd.evolve_residual_corrected(
        observable,
        circuit,
        backbone_budget=1,
        correction_budget=8,
        seed=7,
    )
    exact, _ = spd.evolve(observable, circuit, 0.0, 100)

    _assert_spo_close(
        _sum_spo(result.final_backbone, result.final_sampled_correction),
        exact,
    )
    assert result.diagnostics.num_backbone_truncations >= 2
    assert result.diagnostics.num_randomized_truncations == 0


def test_exact_expectation_when_correction_budget_covers_candidates():
    observable = spd.create_spo({"XX": 1.0}, precision="double")
    circuit = CircuitIR(
        2,
        (
            PauliRotation("RZ", "ZI", 0.37),
            PauliRotation("RZ", "IZ", -0.61),
        ),
    )
    result = spd.run_residual_corrected_spd(
        observable,
        circuit,
        backbone_budget=1,
        correction_budget=8,
        seed=11,
        basis="X",
    )

    assert np.isclose(result.estimate, np.cos(0.37) * np.cos(-0.61), atol=1e-12)
    assert result.diagnostics.num_randomized_truncations == 0


def test_empirical_end_to_end_unbiasedness():
    observable = spd.create_spo(
        {"I": 4.0, "X": 3.0, "Y": 2.0, "Z": 1.0},
        precision="double",
    )
    exact = 5.0

    result = spd.run_residual_corrected_ensemble(
        observable,
        CircuitIR(1, ()),
        backbone_budget=2,
        correction_budget=1,
        runs=4_000,
        master_seed=20260818,
        basis="Z",
    )

    assert abs(result.mean - exact) < 6.0 * result.standard_error
    assert np.isclose(result.mean, result.backbone_estimate + result.correction_mean)
    assert result.sample_variance > 0.0


def test_multiple_residual_injections_and_diagnostics():
    observable = spd.create_spo({"X": 1.0}, precision="double")
    result = spd.evolve_residual_corrected(
        observable,
        _rz_circuit(0.31, 0.27, -0.19),
        backbone_budget=1,
        correction_budget=2,
        seed=5,
    )

    injection_records = [
        record for record in result.diagnostics.records if record.residual_support
    ]
    assert len(injection_records) >= 2
    assert result.diagnostics.max_backbone_candidate_support == 2
    assert all(record.discarded_l2_fraction > 0.0 for record in injection_records)
    assert np.isclose(
        result.diagnostics.records[-1].sampled_correction_l2_norm_sq,
        result.final_sampled_correction.get_norm_square(),
    )


def test_exact_merge_cancellation_and_numerical_zero_reporting():
    left = spd.create_spo({"X": 1.0}, precision="double")
    right = spd.create_spo({"X": -1.0, "Y": 0.5}, precision="double")
    merged, overlap, stats = _merge_spo(left, right, 0.0)

    assert overlap == 1
    assert merged == spd.create_spo({"Y": 0.5}, precision="double")
    assert stats.count == 0

    near = spd.create_spo({"X": -1.0 + 5.0e-10}, precision="double")
    merged, _, stats = _merge_spo(left, near, 1.0e-9)
    assert not merged
    assert stats.count == 1
    assert stats.max_abs_coefficient < 1.0e-9


def test_top_k_equal_magnitude_ties_are_canonical():
    first = spd.create_spo({"X": 1.0, "Y": -1.0}, precision="double")
    second = spd.create_spo({"Y": -1.0, "X": 1.0}, precision="double")

    first_backbone, first_residual = _top_k_split(first, 1)
    second_backbone, second_residual = _top_k_split(second, 1)

    assert first_backbone == second_backbone
    assert first_residual == second_residual


def test_ensemble_seeds_and_samples_are_reproducible_by_run_index():
    observable = spd.create_spo({"X": 1.0}, precision="double")
    kwargs = dict(
        backbone_budget=1,
        correction_budget=1,
        runs=64,
        master_seed=1234,
        basis="X",
    )
    first = spd.run_residual_corrected_ensemble(
        observable,
        _rz_circuit(np.pi / 4, np.pi / 4),
        **kwargs,
    )
    second = spd.run_residual_corrected_ensemble(
        observable,
        _rz_circuit(np.pi / 4, np.pi / 4),
        **kwargs,
    )

    assert np.array_equal(first.estimates, second.estimates)
    assert first.seeds == second.seeds
    assert first.seeds[:3] == tuple(
        _residual_corrected_run_seed(1234, index) for index in range(3)
    )
    assert len(set(first.seeds)) == len(first.seeds)


def test_initial_observable_above_both_budgets_is_handled():
    observable = spd.create_spo(
        {"I": 4.0, "X": 3.0, "Y": 2.0, "Z": 1.0},
        precision="double",
    )
    result = spd.evolve_residual_corrected(
        observable,
        CircuitIR(1, ()),
        backbone_budget=2,
        correction_budget=1,
        seed=13,
    )

    assert len(result.final_backbone) == 2
    assert len(result.final_sampled_correction) == 1
    assert result.diagnostics.num_backbone_truncations == 1
    assert result.diagnostics.num_randomized_truncations == 1


def test_empty_residual_and_correction_are_stable():
    observable = spd.create_spo({"Z": 1.0}, precision="double")
    result = spd.run_residual_corrected_spd(
        observable,
        _rz_circuit(0.2, -0.3),
        backbone_budget=2,
        correction_budget=1,
        seed=17,
        basis="Z",
    )

    assert result.estimate == 1.0
    assert result.correction_estimate == 0.0
    assert not result.evolution.final_sampled_correction
    assert result.diagnostics.num_randomized_truncations == 0


def test_repeated_rz_pi_over_four_stress_case_is_unbiased():
    depth = 6
    observable = spd.create_spo({"X": 1.0}, precision="double")
    result = spd.run_residual_corrected_ensemble(
        observable,
        _rz_circuit(*(np.pi / 4 for _ in range(depth))),
        backbone_budget=1,
        correction_budget=1,
        runs=4_096,
        master_seed=2718,
        basis="X",
    )
    exact = float(np.cos(depth * np.pi / 4))

    assert abs(result.mean - exact) < 6.0 * result.standard_error
    assert result.diagnostics.total_randomized_truncations > 0


def test_zero_tolerance_preserves_tiny_physical_coefficient():
    observable = spd.create_spo({"X": 1.0}, precision="double")
    result = spd.evolve_residual_corrected(
        observable,
        _rz_circuit(1.0e-10),
        backbone_budget=1,
        correction_budget=2,
        seed=19,
        numerical_zero_tolerance=0.0,
    )

    correction_magnitudes = [
        abs(value) for value in result.final_sampled_correction.values()
    ]
    assert len(correction_magnitudes) == 1
    assert correction_magnitudes[0] > 0.0
    assert np.isclose(correction_magnitudes[0], 1.0e-10, rtol=1.0e-12)
    assert result.diagnostics.numerical_zero_stats.count == 0


def test_nonzero_numerical_tolerance_warns_once_per_run():
    observable = spd.create_spo({"X": 1.0}, precision="double")
    with pytest.warns(RuntimeWarning, match="unbiased only up to"):
        result = spd.evolve_residual_corrected(
            observable,
            _rz_circuit(1.0e-10),
            backbone_budget=1,
            correction_budget=1,
            seed=23,
            numerical_zero_tolerance=1.0e-9,
        )

    assert result.diagnostics.numerical_zero_stats.count == 1
    assert not result.final_sampled_correction
