from math import sqrt

import numpy as np
import pytest

import spd
from examples.benchmark_time_blocked_residual_2d_obc_xx_z import (
    _load_runs,
    _save_run,
    write_pilot_summary,
)
from spd.circuit_ir import CircuitIR, PauliRotation
from spd.time_blocked_residual import (
    _prepare_time_blocked_backbone,
    _warn_numerical_zeros,
    prepare_time_blocked_backbone,
    run_time_blocked_residual_block,
    summarize_time_blocked_block_runs,
    time_blocked_run_seed,
)


def _circuit(*operations):
    return CircuitIR(1, tuple(operations))


def _rz(angle):
    return PauliRotation("RZ", "Z", angle)


def _rx(angle):
    return PauliRotation("RX", "X", angle)


def _add_spo(target, source):
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + value
        if target[key] == 0.0:
            target.pop(key)


def _assert_spo_close(actual, expected, atol=1.0e-12):
    assert set(actual) == set(expected)
    for key in actual:
        assert np.isclose(actual[key], expected[key], atol=atol)


def test_blockwise_identity_uses_repository_reverse_gate_order():
    observable = spd.create_spo({"Y": 1.0}, precision="double")
    circuit = _circuit(_rz(0.37), _rx(-0.61), _rz(0.23))
    plan = prepare_time_blocked_backbone(
        observable,
        circuit,
        backbone_budget=1,
        block_sizes=(1, 2),
        numerical_zero_tolerance=0.0,
    )

    reconstructed = dict(plan.final_backbone)
    for block_index in range(plan.num_blocks):
        run = run_time_blocked_residual_block(
            plan,
            block_index,
            correction_budget=16,
            seed=block_index + 1,
            basis="Z",
        )
        _add_spo(reconstructed, run.final_sampled_correction)
    exact, _ = spd.evolve(observable, circuit, 0.0, 100)

    _assert_spo_close(reconstructed, exact)
    assert plan.circuit_gate_indices == (2, 1, 0)
    assert plan.block_diagnostics[0].reverse_gate_start == 0
    assert plan.block_diagnostics[1].reverse_gate_start == 1


def test_initial_residual_is_assigned_only_to_first_block():
    observable = spd.create_spo(
        {"I": 4.0, "X": 3.0, "Y": 2.0, "Z": 1.0},
        precision="double",
    )
    plan = prepare_time_blocked_backbone(
        observable,
        _circuit(_rz(0.0), _rz(0.0)),
        backbone_budget=2,
        block_sizes=(1, 1),
        numerical_zero_tolerance=0.0,
    )

    assert plan.block_diagnostics[0].injection_support == 2
    assert plan.block_diagnostics[1].injection_support == 0
    first = run_time_blocked_residual_block(
        plan, 0, 4, seed=1, basis="Z"
    )
    second = run_time_blocked_residual_block(
        plan, 1, 4, seed=2, basis="Z"
    )
    backbone = plan.final_backbone.get_expectation_value(basis="Z")

    assert np.isclose(
        backbone + first.raw_correction_estimate + second.raw_correction_estimate,
        5.0,
    )
    assert not second.final_sampled_correction


def test_empirical_time_blocked_estimator_is_unbiased():
    observable = spd.create_spo({"X": 1.0}, precision="double")
    circuit = _circuit(*(_rz(np.pi / 4) for _ in range(4)))
    plan = prepare_time_blocked_backbone(
        observable,
        circuit,
        backbone_budget=1,
        block_sizes=(2, 2),
        numerical_zero_tolerance=0.0,
    )
    summaries = []
    for block_index in range(plan.num_blocks):
        runs = [
            run_time_blocked_residual_block(
                plan,
                block_index,
                correction_budget=1,
                seed=time_blocked_run_seed(
                    20260819, block_index, 1, run_index
                ),
                basis="X",
            )
            for run_index in range(2_048)
        ]
        summaries.append(summarize_time_blocked_block_runs(runs))
    corrected_estimate = float(
        plan.final_backbone.get_expectation_value(basis="X").real
    ) + sum(summary.mean for summary in summaries)
    standard_error = sqrt(
        sum(summary.sample_variance / len(summary.runs) for summary in summaries)
    )
    exact = float(np.cos(np.pi))

    assert abs(corrected_estimate - exact) < 6.0 * standard_error
    assert all(summary.second_moment >= 0.0 for summary in summaries)
    assert set(summaries[0].seeds).isdisjoint(summaries[1].seeds)


def test_seed_hierarchy_is_stable_and_separates_stage_block_budget_and_run():
    seed = time_blocked_run_seed(1234, 2, 1000, 7, stage=0)
    assert seed == time_blocked_run_seed(1234, 2, 1000, 7, stage=0)
    variants = {
        seed,
        time_blocked_run_seed(1234, 2, 1000, 7, stage=1),
        time_blocked_run_seed(1234, 3, 1000, 7, stage=0),
        time_blocked_run_seed(1234, 2, 10000, 7, stage=0),
        time_blocked_run_seed(1234, 2, 1000, 8, stage=0),
    }
    assert len(variants) == 5


def test_false_convergence_requires_zero_variance_and_zero_hits():
    observable = spd.create_spo({"X": 1.0}, precision="double")
    plan = prepare_time_blocked_backbone(
        observable,
        _circuit(_rz(0.2)),
        backbone_budget=1,
        block_sizes=(1,),
    )
    runs = [
        run_time_blocked_residual_block(plan, 0, 1, seed=seed, basis="Z")
        for seed in (1, 2)
    ]
    summary = summarize_time_blocked_block_runs(runs)

    assert summary.sample_variance == 0.0
    assert summary.terminal_hit_fraction == 0.0
    assert summary.false_convergence


def test_numerical_zero_warning_is_aggregated_once_for_a_plan():
    observable = spd.create_spo({"X": 1.0}, precision="double")
    with pytest.warns(RuntimeWarning, match="unbiased only up to") as warning_records:
        plan = _prepare_time_blocked_backbone(
            observable,
            _circuit(_rz(1.0e-10)),
            backbone_budget=1,
            block_sizes=(1,),
            numerical_zero_tolerance=1.0e-9,
            rebase=False,
            emit_warning=False,
        )
        _warn_numerical_zeros(plan.numerical_zero_stats, 1.0e-9)

    assert len(warning_records) == 1
    assert plan.numerical_zero_stats.count == 1


def test_block_size_validation_counts_active_reverse_operations():
    observable = spd.create_spo({"X": 1.0}, precision="double")
    with pytest.raises(ValueError, match="must sum"):
        prepare_time_blocked_backbone(
            observable,
            _circuit(_rz(0.1), _rz(0.2)),
            backbone_budget=1,
            block_sizes=(1,),
        )


def test_pilot_checkpoint_files_are_appendable_and_summarized(tmp_path):
    observable = spd.create_spo({"X": 1.0}, precision="double")
    plan = prepare_time_blocked_backbone(
        observable,
        _circuit(_rz(0.2), _rz(0.3)),
        backbone_budget=1,
        block_sizes=(1, 1),
        numerical_zero_tolerance=0.0,
    )
    metadata = {"algorithm": "test"}
    for block_index in range(2):
        for run_index in range(2):
            result = run_time_blocked_residual_block(
                plan,
                block_index,
                correction_budget=2,
                seed=10 * block_index + run_index,
                basis="Z",
            )
            _save_run(
                tmp_path,
                metadata,
                block_index,
                run_index,
                elapsed=block_index + 1.0,
                result=result,
            )

    loaded = _load_runs(tmp_path, metadata)
    rows, total = write_pilot_summary(tmp_path, metadata, plan, loaded)

    assert len(loaded) == 4
    assert len(rows) == 2
    assert total is not None
    assert (tmp_path / "pilot_blocks.csv").exists()
    assert (tmp_path / "pilot_summary.pkl").exists()
