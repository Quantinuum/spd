import numpy as np

import spd
from spd.circuit_ir import CircuitIR, PauliRotation


def _rotation_circuit(*rotations):
    return CircuitIR(
        system_size=len(rotations),
        operations=tuple(
            PauliRotation("RZ", "I" * qubit + "Z" + "I" * (len(rotations) - qubit - 1), theta)
            for qubit, theta in enumerate(rotations)
        ),
    )


def test_randomized_small_circuit_is_unbiased_end_to_end():
    angles = (0.37, -0.61)
    observable = spd.create_spo({"XX": 1.0}, precision="double")
    circuit = _rotation_circuit(*angles)
    exact = float(np.cos(angles[0]) * np.cos(angles[1]))

    result = spd.run_ensemble(
        observable,
        circuit,
        pauli_budget=2,
        runs=4_000,
        master_seed=20260812,
        basis="X",
    )

    assert abs(result.mean - exact) < 6.0 * result.standard_error
    assert result.estimates.shape == (4_000,)
    assert result.sample_variance > 0.0
    assert np.isclose(
        result.standard_error,
        np.sqrt(result.sample_variance / len(result.estimates)),
    )
    assert result.diagnostics.total_randomized_truncations > 0


def test_initial_observable_is_truncated_to_exact_pauli_budget():
    observable = spd.create_spo(
        {"I": 4.0, "X": 3.0, "Y": 2.0, "Z": 1.0},
        precision="double",
    )

    result = spd.evolve_randomized(
        observable,
        CircuitIR(1, ()),
        pauli_budget=2,
        seed=7,
    )

    assert len(result.final_sampled_spo) == 2
    assert result.diagnostics.num_randomized_truncations == 1
    record = result.diagnostics.randomized_truncation_records[0]
    assert record.gate_index is None
    assert record.stats.support_before == 4
    assert record.stats.support_after == 2


def test_ensemble_is_reproducible_and_reports_child_seeds():
    observable = spd.create_spo({"X": 1.0}, precision="double")
    circuit = _rotation_circuit(np.pi / 4)
    kwargs = dict(
        pauli_budget=1,
        runs=32,
        master_seed=12345,
        basis="X",
    )

    first = spd.run_ensemble(observable, circuit, **kwargs)
    second = spd.run_ensemble(observable, circuit, **kwargs)

    assert np.array_equal(first.estimates, second.estimates)
    assert first.seeds == second.seeds
    assert len(set(first.seeds)) == len(first.seeds)
    assert first.diagnostics == second.diagnostics


def test_repeated_rz_pi_over_four_has_expected_k1_second_moment():
    depth = 6
    observable = spd.create_spo({"X": 1.0}, precision="double")
    circuit = CircuitIR(
        1,
        tuple(
            PauliRotation("RZ", "Z", np.pi / 4) for _ in range(depth)
        ),
    )

    result = spd.run_ensemble(
        observable,
        circuit,
        pauli_budget=1,
        runs=4_096,
        master_seed=2718,
        basis="X",
    )

    expected_second_moment = 2.0 ** (depth - 1)
    assert np.isclose(
        np.mean(result.estimates ** 2),
        expected_second_moment,
        rtol=0.08,
    )
    assert np.isclose(
        result.runs[0]
        .diagnostics.randomized_truncation_records[0]
        .relative_local_noise,
        1.0,
    )


def test_repeated_rz_is_exact_at_k2():
    depth = 7
    angle = np.pi / 4
    observable = spd.create_spo({"X": 1.0}, precision="double")
    circuit = CircuitIR(
        1,
        tuple(PauliRotation("RZ", "Z", angle) for _ in range(depth)),
    )

    result = spd.run_randomized_spd(
        observable,
        circuit,
        pauli_budget=2,
        seed=99,
        basis="X",
    )

    assert np.isclose(result.estimate, np.cos(depth * angle), atol=1e-12)
    assert result.diagnostics.num_randomized_truncations == 0
    assert len(result.final_sampled_spo) == 2


def test_finite_budget_merges_and_cancels_before_randomized_truncation():
    theta = np.pi / 4
    observable = spd.create_spo(
        {"X": 1.0, "Y": -np.cos(theta) / np.sin(theta)},
        precision="double",
    )
    circuit = CircuitIR(1, (PauliRotation("RZ", "Z", theta),))

    result = spd.evolve_randomized(
        observable,
        circuit,
        pauli_budget=2,
        seed=1,
    )

    assert result.diagnostics.total_raw_children == 4
    assert result.diagnostics.total_merged_children == 1
    assert result.diagnostics.num_randomized_truncations == 0
    assert len(result.final_sampled_spo) == 1
    assert np.isclose(next(iter(result.final_sampled_spo.values())), -np.sqrt(2.0))


def test_empty_sampled_spo_has_zero_expectation():
    assert spd.estimate_expectation(spd.numpy_backend.SparsePauliOp()) == 0.0j


def test_randomized_evolution_does_not_prune_small_nonzero_coefficients():
    observable = spd.create_spo({"X": 1.0}, precision="double")
    circuit = CircuitIR(1, (PauliRotation("RZ", "Z", 1.0e-10),))

    result = spd.evolve_randomized(
        observable,
        circuit,
        pauli_budget=2,
        seed=5,
    )

    coefficients = sorted(abs(value) for value in result.final_sampled_spo.values())
    assert len(coefficients) == 2
    assert coefficients[0] > 0.0
    assert np.isclose(coefficients[0], 1.0e-10, rtol=1e-12)


def test_prototype_population_names_remain_compatible_aliases():
    observable = spd.create_spo({"X": 1.0}, precision="double")
    circuit = CircuitIR(1, (PauliRotation("RZ", "Z", 0.3),))

    canonical = spd.run_randomized_spd(
        observable,
        circuit,
        pauli_budget=1,
        seed=91,
        basis="X",
    )
    legacy = spd.run_population(
        observable,
        circuit,
        population_size=1,
        seed=91,
        basis="X",
    )

    assert isinstance(canonical, spd.RunResult)
    assert spd.PopulationResult is spd.RunResult
    assert canonical.estimate == legacy.estimate
    assert canonical.final_sampled_spo == legacy.final_population
