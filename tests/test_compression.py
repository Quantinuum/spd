import numpy as np
import pytest

from spd.randomized_truncation import (
    optimal_inclusion_probabilities,
    pivotal_select,
    pivotal_truncate,
)


def test_water_filling_retains_heavy_core_and_sums_to_k():
    coefficients = {0: 10.0, 1: 2.0, 2: -1.0, 3: 0.5}

    probabilities, tau, heavy_count = optimal_inclusion_probabilities(
        coefficients, 2
    )

    assert heavy_count == 1
    assert probabilities[0] == 1.0
    assert np.isclose(tau, 3.5)
    assert np.isclose(sum(probabilities.values()), 2.0)
    assert all(probability > 0.0 for probability in probabilities.values())


def test_water_filling_roundoff_tie_does_not_oscillate():
    boundary = 0.0056686317422069291
    first_tail = 0.0042600720953256399
    rounded_tail_sum = np.nextafter(boundary, np.inf)
    second_tail = rounded_tail_sum - first_tail
    coefficients = {
        0: boundary,
        1: first_tail,
        2: second_tail,
    }

    probabilities, tau, heavy_count = optimal_inclusion_probabilities(
        coefficients, 2
    )
    compressed, stats = pivotal_truncate(
        coefficients, 2, np.random.default_rng(17)
    )

    assert abs(tau - boundary) <= np.spacing(boundary)
    assert heavy_count == 1
    assert np.isclose(
        sum(probabilities.values()),
        2.0,
        rtol=0.0,
        atol=8.0 * np.finfo(float).eps,
    )
    assert len(compressed) == 2
    assert stats.support_after == 2


def test_pivotal_selection_has_fixed_size_and_requested_marginals():
    probabilities = np.asarray([1.0, 0.55, 0.35, 0.10])
    repetitions = 20_000
    counts = np.zeros(len(probabilities))
    rng = np.random.default_rng(1123)

    for _ in range(repetitions):
        selected = pivotal_select(probabilities, 2, rng)
        assert np.sum(selected) == 2
        counts += selected

    frequencies = counts / repetitions
    standard_errors = np.sqrt(
        probabilities * (1.0 - probabilities) / repetitions
    )
    assert np.all(
        np.abs(frequencies - probabilities) <= 6.0 * standard_errors + 2e-3
    )


def test_pivotal_truncation_is_unbiased_and_matches_predicted_mse():
    coefficients = {0: 10.0, 1: 2.0, 2: -1.0, 3: 0.5}
    repetitions = 20_000
    rng = np.random.default_rng(9876)
    coordinate_sum = {key: 0.0 for key in coefficients}
    squared_errors = []

    for _ in range(repetitions):
        compressed, stats = pivotal_truncate(coefficients, 2, rng)
        assert len(compressed) == 2
        assert compressed[0] == coefficients[0]
        for key in coefficients:
            coordinate_sum[key] += compressed.get(key, 0.0)
        squared_errors.append(
            sum(
                abs(compressed.get(key, 0.0) - value) ** 2
                for key, value in coefficients.items()
            )
        )

    empirical_mean = {
        key: value / repetitions for key, value in coordinate_sum.items()
    }
    assert np.allclose(
        list(empirical_mean.values()),
        list(coefficients.values()),
        atol=0.04,
    )
    assert np.isclose(np.mean(squared_errors), stats.predicted_mse, rtol=0.03)


def test_randomized_truncation_is_exact_when_support_does_not_exceed_k():
    coefficients = {"a": 1.0 + 2.0j, "b": -0.25j, "zero": 0.0}

    compressed, stats = pivotal_truncate(
        coefficients, 2, np.random.default_rng(4)
    )

    assert compressed == {"a": 1.0 + 2.0j, "b": -0.25j}
    assert stats.support_before == 2
    assert stats.support_after == 2
    assert stats.predicted_mse == 0.0


def test_pivotal_selection_rejects_probabilities_below_rng_resolution():
    tiny = 2.0 ** -54
    probabilities = [1.0 - tiny, tiny]

    with pytest.raises(FloatingPointError, match="RNG resolution"):
        pivotal_select(probabilities, 1, np.random.default_rng(2))


def test_large_k_pivotal_rounding_repairs_only_accumulated_roundoff():
    support = 200_000
    k = 100_000
    coefficients = {
        index: 1.0 / float(index + 1) ** 0.7
        for index in range(support)
    }
    probabilities, _, _ = optimal_inclusion_probabilities(coefficients, k)

    selected = pivotal_select(
        list(probabilities.values()),
        k,
        np.random.default_rng(7),
    )

    assert np.count_nonzero(selected) == k


def test_large_dynamic_range_water_filling_has_roundoff_scale_mass_error():
    support = 200_000
    k = 100_000
    magnitudes = np.logspace(0.0, -250.0, num=support)
    coefficients = dict(enumerate(magnitudes))

    probabilities, tau, heavy_count = optimal_inclusion_probabilities(
        coefficients, k
    )

    assert 0 <= heavy_count < k
    assert tau > 0.0
    assert np.isclose(
        sum(probabilities.values()),
        k,
        rtol=0.0,
        atol=64.0 * np.finfo(float).eps * support,
    )
