"""Unbiased randomized truncation for sparse coefficient mappings.

This module is independent of Pauli propagation.  It implements the water-filled
inclusion probabilities and pairwise pivotal sampling from the R-SPD note.
"""

from collections import deque
from dataclasses import dataclass
from math import fsum, sqrt
from operator import index as integer_index

import numpy as np


def _validate_k(k):
    if isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer.")
    try:
        return integer_index(k)
    except TypeError as error:
        raise TypeError("k must be an integer.") from error


@dataclass(frozen=True)
class RandomizedTruncationStats:
    """Diagnostics for one call to :func:`pivotal_truncate`."""

    support_before: int
    support_after: int
    heavy_count: int
    tau: float
    l1_norm: float
    l2_norm_sq: float
    predicted_mse: float
    max_abs_coefficient: float
    min_inclusion_probability: float
    max_reweighting_factor: float

    @property
    def l2_norm(self):
        return sqrt(self.l2_norm_sq)


def _nonzero_items(coefficients):
    """Return exact-nonzero items in a deterministic canonical-key order."""
    return sorted(
        ((key, value) for key, value in coefficients.items() if value != 0.0),
        key=lambda pair: pair[0],
    )


def _strictly_above_roundoff(weight, threshold):
    """Return whether ``weight`` is meaningfully above ``threshold``.

    A coefficient tied with the water-filling threshold has inclusion
    probability one whether it is labelled heavy or left in the saturated
    tail. Treat values separated only by floating-point roundoff as tail
    entries so boundary refinement cannot oscillate across such a tie.
    """
    tolerance = 8.0 * max(
        abs(float(np.spacing(weight))),
        abs(float(np.spacing(threshold))),
    )
    return weight - threshold > tolerance


def _optimal_probabilities(items, k):
    m = len(items)
    if not 1 <= k < m:
        raise ValueError("Require 1 <= k < number of nonzero coefficients.")

    magnitudes = [float(abs(value)) for _, value in items]
    if any(not np.isfinite(weight) for weight in magnitudes):
        raise ValueError("Coefficient magnitudes must be finite.")
    if any(weight <= 0.0 for weight in magnitudes):
        raise ValueError(
            "Zero coefficients must be removed before randomized truncation."
        )

    scale = max(magnitudes)
    scaled = [weight / scale for weight in magnitudes]
    if any(weight > 0.0 and scaled_weight == 0.0
           for weight, scaled_weight in zip(magnitudes, scaled)):
        raise FloatingPointError(
            "Scaled tail underflowed; use higher-precision probabilities."
        )

    order = sorted(range(m), key=lambda i: (-scaled[i], items[i][0]))
    ordered = [scaled[i] for i in order]
    suffix = [0.0] * (m + 1)
    running = 0.0
    for index in range(m - 1, -1, -1):
        running = fsum((ordered[index], running))
        suffix[index] = running

    heavy_count = 0
    while heavy_count < k - 1:
        threshold_scaled = suffix[heavy_count] / (k - heavy_count)
        if not _strictly_above_roundoff(
            ordered[heavy_count], threshold_scaled
        ):
            break
        heavy_count += 1

    if (
        heavy_count == k - 1
        and m > k
        and suffix[heavy_count] == ordered[heavy_count]
        and any(weight > 0.0 for weight in ordered[heavy_count + 1:])
    ):
        raise FloatingPointError(
            "Positive tail was lost in a floating suffix sum; use higher precision."
        )

    # Recompute the selected tail with one accurate summation.  The suffix
    # array is sufficient to locate the boundary, but its repeated additions
    # can accumulate enough relative error to be amplified by a large k.
    for _ in range(32):
        tail_sum_scaled = fsum(ordered[heavy_count:])
        tau_scaled = tail_sum_scaled / (k - heavy_count)

        low = 0
        high = min(k, m)
        while low < high:
            middle = (low + high) // 2
            if _strictly_above_roundoff(ordered[middle], tau_scaled):
                low = middle + 1
            else:
                high = middle
        refined_heavy_count = min(low, k - 1)
        if refined_heavy_count == heavy_count:
            break
        heavy_count = refined_heavy_count
    else:
        raise FloatingPointError("Water-filling heavy boundary did not converge.")

    tau = tau_scaled * scale
    if not tau > 0.0:
        raise FloatingPointError(
            "Tail threshold underflowed; use higher-precision probabilities."
        )

    probabilities = [min(1.0, weight / tau_scaled) for weight in scaled]
    tolerance = 64.0 * np.finfo(float).eps * max(1, m, k)
    residual = float(k) - fsum(probabilities)
    if abs(residual) > tolerance:
        raise FloatingPointError(
            "Probability drift exceeds roundoff: "
            f"residual={residual:.3e}, bound={tolerance:.3e}, "
            f"support={m}, k={k}. Use higher precision."
        )
    if residual != 0.0:
        fractional = [i for i, p in enumerate(probabilities) if 0.0 < p < 1.0]
        if not fractional:
            raise FloatingPointError("No fractional probability is available for repair.")
        repair_index = max(
            fractional,
            key=lambda i: min(probabilities[i], 1.0 - probabilities[i]),
        )
        repaired = probabilities[repair_index] + residual
        if not 0.0 < repaired < 1.0:
            raise FloatingPointError("Roundoff repair left the unit interval.")
        probabilities[repair_index] = repaired

    if abs(fsum(probabilities) - float(k)) > tolerance:
        raise FloatingPointError("Inclusion probabilities do not sum to k.")
    heavy_count = sum(probability == 1.0 for probability in probabilities)
    return probabilities, tau, heavy_count


def optimal_inclusion_probabilities(coefficients, k):
    """Return ``({key: pi}, tau, heavy_count)`` for exact-nonzero entries.

    Probabilities obey ``pi_i = min(1, |v_i| / tau)`` and sum to ``k``.
    """
    k = _validate_k(k)
    if k < 1:
        raise ValueError("k must be positive.")
    items = _nonzero_items(coefficients)
    if len(items) <= k:
        return {key: 1.0 for key, _ in items}, 0.0, len(items)
    probabilities, tau, heavy_count = _optimal_probabilities(items, k)
    return (
        {key: probability for (key, _), probability in zip(items, probabilities)},
        tau,
        heavy_count,
    )


def pivotal_select(probabilities, k, rng, tolerance=1.0e-12):
    """Draw exactly ``k`` indicators with the requested marginals."""
    k = _validate_k(k)
    q = [float(probability) for probability in probabilities]
    if k < 0 or k > len(q):
        raise ValueError("k must lie between zero and the number of probabilities.")
    if any(not 0.0 <= probability <= 1.0 for probability in q):
        raise ValueError("Inclusion probabilities must lie in [0, 1].")
    minimum_probability = 2.0 ** -53
    if any(
        0.0 < probability < minimum_probability
        or 0.0 < 1.0 - probability < minimum_probability
        for probability in q
    ):
        raise FloatingPointError(
            "An inclusion probability is below float RNG resolution; use a "
            "finer random-bit source or higher-precision implementation."
        )
    probability_sum = fsum(q)
    sum_tolerance = 64.0 * np.finfo(float).eps * max(1, len(q), k)
    if abs(probability_sum - float(k)) > sum_tolerance:
        raise ValueError("Inclusion probabilities must sum to k.")

    active = [i for i, probability in enumerate(q) if probability not in (0.0, 1.0)]
    rng.shuffle(active)
    queue = deque(active)

    while len(queue) >= 2:
        i = queue.popleft()
        j = queue.popleft()
        a, b = q[i], q[j]
        pair_sum = a + b
        if pair_sum <= 1.0:
            if not pair_sum > 0.0:
                raise FloatingPointError("Fractional pair sum underflowed.")
            first_probability = a / pair_sum
            if _bernoulli(first_probability, rng):
                q[i], q[j] = pair_sum, 0.0
            else:
                q[i], q[j] = 0.0, pair_sum
        else:
            denominator = 2.0 - pair_sum
            if not denominator > 0.0:
                raise FloatingPointError("Pivotal denominator underflowed.")
            first_probability = (1.0 - b) / denominator
            if _bernoulli(first_probability, rng):
                q[i], q[j] = 1.0, pair_sum - 1.0
            else:
                q[i], q[j] = pair_sum - 1.0, 1.0

        for index in (i, j):
            if -tolerance <= q[index] < 0.0:
                q[index] = 0.0
            elif 1.0 < q[index] <= 1.0 + tolerance:
                q[index] = 1.0
            elif not 0.0 <= q[index] <= 1.0:
                raise FloatingPointError("Pivotal update left [0, 1].")
            if q[index] not in (0.0, 1.0):
                queue.append(index)

    if len(queue) == 1:
        index = queue.popleft()
        needed = k - sum(value == 1.0 for value in q)
        # One fractional survivor accumulates the addition roundoff from its
        # complete pivot chain.  The bound must therefore scale with the number
        # of pivots, while remaining only O(machine-epsilon * M).
        endpoint_tolerance = max(
            tolerance,
            8.0 * np.finfo(float).eps * max(1, len(q)),
        )
        residue = abs(q[index] - needed)
        if needed not in (0, 1) or residue > endpoint_tolerance:
            raise FloatingPointError(
                "Non-roundoff fractional residue after pivotal sampling: "
                f"residue={residue:.3e}, bound={endpoint_tolerance:.3e}."
            )
        q[index] = float(needed)

    selected = np.asarray([value == 1.0 for value in q], dtype=bool)
    if int(np.sum(selected)) != k:
        raise RuntimeError("Pivotal selection did not produce exactly k entries.")
    return selected


def _bernoulli(probability, rng):
    """Draw a Bernoulli event without silently losing sub-resolution mass."""
    minimum_probability = 2.0 ** -53
    if (
        0.0 < probability < minimum_probability
        or 0.0 < 1.0 - probability < minimum_probability
    ):
        raise FloatingPointError(
            "Pivotal probability is below float RNG resolution; use a finer "
            "random-bit source or higher-precision implementation."
        )
    return float(rng.random()) < probability


def pivotal_truncate(coefficients, k, rng):
    """Return an unbiased minimum-MSE mapping with at most ``k`` entries.

    Selected coordinates are Horvitz--Thompson reweighted by their original
    water-filled inclusion probabilities.
    """
    k = _validate_k(k)
    if k < 1:
        raise ValueError("k must be positive.")
    items = _nonzero_items(coefficients)
    support_before = len(items)
    magnitudes = [float(abs(value)) for _, value in items]
    if any(not np.isfinite(weight) for weight in magnitudes):
        raise ValueError("Coefficient magnitudes must be finite.")
    l1_norm = fsum(magnitudes)
    l2_norm_sq = fsum(weight * weight for weight in magnitudes)
    max_abs = max(magnitudes, default=0.0)

    if support_before <= k:
        return dict(items), RandomizedTruncationStats(
            support_before=support_before,
            support_after=support_before,
            heavy_count=support_before,
            tau=0.0,
            l1_norm=l1_norm,
            l2_norm_sq=l2_norm_sq,
            predicted_mse=0.0,
            max_abs_coefficient=max_abs,
            min_inclusion_probability=1.0 if support_before else 0.0,
            max_reweighting_factor=1.0 if support_before else 0.0,
        )

    probabilities, tau, heavy_count = _optimal_probabilities(items, k)
    selected = pivotal_select(probabilities, k, rng)
    output = {}
    predicted_mse = 0.0
    for ((key, value), inclusion, keep) in zip(items, probabilities, selected):
        predicted_mse += abs(value) ** 2 * (1.0 / inclusion - 1.0)
        if keep:
            output[key] = value if inclusion == 1.0 else value / inclusion

    if len(output) != k:
        raise RuntimeError("Randomly truncated support is not exactly k.")
    if any(not np.isfinite(value.real) or not np.isfinite(value.imag)
           for value in map(complex, output.values())):
        raise FloatingPointError(
            "Randomized truncation produced a non-finite coefficient."
        )

    return output, RandomizedTruncationStats(
        support_before=support_before,
        support_after=len(output),
        heavy_count=heavy_count,
        tau=tau,
        l1_norm=l1_norm,
        l2_norm_sq=l2_norm_sq,
        predicted_mse=float(predicted_mse),
        max_abs_coefficient=max_abs,
        min_inclusion_probability=min(probabilities),
        max_reweighting_factor=max(1.0 / probability for probability in probabilities),
    )


# Compatibility aliases for the first prototype API and persisted result files.
# New code should use RandomizedTruncationStats and pivotal_truncate.
CompressionStats = RandomizedTruncationStats
pivotal_compress = pivotal_truncate
