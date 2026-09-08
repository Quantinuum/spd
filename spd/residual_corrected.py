"""Deterministic-backbone residual-corrected R-SPD for the NumPy backend."""

from dataclasses import dataclass
from math import fsum, sqrt
import warnings

import numpy as np

from . import numpy_backend
from .circuit_ir import CircuitIR, SkippedOperation
from .randomized import (
    _integer_seed,
    _positive_integer,
    estimate_expectation,
)
from .randomized_truncation import RandomizedTruncationStats, pivotal_truncate
from .run_circuit import _normalize_input_circuit, _resolve_backend_from_state


ALGORITHM_VERSION = "residual-corrected-rspd-v1"


@dataclass(frozen=True)
class NumericalZeroStats:
    count: int
    l1_norm: float
    l2_norm_sq: float
    max_abs_coefficient: float

    @property
    def l2_norm(self):
        return sqrt(self.l2_norm_sq)


@dataclass(frozen=True)
class ResidualCorrectionRecord:
    gate_index: object
    gate_name: str
    backbone_candidate_support: int
    backbone_support: int
    residual_support: int
    residual_l1_norm: float
    residual_l2_norm_sq: float
    discarded_l2_fraction: float
    propagated_correction_support: int
    correction_candidate_support: int
    injection_key_overlap: int
    injection_cancellation_ratio: float
    sampled_correction_l2_norm_sq: float
    randomized_truncation_stats: RandomizedTruncationStats
    numerical_zero_stats: NumericalZeroStats


@dataclass(frozen=True)
class ResidualCorrectedDiagnostics:
    records: tuple
    num_backbone_truncations: int
    num_randomized_truncations: int
    accumulated_predicted_mse: float
    max_backbone_candidate_support: int
    max_correction_candidate_support: int
    numerical_zero_stats: NumericalZeroStats


@dataclass(frozen=True)
class ResidualCorrectedEvolutionResult:
    final_backbone: numpy_backend.SparsePauliOp
    final_sampled_correction: numpy_backend.SparsePauliOp
    diagnostics: ResidualCorrectedDiagnostics
    seed: int
    algorithm_version: str
    precision: str


@dataclass(frozen=True)
class ResidualCorrectedRunResult:
    estimate: float
    raw_estimate: complex
    backbone_estimate: float
    raw_backbone_estimate: complex
    correction_estimate: float
    raw_correction_estimate: complex
    imaginary_leakage: float
    evolution: ResidualCorrectedEvolutionResult

    @property
    def diagnostics(self):
        return self.evolution.diagnostics

    @property
    def seed(self):
        return self.evolution.seed


@dataclass(frozen=True)
class ResidualCorrectedEnsembleDiagnostics:
    total_backbone_truncations: int
    total_randomized_truncations: int
    accumulated_predicted_mse: float
    max_backbone_candidate_support: int
    max_correction_candidate_support: int
    max_imaginary_leakage: float
    numerical_zero_stats: NumericalZeroStats


@dataclass(frozen=True)
class ResidualCorrectedEnsembleResult:
    estimates: np.ndarray
    correction_estimates: np.ndarray
    mean: float
    correction_mean: float
    backbone_estimate: float
    sample_variance: float
    standard_error: float
    seeds: tuple
    runs: tuple
    master_seed: int
    diagnostics: ResidualCorrectedEnsembleDiagnostics


def _validate_numerical_zero_tolerance(value):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("numerical_zero_tolerance must be finite and nonnegative.")
    return value


def _empty_zero_stats():
    return NumericalZeroStats(0, 0.0, 0.0, 0.0)


def _combine_zero_stats(*stats):
    return NumericalZeroStats(
        count=sum(item.count for item in stats),
        l1_norm=fsum(item.l1_norm for item in stats),
        l2_norm_sq=fsum(item.l2_norm_sq for item in stats),
        max_abs_coefficient=max(
            (item.max_abs_coefficient for item in stats),
            default=0.0,
        ),
    )


def _zero_stats(magnitudes):
    return NumericalZeroStats(
        count=len(magnitudes),
        l1_norm=fsum(magnitudes),
        l2_norm_sq=fsum(value * value for value in magnitudes),
        max_abs_coefficient=max(magnitudes, default=0.0),
    )


def _new_spo():
    return numpy_backend.SparsePauliOp()


def _copy_spo(spo):
    result = _new_spo()
    for key, value in spo.items():
        result[key] = value
    return result


def _is_numerical_zero(value, tolerance):
    return bool(np.isclose(value, 0.0, rtol=0.0, atol=tolerance))


def _prune_numerical_zeros(spo, tolerance):
    kept = _new_spo()
    removed = []
    for key, value in spo.items():
        if _is_numerical_zero(value, tolerance):
            if value != 0.0:
                removed.append(float(abs(value)))
        else:
            kept[key] = value
            if not np.isfinite(kept[key]):
                raise FloatingPointError("SPO contains a non-finite coefficient.")
    return kept, _zero_stats(removed)


def _top_k_split(candidate, budget):
    """Return a reproducible top-K operator and its complete residual."""
    if len(candidate) <= budget:
        return _copy_spo(candidate), _new_spo()

    ranked_keys = sorted(
        candidate,
        key=lambda key: (-abs(candidate[key]), key),
    )
    kept_keys = set(ranked_keys[:budget])
    backbone = _new_spo()
    residual = _new_spo()
    for key, value in candidate.items():
        target = backbone if key in kept_keys else residual
        target[key] = value
    return backbone, residual


def _merge_spo(left, right, tolerance):
    merged = _copy_spo(left)
    overlap_count = 0
    removed = []
    for key, value in right.items():
        if key not in merged:
            merged[key] = value
            continue
        overlap_count += 1
        new_value = merged[key] + value
        if _is_numerical_zero(new_value, tolerance):
            merged.pop(key)
            if new_value != 0.0:
                removed.append(float(abs(new_value)))
        else:
            merged[key] = new_value
    return merged, overlap_count, _zero_stats(removed)


def _propagate_exact(spo, operation, backend, tolerance):
    propagated, _, _, _ = backend.apply_forward(
        spo,
        operation,
        trunc_val=0.0,
        max_num_str=None,
    )
    return _prune_numerical_zeros(propagated, tolerance)


def _l1_norm(spo):
    return fsum(float(abs(value)) for value in spo.values())


def _l2_norm_sq(spo):
    return fsum(float(abs(value)) ** 2 for value in spo.values())


def _as_spo(coefficients):
    result = _new_spo()
    for key, value in coefficients.items():
        if value != 0.0:
            result[key] = value
    return result


def _truncate_correction(candidate, budget, rng):
    truncated, stats = pivotal_truncate(candidate, budget, rng)
    return _as_spo(truncated), stats


def _record(
    gate_index,
    gate_name,
    candidate,
    backbone,
    residual,
    propagated_correction,
    correction_candidate,
    sampled_correction,
    overlap_count,
    randomized_stats,
    numerical_zero_stats,
):
    residual_l1 = _l1_norm(residual)
    residual_l2_sq = _l2_norm_sq(residual)
    candidate_l2_sq = _l2_norm_sq(candidate)
    raw_injection_l1 = _l1_norm(propagated_correction) + residual_l1
    merged_l1 = _l1_norm(correction_candidate)
    return ResidualCorrectionRecord(
        gate_index=gate_index,
        gate_name=gate_name,
        backbone_candidate_support=len(candidate),
        backbone_support=len(backbone),
        residual_support=len(residual),
        residual_l1_norm=residual_l1,
        residual_l2_norm_sq=residual_l2_sq,
        discarded_l2_fraction=(
            residual_l2_sq / candidate_l2_sq if candidate_l2_sq else 0.0
        ),
        propagated_correction_support=len(propagated_correction),
        correction_candidate_support=len(correction_candidate),
        injection_key_overlap=overlap_count,
        injection_cancellation_ratio=(
            1.0 - merged_l1 / raw_injection_l1 if raw_injection_l1 else 0.0
        ),
        sampled_correction_l2_norm_sq=float(sampled_correction.get_norm_square()),
        randomized_truncation_stats=randomized_stats,
        numerical_zero_stats=numerical_zero_stats,
    )


def _warn_numerical_zeros(stats, tolerance):
    if not stats.count:
        return
    warnings.warn(
        "Residual-corrected R-SPD removed {} non-exact numerical-zero values "
        "with atol={:.3g} (l1={:.3g}, l2={:.3g}, max={:.3g}). The estimator "
        "is unbiased only up to this numerical-zero policy.".format(
            stats.count,
            tolerance,
            stats.l1_norm,
            stats.l2_norm,
            stats.max_abs_coefficient,
        ),
        RuntimeWarning,
        stacklevel=3,
    )


def _evolve_residual_corrected(
    spo,
    input_circuit,
    backbone_budget,
    correction_budget,
    *,
    seed,
    numerical_zero_tolerance,
    rebase,
    emit_warning,
):
    backbone_budget = _positive_integer(backbone_budget, "backbone_budget")
    correction_budget = _positive_integer(correction_budget, "correction_budget")
    seed = _integer_seed(seed, "seed")
    tolerance = _validate_numerical_zero_tolerance(numerical_zero_tolerance)
    backend = _resolve_backend_from_state(spo, None, state_name="spo")
    if backend.name != "numpy" or not backend.is_spo_instance(spo):
        raise TypeError("residual-corrected R-SPD requires a NumPy SparsePauliOp.")

    normalized = _normalize_input_circuit(input_circuit, backend, rebase)
    operations = normalized.operations if isinstance(normalized, CircuitIR) else normalized
    rng = np.random.default_rng(seed)
    records = []
    zero_stats_history = []

    initial_candidate, initial_zero_stats = _prune_numerical_zeros(spo, tolerance)
    backbone, initial_residual = _top_k_split(initial_candidate, backbone_budget)
    correction_candidate = initial_residual
    correction, randomized_stats = _truncate_correction(
        correction_candidate,
        correction_budget,
        rng,
    )
    initial_record = _record(
        None,
        "initial observable",
        initial_candidate,
        backbone,
        initial_residual,
        _new_spo(),
        correction_candidate,
        correction,
        0,
        randomized_stats,
        initial_zero_stats,
    )
    records.append(initial_record)
    zero_stats_history.append(initial_zero_stats)

    for reverse_step, operation in enumerate(reversed(operations)):
        if isinstance(operation, SkippedOperation):
            continue
        candidate, backbone_zero_stats = _propagate_exact(
            backbone,
            operation,
            backend,
            tolerance,
        )
        next_backbone, residual = _top_k_split(candidate, backbone_budget)
        propagated_correction, correction_zero_stats = _propagate_exact(
            correction,
            operation,
            backend,
            tolerance,
        )
        correction_candidate, overlap_count, merge_zero_stats = _merge_spo(
            propagated_correction,
            residual,
            tolerance,
        )
        next_correction, randomized_stats = _truncate_correction(
            correction_candidate,
            correction_budget,
            rng,
        )
        gate_zero_stats = _combine_zero_stats(
            backbone_zero_stats,
            correction_zero_stats,
            merge_zero_stats,
        )
        gate_index = len(operations) - 1 - reverse_step
        records.append(
            _record(
                gate_index,
                operation.gate_name,
                candidate,
                next_backbone,
                residual,
                propagated_correction,
                correction_candidate,
                next_correction,
                overlap_count,
                randomized_stats,
                gate_zero_stats,
            )
        )
        zero_stats_history.append(gate_zero_stats)
        backbone = next_backbone
        correction = next_correction

    all_zero_stats = _combine_zero_stats(*zero_stats_history)
    diagnostics = ResidualCorrectedDiagnostics(
        records=tuple(records),
        num_backbone_truncations=sum(
            record.residual_support > 0 for record in records
        ),
        num_randomized_truncations=sum(
            record.randomized_truncation_stats.support_before > correction_budget
            for record in records
        ),
        accumulated_predicted_mse=fsum(
            record.randomized_truncation_stats.predicted_mse for record in records
        ),
        max_backbone_candidate_support=max(
            record.backbone_candidate_support for record in records
        ),
        max_correction_candidate_support=max(
            record.correction_candidate_support for record in records
        ),
        numerical_zero_stats=all_zero_stats,
    )
    if emit_warning:
        _warn_numerical_zeros(all_zero_stats, tolerance)
    return ResidualCorrectedEvolutionResult(
        final_backbone=backbone,
        final_sampled_correction=correction,
        diagnostics=diagnostics,
        seed=int(seed),
        algorithm_version=ALGORITHM_VERSION,
        precision=numpy_backend.utils.get_precision(),
    )


def evolve_residual_corrected(
    spo,
    input_circuit,
    backbone_budget,
    correction_budget,
    *,
    seed,
    numerical_zero_tolerance=1.0e-12,
    rebase=False,
):
    """Evolve one deterministic backbone and one sampled residual correction."""
    return _evolve_residual_corrected(
        spo,
        input_circuit,
        backbone_budget,
        correction_budget,
        seed=seed,
        numerical_zero_tolerance=numerical_zero_tolerance,
        rebase=rebase,
        emit_warning=True,
    )


def _run_residual_corrected_spd(
    spo,
    input_circuit,
    backbone_budget,
    correction_budget,
    *,
    seed,
    basis,
    state_overlap,
    numerical_zero_tolerance,
    rebase,
    emit_warning,
):
    evolution = _evolve_residual_corrected(
        spo,
        input_circuit,
        backbone_budget,
        correction_budget,
        seed=seed,
        numerical_zero_tolerance=numerical_zero_tolerance,
        rebase=rebase,
        emit_warning=emit_warning,
    )
    raw_backbone = estimate_expectation(
        evolution.final_backbone,
        basis=basis,
        state_overlap=state_overlap,
    )
    raw_correction = estimate_expectation(
        evolution.final_sampled_correction,
        basis=basis,
        state_overlap=state_overlap,
    )
    raw_estimate = raw_backbone + raw_correction
    return ResidualCorrectedRunResult(
        estimate=float(raw_estimate.real),
        raw_estimate=raw_estimate,
        backbone_estimate=float(raw_backbone.real),
        raw_backbone_estimate=raw_backbone,
        correction_estimate=float(raw_correction.real),
        raw_correction_estimate=raw_correction,
        imaginary_leakage=float(abs(raw_estimate.imag)),
        evolution=evolution,
    )


def run_residual_corrected_spd(
    spo,
    input_circuit,
    backbone_budget,
    correction_budget,
    *,
    seed,
    basis="0",
    state_overlap=None,
    numerical_zero_tolerance=1.0e-12,
    rebase=False,
):
    """Return one residual-corrected estimator sample."""
    return _run_residual_corrected_spd(
        spo,
        input_circuit,
        backbone_budget,
        correction_budget,
        seed=seed,
        basis=basis,
        state_overlap=state_overlap,
        numerical_zero_tolerance=numerical_zero_tolerance,
        rebase=rebase,
        emit_warning=True,
    )


def _residual_corrected_run_seed(master_seed, run_index):
    return int(
        np.random.SeedSequence(master_seed, spawn_key=(run_index,)).generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )


def run_residual_corrected_ensemble(
    spo,
    input_circuit,
    backbone_budget,
    correction_budget,
    runs,
    *,
    master_seed,
    basis="0",
    state_overlap=None,
    numerical_zero_tolerance=1.0e-12,
    rebase=False,
):
    """Run independent residual-corrected realizations and summarize them."""
    runs = _positive_integer(runs, "runs")
    master_seed = _integer_seed(master_seed, "master_seed")
    tolerance = _validate_numerical_zero_tolerance(numerical_zero_tolerance)
    seeds = tuple(
        _residual_corrected_run_seed(master_seed, run_index)
        for run_index in range(runs)
    )
    run_results = tuple(
        _run_residual_corrected_spd(
            spo,
            input_circuit,
            backbone_budget,
            correction_budget,
            seed=seed,
            basis=basis,
            state_overlap=state_overlap,
            numerical_zero_tolerance=tolerance,
            rebase=rebase,
            emit_warning=False,
        )
        for seed in seeds
    )
    estimates = np.asarray([run.estimate for run in run_results], dtype=np.float64)
    correction_estimates = np.asarray(
        [run.correction_estimate for run in run_results],
        dtype=np.float64,
    )
    mean = float(np.mean(estimates, dtype=np.float64))
    correction_mean = float(np.mean(correction_estimates, dtype=np.float64))
    sample_variance = (
        float(np.var(estimates, ddof=1, dtype=np.float64))
        if runs > 1
        else float("nan")
    )
    standard_error = sqrt(sample_variance / runs) if runs > 1 else float("nan")
    all_zero_stats = _combine_zero_stats(
        *(run.diagnostics.numerical_zero_stats for run in run_results)
    )
    _warn_numerical_zeros(all_zero_stats, tolerance)
    diagnostics = ResidualCorrectedEnsembleDiagnostics(
        total_backbone_truncations=sum(
            run.diagnostics.num_backbone_truncations for run in run_results
        ),
        total_randomized_truncations=sum(
            run.diagnostics.num_randomized_truncations for run in run_results
        ),
        accumulated_predicted_mse=fsum(
            run.diagnostics.accumulated_predicted_mse for run in run_results
        ),
        max_backbone_candidate_support=max(
            run.diagnostics.max_backbone_candidate_support for run in run_results
        ),
        max_correction_candidate_support=max(
            run.diagnostics.max_correction_candidate_support for run in run_results
        ),
        max_imaginary_leakage=max(run.imaginary_leakage for run in run_results),
        numerical_zero_stats=all_zero_stats,
    )
    return ResidualCorrectedEnsembleResult(
        estimates=estimates,
        correction_estimates=correction_estimates,
        mean=mean,
        correction_mean=correction_mean,
        backbone_estimate=run_results[0].backbone_estimate,
        sample_variance=sample_variance,
        standard_error=standard_error,
        seeds=seeds,
        runs=run_results,
        master_seed=int(master_seed),
        diagnostics=diagnostics,
    )
