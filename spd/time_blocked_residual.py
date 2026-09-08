"""Independent time-blocked residual corrections for the NumPy backend."""

from dataclasses import dataclass
from math import fsum, sqrt
from operator import index as integer_index
import warnings

import numpy as np

from . import numpy_backend
from .circuit_ir import CircuitIR, SkippedOperation
from .randomized import _integer_seed, _positive_integer, estimate_expectation
from .randomized_truncation import RandomizedTruncationStats
from .residual_corrected import (
    NumericalZeroStats,
    _combine_zero_stats,
    _copy_spo,
    _empty_zero_stats,
    _l1_norm,
    _l2_norm_sq,
    _merge_spo,
    _new_spo,
    _propagate_exact,
    _prune_numerical_zeros,
    _top_k_split,
    _truncate_correction,
    _validate_numerical_zero_tolerance,
)
from .run_circuit import _normalize_input_circuit, _resolve_backend_from_state


ALGORITHM_VERSION = "time-blocked-residual-rspd-v1"
SEED_DOMAIN = 4
DEGENERACY_NORM_INFLATION = 1.0e6
DEGENERACY_HEAVY_FRACTION = 0.01


@dataclass(frozen=True)
class BlockBackboneDiagnostics:
    block_index: int
    reverse_gate_start: int
    reverse_gate_stop: int
    injection_support: int
    injection_l1_norm: float
    injection_l2_norm_sq: float
    max_residual_support: int
    max_backbone_candidate_support: int


@dataclass(frozen=True)
class TimeBlockedBackbonePlan:
    reverse_operations: tuple
    circuit_gate_indices: tuple
    block_sizes: tuple
    block_starts: tuple
    boundary_backbones: tuple
    initial_residual: numpy_backend.SparsePauliOp
    final_backbone: numpy_backend.SparsePauliOp
    block_diagnostics: tuple
    numerical_zero_stats: NumericalZeroStats
    backbone_budget: int
    numerical_zero_tolerance: float
    algorithm_version: str
    precision: str

    @property
    def num_blocks(self):
        return len(self.block_sizes)


@dataclass(frozen=True)
class TimeBlockedCorrectionRecord:
    reverse_gate_index: object
    circuit_gate_index: object
    gate_name: str
    residual_support: int
    residual_l1_norm: float
    residual_l2_norm_sq: float
    correction_candidate_support: int
    sampled_correction_support: int
    sampled_correction_l2_norm_sq: float
    cumulative_injection_l2_norm_sq: float
    norm_inflation_ratio: float
    randomized_truncation_stats: RandomizedTruncationStats
    numerical_zero_stats: NumericalZeroStats


@dataclass(frozen=True)
class TimeBlockedBlockDiagnostics:
    records: tuple
    num_randomized_truncations: int
    accumulated_predicted_mse: float
    max_correction_candidate_support: int
    max_sampled_correction_l2_norm_sq: float
    max_norm_inflation_ratio: float
    min_heavy_fraction: float
    first_degeneracy_warning_gate: object
    numerical_zero_stats: NumericalZeroStats


@dataclass(frozen=True)
class TimeBlockedBlockRunResult:
    block_index: int
    correction_budget: int
    seed: int
    correction_estimate: float
    raw_correction_estimate: complex
    imaginary_leakage: float
    terminal_hit: bool
    final_sampled_correction: numpy_backend.SparsePauliOp
    diagnostics: TimeBlockedBlockDiagnostics


@dataclass(frozen=True)
class TimeBlockedBlockSummary:
    block_index: int
    correction_budget: int
    estimates: np.ndarray
    mean: float
    second_moment: float
    sample_variance: float
    standard_error: float
    terminal_hit_fraction: float
    quantiles: tuple
    false_convergence: bool
    seeds: tuple
    runs: tuple


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer.")
    try:
        value = integer_index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer.") from error
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _validate_block_sizes(block_sizes, num_operations):
    sizes = tuple(
        _positive_integer(value, "each block size") for value in block_sizes
    )
    if not sizes:
        raise ValueError("block_sizes must contain at least one block.")
    if sum(sizes) != num_operations:
        raise ValueError(
            "block_sizes must sum to the number of active reverse operations "
            f"({num_operations})."
        )
    return sizes


def _warn_numerical_zeros(stats, tolerance):
    if not stats.count:
        return
    warnings.warn(
        "Time-blocked residual correction removed {} non-exact numerical-zero "
        "values with atol={:.3g} (l1={:.3g}, l2={:.3g}, max={:.3g}). The "
        "estimator is unbiased only up to this numerical-zero policy.".format(
            stats.count,
            tolerance,
            stats.l1_norm,
            stats.l2_norm,
            stats.max_abs_coefficient,
        ),
        RuntimeWarning,
        stacklevel=3,
    )


def _active_reverse_operations(input_circuit, backend, rebase):
    normalized = _normalize_input_circuit(input_circuit, backend, rebase)
    operations = normalized.operations if isinstance(normalized, CircuitIR) else normalized
    active = [
        (gate_index, operation)
        for gate_index, operation in reversed(tuple(enumerate(operations)))
        if not isinstance(operation, SkippedOperation)
    ]
    return tuple(operation for _, operation in active), tuple(
        gate_index for gate_index, _ in active
    )


def _block_starts(block_sizes):
    starts = [0]
    for size in block_sizes[:-1]:
        starts.append(starts[-1] + size)
    return tuple(starts)


def _prepare_time_blocked_backbone(
    spo,
    input_circuit,
    backbone_budget,
    block_sizes,
    *,
    numerical_zero_tolerance,
    rebase,
    emit_warning,
):
    backbone_budget = _positive_integer(backbone_budget, "backbone_budget")
    tolerance = _validate_numerical_zero_tolerance(numerical_zero_tolerance)
    backend = _resolve_backend_from_state(spo, None, state_name="spo")
    if backend.name != "numpy" or not backend.is_spo_instance(spo):
        raise TypeError("time-blocked residual correction requires a NumPy SparsePauliOp.")

    reverse_operations, circuit_gate_indices = _active_reverse_operations(
        input_circuit, backend, rebase
    )
    sizes = _validate_block_sizes(block_sizes, len(reverse_operations))
    starts = _block_starts(sizes)

    initial_candidate, initial_zero_stats = _prune_numerical_zeros(spo, tolerance)
    backbone, initial_residual = _top_k_split(initial_candidate, backbone_budget)
    boundary_backbones = []
    block_diagnostics = []
    zero_stats = [initial_zero_stats]

    for block_index, (start, size) in enumerate(zip(starts, sizes)):
        stop = start + size
        boundary_backbones.append(_copy_spo(backbone))
        injection_support = len(initial_residual) if block_index == 0 else 0
        injection_l1 = _l1_norm(initial_residual) if block_index == 0 else 0.0
        injection_l2_sq = _l2_norm_sq(initial_residual) if block_index == 0 else 0.0
        max_residual_support = len(initial_residual) if block_index == 0 else 0
        max_candidate_support = len(initial_candidate) if block_index == 0 else 0

        for reverse_gate_index in range(start, stop):
            candidate, gate_zero_stats = _propagate_exact(
                backbone,
                reverse_operations[reverse_gate_index],
                backend,
                tolerance,
            )
            backbone, residual = _top_k_split(candidate, backbone_budget)
            zero_stats.append(gate_zero_stats)
            injection_support += len(residual)
            injection_l1 += _l1_norm(residual)
            injection_l2_sq += _l2_norm_sq(residual)
            max_residual_support = max(max_residual_support, len(residual))
            max_candidate_support = max(max_candidate_support, len(candidate))

        block_diagnostics.append(
            BlockBackboneDiagnostics(
                block_index=block_index,
                reverse_gate_start=start,
                reverse_gate_stop=stop,
                injection_support=injection_support,
                injection_l1_norm=injection_l1,
                injection_l2_norm_sq=injection_l2_sq,
                max_residual_support=max_residual_support,
                max_backbone_candidate_support=max_candidate_support,
            )
        )

    all_zero_stats = _combine_zero_stats(*zero_stats)
    if emit_warning:
        _warn_numerical_zeros(all_zero_stats, tolerance)
    return TimeBlockedBackbonePlan(
        reverse_operations=reverse_operations,
        circuit_gate_indices=circuit_gate_indices,
        block_sizes=sizes,
        block_starts=starts,
        boundary_backbones=tuple(boundary_backbones),
        initial_residual=initial_residual,
        final_backbone=backbone,
        block_diagnostics=tuple(block_diagnostics),
        numerical_zero_stats=all_zero_stats,
        backbone_budget=backbone_budget,
        numerical_zero_tolerance=tolerance,
        algorithm_version=ALGORITHM_VERSION,
        precision=numpy_backend.utils.get_precision(),
    )


def prepare_time_blocked_backbone(
    spo,
    input_circuit,
    backbone_budget,
    block_sizes,
    *,
    numerical_zero_tolerance=1.0e-12,
    rebase=False,
):
    """Build deterministic block-boundary snapshots without storing residuals."""
    return _prepare_time_blocked_backbone(
        spo,
        input_circuit,
        backbone_budget,
        block_sizes,
        numerical_zero_tolerance=numerical_zero_tolerance,
        rebase=rebase,
        emit_warning=True,
    )


def _correction_record(
    reverse_gate_index,
    circuit_gate_index,
    gate_name,
    residual,
    candidate,
    sampled,
    cumulative_injection_l2_norm_sq,
    stats,
    zero_stats,
):
    sampled_norm_sq = float(sampled.get_norm_square())
    return TimeBlockedCorrectionRecord(
        reverse_gate_index=reverse_gate_index,
        circuit_gate_index=circuit_gate_index,
        gate_name=gate_name,
        residual_support=len(residual),
        residual_l1_norm=_l1_norm(residual),
        residual_l2_norm_sq=_l2_norm_sq(residual),
        correction_candidate_support=len(candidate),
        sampled_correction_support=len(sampled),
        sampled_correction_l2_norm_sq=sampled_norm_sq,
        cumulative_injection_l2_norm_sq=cumulative_injection_l2_norm_sq,
        norm_inflation_ratio=(
            sampled_norm_sq / cumulative_injection_l2_norm_sq
            if cumulative_injection_l2_norm_sq
            else 0.0
        ),
        randomized_truncation_stats=stats,
        numerical_zero_stats=zero_stats,
    )


def _run_time_blocked_residual_block(
    plan,
    block_index,
    correction_budget,
    *,
    seed,
    basis,
    state_overlap,
    emit_warning,
):
    if not isinstance(plan, TimeBlockedBackbonePlan):
        raise TypeError("plan must be a TimeBlockedBackbonePlan.")
    block_index = _nonnegative_integer(block_index, "block_index")
    if block_index >= plan.num_blocks:
        raise ValueError("block_index is outside the plan.")
    correction_budget = _positive_integer(correction_budget, "correction_budget")
    seed = _integer_seed(seed, "seed")
    rng = np.random.default_rng(seed)
    backend = _resolve_backend_from_state(
        plan.final_backbone, None, state_name="plan.final_backbone"
    )
    tolerance = plan.numerical_zero_tolerance
    start = plan.block_starts[block_index]
    stop = start + plan.block_sizes[block_index]
    backbone = _copy_spo(plan.boundary_backbones[block_index])
    correction_candidate = (
        _copy_spo(plan.initial_residual) if block_index == 0 else _new_spo()
    )
    correction, initial_stats = _truncate_correction(
        correction_candidate, correction_budget, rng
    )
    cumulative_injection_l2_norm_sq = _l2_norm_sq(correction_candidate)
    records = []
    zero_stats = []
    if correction_candidate:
        records.append(
            _correction_record(
                None,
                None,
                "initial observable",
                correction_candidate,
                correction_candidate,
                correction,
                cumulative_injection_l2_norm_sq,
                initial_stats,
                _empty_zero_stats(),
            )
        )

    for reverse_gate_index in range(start, len(plan.reverse_operations)):
        operation = plan.reverse_operations[reverse_gate_index]
        propagated, correction_zero_stats = _propagate_exact(
            correction, operation, backend, tolerance
        )
        residual = _new_spo()
        merge_zero_stats = _empty_zero_stats()
        if reverse_gate_index < stop:
            backbone_candidate, _ = _propagate_exact(
                backbone, operation, backend, tolerance
            )
            backbone, residual = _top_k_split(
                backbone_candidate, plan.backbone_budget
            )
            cumulative_injection_l2_norm_sq += _l2_norm_sq(residual)
            candidate, _, merge_zero_stats = _merge_spo(
                propagated, residual, tolerance
            )
        else:
            candidate = propagated
        correction, stats = _truncate_correction(candidate, correction_budget, rng)
        gate_zero_stats = _combine_zero_stats(
            correction_zero_stats, merge_zero_stats
        )
        zero_stats.append(gate_zero_stats)
        records.append(
            _correction_record(
                reverse_gate_index,
                plan.circuit_gate_indices[reverse_gate_index],
                operation.gate_name,
                residual,
                candidate,
                correction,
                cumulative_injection_l2_norm_sq,
                stats,
                gate_zero_stats,
            )
        )

    all_zero_stats = _combine_zero_stats(*zero_stats)
    if emit_warning:
        _warn_numerical_zeros(all_zero_stats, tolerance)
    raw_estimate = estimate_expectation(
        correction, basis=basis, state_overlap=state_overlap
    )
    truncation_records = [
        record
        for record in records
        if record.randomized_truncation_stats.support_before > correction_budget
    ]
    heavy_fractions = [
        record.randomized_truncation_stats.heavy_count / correction_budget
        for record in truncation_records
    ]
    degeneracy_records = [
        record
        for record in truncation_records
        if record.norm_inflation_ratio >= DEGENERACY_NORM_INFLATION
        and record.randomized_truncation_stats.heavy_count / correction_budget
        <= DEGENERACY_HEAVY_FRACTION
    ]
    diagnostics = TimeBlockedBlockDiagnostics(
        records=tuple(records),
        num_randomized_truncations=len(truncation_records),
        accumulated_predicted_mse=fsum(
            record.randomized_truncation_stats.predicted_mse for record in records
        ),
        max_correction_candidate_support=max(
            (record.correction_candidate_support for record in records), default=0
        ),
        max_sampled_correction_l2_norm_sq=max(
            (record.sampled_correction_l2_norm_sq for record in records), default=0.0
        ),
        max_norm_inflation_ratio=max(
            (record.norm_inflation_ratio for record in records), default=0.0
        ),
        min_heavy_fraction=min(heavy_fractions, default=1.0),
        first_degeneracy_warning_gate=(
            degeneracy_records[0].reverse_gate_index if degeneracy_records else None
        ),
        numerical_zero_stats=all_zero_stats,
    )
    return TimeBlockedBlockRunResult(
        block_index=block_index,
        correction_budget=correction_budget,
        seed=int(seed),
        correction_estimate=float(raw_estimate.real),
        raw_correction_estimate=raw_estimate,
        imaginary_leakage=float(abs(raw_estimate.imag)),
        terminal_hit=bool(raw_estimate != 0.0),
        final_sampled_correction=correction,
        diagnostics=diagnostics,
    )


def run_time_blocked_residual_block(
    plan,
    block_index,
    correction_budget,
    *,
    seed,
    basis="0",
    state_overlap=None,
):
    """Return one independently seeded terminal correction for one block."""
    return _run_time_blocked_residual_block(
        plan,
        block_index,
        correction_budget,
        seed=seed,
        basis=basis,
        state_overlap=state_overlap,
        emit_warning=True,
    )


def time_blocked_run_seed(
    master_seed,
    block_index,
    correction_budget,
    run_index,
    *,
    stage=0,
):
    """Derive a stable seed from stage, block, budget, and run index."""
    master_seed = _integer_seed(master_seed, "master_seed")
    block_index = _nonnegative_integer(block_index, "block_index")
    correction_budget = _positive_integer(correction_budget, "correction_budget")
    run_index = _nonnegative_integer(run_index, "run_index")
    stage = _nonnegative_integer(stage, "stage")
    spawn_key = (SEED_DOMAIN, stage, block_index, correction_budget, run_index)
    if any(value >= 2**32 for value in spawn_key):
        raise ValueError("seed hierarchy indices and budgets must be below 2**32.")
    return int(
        np.random.SeedSequence(master_seed, spawn_key=spawn_key).generate_state(
            1, dtype=np.uint64
        )[0]
    )


def summarize_time_blocked_block_runs(runs):
    """Summarize independent runs that share a block and correction budget."""
    runs = tuple(runs)
    if not runs:
        raise ValueError("runs must not be empty.")
    block_index = runs[0].block_index
    correction_budget = runs[0].correction_budget
    if any(
        run.block_index != block_index
        or run.correction_budget != correction_budget
        for run in runs
    ):
        raise ValueError("all runs must share a block index and correction budget.")
    estimates = np.asarray(
        [run.correction_estimate for run in runs], dtype=np.float64
    )
    count = len(runs)
    sample_variance = (
        float(np.var(estimates, ddof=1, dtype=np.float64))
        if count > 1
        else float("nan")
    )
    hit_fraction = float(np.mean([run.terminal_hit for run in runs]))
    has_injected_correction = any(
        record.residual_support > 0
        for run in runs
        for record in run.diagnostics.records
    )
    return TimeBlockedBlockSummary(
        block_index=block_index,
        correction_budget=correction_budget,
        estimates=estimates,
        mean=float(np.mean(estimates, dtype=np.float64)),
        second_moment=float(np.mean(estimates * estimates, dtype=np.float64)),
        sample_variance=sample_variance,
        standard_error=sqrt(sample_variance / count) if count > 1 else float("nan"),
        terminal_hit_fraction=hit_fraction,
        quantiles=tuple(
            float(value) for value in np.quantile(estimates, [0.0, 0.25, 0.5, 0.75, 1.0])
        ),
        false_convergence=bool(
            has_injected_correction
            and count > 1
            and sample_variance == 0.0
            and hit_fraction == 0.0
        ),
        seeds=tuple(run.seed for run in runs),
        runs=runs,
    )


__all__ = [
    "ALGORITHM_VERSION",
    "DEGENERACY_HEAVY_FRACTION",
    "DEGENERACY_NORM_INFLATION",
    "BlockBackboneDiagnostics",
    "TimeBlockedBackbonePlan",
    "TimeBlockedCorrectionRecord",
    "TimeBlockedBlockDiagnostics",
    "TimeBlockedBlockRunResult",
    "TimeBlockedBlockSummary",
    "prepare_time_blocked_backbone",
    "run_time_blocked_residual_block",
    "summarize_time_blocked_block_runs",
    "time_blocked_run_seed",
]
