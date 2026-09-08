"""Randomized Sparse Pauli Dynamics (R-SPD) for the NumPy backend."""

from dataclasses import dataclass
from math import fsum, sqrt
from operator import index as integer_index

import numpy as np

from . import numpy_backend
from .circuit_ir import CircuitIR, PauliRotation, SkippedOperation
from .randomized_truncation import RandomizedTruncationStats, pivotal_truncate
from .run_circuit import _normalize_input_circuit, _resolve_backend_from_state


# Retained as a persisted compatibility identifier for existing checkpoints.
# The algorithm's canonical name is R-SPD.
ALGORITHM_VERSION = "fp-spp-v1"


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("{} must be an integer.".format(name))
    try:
        value = integer_index(value)
    except TypeError as error:
        raise TypeError("{} must be an integer.".format(name)) from error
    if value < 1:
        raise ValueError("{} must be positive.".format(name))
    return value


def _integer_seed(seed, name):
    if isinstance(seed, (bool, np.bool_)):
        raise TypeError("{} must be an integer.".format(name))
    try:
        seed = integer_index(seed)
    except TypeError as error:
        raise TypeError("{} must be an integer.".format(name)) from error
    if seed < 0:
        raise ValueError("{} must be nonnegative.".format(name))
    return seed


@dataclass(frozen=True)
class RandomizedTruncationRecord:
    gate_index: object
    gate_name: str
    raw_children: int
    merged_children: int
    cancellation_ratio: float
    population_l2_norm_sq: float
    relative_local_noise: float
    stats: RandomizedTruncationStats

    @property
    def sampled_spo_l2_norm_sq(self):
        """Squared norm after truncation (canonical R-SPD terminology)."""
        return self.population_l2_norm_sq


@dataclass(frozen=True)
class EvolutionDiagnostics:
    compression_records: tuple
    num_compressions: int
    accumulated_predicted_mse: float
    total_raw_children: int
    total_merged_children: int
    max_candidate_support: int

    @property
    def randomized_truncation_records(self):
        return self.compression_records

    @property
    def num_randomized_truncations(self):
        return self.num_compressions


@dataclass(frozen=True)
class RandomizedEvolutionResult:
    final_sampled_spo: numpy_backend.SparsePauliOp
    diagnostics: EvolutionDiagnostics
    seed: int
    algorithm_version: str
    precision: str

    @property
    def final_population(self):
        """Compatibility alias for ``final_sampled_spo``."""
        return self.final_sampled_spo


@dataclass(frozen=True)
class RunResult:
    estimate: float
    raw_estimate: complex
    imaginary_leakage: float
    evolution: RandomizedEvolutionResult

    @property
    def final_population(self):
        """Compatibility alias for ``final_sampled_spo``."""
        return self.evolution.final_sampled_spo

    @property
    def final_sampled_spo(self):
        return self.evolution.final_sampled_spo

    @property
    def diagnostics(self):
        return self.evolution.diagnostics

    @property
    def seed(self):
        return self.evolution.seed


@dataclass(frozen=True)
class EnsembleDiagnostics:
    total_compressions: int
    accumulated_predicted_mse: float
    total_raw_children: int
    total_merged_children: int
    max_candidate_support: int
    max_imaginary_leakage: float

    @property
    def total_randomized_truncations(self):
        return self.total_compressions


@dataclass(frozen=True)
class EnsembleResult:
    estimates: np.ndarray
    mean: float
    sample_variance: float
    standard_error: float
    seeds: tuple
    runs: tuple
    master_seed: int
    diagnostics: EnsembleDiagnostics


def _as_sampled_spo(coefficients):
    result = numpy_backend.SparsePauliOp()
    for key, value in coefficients.items():
        if value != 0.0:
            result[key] = value
            if not np.isfinite(result[key]):
                raise FloatingPointError(
                    "Sampled SPO contains a non-finite coefficient in NumPy precision."
                )
    return result


def _raw_rotation_diagnostics(sampled_spo, operation):
    generator = numpy_backend.utils.pauli_str_to_uint(operation.pauli)
    cosine = float(np.cos(operation.theta))
    sine = float(np.sin(operation.theta))
    raw_children = 0
    raw_l1_norm = 0.0
    for key, value in sampled_spo.items():
        anticommutes = numpy_backend.kernels.check_anticommute_uint(
            np.asarray(key), generator
        )
        if anticommutes:
            child_weights = (abs(value * cosine), abs(value * sine))
            raw_children += sum(weight != 0.0 for weight in child_weights)
            raw_l1_norm += fsum(child_weights)
        else:
            raw_children += 1
            raw_l1_norm += abs(value)
    return raw_children, raw_l1_norm


def _candidate_diagnostics(sampled_spo, operation):
    if isinstance(operation, PauliRotation):
        return _raw_rotation_diagnostics(sampled_spo, operation)
    return len(sampled_spo), fsum(abs(value) for value in sampled_spo.values())


def _randomly_truncate(sampled_spo, pauli_budget, rng, gate_index, gate_name,
                       raw_children, raw_l1_norm):
    truncated, stats = pivotal_truncate(sampled_spo, pauli_budget, rng)
    truncated = _as_sampled_spo(truncated)
    merged_l1_norm = fsum(abs(value) for value in sampled_spo.values())
    cancellation_ratio = (
        0.0 if raw_l1_norm == 0.0 else 1.0 - merged_l1_norm / raw_l1_norm
    )
    sampled_spo_l2_norm_sq = truncated.get_norm_square()
    relative_local_noise = (
        0.0
        if stats.l2_norm_sq == 0.0
        else stats.predicted_mse / stats.l2_norm_sq
    )
    record = RandomizedTruncationRecord(
        gate_index=gate_index,
        gate_name=gate_name,
        raw_children=int(raw_children),
        merged_children=len(sampled_spo),
        cancellation_ratio=float(cancellation_ratio),
        population_l2_norm_sq=float(sampled_spo_l2_norm_sq),
        relative_local_noise=float(relative_local_noise),
        stats=stats,
    )
    return truncated, record


def evolve_randomized(spo, input_circuit, pauli_budget=None, *, seed,
                      rebase=False, population_size=None):
    """Evolve one sampled SPO with unbiased R-SPD on the NumPy backend.

    Each Pauli rotation is evaluated by the existing NumPy propagation kernel
    with zero deterministic threshold and transient capacity ``2 * pauli_budget``.

    ``population_size`` is a compatibility alias for ``pauli_budget``.
    """
    pauli_budget = _resolve_renamed_argument(
        pauli_budget,
        population_size,
        canonical_name="pauli_budget",
        legacy_name="population_size",
    )
    pauli_budget = _positive_integer(pauli_budget, "pauli_budget")
    seed = _integer_seed(seed, "seed")
    backend = _resolve_backend_from_state(spo, None, state_name="spo")
    if backend.name != "numpy" or not backend.is_spo_instance(spo):
        raise TypeError("evolve_randomized requires a NumPy SparsePauliOp.")

    normalized = _normalize_input_circuit(input_circuit, backend, rebase)
    operations = normalized.operations if isinstance(normalized, CircuitIR) else normalized
    rng = np.random.default_rng(seed)
    sampled_spo = _as_sampled_spo(spo)
    records = []
    total_raw_children = 0
    total_merged_children = 0
    max_candidate_support = len(sampled_spo)

    if len(sampled_spo) > pauli_budget:
        sampled_spo, record = _randomly_truncate(
            sampled_spo,
            pauli_budget,
            rng,
            gate_index=None,
            gate_name="initial observable",
            raw_children=len(sampled_spo),
            raw_l1_norm=fsum(abs(value) for value in sampled_spo.values()),
        )
        records.append(record)

    for reverse_step, operation in enumerate(reversed(operations)):
        if isinstance(operation, SkippedOperation):
            continue
        raw_children, raw_l1_norm = _candidate_diagnostics(sampled_spo, operation)
        next_sampled_spo, _, _, _ = backend.apply_forward(
            sampled_spo,
            operation,
            trunc_val=0.0,
            max_num_str=2 * pauli_budget,
        )
        candidates = _as_sampled_spo(next_sampled_spo)
        merged_children = len(candidates)
        total_raw_children += raw_children
        total_merged_children += merged_children
        max_candidate_support = max(max_candidate_support, merged_children)

        if merged_children > pauli_budget:
            gate_index = len(operations) - 1 - reverse_step
            sampled_spo, record = _randomly_truncate(
                candidates,
                pauli_budget,
                rng,
                gate_index=gate_index,
                gate_name=operation.gate_name,
                raw_children=raw_children,
                raw_l1_norm=raw_l1_norm,
            )
            records.append(record)
        else:
            sampled_spo = candidates

    diagnostics = EvolutionDiagnostics(
        compression_records=tuple(records),
        num_compressions=len(records),
        accumulated_predicted_mse=fsum(
            record.stats.predicted_mse for record in records
        ),
        total_raw_children=int(total_raw_children),
        total_merged_children=int(total_merged_children),
        max_candidate_support=int(max_candidate_support),
    )
    return RandomizedEvolutionResult(
        final_sampled_spo=sampled_spo,
        diagnostics=diagnostics,
        seed=int(seed),
        algorithm_version=ALGORITHM_VERSION,
        precision=numpy_backend.utils.get_precision(),
    )


def estimate_expectation(sampled_spo, *, basis="0", state_overlap=None):
    """Estimate the expectation from one final sampled SPO."""
    if not sampled_spo:
        return 0.0j
    if state_overlap is None:
        return complex(sampled_spo.get_expectation_value(basis=basis))
    return sum(
        complex(coefficient) * complex(state_overlap(key))
        for key, coefficient in sampled_spo.items()
    )


def run_randomized_spd(spo, input_circuit, pauli_budget=None, *, seed, basis="0",
                       state_overlap=None, rebase=False, population_size=None):
    """Return one estimator sample from an independently seeded R-SPD run.

    ``population_size`` is a compatibility alias for ``pauli_budget``.
    """
    pauli_budget = _resolve_renamed_argument(
        pauli_budget,
        population_size,
        canonical_name="pauli_budget",
        legacy_name="population_size",
    )
    evolution = evolve_randomized(
        spo,
        input_circuit,
        pauli_budget,
        seed=seed,
        rebase=rebase,
    )
    raw_estimate = estimate_expectation(
        evolution.final_sampled_spo,
        basis=basis,
        state_overlap=state_overlap,
    )
    return RunResult(
        estimate=float(raw_estimate.real),
        raw_estimate=raw_estimate,
        imaginary_leakage=float(abs(raw_estimate.imag)),
        evolution=evolution,
    )


def _run_seeds(master_seed, runs):
    seed_sequence = np.random.SeedSequence(master_seed)
    return tuple(
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in seed_sequence.spawn(runs)
    )


def run_ensemble(spo, input_circuit, pauli_budget=None, runs=None, *,
                 master_seed, basis="0", state_overlap=None, rebase=False,
                 population_size=None, num_populations=None):
    """Run independent R-SPD realizations and return Monte Carlo statistics.

    ``population_size`` and ``num_populations`` are compatibility aliases for
    ``pauli_budget`` and ``runs`` respectively.
    """
    pauli_budget = _resolve_renamed_argument(
        pauli_budget,
        population_size,
        canonical_name="pauli_budget",
        legacy_name="population_size",
    )
    runs = _resolve_renamed_argument(
        runs,
        num_populations,
        canonical_name="runs",
        legacy_name="num_populations",
    )
    pauli_budget = _positive_integer(pauli_budget, "pauli_budget")
    runs = _positive_integer(runs, "runs")
    master_seed = _integer_seed(master_seed, "master_seed")
    seeds = _run_seeds(master_seed, runs)
    run_results = tuple(
        run_randomized_spd(
            spo,
            input_circuit,
            pauli_budget,
            seed=seed,
            basis=basis,
            state_overlap=state_overlap,
            rebase=rebase,
        )
        for seed in seeds
    )
    estimates = np.asarray([run.estimate for run in run_results], dtype=np.float64)
    mean = float(np.mean(estimates, dtype=np.float64))
    sample_variance = (
        float(np.var(estimates, ddof=1, dtype=np.float64))
        if runs > 1
        else float("nan")
    )
    standard_error = (
        sqrt(sample_variance / runs)
        if runs > 1
        else float("nan")
    )
    diagnostics = EnsembleDiagnostics(
        total_compressions=sum(
            run.diagnostics.num_compressions for run in run_results
        ),
        accumulated_predicted_mse=fsum(
            run.diagnostics.accumulated_predicted_mse for run in run_results
        ),
        total_raw_children=sum(
            run.diagnostics.total_raw_children for run in run_results
        ),
        total_merged_children=sum(
            run.diagnostics.total_merged_children for run in run_results
        ),
        max_candidate_support=max(
            run.diagnostics.max_candidate_support for run in run_results
        ),
        max_imaginary_leakage=max(
            run.imaginary_leakage for run in run_results
        ),
    )
    return EnsembleResult(
        estimates=estimates,
        mean=mean,
        sample_variance=sample_variance,
        standard_error=standard_error,
        seeds=seeds,
        runs=run_results,
        master_seed=int(master_seed),
        diagnostics=diagnostics,
    )


def run_population(spo, input_circuit, population_size=None, *, seed, basis="0",
                   state_overlap=None, rebase=False, pauli_budget=None):
    """Compatibility wrapper for :func:`run_randomized_spd`."""
    return run_randomized_spd(
        spo,
        input_circuit,
        pauli_budget=pauli_budget,
        population_size=population_size,
        seed=seed,
        basis=basis,
        state_overlap=state_overlap,
        rebase=rebase,
    )


def _resolve_renamed_argument(canonical, legacy, *, canonical_name, legacy_name):
    if canonical is None:
        if legacy is None:
            raise TypeError("{} is required.".format(canonical_name))
        return legacy
    if legacy is not None and legacy != canonical:
        raise ValueError(
            "{} and its compatibility alias {} disagree.".format(
                canonical_name, legacy_name
            )
        )
    return canonical


# Compatibility aliases for the first prototype API and persisted pickles.
CompressionRecord = RandomizedTruncationRecord
PopulationResult = RunResult
