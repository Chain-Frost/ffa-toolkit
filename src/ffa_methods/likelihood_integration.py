"""Internal source-preserving WP02/WP03/WP05 mathematical integration.

This boundary admits *explicitly synthetic* evidence only. The native Flike
file3.10 parser, unknown flag meanings, default priors, MGBT censor rules and
rating law do not become verified merely because numeric components work.
This is not a separately supported Python API or a CLI fitting route.
"""

import re
from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

from ffa_methods.contracts import DistributionKernel, ExposureKind, ObservationKind, ParameterCoordinate
from ffa_methods.distributions import GEV, GeneralizedPareto, Gumbel, Lognormal, LogPearsonIII
from ffa_methods.legacy.adapter import MappedAnnualSeries
from ffa_methods.legacy.models import ExposureKind as LegacyExposureKind
from ffa_methods.legacy.models import ObservationKind as LegacyObservationKind
from ffa_methods.likelihood import HistoricalPeriod, LikelihoodResult, log_likelihood
from ffa_methods.priors import CoordinateTransform, LegacyPriorSelection, PriorEvidenceError
from ffa_methods.rating_errors import (
    JointRatingScenario,
    RatingEpoch,
    marginal_shared_rating_log_likelihood,
)

WP03Kernel: TypeAlias = Lognormal | LogPearsonIII | Gumbel | GEV | GeneralizedPareto


class SyntheticIntegrationError(ValueError):
    """Source mapping or assumptions are insufficient for a mathematical check."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SyntheticEvidenceDeclaration:
    """Explicit synthetic-only source/unit and historical-record identity.

    A digest is a caller-supplied identifier, not independent custody proof.
    Every source row is retained in MappedAnnualSeries; none is reclassified.
    """

    purpose: str
    evidence_id: str
    source_sha256: str
    discharge_unit: str
    kernel_discharge_unit: str
    source_period_ids: tuple[str, ...]
    historical_record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.purpose != "independent_synthetic_mathematics":
            msg = "only explicitly independent synthetic mathematical checks are admitted"
            raise SyntheticIntegrationError(msg)
        if not self.evidence_id.startswith("synthetic:") or not self.evidence_id.strip():
            msg = "synthetic evidence needs a source-identified synthetic: label"
            raise SyntheticIntegrationError(msg)
        if re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None:
            msg = "synthetic source digest must be a lowercase SHA-256 value"
            raise SyntheticIntegrationError(msg)
        if not self.discharge_unit.strip() or self.discharge_unit.strip() != self.kernel_discharge_unit.strip():
            msg = "synthetic observations and bound kernel require one recorded discharge unit"
            raise SyntheticIntegrationError(msg)
        if len(set(self.source_period_ids)) != len(self.source_period_ids):
            msg = "synthetic source period IDs must be unique"
            raise SyntheticIntegrationError(msg)
        if len(set(self.historical_record_ids)) != len(self.historical_record_ids):
            msg = "synthetic historical record IDs must be unique"
            raise SyntheticIntegrationError(msg)


@dataclass(frozen=True, slots=True)
class SyntheticLikelihoodEvaluation:
    """Auditable component result; it is never a qualified legacy fit."""

    likelihood: LikelihoodResult
    evidence_id: str
    source_sha256: str
    discharge_unit: str
    annual_years: tuple[int, ...]
    historical_records: tuple[str, ...]
    legacy_qualified: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class SyntheticRatingEpoch:
    """One mapped, source-identified synthetic epoch with an explicit rating ID."""

    identifier: str
    mapped: MappedAnnualSeries
    declaration: SyntheticEvidenceDeclaration
    historical_periods: tuple[HistoricalPeriod, ...] = ()


@dataclass(frozen=True, slots=True)
class SyntheticRatingEvaluation:
    """Joint error result with source IDs and no asserted Flike error law."""

    log_value: float
    evidence_ids: tuple[str, ...]
    legacy_qualified: bool = False


@dataclass(frozen=True, slots=True)
class _WP03KernelContract:
    """Delegate a WP03 bound kernel under WP01's explicitly named PPF/ISF protocol.

    WP03 calls the argument 'probability'; WP01's DistributionKernel calls
    the same value 'non_exceedance_probability' or 'survival_probability'.
    This is a keyword-only signature bridge, not a second numerical kernel.
    """

    model: WP03Kernel

    def logpdf(self, discharge: float) -> float:
        return self.model.logpdf(discharge)

    def logcdf(self, discharge: float) -> float:
        return self.model.logcdf(discharge)

    def logsf(self, discharge: float) -> float:
        return self.model.logsf(discharge)

    def cdf(self, discharge: float) -> float:
        return self.model.cdf(discharge)

    def sf(self, discharge: float) -> float:
        return self.model.sf(discharge)

    def ppf(self, non_exceedance_probability: float) -> float:
        return self.model.ppf(non_exceedance_probability)

    def isf(self, survival_probability: float) -> float:
        return self.model.isf(survival_probability)


def _as_shared_kernel(model: WP03Kernel) -> DistributionKernel:
    """Enforce structural agreement at this WP03/WP05 consumer boundary."""
    return _WP03KernelContract(model)


def _validate_synthetic_periods(mapped: MappedAnnualSeries, declaration: SyntheticEvidenceDeclaration) -> None:
    """Verify synthetic unit/exposure controls and source period identities."""
    if mapped.ready_for_likelihood or mapped.series.source_file_version != "synthetic-component-v1":
        msg = "native/project control evidence cannot be admitted as a synthetic likelihood"
        raise SyntheticIntegrationError(msg)
    if mapped.series.zero_threshold is not None or mapped.series.groups:
        msg = "zero-threshold or grouped WP02 evidence has no admitted likelihood mapping"
        raise SyntheticIntegrationError(msg)
    if mapped.series.discharge_unit.strip() != declaration.discharge_unit.strip():
        msg = "source observation discharge unit differs from the declared bound kernel unit"
        raise SyntheticIntegrationError(msg)
    if not mapped.exposure_audit.ready_for_likelihood or mapped.exposure_audit.grouped_years:
        msg = "unresolved or grouped WP02 exposure cannot be admitted"
        raise SyntheticIntegrationError(msg)

    periods = mapped.source_periods
    if declaration.source_period_ids != tuple(period.identifier for period in periods):
        msg = "source exposure period IDs do not match their admitted provenance"
        raise SyntheticIntegrationError(msg)
    if any(
        period.kind != LegacyExposureKind.SYSTEMATIC or period.perception_threshold is not None for period in periods
    ):
        msg = "historical source perception/selection rules remain unverified"
        raise SyntheticIntegrationError(msg)


def _validate_synthetic_rows(mapped: MappedAnnualSeries, declaration: SyntheticEvidenceDeclaration) -> None:
    """Cross-check every typed row against retained WP02 Decimal/raw evidence."""
    periods = mapped.source_periods
    rows = mapped.series.annual
    records = mapped.source_records
    covered = frozenset(year for period in periods for year in period.years)
    if (
        len(rows) != len(records)
        or covered != mapped.exposure_audit.covered_years
        or covered != frozenset(row.year for row in rows)
        or len(covered) != len(rows)
    ):
        msg = "annual rows and source exposure membership are inconsistent"
        raise SyntheticIntegrationError(msg)
    for row, record in zip(rows, records, strict=True):
        if record.annual != row or record.legacy.year != row.year or row.exposure != ExposureKind.SYSTEMATIC:
            msg = "annual row differs from its original source-mapped observation"
            raise SyntheticIntegrationError(msg)
        if record.source_sha256 != declaration.source_sha256 or not record.raw_source_line.strip():
            msg = "annual source line or SHA-256 provenance does not match declaration"
            raise SyntheticIntegrationError(msg)
        if record.legacy.source != declaration.evidence_id or record.legacy.raw_flag is not None:
            msg = "real/flagged project rows cannot be silently admitted as synthetic observations"
            raise SyntheticIntegrationError(msg)
        if record.legacy.kind == LegacyObservationKind.VERIFIED_TRUE_ZERO or (
            row.kind == ObservationKind.EXACT and (row.value is None or row.value <= 0.0)
        ):
            msg = "zero-flow/zero-threshold treatment is not admitted in the synthetic continuous path"
            raise SyntheticIntegrationError(msg)
        if row.kind == ObservationKind.AT_OR_BELOW_ZERO_THRESHOLD:
            msg = "legacy zero-threshold likelihood remains unavailable"
            raise SyntheticIntegrationError(msg)


def _validate_synthetic_history(
    mapped: MappedAnnualSeries,
    declaration: SyntheticEvidenceDeclaration,
    historical_periods: tuple[HistoricalPeriod, ...],
) -> None:
    """Require explicit source-labelled disjoint history before likelihood use."""
    rows = mapped.series.annual
    record_ids = tuple(period.source_record for period in historical_periods)
    if any(record_id is None or not record_id.startswith("synthetic:") for record_id in record_ids):
        msg = "historical periods need explicit synthetic source-record identities"
        raise SyntheticIntegrationError(msg)
    if record_ids != declaration.historical_record_ids:
        msg = "historical record identity differs from declared evidence"
        raise SyntheticIntegrationError(msg)
    if any(period.threshold < 0.0 for period in historical_periods):
        msg = "synthetic historical discharge thresholds must not be negative"
        raise SyntheticIntegrationError(msg)
    if not rows and not historical_periods:
        msg = "synthetic likelihood requires actual classified observations or history"
        raise SyntheticIntegrationError(msg)


def _admit_synthetic_source(
    mapped: MappedAnnualSeries,
    declaration: SyntheticEvidenceDeclaration,
    historical_periods: tuple[HistoricalPeriod, ...],
) -> None:
    """Check original WP02 evidence *before* using its numeric annual view."""
    _validate_synthetic_periods(mapped, declaration)
    _validate_synthetic_rows(mapped, declaration)
    _validate_synthetic_history(mapped, declaration, historical_periods)


def evaluate_synthetic_likelihood(
    kernel: WP03Kernel,
    mapped: MappedAnnualSeries,
    *,
    declaration: SyntheticEvidenceDeclaration,
    historical_periods: tuple[HistoricalPeriod, ...] = (),
) -> SyntheticLikelihoodEvaluation:
    """Evaluate admitted mathematical evidence; never activate a Flike fit."""
    _admit_synthetic_source(mapped, declaration, historical_periods)
    likelihood = log_likelihood(_as_shared_kernel(kernel), mapped.series, historical_periods=historical_periods)
    return SyntheticLikelihoodEvaluation(
        likelihood=likelihood,
        evidence_id=declaration.evidence_id,
        source_sha256=declaration.source_sha256,
        discharge_unit=mapped.series.discharge_unit,
        annual_years=tuple(row.year for row in mapped.series.annual),
        historical_records=declaration.historical_record_ids,
    )


def evaluate_synthetic_shared_rating(
    kernel: WP03Kernel,
    epochs: tuple[SyntheticRatingEpoch, ...],
    scenarios: tuple[JointRatingScenario, ...],
) -> SyntheticRatingEvaluation:
    """Reuse WP05 joint rating math with all WP02 source records admitted first."""
    admitted: list[RatingEpoch] = []
    ids: list[str] = []
    for epoch in epochs:
        _admit_synthetic_source(epoch.mapped, epoch.declaration, epoch.historical_periods)
        admitted.append(
            RatingEpoch(
                identifier=epoch.identifier, series=epoch.mapped.series, historical_periods=epoch.historical_periods
            )
        )
        ids.append(epoch.declaration.evidence_id)
    value = marginal_shared_rating_log_likelihood(_as_shared_kernel(kernel), tuple(admitted), scenarios)
    if not isfinite(value) and value != float("-inf"):
        msg = "joint synthetic rating result is numerically invalid"
        raise ValueError(msg)
    return SyntheticRatingEvaluation(value, tuple(ids))


def evaluate_synthetic_source_prior(
    prior: LegacyPriorSelection,
    *,
    source: str,
    coordinates: tuple[ParameterCoordinate | str, ...],
    values: tuple[float, ...],
    transformations: tuple[CoordinateTransform, ...],
) -> float:
    """Evaluate one explicitly synthetic source-coordinate density, not a fit.

    No default/no-prior density, proposal sampler or source-to-kernel parameter
    binding is inferred. The calling test supplies and verifies that binding.
    """
    if prior.supplied is None:
        return prior.logpdf(values)  # Explicitly rejects unknown legacy default/no-prior.
    if not source.startswith("synthetic:") or source != prior.supplied.source:
        msg = "regional/supplied synthetic prior source has not been identified"
        raise PriorEvidenceError(msg)
    if coordinates != prior.supplied.coordinates:
        msg = "prior source coordinates/order do not match the declared parameter vector"
        raise PriorEvidenceError(msg)
    return prior.supplied.logpdf_in_evaluator_coordinates(values, transformations=transformations)
