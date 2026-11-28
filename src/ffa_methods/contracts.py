"""Shared WP01 interfaces; these preserve evidence without imposing unverified model rules.

These types are internal composition boundaries for the supported CLI, not a separately
supported Python API. WP02 owns parsing and source custody; WP03 owns numerical kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import TYPE_CHECKING, Literal, Protocol

from ffa_methods.gev_support import GEVFittedSupport

if TYPE_CHECKING:
    from ffa_methods.importance_precision import PrecisionAudit


class ExposureKind(StrEnum):
    """Observed annual exposure. An absent year is not a zero discharge."""

    SYSTEMATIC = "systematic"
    HISTORICAL = "historical"
    UNRESOLVED = "unresolved"


class ObservationKind(StrEnum):
    """Distinct observation states; do not silently equate their likelihoods."""

    EXACT = "exact"
    LEFT_CENSORED = "left_censored"
    RIGHT_CENSORED = "right_censored"
    INTERVAL_CENSORED = "interval_censored"
    AT_OR_BELOW_ZERO_THRESHOLD = "at_or_below_zero_threshold"
    EXCLUDED = "excluded"
    MISSING = "missing"


class EvidenceStatus(StrEnum):
    """Admissibility of a legacy case, independent of mathematical verification."""

    OBSERVED_ARTIFACT = "observed_artifact"
    CANDIDATE_PAIR = "candidate_pair"
    VERIFIED_PAIR = "verified_pair"
    MISMATCHED_OR_UNRESOLVED = "mismatched_or_unresolved"
    QUALIFIED_REFERENCE = "qualified_reference"


class ProbabilityKind(StrEnum):
    """AEP=1-F is not the same column as a non-exceedance probability F."""

    NON_EXCEEDANCE = "non_exceedance"
    ANNUAL_EXCEEDANCE = "annual_exceedance"


class ParameterCoordinate(StrEnum):
    """The report's LP3/posterior coordinate names; not their transformed values."""

    LP3_MEAN_LOGE_FLOW = "lp3.mean_loge_flow"
    LP3_LOGE_SD_LOGE_FLOW = "lp3.loge_sd_loge_flow"
    LP3_SKEW_LOGE_FLOW = "lp3.skew_loge_flow"
    ZERO_THRESHOLD_LOGIT_PROBABILITY = "zero_threshold.logit_p_flow_at_or_below_threshold"


class ResultMeaning(StrEnum):
    """Different legacy report quantities must never be combined by label alone."""

    QUANTILE_AT_EXPECTED_PARAMETERS = "quantile_at_expected_parameters"
    DETERMINISTIC_FITTED_QUANTILE = "deterministic_fitted_quantile"
    POSTERIOR_MEAN_QUANTILE = "posterior_mean_quantile"
    POSTERIOR_MEDIAN_QUANTILE = "posterior_median_quantile"
    POSTERIOR_PREDICTIVE_QUANTILE = "posterior_predictive_quantile"
    EXPECTED_NON_EXCEEDANCE_PROBABILITY = "expected_non_exceedance_probability"
    EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY = "expected_annual_exceedance_probability"
    RETURN_PERIOD_FROM_EXPECTED_AEP = "return_period_from_expected_aep"
    QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT = "quantile_monte_carlo_probability_limit"
    RETURN_PERIOD_MONTE_CARLO_PROBABILITY_LIMIT = "return_period_monte_carlo_probability_limit"
    LOG10_QUANTILE_POSTERIOR_MOMENT = "log10_quantile_posterior_moment"


@dataclass(frozen=True, slots=True, kw_only=True)
class AnnualObservation:
    """One observed annual-maximum record, with original flags retained by WP02.

    `value` holds the original numeric reading when present, even for EXCLUDED.
    `lower_bound` and `upper_bound` are constraints, not replacement values.
    A zero threshold is an independent source field, not a fabricated positive flow.
    """

    year: int
    kind: ObservationKind
    exposure: ExposureKind
    value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    source_flag: str | None = None

    def __post_init__(self) -> None:
        """Reject malformed boundaries while leaving legacy meaning unresolved."""
        if self.year <= 0:
            msg = "observation year must be a positive calendar-year identifier"
            raise ValueError(msg)
        values = (self.value, self.lower_bound, self.upper_bound)
        if any(value is not None and not isfinite(value) for value in values):
            msg = "observation values and bounds must be finite when provided"
            raise ValueError(msg)
        if self.kind == ObservationKind.EXACT and self.value is None:
            msg = "exact observations require a recorded value, including an explicit zero"
            raise ValueError(msg)
        if self.kind == ObservationKind.LEFT_CENSORED and self.upper_bound is None:
            msg = "left-censored observations require an upper bound"
            raise ValueError(msg)
        if self.kind == ObservationKind.RIGHT_CENSORED and self.lower_bound is None:
            msg = "right-censored observations require a lower bound"
            raise ValueError(msg)
        if self.kind == ObservationKind.INTERVAL_CENSORED and (self.lower_bound is None or self.upper_bound is None):
            msg = "interval-censored observations require both bounds"
            raise ValueError(msg)
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound > self.upper_bound:
            msg = "lower observation bound must not exceed upper bound"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class CensorGroup:
    """Original grouped threshold/count record; not a collection of deleted years."""

    threshold: float
    exposure: ExposureKind
    count_at_or_below: int | None = None
    count_above: int | None = None
    start_year: int | None = None
    end_year: int | None = None
    source_record: str | None = None

    def __post_init__(self) -> None:
        """Validate counts without assuming undocumented endpoint conventions."""
        if not isfinite(self.threshold):
            msg = "censor-group threshold must be finite"
            raise ValueError(msg)
        if self.count_at_or_below is None and self.count_above is None:
            msg = "at least one grouped count is required"
            raise ValueError(msg)
        if any(count is not None and count < 0 for count in (self.count_at_or_below, self.count_above)):
            msg = "grouped counts cannot be negative"
            raise ValueError(msg)
        if self.start_year is not None and self.end_year is not None and self.start_year > self.end_year:
            msg = "censor-group start year must not exceed end year"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationSeries:
    """One source unit convention, retaining annual and grouped evidence separately.

    WP02/WP05 must reconcile overlapping exposure before likelihood evaluation.
    A group's counts must not be added to annual records without provenance checks.
    """

    discharge_unit: str
    annual: tuple[AnnualObservation, ...]
    groups: tuple[CensorGroup, ...] = ()
    zero_threshold: float | None = None
    source_file_version: str | None = None
    year_definition: str | None = None

    def __post_init__(self) -> None:
        """Keep discharge units and recorded threshold explicit."""
        if not self.discharge_unit.strip():
            msg = "observation discharge units must be recorded"
            raise ValueError(msg)
        if self.zero_threshold is not None and not isfinite(self.zero_threshold):
            msg = "zero-threshold discharge must be finite"
            raise ValueError(msg)


class DistributionKernel(Protocol):
    """Bound-parameter continuous kernel for WP03, with no IO or random state.

    Invalid parameters/observed likelihoods must be distinguished from prediction
    points outside the kernel support (where CDF/SF remain defined).
    Probability arguments are fractions and discharge values use declared units.
    """

    def logpdf(self, discharge: float) -> float: ...
    def logcdf(self, discharge: float) -> float: ...
    def logsf(self, discharge: float) -> float: ...
    def cdf(self, discharge: float) -> float: ...
    def sf(self, discharge: float) -> float: ...
    def ppf(self, non_exceedance_probability: float) -> float: ...
    def isf(self, survival_probability: float) -> float: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ParameterSummary:
    """A source-coordinate parameter summary with its quantity type explicit.

    The legacy `mean` constructor field remains for backwards compatibility.
    A deterministic point estimate is emitted as `point_estimate`, never a
    posterior mean, by the downstream JSON/text reporter.
    """

    name: str
    coordinate: ParameterCoordinate | str
    mean: float
    standard_deviation: float | None = None
    summary_kind: Literal["posterior_mean", "point_estimate"] = "posterior_mean"


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultPoint:
    """One reported quantity, with x variable and probability convention explicit."""

    meaning: ResultMeaning
    value: float
    unit: str
    input_probability: float | None = None
    input_probability_kind: ProbabilityKind | None = None
    input_discharge: float | None = None
    input_discharge_unit: str | None = None
    confidence_level: float | None = None
    probability_limit_side: Literal["lower", "upper"] | None = None
    probability_limit_construction: str | None = None
    log10_moment_kind: Literal["mean", "standard_deviation", "variance"] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunProvenance:
    """Method/implementation versions remain distinct from observed vendor versions."""

    compatibility_profile: str
    method_id: str
    method_version: str
    package_version: str
    evidence_status: EvidenceStatus
    input_sha256: str | None = None
    settings_sha256: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunDiagnostic:
    """Explicit mathematical-fit diagnostic; never an inferred run setting."""

    name: str
    value: float
    unit: str


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisResult:
    """Computed result container; no inference/qualification follows from its existence."""

    provenance: RunProvenance
    parameters: tuple[ParameterSummary, ...] = ()
    values: tuple[ResultPoint, ...] = ()
    warnings: tuple[str, ...] = ()
    diagnostics: tuple[RunDiagnostic, ...] = ()
    fitted_gev_support: GEVFittedSupport | None = None
    precision_audit: PrecisionAudit | None = None
