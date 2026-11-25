"""Explicit independent annual-maximum lognormal target binding (WP06).

No legacy project flags, default priors or unverified zero/rating laws are
interpreted. The binding is an internal component for later CLI integration.
"""

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from re import fullmatch

from ffa_methods import __version__
from ffa_methods.compatibility import LEGACY_FLIKE_PROFILE
from ffa_methods.contracts import (
    AnnualObservation,
    DistributionKernel,
    EvidenceStatus,
    ExposureKind,
    ObservationKind,
    ObservationSeries,
    RunProvenance,
)
from ffa_methods.distributions import LogCoordinates, Lognormal
from ffa_methods.importance_proposal import CurvatureControls, CurvatureProposalResult, build_curvature_proposal
from ffa_methods.importance_sampling import (
    ExplicitPosteriorTarget,
    ExplicitSamplingControls,
    IndependentSamplingResult,
    independent_importance_sample,
)
from ffa_methods.legacy.models import ObservationKind as LegacyObservationKind
from ffa_methods.legacy.project import parse_ams_csv
from ffa_methods.likelihood import exposure_intervals, log_likelihood
from ffa_methods.likelihood_integration import (
    _as_shared_kernel,  # pyright: ignore[reportPrivateUsage]  # existing internal bridge
)
from ffa_methods.priors import JointGaussianPrior, LegacyPriorMode

MEAN_COORDINATE = "lognormal.mean_loge_flow"
LOG_SD_COORDINATE = "lognormal.loge_sd_loge_flow"
METHOD_VERSION = "explicit-lognormal-posterior/1"
SOURCE_CLASSIFICATION = "caller_asserted_classified_systematic_annual_maxima"


class PosteriorBindingError(ValueError):
    """The declared evidence or settings do not identify a supported target."""


def _check_row(row: AnnualObservation) -> None:
    if row.exposure != ExposureKind.SYSTEMATIC or row.source_flag is not None:
        msg = "only explicitly classified systematic annual observations without native flags are admitted"
        raise PosteriorBindingError(msg)
    if row.kind == ObservationKind.EXACT:
        if row.value is None or row.value <= 0 or row.lower_bound is not None or row.upper_bound is not None:
            msg = "exact lognormal maxima must be strictly positive with no censor bounds"
            raise PosteriorBindingError(msg)
    elif row.kind in (ObservationKind.MISSING, ObservationKind.EXCLUDED):
        if row.lower_bound is not None or row.upper_bound is not None:
            msg = "missing and excluded rows cannot carry active censor bounds"
            raise PosteriorBindingError(msg)
        if row.kind == ObservationKind.MISSING and row.value is not None:
            msg = "missing observations cannot carry a recorded value"
            raise PosteriorBindingError(msg)
    else:
        _check_censor(row)


def _check_censor(row: AnnualObservation) -> None:
    lower, upper = row.lower_bound, row.upper_bound
    if row.kind == ObservationKind.LEFT_CENSORED:
        valid = lower is None and upper is not None and upper > 0
    elif row.kind == ObservationKind.RIGHT_CENSORED:
        valid = upper is None and lower is not None and lower > 0
    elif row.kind == ObservationKind.INTERVAL_CENSORED:
        valid = lower is not None and upper is not None and 0 < lower < upper
    else:
        valid = False
    if not valid:
        msg = "unsupported observation or nonpositive/invalid lognormal censor bounds"
        raise PosteriorBindingError(msg)
    if row.value is not None and (
        row.value <= 0 or (lower is not None and row.value < lower) or (upper is not None and row.value > upper)
    ):
        msg = "retained censored measurement conflicts with the stated positive interval"
        raise PosteriorBindingError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassifiedLognormalPosterior:
    """One explicit model, likelihood and prior in the same named log coordinates.

    Reference discharge is in series.discharge_unit. A numeric fixed_log_sd
    is the natural log of a known SD of ln(Q/reference); None estimates it.
    The supplied Gaussian prior is in the free parameter coordinates only.
    """

    series: ObservationSeries
    source: str
    source_sha256: str
    source_classification: str
    reference_discharge: float
    fixed_log_sd: float | None
    prior: JointGaussianPrior

    def __post_init__(self) -> None:
        self._validate_source()
        if not isfinite(self.reference_discharge) or self.reference_discharge <= 0:
            msg = "reference discharge must be finite and strictly positive"
            raise PosteriorBindingError(msg)
        if self.fixed_log_sd is not None:
            LogCoordinates(0.0, self.fixed_log_sd, reference_discharge=self.reference_discharge)
        if self.prior.mode != LegacyPriorMode.SUPPLIED or self.prior.coordinates != self.coordinates:
            msg = "a supplied Gaussian prior must use the exact ordered free lognormal coordinates"
            raise PosteriorBindingError(msg)

    def _validate_source(self) -> None:
        if (
            self.source_classification != SOURCE_CLASSIFICATION
            or not self.source.strip()
            or fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None
            or not self.series.year_definition
            or not self.series.year_definition.strip()
        ):
            msg = "explicit source, SHA-256, classification and year definition are required"
            raise PosteriorBindingError(msg)
        if self.series.source_file_version not in (None, "year-ams-csv/1"):
            msg = "native project source classifications are not admitted"
            raise PosteriorBindingError(msg)
        if self.series.groups or self.series.zero_threshold is not None:
            msg = "grouped or zero-threshold likelihood is not admitted by this binding"
            raise PosteriorBindingError(msg)
        exposure_intervals(self.series)
        for row in self.series.annual:
            _check_row(row)
        if not any(row.kind not in (ObservationKind.MISSING, ObservationKind.EXCLUDED) for row in self.series.annual):
            msg = "posterior binding requires at least one included observation"
            raise PosteriorBindingError(msg)

    @property
    def coordinates(self) -> tuple[str, ...]:
        if self.fixed_log_sd is not None:
            return (MEAN_COORDINATE,)
        return (MEAN_COORDINATE, LOG_SD_COORDINATE)

    def kernel(self, values: tuple[float, ...]) -> DistributionKernel:
        """Same coordinate mapping for likelihood evaluation and every output."""
        if len(values) != len(self.coordinates) or any(not isfinite(value) for value in values):
            msg = "lognormal parameter vector must be finite and match the free coordinates"
            raise PosteriorBindingError(msg)
        log_sd = values[1] if self.fixed_log_sd is None else self.fixed_log_sd
        return _as_shared_kernel(
            Lognormal(LogCoordinates(values[0], log_sd, reference_discharge=self.reference_discharge))
        )

    def log_likelihood(self, values: tuple[float, ...]) -> float:
        return log_likelihood(self.kernel(values), self.series).log_value

    def target(self) -> ExplicitPosteriorTarget:
        """Return the existing sampler's target; no extra parameter Jacobian."""
        return ExplicitPosteriorTarget(
            coordinates=self.coordinates,
            log_likelihood=self.log_likelihood,
            log_prior=self.prior.logpdf,
            source=(
                f"{METHOD_VERSION}; {self.source}; sha256={self.source_sha256}; "
                f"unit={self.series.discharge_unit}; reference={self.reference_discharge}; "
                f"year_definition={self.series.year_definition}; {self.source_classification}; "
                f"fixed_log_sd={self.fixed_log_sd}; supplied prior={self.prior.source}"
            ),
        )


def bind_exact_lognormal_csv(
    data: bytes,
    *,
    source: str,
    discharge_unit: str,
    year_definition: str,
    source_classification: str,
    reference_discharge: float,
    fixed_log_sd: float | None,
    prior: JointGaussianPrior,
) -> ClassifiedLognormalPosterior:
    """Read actual Year,AMS bytes with WP02; refuse zero, missing or flagged values."""
    rows: list[AnnualObservation] = []
    for record in parse_ams_csv(data):
        if record.kind != LegacyObservationKind.EXACT or record.value is None:
            msg = f"year {record.year}: the exact CSV path requires strictly positive annual maxima"
            raise PosteriorBindingError(msg)
        value = float(record.value)
        if not isfinite(value) or value <= 0:
            msg = f"year {record.year}: discharge is not representable as a positive finite binary64 value"
            raise PosteriorBindingError(msg)
        rows.append(
            AnnualObservation(
                year=record.year, kind=ObservationKind.EXACT, exposure=ExposureKind.SYSTEMATIC, value=value
            )
        )
    return ClassifiedLognormalPosterior(
        series=ObservationSeries(
            discharge_unit=discharge_unit,
            annual=tuple(rows),
            source_file_version="year-ams-csv/1",
            year_definition=year_definition,
        ),
        source=source,
        source_sha256=sha256(data).hexdigest(),
        source_classification=source_classification,
        reference_discharge=reference_discharge,
        fixed_log_sd=fixed_log_sd,
        prior=prior,
    )


@dataclass(frozen=True, slots=True)
class ExecutedLognormalPosterior:
    """Source binding, proposal evidence and actual weighted sampling result."""

    model: ClassifiedLognormalPosterior
    construction: CurvatureProposalResult
    sampling: IndependentSamplingResult


def execute_lognormal_posterior(
    model: ClassifiedLognormalPosterior,
    *,
    starting_point: tuple[tuple[str, float], ...],
    curvature: CurvatureControls,
    sampling: ExplicitSamplingControls,
) -> ExecutedLognormalPosterior:
    """Execute the reviewed WP06 components; no native qualification or CLI implied."""
    target = model.target()
    construction = build_curvature_proposal(target=target, starting_point=starting_point, controls=curvature, gamma=1.0)
    result = independent_importance_sample(
        proposal=construction.proposal,
        target=target,
        controls=sampling,
        provenance=RunProvenance(
            compatibility_profile=LEGACY_FLIKE_PROFILE.identifier,
            method_id="importance-sampling-reference",
            method_version=METHOD_VERSION,
            package_version=__version__,
            evidence_status=EvidenceStatus.OBSERVED_ARTIFACT,
            input_sha256=model.source_sha256,
        ),
    )
    return ExecutedLognormalPosterior(model, construction, result)
