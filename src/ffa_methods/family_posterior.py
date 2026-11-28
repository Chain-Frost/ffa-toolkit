"""Classified systematic GEV and LP3 posteriors with explicitly fixed shape."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from re import fullmatch
from sys import float_info
from typing import ClassVar

from ffa_methods import __version__
from ffa_methods.compatibility import LEGACY_FLIKE_PROFILE
from ffa_methods.contracts import (
    AnnualObservation,
    DistributionKernel,
    EvidenceStatus,
    ExposureKind,
    ObservationKind,
    ObservationSeries,
    ParameterCoordinate,
    RunProvenance,
)
from ffa_methods.distributions import GEV, LogCoordinates, LogPearsonIII
from ffa_methods.estimated_scale_kernel import EstimatedScaleKernel
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
from ffa_methods.lognormal_posterior import SOURCE_CLASSIFICATION, PosteriorBindingError
from ffa_methods.priors import JointGaussianPrior, LegacyPriorMode

GEV_LOCATION = "gev.location"
GEV_LOG_SCALE = "gev.log_scale"
LP3_LOG_SD = ParameterCoordinate.LP3_LOGE_SD_LOGE_FLOW.value
LP3_MEAN = ParameterCoordinate.LP3_MEAN_LOGE_FLOW.value
METHOD_VERSION = "explicit-fixed-family-posterior/1"


class PosteriorFamily(StrEnum):
    GEV = "gev"
    LP3 = "lp3"


def _check_row(row: AnnualObservation) -> None:
    if row.exposure != ExposureKind.SYSTEMATIC or row.source_flag is not None:
        msg = "only explicitly classified systematic annual observations without native flags are admitted"
        raise PosteriorBindingError(msg)
    if row.kind == ObservationKind.EXACT:
        if row.value is None or row.value <= 0 or row.lower_bound is not None or row.upper_bound is not None:
            msg = "exact annual maxima must be strictly positive with no censor bounds"
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
        msg = "unsupported observation or nonpositive/invalid continuous censor bounds"
        raise PosteriorBindingError(msg)
    if row.value is not None and (
        row.value <= 0 or (lower is not None and row.value < lower) or (upper is not None and row.value > upper)
    ):
        msg = "retained censored measurement conflicts with the stated positive interval"
        raise PosteriorBindingError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ClassifiedFamilyPosterior(ABC):
    """Shared source, fixed-shape and supplied-prior contract."""

    version: ClassVar[str]
    family: PosteriorFamily
    series: ObservationSeries
    source: str
    source_sha256: str
    source_classification: str
    fixed_shape: float
    reference_discharge: float | None
    prior: JointGaussianPrior

    def __post_init__(self) -> None:
        self._validate_source()
        self._validate_model()
        if self.prior.mode != LegacyPriorMode.SUPPLIED or self.prior.coordinates != self.coordinates:
            msg = "a supplied Gaussian prior must use the exact free family coordinate"
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

    def _validate_model(self) -> None:
        if self.family not in (PosteriorFamily.GEV, PosteriorFamily.LP3):
            msg = "only fixed-shape GEV and LP3 are admitted"
            raise PosteriorBindingError(msg)
        if not isfinite(self.fixed_shape):
            msg = "fixed log-scale and shape must be finite"
            raise PosteriorBindingError(msg)
        if self.family == PosteriorFamily.GEV:
            if 0.0 < abs(self.fixed_shape) < float_info.min:
                message = "GEV EVT xi must be zero or have a normal binary64 magnitude"
                raise PosteriorBindingError(message)
            if self.reference_discharge is not None or self.fixed_shape <= -1.0:
                msg = "GEV requires no log reference and EVT xi > -1 for this bounded binding"
                raise PosteriorBindingError(msg)
        else:
            self._validate_lp3_shape()

    def _validate_lp3_shape(self) -> None:
        if abs(self.fixed_shape) >= 2.0 or 0.0 < abs(self.fixed_shape) < 1.6e-5:
            msg = "LP3 requires |skew| < 2 and excludes the nonzero near-normal approximation band"
            raise PosteriorBindingError(msg)
        if self.reference_discharge is None or not isfinite(self.reference_discharge) or self.reference_discharge <= 0:
            msg = "LP3 requires a finite positive reference discharge in source units"
            raise PosteriorBindingError(msg)

    @property
    @abstractmethod
    def coordinates(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def _scale_description(self) -> str: ...

    @abstractmethod
    def kernel(self, values: tuple[float, ...]) -> DistributionKernel: ...

    def _raw_kernel(self, location: float, log_scale: float) -> GEV | LogPearsonIII:
        if self.family == PosteriorFamily.GEV:
            return GEV(location, log_scale, self.fixed_shape)
        reference = self.reference_discharge
        if reference is None:
            msg = "LP3 requires a reference discharge"
            raise PosteriorBindingError(msg)
        return LogPearsonIII(LogCoordinates(location, log_scale, reference_discharge=reference), self.fixed_shape)

    def log_likelihood(self, values: tuple[float, ...]) -> float:
        return log_likelihood(self.kernel(values), self.series).log_value

    def target(self) -> ExplicitPosteriorTarget:
        """The supplied density is already in the free coordinate (Jacobian one)."""
        shape_name = "EVT xi" if self.family == PosteriorFamily.GEV else "log-flow skew"
        return ExplicitPosteriorTarget(
            coordinates=self.coordinates,
            log_likelihood=self.log_likelihood,
            log_prior=self.prior.logpdf,
            source=(
                f"{self.version}; family={self.family}; {self.source}; sha256={self.source_sha256}; "
                f"unit={self.series.discharge_unit}; reference={self.reference_discharge}; "
                f"year_definition={self.series.year_definition}; {self.source_classification}; "
                f"{self._scale_description}; fixed {shape_name}={self.fixed_shape}; "
                f"supplied prior={self.prior.source}"
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassifiedFixedFamilyPosterior(_ClassifiedFamilyPosterior):
    fixed_log_scale: float
    version: ClassVar[str] = METHOD_VERSION

    def __post_init__(self) -> None:
        _ClassifiedFamilyPosterior.__post_init__(self)
        if not isfinite(self.fixed_log_scale):
            msg = "fixed log-scale and shape must be finite"
            raise PosteriorBindingError(msg)
        _ = self._raw_kernel(0.0, self.fixed_log_scale)

    @property
    def _scale_description(self) -> str:
        return f"fixed_log_scale={self.fixed_log_scale}"

    @property
    def coordinates(self) -> tuple[str, ...]:
        return (GEV_LOCATION,) if self.family == PosteriorFamily.GEV else (LP3_MEAN,)

    def kernel(self, values: tuple[float, ...]) -> DistributionKernel:
        if len(values) != 1 or not isfinite(values[0]):
            msg = "family parameter vector must contain exactly one finite location/mean"
            raise PosteriorBindingError(msg)
        return _as_shared_kernel(self._raw_kernel(values[0], self.fixed_log_scale))


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassifiedEstimatedScalePosterior(_ClassifiedFamilyPosterior):
    """Gaussian prior on (location, ln(scale)) or (log-flow mean, ln(log-flow SD))."""

    version: ClassVar[str] = "explicit-estimated-scale-family-posterior/1"

    def __post_init__(self) -> None:
        _ClassifiedFamilyPosterior.__post_init__(self)
        if self.prior.covariance[0][1] != self.prior.covariance[1][0]:
            msg = "estimated-scale prior covariance must be symmetric in the declared coordinates"
            raise PosteriorBindingError(msg)

    @property
    def _scale_description(self) -> str:
        return f"estimated {self.coordinates}; prior means={self.prior.means}; covariance={self.prior.covariance}"

    @property
    def coordinates(self) -> tuple[str, ...]:
        return (GEV_LOCATION, GEV_LOG_SCALE) if self.family == PosteriorFamily.GEV else (LP3_MEAN, LP3_LOG_SD)

    def kernel(self, values: tuple[float, ...]) -> DistributionKernel:
        if len(values) != 2 or not all(isfinite(v) for v in values):
            msg = "estimated scale requires a finite location/mean and log-scale in declared order"
            raise PosteriorBindingError(msg)
        return EstimatedScaleKernel(self._raw_kernel(*values))


def _exact_series(data: bytes, discharge_unit: str, year_definition: str) -> ObservationSeries:
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
    return ObservationSeries(
        discharge_unit=discharge_unit,
        annual=tuple(rows),
        source_file_version="year-ams-csv/1",
        year_definition=year_definition,
    )


def bind_exact_fixed_family_csv(
    data: bytes,
    *,
    family: PosteriorFamily,
    source: str,
    discharge_unit: str,
    year_definition: str,
    source_classification: str,
    fixed_log_scale: float,
    fixed_shape: float,
    reference_discharge: float | None,
    prior: JointGaussianPrior,
) -> ClassifiedFixedFamilyPosterior:
    """Read original Year,AMS bytes without dropping or coercing observations."""
    return ClassifiedFixedFamilyPosterior(
        family=family,
        series=_exact_series(data, discharge_unit, year_definition),
        source=source,
        source_sha256=sha256(data).hexdigest(),
        source_classification=source_classification,
        fixed_log_scale=fixed_log_scale,
        fixed_shape=fixed_shape,
        reference_discharge=reference_discharge,
        prior=prior,
    )


def bind_exact_estimated_scale_csv(
    data: bytes,
    *,
    family: PosteriorFamily,
    source: str,
    discharge_unit: str,
    year_definition: str,
    source_classification: str,
    fixed_shape: float,
    reference_discharge: float | None,
    prior: JointGaussianPrior,
) -> ClassifiedEstimatedScalePosterior:
    """Read original Year,AMS bytes without dropping or coercing observations."""
    return ClassifiedEstimatedScalePosterior(
        family=family,
        series=_exact_series(data, discharge_unit, year_definition),
        source=source,
        source_sha256=sha256(data).hexdigest(),
        source_classification=source_classification,
        fixed_shape=fixed_shape,
        reference_discharge=reference_discharge,
        prior=prior,
    )


@dataclass(frozen=True, slots=True)
class ExecutedFixedFamilyPosterior:
    model: ClassifiedFixedFamilyPosterior
    construction: CurvatureProposalResult
    sampling: IndependentSamplingResult


def execute_fixed_family_posterior(
    model: ClassifiedFixedFamilyPosterior,
    *,
    starting_point: tuple[tuple[str, float], ...],
    curvature: CurvatureControls,
    sampling: ExplicitSamplingControls,
    gamma: float,
) -> ExecutedFixedFamilyPosterior:
    """Execute existing local curvature and sampling components with explicit gamma."""
    construction, result = _execute(model, starting_point, curvature, sampling, gamma)
    return ExecutedFixedFamilyPosterior(model, construction, result)


@dataclass(frozen=True, slots=True)
class ExecutedEstimatedScalePosterior:
    model: ClassifiedEstimatedScalePosterior
    construction: CurvatureProposalResult
    sampling: IndependentSamplingResult


def execute_estimated_scale_posterior(
    model: ClassifiedEstimatedScalePosterior,
    *,
    starting_point: tuple[tuple[str, float], ...],
    curvature: CurvatureControls,
    sampling: ExplicitSamplingControls,
    gamma: float,
) -> ExecutedEstimatedScalePosterior:
    construction, result = _execute(model, starting_point, curvature, sampling, gamma)
    particles = result.particles
    for weight, likelihood, prior in zip(particles.weights, particles.log_likelihood, particles.log_prior, strict=True):
        if not isfinite(prior) or (isfinite(likelihood) and weight == 0.0):
            msg = "a positive Gaussian prior or posterior weight was lost numerically"
            raise PosteriorBindingError(msg)
    return ExecutedEstimatedScalePosterior(model, construction, result)


def _execute(
    model: _ClassifiedFamilyPosterior,
    starting_point: tuple[tuple[str, float], ...],
    curvature: CurvatureControls,
    sampling: ExplicitSamplingControls,
    gamma: float,
) -> tuple[CurvatureProposalResult, IndependentSamplingResult]:
    target = model.target()
    construction = build_curvature_proposal(
        target=target, starting_point=starting_point, controls=curvature, gamma=gamma
    )
    result = independent_importance_sample(
        proposal=construction.proposal,
        target=target,
        controls=sampling,
        provenance=RunProvenance(
            compatibility_profile=LEGACY_FLIKE_PROFILE.identifier,
            method_id="importance-sampling-reference",
            method_version=model.version,
            package_version=__version__,
            evidence_status=EvidenceStatus.OBSERVED_ARTIFACT,
            input_sha256=model.source_sha256,
        ),
    )
    return construction, result
