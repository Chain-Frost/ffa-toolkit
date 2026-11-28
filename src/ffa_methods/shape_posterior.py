"""Three-coordinate GEV and LP3 posteriors under a supplied bounded logit-Normal shape law."""

from dataclasses import dataclass, field
from fractions import Fraction
from math import exp, fsum, inf, isclose, isfinite, log, log1p, nextafter, pi, tanh
from random import Random
from sys import float_info
from typing import ClassVar

from scipy.special import gammainc, gammaincc

from ffa_methods import __version__
from ffa_methods.bayesian_outputs import ImportanceParticles, PosteriorMoments
from ffa_methods.compatibility import LEGACY_FLIKE_PROFILE
from ffa_methods.contracts import DistributionKernel, EvidenceStatus, ParameterCoordinate, RunProvenance
from ffa_methods.distributions import GEV, LogCoordinates, LogPearsonIII
from ffa_methods.estimated_scale_kernel import EstimatedScaleKernel
from ffa_methods.family_posterior import (
    GEV_LOCATION,
    GEV_LOG_SCALE,
    LP3_LOG_SD,
    LP3_MEAN,
    ClassifiedFamilySource,
    PosteriorFamily,
)
from ffa_methods.importance_sampling import (
    ExplicitNormalProposal,
    ExplicitPosteriorTarget,
    ExplicitSamplingControls,
    independent_importance_sample,
)
from ffa_methods.likelihood import log_likelihood
from ffa_methods.lognormal_posterior import PosteriorBindingError

GEV_COORDINATES = (GEV_LOCATION, GEV_LOG_SCALE, "gev.shape_xi")
LP3_COORDINATES = (LP3_MEAN, LP3_LOG_SD, ParameterCoordinate.LP3_SKEW_LOGE_FLOW.value)
GEV_LATENT = (GEV_LOCATION, GEV_LOG_SCALE, "gev.logit_shape_fraction")
LP3_LATENT = (LP3_MEAN, LP3_LOG_SD, "lp3.logit_skew_fraction")
METHOD_VERSION = "bounded-shape-posterior/1"


@dataclass(frozen=True, slots=True, kw_only=True)
class BoundedShapePrior:
    """Push forward three independent standard Normals using a named lower factor and a scaled logit."""

    coordinates: tuple[str, ...]
    means: tuple[float, ...]
    lower_factor: tuple[tuple[float, ...], ...]
    bounds: tuple[float, float]
    source: str
    _log_det: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.coordinates not in (GEV_LATENT, LP3_LATENT) or not self.source.strip():
            msg = "supply the ordered latent family coordinates and prior source"
            raise PosteriorBindingError(msg)
        if len(self.means) != 3 or not all(isfinite(v) for v in self.means) or len(self.lower_factor) != 3:
            msg = "supply three finite prior means and the full 3 by 3 lower factor"
            raise PosteriorBindingError(msg)
        for i, row in enumerate(self.lower_factor):
            if len(row) != 3 or not all(isfinite(x) for x in row) or row[i] <= 0 or any(row[i + 1 :]):
                msg = "prior factor must be finite, lower triangular and have strictly positive diagonal"
                raise PosteriorBindingError(msg)
        a, b = self.bounds
        if not isfinite(b - a) or nextafter(a, inf) >= b:
            msg = "shape bounds require a finite width and a representable open interior"
            raise PosteriorBindingError(msg)
        object.__setattr__(self, "_log_det", fsum(log(row[i]) for i, row in enumerate(self.lower_factor)))

    def physical(self, u: tuple[float, ...]) -> tuple[float, ...]:
        if len(u) != 3 or not all(isfinite(x) for x in u):
            msg = "three finite Gaussian innovations are required"
            raise PosteriorBindingError(msg)
        mu, eta, z = (
            fsum((mean, *(v * x for v, x in zip(row, u, strict=True))))
            for mean, row in zip(self.means, self.lower_factor, strict=True)
        )
        a, b = self.bounds
        if abs(z) < 1:
            shape = fsum((a, (b - a) / 2, ((b - a) / 2) * tanh(z / 2)))
        else:
            tail = exp(-abs(z))
            distance = (b - a) * (tail / (1 + tail))
            shape = b - distance if z > 0 else a + distance
        if not a < shape < b or not all(isfinite(v) for v in (mu, eta, z)):
            msg = "bounded shape or location/scale transform is unresolved in binary64"
            raise PosteriorBindingError(msg)
        return mu, eta, shape

    def innovations(self, physical: tuple[float, ...]) -> tuple[float, ...]:
        """Invert the exact supplied triangular factor without forming a rounded covariance."""
        mu, eta, s = physical
        a, b = self.bounds
        left, right = Fraction(s) - Fraction(a), Fraction(b) - Fraction(s)
        if min(left, right) <= 0:
            msg = "shape must be inside the declared open interval"
            raise PosteriorBindingError(msg)
        z = (
            log1p(float((left - right) / right))
            if right / 2 < left < 2 * right
            else log(float(left)) - log(float(right))
        )
        residuals: list[Fraction] = []
        for i, (value, center, row) in enumerate(zip((mu, eta, z), self.means, self.lower_factor, strict=True)):
            residual = Fraction(value) - Fraction(center) - sum(Fraction(row[j]) * u for j, u in enumerate(residuals))
            residuals.append(residual / Fraction(row[i]))
        return tuple(float(u) for u in residuals)

    def logpdf(self, values: tuple[float, ...], *, gamma: float = 1.0) -> float:
        if len(values) != 3 or not all(isfinite(v) for v in values) or not isfinite(gamma) or gamma <= 0:
            msg = "density requires finite physical coordinates and a positive finite innovation scale"
            raise PosteriorBindingError(msg)
        a, b = self.bounds
        shape = values[2]
        if not a < shape < b:
            return -inf
        u = self.innovations(values)
        log_jacobian = self._log_det + log(shape - a) + log(b - shape) - log(b - a)
        value = -1.5 * log(2 * pi) - 3 * log(gamma) - 0.5 * fsum((x / gamma) ** 2 for x in u) - log_jacobian
        if not isfinite(value):
            msg = "interior shape prior/proposal density is unresolved"
            raise PosteriorBindingError(msg)
        return value


class _ShapeKernel(EstimatedScaleKernel):
    def _checked_inverse(self, discharge: float, probability: float, *, survival: bool) -> float:
        if not 0 < probability < 1:
            return discharge
        if not isfinite(discharge) or (isinstance(self.raw, LogPearsonIII) and discharge <= 0):
            msg = "interior design quantile is not representable in flow units"
            raise PosteriorBindingError(msg)
        use_sf = survival if probability <= 0.5 else not survival
        log_target = log(probability) if probability <= 0.5 else log1p(-probability)
        observed = self.logsf(discharge) if use_sf else self.logcdf(discharge)
        if abs(observed - log_target) > 1e-8:
            msg = "conditional design quantile does not resolve its requested tail probability"
            raise PosteriorBindingError(msg)
        return discharge

    def ppf(self, non_exceedance_probability: float) -> float:
        return self._checked_inverse(
            self.raw.ppf(non_exceedance_probability), non_exceedance_probability, survival=False
        )

    def isf(self, survival_probability: float) -> float:
        return self._checked_inverse(self.raw.isf(survival_probability), survival_probability, survival=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassifiedShapePosterior(ClassifiedFamilySource):
    prior: BoundedShapePrior

    def __post_init__(self) -> None:
        self.validate_source()
        a, b = self.prior.bounds
        if self.family == PosteriorFamily.GEV:
            if self.prior.coordinates != GEV_LATENT or self.reference_discharge is not None or a <= -1:
                msg = "GEV needs EVT shape bounds strictly above -1, ordered GEV coordinates and no log reference"
                raise PosteriorBindingError(msg)
        elif self.family == PosteriorFamily.LP3:
            if (
                self.reference_discharge is None
                or not isfinite(self.reference_discharge)
                or self.reference_discharge <= 0
                or self.prior.coordinates != LP3_LATENT
            ):
                msg = "LP3 requires a positive finite reference and ordered LP3 coordinates"
                raise PosteriorBindingError(msg)
            if not (-2 < a < b <= -1.6e-5 or 1.6e-5 <= a < b < 2):
                msg = "LP3 skew support must be nonsingular and entirely outside the near-normal band on one side"
                raise PosteriorBindingError(msg)
        else:
            msg = "unsupported estimated-shape family"
            raise PosteriorBindingError(msg)

    @property
    def coordinates(self) -> tuple[str, ...]:
        return GEV_COORDINATES if self.family == PosteriorFamily.GEV else LP3_COORDINATES

    def kernel(self, values: tuple[float, ...]) -> DistributionKernel:
        if (
            len(values) != 3
            or not all(isfinite(v) for v in values)
            or not self.prior.bounds[0] < values[2] < self.prior.bounds[1]
        ):
            msg = "expected three finite physical coordinates with strictly interior shape"
            raise PosteriorBindingError(msg)
        mu, eta, s = values
        if self.family == PosteriorFamily.GEV:
            if 0 < abs(s) < float_info.min:
                msg = "nonzero subnormal GEV shape is unresolved"
                raise PosteriorBindingError(msg)
            return _ShapeKernel(GEV(mu, eta, s))
        if self.reference_discharge is None:
            msg = "missing LP3 reference"
            raise PosteriorBindingError(msg)
        return _ShapeKernel(LogPearsonIII(LogCoordinates(mu, eta, reference_discharge=self.reference_discharge), s))

    def log_likelihood(self, values: tuple[float, ...]) -> float:
        return log_likelihood(self.kernel(values), self.series).log_value

    def raw_mean_undefined_reason(self, p: float) -> str | None:
        if not 0 < p < 1:
            msg = "design probability must be interior"
            raise PosteriorBindingError(msg)
        if self.family == PosteriorFamily.GEV:
            return None
        # Pinelis, Thm 1(c=0): F_standard(0;g) increases with signed skew.
        g = self.prior.bounds[0]
        alpha = (2 / g) ** 2
        cutoff = float(gammainc(alpha, alpha) if g > 0 else gammaincc(alpha, alpha))
        if abs(p - cutoff) < 1e-12:
            msg = "LP3 design moment classification is numerically unresolved at the shape-support cutoff"
            raise PosteriorBindingError(msg)
        return (
            "raw LP3 design-quantile mean diverges on an open set of positive standardized quantiles"
            if p > cutoff
            else None
        )


@dataclass(frozen=True, slots=True)
class _ShapeProposal:
    prior: BoundedShapePrior
    normal: ExplicitNormalProposal
    gamma: float
    coordinates: tuple[str, ...]
    algorithm: ClassVar[str] = "independent transformed-Normal self-normalized importance sampling"

    @property
    def source(self) -> str:
        return f"pushforward of {self.normal.source}; prior={self.prior!r}"

    def draw(self, rng: Random) -> tuple[float, ...]:
        u = self.normal.draw(rng)
        values = self.prior.physical(u)
        recovered = self.prior.innovations(values)
        if any(not isclose(x, y, rel_tol=1e-10, abs_tol=1e-10) for x, y in zip(u, recovered, strict=True)):
            msg = "sampled physical coordinates cannot resolve the Gaussian innovations"
            raise PosteriorBindingError(msg)
        return values

    def logpdf(self, values: tuple[float, ...]) -> float:
        return self.prior.logpdf(values, gamma=self.gamma)


@dataclass(frozen=True, slots=True)
class ExecutedShapePosterior:
    particles: ImportanceParticles
    moments: PosteriorMoments
    log_evidence: float


def execute_shape_posterior(
    model: ClassifiedShapePosterior,
    *,
    gamma: float,
    sampling: ExplicitSamplingControls,
    settings_sha256: str | None = None,
) -> ExecutedShapePosterior:
    if not isfinite(gamma) or gamma <= 0 or 2 * Fraction(gamma) ** 2 <= 1 or not isfinite(gamma * gamma):
        msg = "proposal requires finite innovation variance and 2 gamma squared > 1 for the second-weight-moment bound"
        raise PosteriorBindingError(msg)
    variance = gamma * gamma
    normal = ExplicitNormalProposal(
        coordinates=("u0", "u1", "u2"),
        means=(0.0, 0.0, 0.0),
        covariance=((variance, 0.0, 0.0), (0.0, variance, 0.0), (0.0, 0.0, variance)),
        source=f"N(0, gamma^2 I), gamma={gamma!r}",
    )
    proposal = _ShapeProposal(model.prior, normal, gamma, model.coordinates)
    target = ExplicitPosteriorTarget(
        coordinates=model.coordinates,
        log_likelihood=model.log_likelihood,
        log_prior=model.prior.logpdf,
        source=f"{model.source}; {model.series.year_definition}; {model.series.discharge_unit}; reference={model.reference_discharge}; {model.source_classification}; prior={model.prior!r}",
    )
    result = independent_importance_sample(
        proposal=proposal,
        target=target,
        controls=sampling,
        provenance=RunProvenance(
            compatibility_profile=LEGACY_FLIKE_PROFILE.identifier,
            method_id="quantile-estimands",
            method_version=METHOD_VERSION,
            package_version=__version__,
            evidence_status=EvidenceStatus.OBSERVED_ARTIFACT,
            input_sha256=model.source_sha256,
            settings_sha256=settings_sha256,
        ),
    )
    particles = result.particles
    ratios = tuple(
        fsum((ll, lp, -lq))
        for ll, lp, lq in zip(particles.log_likelihood, particles.log_prior, particles.log_proposal, strict=True)
    )
    if any(isfinite(r) and w == 0 for r, w in zip(ratios, particles.weights, strict=True)):
        msg = "positive interior importance weight was lost numerically"
        raise PosteriorBindingError(msg)
    peak = max(ratios)
    log_evidence = peak + log(fsum(exp(r - peak) for r in ratios) / sampling.draw_count)
    return ExecutedShapePosterior(particles, result.moments, log_evidence)
