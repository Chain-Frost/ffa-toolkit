"""WP06: explicit independent Gaussian-proposal importance sampling.

ARR Book 3 (2019 v4.2), section 2.6.3.6, describes a Normal proposal
around an estimated posterior mode. This module implements *only* the
importance-draw/weight stage from a fully caller-supplied proposal. It does
not choose a mode, Hessian, prior, vendor setting or proposal default.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from math import fsum, hypot, isfinite, log, pi, sqrt, ulp
from random import Random
from sys import version
from typing import TypeAlias

from .bayesian_outputs import ImportanceParticles, PosteriorMoments, require_weight_adequacy
from .contracts import EvidenceStatus, RunProvenance

LogDensity: TypeAlias = Callable[[tuple[float, ...]], float]


class SamplingError(ValueError):
    """Invalid explicit sampling configuration or an unrepresentable result."""


def _covariance_factor(covariance: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """Factor a full finite symmetric positive-definite matrix."""
    n = len(covariance)
    rows: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            try:
                remainder = covariance[i][j] - fsum(rows[i][k] * rows[j][k] for k in range(j))
                if i == j:
                    if remainder <= 0.0 or not isfinite(remainder):
                        msg = "proposal covariance must be positive definite"
                        raise SamplingError(msg)
                    rows[i][j] = sqrt(remainder)
                else:
                    rows[i][j] = remainder / rows[j][j]
            except OverflowError as exc:
                msg = "proposal covariance factorization is not representable"
                raise SamplingError(msg) from exc
            if not isfinite(rows[i][j]) or (rows[i][j] == 0.0 and i == j):
                msg = "proposal covariance factor is not representable"
                raise SamplingError(msg)
    return tuple(tuple(row) for row in rows)


def _scaled_euclidean_norm(components: Iterable[float]) -> float:
    """A two-pass-free scaled L2 norm, without premature residual-square underflow."""
    scale = 0.0
    sum_squares = 1.0
    for component in components:
        magnitude = abs(component)
        if not isfinite(magnitude):
            msg = "weighted residual is not finite"
            raise SamplingError(msg)
        if magnitude == 0.0:
            continue
        if scale < magnitude:
            sum_squares = 1.0 + sum_squares * (scale / magnitude) ** 2
            scale = magnitude
        else:
            sum_squares += (magnitude / scale) ** 2
    return scale * sqrt(sum_squares)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExplicitNormalProposal:
    """Full multivariate Normal density in named original/source coordinates.

    All locations and covariance entries must be supplied. No optimization,
    covariance shrinkage, prior substitution or independence assumption is
    introduced. Covariance is an SPD matrix, not a vector of variances.
    """

    coordinates: tuple[str, ...]
    means: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    source: str
    _lower: tuple[tuple[float, ...], ...] = field(init=False, repr=False, compare=False)
    _log_normalizer: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = len(self.coordinates)
        if (
            not n
            or len(self.means) != n
            or len(self.covariance) != n
            or len(set(self.coordinates)) != n
            or any(not item.strip() for item in self.coordinates)
            or not self.source.strip()
            or any(not isfinite(mean) for mean in self.means)
            or any(len(row) != n or any(not isfinite(v) for v in row) for row in self.covariance)
        ):
            msg = "proposal requires source, unique coordinates, finite means and full finite covariance"
            raise SamplingError(msg)
        if any(
            abs(self.covariance[i][j] - self.covariance[j][i])
            > 1e-12 * max(abs(self.covariance[i][j]), abs(self.covariance[j][i]))
            for i in range(n)
            for j in range(i)
        ):
            msg = "proposal covariance must be symmetric"
            raise SamplingError(msg)
        lower = _covariance_factor(self.covariance)
        for index, mean in enumerate(self.means):
            # A Normal coordinate is not faithfully representable when one
            # marginal SD fits inside a single binary64 spacing at its mean.
            marginal_sd = sqrt(self.covariance[index][index])
            if marginal_sd <= ulp(mean):
                msg = "proposal coordinate resolution cannot represent one marginal standard deviation"
                raise SamplingError(msg)
        try:
            log_normalizer = -0.5 * n * log(2.0 * pi) - fsum(log(lower[i][i]) for i in range(n))
        except (OverflowError, ValueError) as exc:
            msg = "proposal normalization is not representable"
            raise SamplingError(msg) from exc
        if not isfinite(log_normalizer):
            msg = "proposal normalization is not finite"
            raise SamplingError(msg)
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(self, "_log_normalizer", log_normalizer)

    def draw(self, rng: Random) -> tuple[float, ...]:
        """Return one actual random vector with the explicitly specified covariance."""
        normals = tuple(rng.gauss(0.0, 1.0) for _ in self.coordinates)
        try:
            sampled = tuple(
                self.means[i] + fsum(self._lower[i][j] * normals[j] for j in range(i + 1))
                for i in range(len(self.coordinates))
            )
        except OverflowError as exc:
            msg = "proposal draw exceeded representable parameter coordinates"
            raise SamplingError(msg) from exc
        if any(not isfinite(value) for value in sampled):
            msg = "proposal draw is not finite"
            raise SamplingError(msg)
        return sampled

    def logpdf(self, values: Sequence[float]) -> float:
        """Multivariate Normal log density of the *stored* generated vector."""
        if len(values) != len(self.coordinates) or any(not isfinite(v) for v in values):
            msg = "proposal log density requires a finite vector in the named coordinates"
            raise SamplingError(msg)
        residuals: list[float] = []
        try:
            for i, value in enumerate(values):
                residual = value - self.means[i] - fsum(self._lower[i][j] * residuals[j] for j in range(i))
                residuals.append(residual / self._lower[i][i])
            value = self._log_normalizer - 0.5 * fsum(v * v for v in residuals)
        except OverflowError as exc:
            msg = "proposal log density is not representable"
            raise SamplingError(msg) from exc
        if not isfinite(value):
            msg = "proposal log density is not finite"
            raise SamplingError(msg)
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class ExplicitPosteriorTarget:
    """Likelihood and prior densities in *exactly* the proposal coordinates.

    Caller must supply the fully specified mathematical data/parameter binding
    and any required Jacobians before evaluation. A callback exception stops
    the run; negative infinity is a valid out-of-support contribution.
    """

    coordinates: tuple[str, ...]
    log_likelihood: LogDensity
    log_prior: LogDensity
    source: str

    def __post_init__(self) -> None:
        if not self.coordinates or len(set(self.coordinates)) != len(self.coordinates) or not self.source.strip():
            msg = "posterior target needs ordered unique coordinates and evidence source"
            raise SamplingError(msg)
        if not callable(self.log_likelihood) or not callable(self.log_prior):
            msg = "posterior target requires explicit likelihood and prior functions"
            raise SamplingError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExplicitSamplingControls:
    """All numerical quality decisions are caller-supplied, never vendor defaults."""

    draw_count: int
    random_seed: int
    minimum_effective_size: float
    maximum_single_weight: float

    def __post_init__(self) -> None:
        if type(self.draw_count) is not int or not 2 <= self.draw_count <= 1_000_000:
            msg = "draw count must be an integer from 2 to 1,000,000"
            raise SamplingError(msg)
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**64:
            msg = "random seed must be a nonnegative integer smaller than 2**64"
            raise SamplingError(msg)
        if not isfinite(self.minimum_effective_size) or not 0 < self.minimum_effective_size <= self.draw_count:
            msg = "minimum ESS must be finite, positive and no larger than draw count"
            raise SamplingError(msg)
        if not isfinite(self.maximum_single_weight) or not 1 / self.draw_count <= self.maximum_single_weight <= 1:
            msg = "maximum single weight must be finite and feasible for draw count"
            raise SamplingError(msg)


@dataclass(frozen=True, slots=True)
class IndependentSamplingResult:
    """Executed and quality-gated particles; MCSE is approximate, not an interval."""

    particles: ImportanceParticles
    moments: PosteriorMoments
    controls: ExplicitSamplingControls
    approximate_mean_mcse: tuple[float, ...]
    algorithm: str = "independent multivariate-normal self-normalized importance sampling"


def independent_importance_sample(
    *,
    proposal: ExplicitNormalProposal,
    target: ExplicitPosteriorTarget,
    controls: ExplicitSamplingControls,
    provenance: RunProvenance,
) -> IndependentSamplingResult:
    """Draw real iid proposal particles and evaluate source-coordinate log ratios.

    Global ESS and weight gates are necessary health checks, NOT evidence of
    rare design-quantile precision. No unverified legacy Flike default exists.
    """
    if proposal.coordinates != target.coordinates:
        msg = "target/proposal source-coordinate names and order must match exactly"
        raise SamplingError(msg)
    if provenance.evidence_status == EvidenceStatus.QUALIFIED_REFERENCE:
        msg = "this independent mathematical sampler cannot assert native qualification"
        raise SamplingError(msg)
    rng = Random(controls.random_seed)  # noqa: S311 - reproducible statistical, not security, randomness
    draws: list[tuple[float, ...]] = []
    likelihoods: list[float] = []
    priors: list[float] = []
    proposals: list[float] = []
    for index in range(controls.draw_count):
        theta = proposal.draw(rng)
        try:
            log_proposal = proposal.logpdf(theta)
            log_likelihood = target.log_likelihood(theta)
            log_prior = target.log_prior(theta)
        except (ArithmeticError, ValueError, OverflowError) as exc:
            msg = f"importance draw {index} density evaluation failed"
            raise SamplingError(msg) from exc
        if any(v != v or v == float("inf") for v in (log_likelihood, log_prior)):
            msg = f"importance draw {index} has invalid target log density"
            raise SamplingError(msg)
        draws.append(theta)
        likelihoods.append(log_likelihood)
        priors.append(log_prior)
        proposals.append(log_proposal)
    stream = (
        f"local random.Random(seed={controls.random_seed}).gauss; CPython {version.split()[0]}; "
        "independent draws, no hidden optimizer; not native Flike RNG"
    )
    try:
        particles = ImportanceParticles(
            coordinates=proposal.coordinates,
            draws=tuple(draws),
            log_likelihood=tuple(likelihoods),
            log_prior=tuple(priors),
            log_proposal=tuple(proposals),
            proposal_description=proposal.source,
            random_stream_description=stream,
            provenance=provenance,
        )
        moments = require_weight_adequacy(
            particles,
            minimum_effective_size=controls.minimum_effective_size,
            maximum_single_weight=controls.maximum_single_weight,
        )
    except ValueError as exc:
        msg = "independent importance run fails weight adequacy or representability"
        raise SamplingError(msg) from exc
    mcse = tuple(
        _scaled_euclidean_norm(
            weight * (draw[i] - moment.mean) for weight, draw in zip(particles.weights, particles.draws, strict=True)
        )
        for i, moment in enumerate(moments.parameters)
    )
    if any(not isfinite(value) for value in mcse):
        msg = "approximate posterior-mean Monte Carlo error is not representable"
        raise SamplingError(msg)
    return IndependentSamplingResult(
        particles=particles,
        moments=moments,
        controls=controls,
        approximate_mean_mcse=mcse,
    )


def standardized_independent_replication_difference(
    first: IndependentSamplingResult, second: IndependentSamplingResult
) -> tuple[float, ...]:
    """Diagnostic |mu1-mu2|/sqrt(MCSE1^2+MCSE2^2) for independent seeds.

    A caller must inspect these diagnostics and choose a scientifically
    justified tolerance. This is *not* an automatic convergence certificate.
    """
    if first.particles.coordinates != second.particles.coordinates:
        msg = "replicates have different parameter coordinates"
        raise SamplingError(msg)
    if first.controls.random_seed == second.controls.random_seed:
        msg = "independent replicate comparison requires distinct seeds"
        raise SamplingError(msg)
    result: list[float] = []
    for a, b, se_a, se_b in zip(
        first.moments.parameters,
        second.moments.parameters,
        first.approximate_mean_mcse,
        second.approximate_mean_mcse,
        strict=True,
    ):
        denominator = hypot(se_a, se_b)
        delta = abs(a.mean - b.mean)
        score = delta / denominator if denominator else (0.0 if delta == 0.0 else float("inf"))
        result.append(score)
    return tuple(result)
