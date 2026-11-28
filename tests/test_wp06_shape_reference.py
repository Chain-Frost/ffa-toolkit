"""Independent 3D Normal quadrature of the bounded-shape synthetic cases.

The regression uses 32 and 56 nodes per innovation axis, the latter at most
175,616 cells. Numerical tolerances, 1,600 draws and seed 87 were fixed before
comparison. Each sampled statistic gets its own reference influence variance.
"""

from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import combinations_with_replacement
from math import exp, fsum, gamma, inf, isclose, log, pi, sqrt

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import gammainc, gammaincc, gammainccinv, gammaincinv, gammaln

from ffa_methods.bayesian_outputs import probability_outputs
from ffa_methods.family_posterior import GEV_LOCATION, GEV_LOG_SCALE, LP3_LOG_SD, LP3_MEAN, PosteriorFamily
from ffa_methods.importance_sampling import ExplicitSamplingControls
from ffa_methods.shape_posterior import BoundedShapePrior, ClassifiedShapePosterior, execute_shape_posterior
from tests.family_posterior_cases import synthetic_flow, synthetic_model

Array = NDArray[np.float64]
CASES = (
    (PosteriorFamily.GEV, (-0.35, 0.35)),
    (PosteriorFamily.LP3, (0.55, 1.25)),
    (PosteriorFamily.LP3, (-1.25, -0.55)),
)
PAIRS = tuple(combinations_with_replacement(range(3), 2))
SAMPLES = ExplicitSamplingControls(
    draw_count=1600, random_seed=87, minimum_effective_size=300, maximum_single_weight=0.03
)


@dataclass(frozen=True)
class ShapeReference:
    values: tuple[float, ...]
    variance_coefficients: tuple[float, ...]


def _gev(z: Array, shape: Array) -> tuple[Array, Array, Array]:
    """Direct CDF, survival and density. At xi=0 use the exact Gumbel limit."""
    offset = z * shape
    interior = offset > -1.0
    log_t = np.log1p(np.where(interior, offset, 0.0))
    log_rate = np.divide(-log_t, shape, where=shape != 0.0, out=np.broadcast_to(-z, shape.shape).copy())
    with np.errstate(over="ignore", under="ignore"):
        rate = np.exp(log_rate)
        density = np.exp(log_rate - rate - log_t)
    return (
        np.where(interior, np.exp(-rate), np.where(shape > 0, 0.0, 1.0)),
        np.where(interior, -np.expm1(-rate), np.where(shape > 0, 1.0, 0.0)),
        np.where(interior, density, 0.0),
    )


def _pearson(z: Array, skew: Array) -> tuple[Array, Array, Array]:
    """Standardized reflected Gamma. No production distribution or likelihood."""
    alpha = 4 / skew**2
    x = np.maximum(0.0, alpha + 2 * z / skew)
    lower = gammainc(alpha, x)
    upper = gammaincc(alpha, x)
    with np.errstate(divide="ignore", under="ignore"):
        density = (2 / np.abs(skew)) * np.exp((alpha - 1) * np.log(x) - x - gammaln(alpha))
    return np.where(skew > 0, lower, upper), np.where(skew > 0, upper, lower), density


def _standard_design(family: PosteriorFamily, shape: Array) -> Array:
    p = 0.99
    if family == PosteriorFamily.GEV:
        limit = -log(-log(p))
        return np.divide(np.expm1(limit * shape), shape, where=shape != 0, out=np.full_like(shape, limit))
    alpha = 4 / shape**2
    x = np.where(shape > 0, gammaincinv(alpha, p), gammainccinv(alpha, p))
    return (shape / 2) * (x - alpha)


def shape_reference(family: PosteriorFamily, bounds: tuple[float, float], count: int) -> ShapeReference:
    """Integrate on R^3. Covariance scores center *both* parameters at posterior means."""
    nodes, weights = hermgauss(count)
    axis = np.sqrt(2.0) * nodes
    weights = weights / sqrt(pi)
    u = axis[:, None, None]
    v = axis[None, :, None]
    w = axis[None, None, :]
    base = weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
    offset = 0.25 * u
    mu = (3.0 if family == PosteriorFamily.GEV else 0.0) + offset
    eta = 0.08 * u + 0.14 * v
    z = 0.20 * u - 0.15 * v + 0.60 * w
    lower, upper = bounds
    shape = lower + (upper - lower) / (1 + np.exp(-z))
    scale = np.exp(eta)
    law = _gev if family == PosteriorFamily.GEV else _pearson
    exact, left, right, low, high, design = (law((x - offset) / scale, shape) for x in (0.2, 0.8, -0.2, -0.3, 1.1, 1.0))
    interval = np.where(low[0] + high[0] < 1, high[0] - low[0], low[1] - high[1])
    jacobian = 1.0 if family == PosteriorFamily.GEV else 1 / (2 * exp(0.2))
    likelihood = (jacobian * exact[2] / scale) ** 2 * left[0] * right[1] * interval
    assert np.all(likelihood >= 0)
    mass = base * likelihood
    evidence = float(mass.sum())
    posterior = mass / evidence

    def expectation(value: Array) -> float:
        return float(np.sum(posterior * value))

    coordinates = (mu, eta, shape)
    centered = tuple(x - expectation(x) for x in coordinates)
    covariances = tuple(centered[i] * centered[j] for i, j in PAIRS)
    quantities = (*coordinates, *covariances, design[1], mu + scale * _standard_design(family, shape))
    means = tuple(expectation(x) for x in quantities)
    second_mass = base * (likelihood / evidence) ** 2
    coefficients = tuple(
        float(np.sum(second_mass * (x - mean) ** 2)) for x, mean in zip(quantities, means, strict=True)
    )
    return ShapeReference(
        values=(evidence, *means),
        variance_coefficients=(float(np.sum(base * (likelihood - evidence) ** 2)), *coefficients),
    )


def synthetic_shape_model(family: PosteriorFamily, bounds: tuple[float, float]) -> ClassifiedShapePosterior:
    """The tied/mixed record from the fixed-shape baseline with a specified joint shape prior."""
    fixed = synthetic_model(family, sum(bounds) / 2)
    rows = (*fixed.series.annual, replace(fixed.series.annual[0], year=2006))
    coordinates = (
        (GEV_LOCATION, GEV_LOG_SCALE, "gev.logit_shape_fraction")
        if family == PosteriorFamily.GEV
        else (LP3_MEAN, LP3_LOG_SD, "lp3.logit_skew_fraction")
    )
    prior = BoundedShapePrior(
        coordinates=coordinates,
        means=(3.0 if family == PosteriorFamily.GEV else 0.0, 0.0, 0.0),
        lower_factor=((0.25, 0.0, 0.0), (0.08, 0.14, 0.0), (0.20, -0.15, 0.60)),
        bounds=bounds,
        source="synthetic correlated Gaussian on location, log-scale, and interval logit",
    )
    return ClassifiedShapePosterior(
        family=family,
        series=replace(fixed.series, annual=rows),
        source="synthetic classified tied exact and censored years",
        source_sha256=sha256(repr(rows).encode("utf-8")).hexdigest(),
        source_classification=fixed.source_classification,
        reference_discharge=fixed.reference_discharge,
        prior=prior,
    )


@pytest.mark.parametrize(("family", "bounds"), CASES)
def test_executed_shape_posterior_against_3d_reference(family: PosteriorFamily, bounds: tuple[float, float]) -> None:
    coarse = shape_reference(family, bounds, 32)
    reference = shape_reference(family, bounds, 56)
    model = synthetic_shape_model(family, bounds)
    run = execute_shape_posterior(model, gamma=1.0, sampling=SAMPLES)
    q = synthetic_flow(family, 1.0)
    output = probability_outputs(run.particles, q, model.kernel, interval_mass=0.9)
    conditional_designs = (model.kernel(draw).ppf(0.99) for draw in run.particles.draws)
    mean_design = fsum(
        weight * (value if family == PosteriorFamily.GEV else log(value / 2.0))
        for weight, value in zip(run.particles.weights, conditional_designs, strict=True)
    )
    actual = (
        exp(run.log_evidence),
        *(parameter.mean for parameter in run.moments.parameters),
        *(run.moments.covariance[i][j] for i, j in PAIRS),
        output.expected_aep,
        mean_design,
    )
    for observed, rough, ref, rough_variance, variance in zip(
        actual,
        coarse.values,
        reference.values,
        coarse.variance_coefficients,
        reference.variance_coefficients,
        strict=True,
    ):
        assert isclose(rough, ref, abs_tol=3e-7, rel_tol=0)
        assert isclose(rough_variance, variance, rel_tol=0.01)
        assert variance > 0
        assert abs(observed - ref) <= 6 * sqrt(variance / SAMPLES.draw_count)


@pytest.mark.parametrize("skew", [-1.25, -0.9, -0.55, 0.55, 0.9, 1.25])
def test_reference_gamma_functions_by_density_integration(skew: float) -> None:
    """Verify the oracle CDF and p=.99 inverse by integrating a normalized density."""
    shape = np.array([skew])
    alpha = 4 / skew**2
    normalizer = gamma(alpha)

    def gamma_density(x: float) -> float:
        return 0.0 if x == 0 else exp((alpha - 1) * log(x) - x) / normalizer

    quantile = float(_standard_design(PosteriorFamily.LP3, shape)[0])
    for index, z in enumerate((-1.0, 0.0, 1.0, quantile)):
        x = max(0.0, alpha + 2 * z / skew)
        lower, lower_error = quad(gamma_density, 0.0, x, epsabs=2e-14, epsrel=2e-14, limit=200)
        upper, upper_error = quad(gamma_density, x, inf, epsabs=2e-14, epsrel=2e-14, limit=200)
        assert max(lower_error, upper_error) < 2e-12
        assert abs(lower + upper - 1.0) < 2e-12
        cdf, survival, _ = _pearson(np.array([z]), shape)
        direct_cdf, direct_survival = (lower, upper) if skew > 0 else (upper, lower)
        assert abs(float(cdf[0]) - direct_cdf) < 2e-12
        assert abs(float(survival[0]) - direct_survival) < 2e-12
        if index == 3:
            assert abs(direct_cdf - 0.99) < 2e-12
