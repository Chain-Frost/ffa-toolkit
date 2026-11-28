"""Two-dimensional quadrature using direct GEV, Normal and Erlang-4 laws."""

from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from math import erf, exp, expm1, fsum, inf, isclose, log, log1p, nextafter, pi, sqrt

import pytest

from ffa_methods.bayesian_outputs import probability_outputs
from ffa_methods.contracts import AnnualObservation
from ffa_methods.family_posterior import (
    GEV_LOCATION,
    GEV_LOG_SCALE,
    LP3_LOG_SD,
    LP3_MEAN,
    ClassifiedEstimatedScalePosterior,
    PosteriorFamily,
    bind_exact_estimated_scale_csv,
    execute_estimated_scale_posterior,
)
from ffa_methods.importance_proposal import CurvatureControls
from ffa_methods.importance_sampling import ExplicitNormalProposal, ExplicitSamplingControls
from ffa_methods.lognormal_posterior import PosteriorBindingError
from ffa_methods.priors import LegacyPriorMode, PriorEvidenceError
from tests.family_posterior_cases import synthetic_flow, synthetic_model

CASES = [(PosteriorFamily.GEV, x) for x in (0.2, 0.0, -0.2)] + [(PosteriorFamily.LP3, x) for x in (1.0, 0.0, -1.0)]
RHO = 0.45
COV = ((0.25**2, 0.25 * 0.16 * RHO), (0.25 * 0.16 * RHO, 0.16**2))
CONTROLS = CurvatureControls(
    scales=(0.25, 0.16),
    initial_step=0.01,
    max_refinements=8,
    max_iterations=30,
    gradient_tolerance=1e-5,
    mode_tolerance=1e-5,
    curvature_tolerance=1e-4,
    max_condition=1e5,
)
SAMPLING = ExplicitSamplingControls(
    draw_count=1200, random_seed=87, minimum_effective_size=300, maximum_single_weight=0.03
)


def _estimated(family: PosteriorFamily, shape: float) -> ClassifiedEstimatedScalePosterior:
    fixed = synthetic_model(family, shape)
    coords = (GEV_LOCATION, GEV_LOG_SCALE) if family == PosteriorFamily.GEV else (LP3_MEAN, LP3_LOG_SD)
    rows = (*fixed.series.annual, replace(fixed.series.annual[0], year=2006))
    return ClassifiedEstimatedScalePosterior(
        family=family,
        series=replace(fixed.series, annual=rows),
        source="synthetic tied exact and censored series",
        source_sha256=sha256(repr(rows).encode("utf-8")).hexdigest(),
        source_classification=fixed.source_classification,
        reference_discharge=fixed.reference_discharge,
        fixed_shape=shape,
        prior=replace(fixed.prior, coordinates=coords, means=(*fixed.prior.means, 0.0), covariance=COV),
    )


def _standard(family: PosteriorFamily, shape: float, z: float) -> tuple[float, float, float]:
    """CDF, survival, density at standardized z, including both exterior limits."""
    if family == PosteriorFamily.GEV:
        t = 1 + shape * z
        if t <= 0:
            return (0.0, 1.0, 0.0) if shape > 0 else (1.0, 0.0, 0.0)
        intensity = t ** (-1 / shape) if shape else exp(-z)
        cdf = exp(-intensity)
        return cdf, -expm1(-intensity), cdf * intensity / t
    if shape == 0:
        return (1 + erf(z / sqrt(2))) / 2, (1 - erf(z / sqrt(2))) / 2, exp(-z * z / 2) / sqrt(2 * pi)
    x = 4 + 2 * shape * z
    if x <= 0:
        return (0.0, 1.0, 0.0) if shape > 0 else (1.0, 0.0, 0.0)
    survival = exp(-x) * (1 + x + x * x / 2 + x**3 / 6)
    density = exp(-x) * x**3 / 3
    return (1 - survival, survival, density) if shape > 0 else (survival, 1 - survival, density)


def _likelihood_aep(family: PosteriorFamily, shape: float, offset: float, eta: float) -> tuple[float, float]:
    scale = exp(eta)
    exact, left, right, low, high, threshold = (
        _standard(family, shape, (x - offset) / scale) for x in (0.2, 0.8, -0.2, -0.3, 1.1, 1.0)
    )
    jacobian = 1.0 if family == PosteriorFamily.GEV else 1 / (2 * exp(0.2))
    like = (exact[2] * jacobian / scale) ** 2 * left[0] * right[1] * (high[0] - low[0])
    return like, threshold[1]


def _proposal_density(mu: float, eta: float, proposal: ExplicitNormalProposal) -> float:
    a, b = proposal.covariance[0]
    d = proposal.covariance[1][1]
    det = a * d - b * b
    u, v = mu - proposal.means[0], eta - proposal.means[1]
    return exp(-(d * u * u - 2 * b * u * v + a * v * v) / (2 * det)) / (2 * pi * sqrt(det))


def _quadrature(
    family: PosteriorFamily,
    shape: float,
    proposal: ExplicitNormalProposal,
    radius: int,
    count: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    # Integrate on whitened prior coordinates. Store target mass and target^2/proposal mass.
    width = 2 * radius / count
    axis = tuple(-radius + (i + 0.5) * width for i in range(count))
    jac = 0.25 * 0.16 * sqrt(1 - RHO**2)
    center = 3.0 if family == PosteriorFamily.GEV else 0.0
    cells: list[tuple[float, float, float, float, float]] = []
    for u in axis:
        for v in axis:
            mu = center + 0.25 * u
            eta = 0.16 * (RHO * u + sqrt(1 - RHO**2) * v)
            likelihood, aep = _likelihood_aep(family, shape, 0.25 * u, eta)
            prior = exp(-(u * u + v * v) / 2) / (2 * pi * jac)
            target = likelihood * prior
            mass = target * width**2 * jac
            cells.append((mass, mass * target / _proposal_density(mu, eta, proposal), mu, eta, aep))
    evidence = fsum(row[0] for row in cells)
    m, e, aep = (fsum(row[0] * row[j] for row in cells) / evidence for j in (2, 3, 4))
    centered = [
        (p, w, (x - m, y - e, (x - m) ** 2, (x - m) * (y - e), (y - e) ** 2, a - aep)) for p, w, x, y, a in cells
    ]
    variances = tuple(fsum(p * h[i] for p, _, h in centered) / evidence for i in (2, 3, 4))
    offsets = (0.0, 0.0, *variances, 0.0)
    coefficients = tuple(
        fsum(w * (h[i] - shift) ** 2 for _, w, h in centered) / evidence**2 for i, shift in enumerate(offsets)
    )
    return (evidence, m, e, *variances, aep), (fsum(row[1] for row in cells) - evidence**2, *coefficients)


@pytest.mark.parametrize(("family", "shape"), CASES)
def test_estimated_posterior(family: PosteriorFamily, shape: float) -> None:
    model = _estimated(family, shape)
    start = tuple(zip(model.coordinates, model.prior.means, strict=True))
    result = execute_estimated_scale_posterior(
        model, starting_point=start, curvature=CONTROLS, sampling=SAMPLING, gamma=1.5
    )
    proposal = result.construction.proposal
    # Sylvester criterion for 2G-P > 0, equivalent to 2P^-1-G^-1 > 0.
    a, b = (2 * x - y for x, y in zip(proposal.covariance[0], COV[0], strict=True))
    d = 2 * proposal.covariance[1][1] - COV[1][1]
    assert a > 0
    assert a * d > b * b
    # The domain argument in the worked example uses the stronger G-P > 0
    # and bounds the whitened center after the squared-likelihood tilt.
    assert proposal.covariance[0][0] > COV[0][0]
    assert (proposal.covariance[0][0] - COV[0][0]) * (proposal.covariance[1][1] - COV[1][1]) > (
        proposal.covariance[0][1] - COV[0][1]
    ) ** 2
    u = (proposal.means[0] - model.prior.means[0]) / 0.25
    v = (proposal.means[1] / 0.16 - RHO * u) / sqrt(1 - RHO**2)
    assert sqrt(u * u + v * v) + 4 * 0.16 < 1.3
    coarse, cvar = _quadrature(family, shape, proposal, 10, 81)
    refined, rvar = _quadrature(family, shape, proposal, 12, 145)
    particles = result.sampling.particles
    evidence = (
        fsum(
            exp(like + prior - proposal_log)
            for like, prior, proposal_log in zip(
                particles.log_likelihood, particles.log_prior, particles.log_proposal, strict=True
            )
        )
        / SAMPLING.draw_count
    )
    moments = result.sampling.moments
    output = probability_outputs(particles, synthetic_flow(family, 1.0), model.kernel, interval_mass=0.9)
    actual = (
        evidence,
        *(v.mean for v in moments.parameters),
        *moments.covariance[0],
        moments.covariance[1][1],
        output.expected_aep,
    )
    for first, ref, low, var, observed in zip(coarse, refined, cvar, rvar, actual, strict=True):
        assert isclose(first, ref, rel_tol=0, abs_tol=2e-8)
        assert isclose(low, var, rel_tol=0.01)
        assert var > 0
        assert abs(observed - ref) <= 6 * sqrt(var / SAMPLING.draw_count)
    assert particles.provenance.method_version == model.version
    if family == PosteriorFamily.LP3 and shape == 0.0:
        replay = execute_estimated_scale_posterior(
            model, starting_point=start, curvature=CONTROLS, sampling=SAMPLING, gamma=1.5
        )
        assert result == replay


@pytest.mark.parametrize(("family", "shape"), CASES)
def test_joint_density_units_and_reference(family: PosteriorFamily, shape: float) -> None:
    model = _estimated(family, shape)
    mu = model.prior.means[0] + 0.4
    eta = -0.3
    like, _ = _likelihood_aep(family, shape, 0.4, eta)
    assert isclose(model.log_likelihood((mu, eta)), log(like), abs_tol=2e-12, rel_tol=0)
    z1, z2 = 0.4 / 0.25, eta / 0.16
    log_prior = -log(2 * pi * 0.25 * 0.16 * sqrt(1 - RHO**2)) - (z1 * z1 - 2 * RHO * z1 * z2 + z2 * z2) / (
        2 * (1 - RHO**2)
    )
    assert isclose(model.target().log_prior((mu, eta)), log_prior, abs_tol=2e-13, rel_tol=0)
    factor = 1000.0

    def rescale(row: AnnualObservation) -> AnnualObservation:
        return replace(
            row,
            value=None if row.value is None else factor * row.value,
            lower_bound=None if row.lower_bound is None else factor * row.lower_bound,
            upper_bound=None if row.upper_bound is None else factor * row.upper_bound,
        )

    series = replace(model.series, discharge_unit="L/s", annual=tuple(rescale(row) for row in model.series.annual))
    if family == PosteriorFamily.GEV:
        coords = (factor * mu, eta + log(factor))
        prior = replace(
            model.prior,
            means=(factor * 3.0, log(factor)),
            covariance=((factor**2 * COV[0][0], factor * COV[0][1]), (factor * COV[1][0], COV[1][1])),
        )
        transformed = replace(model, series=series, prior=prior)
        log_det = log(factor)
    else:
        # Convert units and double the physical reference in one transformation.
        coords = (mu - log(2), eta)
        prior = replace(model.prior, means=(-log(2), 0.0))
        transformed = replace(model, series=series, prior=prior, reference_discharge=4 * factor)
        log_det = 0.0
    assert isclose(transformed.prior.logpdf(coords) + log_det, log_prior, abs_tol=3e-13, rel_tol=0)
    assert isclose(transformed.log_likelihood(coords) + 2 * log(factor), log(like), abs_tol=2e-12, rel_tol=0)


@pytest.mark.parametrize(("family", "shape"), [case for case in CASES if case[1]])
def test_moving_endpoint_and_ties(family: PosteriorFamily, shape: float) -> None:
    model = _estimated(family, shape)
    row = model.series.annual[0]
    model = replace(model, series=replace(model.series, annual=(row, replace(row, year=2007))))
    mu = model.prior.means[0] + (6 if shape > 0 else -6)
    assert model.log_likelihood((mu, -1.0)) == -inf
    assert model.log_likelihood((mu, 2.0)) > -inf


def test_estimated_parameter_failures() -> None:
    model = _estimated(PosteriorFamily.GEV, 0.0)
    for coords in ((GEV_LOG_SCALE, GEV_LOCATION), (GEV_LOCATION, "gev.scale")):
        with pytest.raises(PosteriorBindingError, match="coordinate"):
            replace(model, prior=replace(model.prior, coordinates=coords))
    with pytest.raises(PosteriorBindingError, match="supplied"):
        replace(model, prior=replace(model.prior, mode=LegacyPriorMode.REGIONAL))
    for values in ((0.0,), (0.0, inf)):
        with pytest.raises(PosteriorBindingError, match="log-scale"):
            model.kernel(values)
    with pytest.raises(ValueError, match="scale"):
        model.kernel((0.0, 1000.0))
    with pytest.raises(PosteriorBindingError, match="interior"):
        model.log_likelihood((1000.0, 0.0))


@pytest.mark.parametrize(("family", "shape"), [(PosteriorFamily.GEV, -0.2), (PosteriorFamily.LP3, 1.0)])
def test_estimated_csv_source(family: PosteriorFamily, shape: float) -> None:
    model = _estimated(family, shape)
    raw = b"Year,AMS\n1980,1.3\n1981,1.3\n"
    bound = bind_exact_estimated_scale_csv(
        raw,
        family=family,
        source="synthetic CSV",
        discharge_unit="L/s",
        year_definition="water year",
        source_classification=model.source_classification,
        fixed_shape=shape,
        reference_discharge=model.reference_discharge,
        prior=model.prior,
    )
    assert bound.source_sha256 == sha256(raw).hexdigest()
    assert bound.series.year_definition == "water year"
    assert bound.series.discharge_unit == "L/s"
    mu, eta = model.prior.means
    assert bound.log_likelihood((mu, eta)) == 2 * bound.kernel((mu, eta)).logpdf(1.3)


@pytest.mark.parametrize("skew", [0.0, 1.0])
def test_lp3_neighbour_of_reference_at_small_scale(skew: float) -> None:
    model = replace(_estimated(PosteriorFamily.LP3, skew), reference_discharge=10.0)
    q = nextafter(10.0, inf)
    eta = log(1e-16)
    mu = 2 * exp(eta) if skew else 0.0
    z = (log1p((q - 10) / 10) - mu) / exp(eta)
    expected = log(_standard(PosteriorFamily.LP3, skew, z)[2]) - eta - log(q)
    assert isclose(model.kernel((mu, eta)).logpdf(q), expected, rel_tol=0, abs_tol=2e-13)


def test_lp3_log_observation_resolution() -> None:
    model = replace(_estimated(PosteriorFamily.LP3, 1.0), reference_discharge=1.0)
    # 70-digit direct Erlang evaluation retains ln(2)-binary64 ln(2) before scaling.
    assert isclose(model.kernel((log(2), log(3e-17))).logpdf(2), 35.8467913373, abs_tol=6e-11, rel_tol=0)


@pytest.mark.parametrize(
    ("mu", "eta", "xi", "q"),
    [
        (0.2, 0.1, -0.4, 2.962927295189119),
        (19.416239798574416, 1.4551134923013125, -0.9366626862027638, 23.99096003906574),
    ],
)
def test_rounded_gev_endpoint_exterior(mu: float, eta: float, xi: float, q: float) -> None:
    kernel = _estimated(PosteriorFamily.GEV, xi).kernel((mu, eta))
    assert kernel.logpdf(q) == -inf
    assert kernel.logcdf(q) == 0.0
    assert kernel.logsf(q) == -inf
    assert kernel.cdf(q) == 1.0
    assert kernel.sf(q) == 0.0


def test_unresolved_interior_endpoint() -> None:
    gev = _estimated(PosteriorFamily.GEV, -0.5).kernel((1.0, 1e-17))
    lp3 = replace(_estimated(PosteriorFamily.LP3, 1.0), reference_discharge=10.0).kernel((2.0, 0.0))
    for kernel, q in ((gev, 3.0), (lp3, nextafter(10.0, inf))):
        with pytest.raises(PosteriorBindingError, match="interior"):
            kernel.logpdf(q)
        with pytest.raises(PosteriorBindingError, match="interior"):
            kernel.sf(q)


def test_small_covariance_symmetry() -> None:
    model = _estimated(PosteriorFamily.GEV, 0.0)
    with pytest.raises(PosteriorBindingError, match="symmetric"):
        replace(model, prior=replace(model.prior, covariance=((1e-18, 1e-15), (0.0, 1e-18))))
    valid = replace(model, prior=replace(model.prior, covariance=((1e-18, 1e-19), (1e-19, 1e-18))))
    assert valid.prior.logpdf(valid.prior.means) > 0
    with pytest.raises(PosteriorBindingError, match="finite"):
        replace(synthetic_model(PosteriorFamily.GEV, 0.0), fixed_log_scale=inf)


def test_exact_covariance_pivot_and_conditional_density() -> None:
    model = _estimated(PosteriorFamily.GEV, 0.0)
    for d in (0.5, nextafter(0.5, 0.0)):
        with pytest.raises(PriorEvidenceError, match="positive definite"):
            replace(model.prior, covariance=((0.5, 0.5), (0.5, d)))
    d = nextafter(0.5, inf)
    prior = replace(model.prior, means=(0.0, 0.0), covariance=((0.5, 0.5), (0.5, d)))
    # Exact inverse of [[a,b],[b,d]], with one observation off the y=x ridge.
    u, v = Fraction(1.0), Fraction(1 + 2**-26)
    det = (Fraction(d) - Fraction(0.5)) / 2
    quad = (Fraction(d) * u * u - u * v + v * v / 2) / det
    center = -log(2 * pi) - log(float(det)) / 2
    assert isclose(prior.logpdf(prior.means), center, abs_tol=1e-14, rel_tol=0)
    assert isclose(prior.logpdf((float(u), float(v))), center - float(quad) / 2, abs_tol=1e-14, rel_tol=0)
