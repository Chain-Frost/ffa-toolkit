"""Independent density/censor and one-dimensional posterior checks for issue #8."""

from dataclasses import replace
from hashlib import sha256
from math import exp, fsum, inf, isclose, log, sqrt
from statistics import NormalDist

import pytest

from ffa_methods.bayesian_outputs import probability_outputs
from ffa_methods.contracts import AnnualObservation, EvidenceStatus, ExposureKind, ObservationKind, ObservationSeries
from ffa_methods.family_posterior import (
    GEV_LOCATION,
    LP3_MEAN,
    ClassifiedFixedFamilyPosterior,
    PosteriorFamily,
    bind_exact_fixed_family_csv,
    execute_fixed_family_posterior,
)
from ffa_methods.importance_proposal import CurvatureControls
from ffa_methods.importance_sampling import ExplicitSamplingControls
from ffa_methods.likelihood import log_likelihood
from ffa_methods.lognormal_posterior import SOURCE_CLASSIFICATION, PosteriorBindingError
from ffa_methods.priors import JointGaussianPrior, LegacyPriorMode


def _flow(family: PosteriorFamily, value: float) -> float:
    return 3.0 + value if family == PosteriorFamily.GEV else 2.0 * exp(value)


def _row(
    year: int,
    kind: ObservationKind,
    *,
    value: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> AnnualObservation:
    return AnnualObservation(
        year=year,
        kind=kind,
        exposure=ExposureKind.SYSTEMATIC,
        value=value,
        lower_bound=lower,
        upper_bound=upper,
    )


def _model(family: PosteriorFamily, shape: float) -> ClassifiedFixedFamilyPosterior:
    q, upper, lower, low, high = (_flow(family, x) for x in (0.2, 0.8, -0.2, -0.3, 1.1))
    rows = (
        _row(2000, ObservationKind.EXACT, value=q),
        _row(2001, ObservationKind.LEFT_CENSORED, upper=upper),
        _row(2002, ObservationKind.RIGHT_CENSORED, lower=lower),
        _row(2003, ObservationKind.INTERVAL_CENSORED, lower=low, upper=high),
        _row(2004, ObservationKind.MISSING),
        _row(2005, ObservationKind.EXCLUDED, value=0.0),
    )
    is_gev = family == PosteriorFamily.GEV
    return ClassifiedFixedFamilyPosterior(
        family=family,
        series=ObservationSeries(discharge_unit="m3/s", annual=rows, year_definition="calendar year"),
        source="synthetic classified annual maxima",
        source_sha256=sha256(b"synthetic classified records").hexdigest(),
        source_classification=SOURCE_CLASSIFICATION,
        fixed_log_scale=0.0,
        fixed_shape=shape,
        reference_discharge=None if is_gev else 2.0,
        prior=JointGaussianPrior(
            coordinates=(GEV_LOCATION if is_gev else LP3_MEAN,),
            means=(3.0 if is_gev else 0.0,),
            covariance=((0.0625,),),
            source="synthetic supplied location prior",
            mode=LegacyPriorMode.SUPPLIED,
        ),
    )


def _law(family: PosteriorFamily, shape: float, mu: float, q: float) -> tuple[float, float]:
    """Independent CDF and raw-flow density: unit scale GEV or Erlang-4 LP3."""
    if family == PosteriorFamily.GEV:
        z = 1.0 + shape * (q - mu)
        if z <= 0:
            return (0.0 if shape > 0 else 1.0), 0.0
        a = z ** (-1.0 / shape)
        return exp(-a), exp(-a) * z ** (-1.0 / shape - 1.0)
    # At skew +/-1, the unit-scale Gamma variate has integer shape 4.
    # Its survival is exp(-x) times a cubic; no SciPy or Toolkit kernel.
    x = (1.0 if shape > 0 else -1.0) * (log(q / 2.0) - mu + 2.0 / shape) / 0.5
    if x <= 0:
        return (0.0 if shape > 0 else 1.0), 0.0
    survival = exp(-x) * (1.0 + x + x**2 / 2.0 + x**3 / 6.0)
    return (1.0 - survival if shape > 0 else survival), exp(-x) * x**3 / (3.0 * q)


def _terms(family: PosteriorFamily, shape: float, mu: float) -> tuple[float, ...]:
    q, upper, lower, low, high = (_flow(family, x) for x in (0.2, 0.8, -0.2, -0.3, 1.1))
    return (
        _law(family, shape, mu, q)[1],
        _law(family, shape, mu, upper)[0],
        1.0 - _law(family, shape, mu, lower)[0],
        _law(family, shape, mu, high)[0] - _law(family, shape, mu, low)[0],
    )


@pytest.mark.parametrize(
    ("family", "shape"),
    [(PosteriorFamily.GEV, 0.2), (PosteriorFamily.GEV, -0.2), (PosteriorFamily.LP3, 1.0), (PosteriorFamily.LP3, -1.0)],
)
def test_family_density_prior_and_censor_terms(family: PosteriorFamily, shape: float) -> None:
    model = _model(family, shape)
    mu = model.prior.means[0] + 0.1
    kernel = model.kernel((mu,))
    actual = log_likelihood(kernel, model.series)
    expected = tuple(log(x) for x in _terms(family, shape, mu))
    assert tuple(term.included for term in actual.terms) == (True, True, True, True, False, False)
    for term, reference in zip(actual.terms[:4], expected, strict=True):
        assert isclose(term.log_value, reference, rel_tol=0, abs_tol=2e-12)
    assert isclose(model.target().log_likelihood((mu,)), fsum(expected), rel_tol=0, abs_tol=3e-12)
    assert isclose(
        model.target().log_prior((mu,)), log(NormalDist(model.prior.means[0], 0.25).pdf(mu)), abs_tol=1e-12
    )
    assert model.source_sha256 in model.target().source
    # Move the parameter far enough that the exact observation becomes impossible.
    impossible_mu = mu + (10.0 if shape > 0 else -10.0)
    assert model.log_likelihood((impossible_mu,)) == -inf


def _reference(
    family: PosteriorFamily, shape: float, points: int, proposal: NormalDist
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Midpoint quadrature, independent of model, sampler and probability kernels."""
    center = 3.0 if family == PosteriorFamily.GEV else 0.0
    prior = NormalDist(center, 0.25)
    width = 4.5 / points
    cells: list[tuple[float, float, tuple[float, float]]] = []
    for i in range(points):
        mu = center - 2.25 + (i + 0.5) * width
        a, b, c, d = _terms(family, shape, mu)
        density = prior.pdf(mu) * a * b * c * d
        survival = 1.0 - _law(family, shape, mu, _flow(family, 1.0))[0]
        cells.append((density, proposal.pdf(mu), (mu, survival)))
    normalizer = fsum(density * width for density, _, _ in cells)
    means = tuple(fsum(density * values[k] * width for density, _, values in cells) / normalizer for k in range(2))
    variances = tuple(
        fsum(
            (density / normalizer) ** 2 / g * (values[k] - means[k]) ** 2 * width
            for density, g, values in cells
        )
        for k in range(2)
    )
    return means, variances


@pytest.mark.parametrize(("family", "shape"), [(PosteriorFamily.GEV, 0.2), (PosteriorFamily.LP3, -1.0)])
def test_executed_posterior_against_independent_quadrature(family: PosteriorFamily, shape: float) -> None:
    model = _model(family, shape)
    controls = ExplicitSamplingControls(
        draw_count=1600, random_seed=87, minimum_effective_size=800.0, maximum_single_weight=0.01
    )
    result = execute_fixed_family_posterior(
        model,
        starting_point=((model.coordinates[0], model.prior.means[0]),),
        curvature=CurvatureControls(
            scales=(0.25,),
            initial_step=0.01,
            max_refinements=8,
            max_iterations=30,
            gradient_tolerance=1e-5,
            mode_tolerance=1e-5,
            curvature_tolerance=1e-4,
            max_condition=1e5,
        ),
        sampling=controls,
        gamma=1.5,
    )
    proposal = result.construction.proposal
    assert result.construction.gamma == 1.5
    # A fixed-scale nonsingular density is bounded. The proposal variance
    # exceeds half the prior variance, making the Gaussian ratio integrable.
    assert proposal.covariance[0][0] > 0.0625 / 2.0
    normal = NormalDist(proposal.means[0], sqrt(proposal.covariance[0][0]))
    coarse, coarse_variance = _reference(family, shape, 801, normal)
    expected, variance = _reference(family, shape, 1601, normal)
    for low, high in zip(coarse, expected, strict=True):
        assert isclose(low, high, rel_tol=0, abs_tol=1e-9)
    for low, high in zip(coarse_variance, variance, strict=True):
        assert isclose(low, high, rel_tol=1e-6)
    run = result.sampling
    output = probability_outputs(run.particles, _flow(family, 1.0), model.kernel, interval_mass=0.9)
    actual = (run.moments.parameters[0].mean, output.expected_aep)
    for observed, reference, coefficient in zip(actual, expected, variance, strict=True):
        assert coefficient > 0
        assert isclose(observed, reference, rel_tol=0, abs_tol=6.0 * sqrt(coefficient / controls.draw_count))
    assert isclose(output.expected_aep + output.expected_non_exceedance, 1.0, abs_tol=1e-12)
    assert run.particles.provenance.input_sha256 == model.source_sha256
    assert run.particles.provenance.evidence_status == EvidenceStatus.OBSERVED_ARTIFACT


@pytest.mark.parametrize(("family", "shape"), [(PosteriorFamily.GEV, 0.2), (PosteriorFamily.LP3, -1.0)])
def test_exact_csv_and_unsupported_sources(family: PosteriorFamily, shape: float) -> None:
    model = _model(family, shape)
    data = b"Year,AMS\n2001,2.0\n2002,3.0\n"

    def bind(raw: bytes) -> ClassifiedFixedFamilyPosterior:
        return bind_exact_fixed_family_csv(
            raw,
            family=family,
            source="synthetic CSV",
            discharge_unit="m3/s",
            year_definition="calendar year",
            source_classification=SOURCE_CLASSIFICATION,
            fixed_log_scale=model.fixed_log_scale,
            fixed_shape=shape,
            reference_discharge=model.reference_discharge,
            prior=model.prior,
        )

    bound = bind(data)
    assert bound.source_sha256 == sha256(data).hexdigest()
    assert tuple(row.value for row in bound.series.annual) == (2.0, 3.0)
    for invalid in (b"Year,AMS\n2001,0\n", b"Year,AMS\n2001,\n", b"Year,AMS\n2001,1e-500\n"):
        with pytest.raises(PosteriorBindingError):
            bind(invalid)
    with pytest.raises(PosteriorBindingError, match="coordinate"):
        replace(model, prior=replace(model.prior, coordinates=("wrong",)))
    with pytest.raises(PosteriorBindingError, match="source"):
        replace(model, source_classification="native")
    with pytest.raises(PosteriorBindingError, match="zero-threshold"):
        replace(model, series=replace(model.series, zero_threshold=0.0))
    with pytest.raises(PosteriorBindingError, match="systematic"):
        replace(model, series=replace(model.series, annual=(replace(model.series.annual[0], exposure=ExposureKind.HISTORICAL),)))
    with pytest.raises(ValueError, match="overlapping"):
        replace(model, series=replace(model.series, annual=(model.series.annual[0],) * 2))
    with pytest.raises(PosteriorBindingError, match="one finite"):
        model.kernel((1.0, 2.0))


def test_shape_and_reference_admission() -> None:
    gev = _model(PosteriorFamily.GEV, 0.2)
    lp3 = _model(PosteriorFamily.LP3, 1.0)
    with pytest.raises(PosteriorBindingError, match="EVT xi"):
        replace(gev, fixed_shape=-1.0)
    with pytest.raises(PosteriorBindingError, match="no log reference"):
        replace(gev, reference_discharge=1.0)
    for shape in (-2.0, 2.0, 1e-6):
        with pytest.raises(PosteriorBindingError, match="skew"):
            replace(lp3, fixed_shape=shape)
    with pytest.raises(PosteriorBindingError, match="reference discharge"):
        replace(lp3, reference_discharge=None)
    # Exact zero limits are not the near-zero LP3 approximation.
    assert isclose(replace(gev, fixed_shape=0.0).kernel((3.0,)).cdf(4.0), exp(-exp(-1.0)), abs_tol=1e-14)
    assert isclose(replace(lp3, fixed_shape=0.0).kernel((0.0,)).cdf(2 * exp(1.0)), NormalDist().cdf(1.0), abs_tol=1e-14)
