"""Independent density and actual generated-particle tests for WP06 lognormal binding."""

from dataclasses import replace
from hashlib import sha256
from math import exp, isclose, log, pi, sqrt
from statistics import NormalDist

import pytest

from ffa_methods.bayesian_outputs import probability_outputs
from ffa_methods.contracts import AnnualObservation, EvidenceStatus, ExposureKind, ObservationKind, ObservationSeries
from ffa_methods.importance_proposal import CurvatureControls
from ffa_methods.importance_sampling import ExplicitSamplingControls
from ffa_methods.lognormal_posterior import (
    LOG_SD_COORDINATE,
    MEAN_COORDINATE,
    SOURCE_CLASSIFICATION,
    ClassifiedLognormalPosterior,
    PosteriorBindingError,
    bind_exact_lognormal_csv,
    execute_lognormal_posterior,
)
from ffa_methods.priors import JointGaussianPrior, LegacyPriorMode


def _prior(*, estimate_scale: bool = False) -> JointGaussianPrior:
    return JointGaussianPrior(
        coordinates=(MEAN_COORDINATE, LOG_SD_COORDINATE) if estimate_scale else (MEAN_COORDINATE,),
        means=(0.0, 0.0) if estimate_scale else (0.0,),
        covariance=((2.0, 0.5), (0.5, 1.0)) if estimate_scale else ((1.0,),),
        source="synthetic explicitly supplied Gaussian prior",
        mode=LegacyPriorMode.SUPPLIED,
    )


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


def _model(rows: tuple[AnnualObservation, ...], *, estimate_scale: bool = False) -> ClassifiedLognormalPosterior:
    return ClassifiedLognormalPosterior(
        series=ObservationSeries(discharge_unit="m3/s", annual=rows, year_definition="calendar year"),
        source="synthetic classified annual maxima",
        source_sha256=sha256(b"synthetic classified annual records").hexdigest(),
        source_classification=SOURCE_CLASSIFICATION,
        reference_discharge=2.0,
        fixed_log_sd=None if estimate_scale else 0.0,
        prior=_prior(estimate_scale=estimate_scale),
    )


def _csv(data: bytes) -> ClassifiedLognormalPosterior:
    return bind_exact_lognormal_csv(
        data,
        source="synthetic exact Year,AMS CSV",
        discharge_unit="m3/s",
        year_definition="calendar year",
        source_classification=SOURCE_CLASSIFICATION,
        reference_discharge=1.0,
        fixed_log_sd=0.0,
        prior=_prior(),
    )


def test_density_prior_and_explicit_censor_terms_against_independent_formulas() -> None:
    # Q=2*exp(Y), Y~N(mu,sigma), prior is in (mu, log(sigma)).
    rows = (
        _row(2000, ObservationKind.EXACT, value=2 * exp(0.7)),
        _row(2001, ObservationKind.LEFT_CENSORED, upper=2 * exp(0.0)),
        _row(2002, ObservationKind.RIGHT_CENSORED, lower=2 * exp(1.0)),
        _row(2003, ObservationKind.INTERVAL_CENSORED, lower=2 * exp(-0.5), upper=2 * exp(0.5)),
        _row(2004, ObservationKind.MISSING),
        _row(2005, ObservationKind.EXCLUDED, value=0.0),
    )
    model = _model(rows, estimate_scale=True)
    theta = (0.4, log(1.2))
    mu, eta = theta
    normal = NormalDist(mu=mu, sigma=1.2)
    expected_likelihood = (
        log(normal.pdf(0.7))
        - log(2 * exp(0.7))
        + log(normal.cdf(0.0))
        + log(1 - normal.cdf(1.0))
        + log(normal.cdf(0.5) - normal.cdf(-0.5))
    )
    # Independent inverse/determinant of [[2,.5],[.5,1]], det=1.75.
    expected_prior = -log(2 * pi) - 0.5 * log(1.75) - (mu**2 - mu * eta + 2 * eta**2) / 3.5
    target = model.target()
    assert target.coordinates == (MEAN_COORDINATE, LOG_SD_COORDINATE)
    assert isclose(target.log_likelihood(theta), expected_likelihood, rel_tol=0, abs_tol=1e-12)
    assert isclose(target.log_prior(theta), expected_prior, rel_tol=0, abs_tol=1e-12)
    assert isclose(model.kernel(theta).cdf(2 * exp(0.5)), normal.cdf(0.5), rel_tol=0, abs_tol=1e-14)
    assert model.series.annual == rows
    assert model.source_sha256 in target.source


def test_reference_rescaling_preserves_posterior_shape() -> None:
    original = _model(
        (
            _row(2000, ObservationKind.EXACT, value=3.0),
            _row(2001, ObservationKind.LEFT_CENSORED, upper=5.0),
        ),
        estimate_scale=True,
    )
    rescaled = replace(
        original,
        series=replace(
            original.series,
            discharge_unit="L/s",
            annual=(
                _row(2000, ObservationKind.EXACT, value=3000.0),
                _row(2001, ObservationKind.LEFT_CENSORED, upper=5000.0),
            ),
        ),
        reference_discharge=2000.0,
    )
    for theta in ((0.2, -0.3), (0.8, 0.4)):
        assert isclose(
            rescaled.log_likelihood(theta) - original.log_likelihood(theta), -log(1000), rel_tol=0, abs_tol=1e-12
        )
        assert isclose(rescaled.kernel(theta).ppf(0.9), 1000 * original.kernel(theta).ppf(0.9), rel_tol=1e-12)


def test_actual_csv_to_curvature_to_sampler_to_probability_outputs() -> None:
    # Three exact log flows 1,2,3, known SD 1, prior N(0,1):
    # posterior mean = 6/4, variance = 1/4; predictive Y ~ N(1.5, 1.25).
    data = ("Year,AMS\n" + "".join(f"{2000 + i},{exp(i):.17g}\n" for i in (1, 2, 3))).encode()
    model = _csv(data)
    result = execute_lognormal_posterior(
        model,
        starting_point=((MEAN_COORDINATE, 0.0),),
        curvature=CurvatureControls(
            scales=(1.0,),
            initial_step=0.01,
            max_refinements=8,
            max_iterations=30,
            gradient_tolerance=1e-5,
            mode_tolerance=1e-5,
            curvature_tolerance=1e-4,
            max_condition=1e5,
        ),
        sampling=ExplicitSamplingControls(
            draw_count=800, random_seed=51, minimum_effective_size=790.0, maximum_single_weight=0.002
        ),
    )
    run = result.sampling
    assert isclose(result.construction.proposal.means[0], 1.5, rel_tol=0, abs_tol=1e-6)
    assert isclose(result.construction.proposal.covariance[0][0], 0.25, rel_tol=0, abs_tol=1e-7)
    assert isclose(run.moments.parameters[0].mean, 1.5, rel_tol=0, abs_tol=5 * sqrt(0.25 / 800))
    assert isclose(run.moments.covariance[0][0], 0.25, rel_tol=0, abs_tol=5 * 0.25 * sqrt(2 / 799))
    assert run.moments.effective_sample_size > 799
    assert run.particles.provenance.input_sha256 == sha256(data).hexdigest()
    assert run.particles.provenance.evidence_status == EvidenceStatus.OBSERVED_ARTIFACT
    assert model.coordinates == (MEAN_COORDINATE,)
    assert model.fixed_log_sd == 0.0
    assert tuple(row.year for row in model.series.annual) == (2001, 2002, 2003)
    assert run.approximate_mean_mcse[0] > 0
    # Bounded probability integrand in [0,1], exact posterior draws to numerical
    # precision here. Five worst-case standard errors are <= 5/(2 sqrt(n)).
    output = probability_outputs(run.particles, exp(2.0), model.kernel, interval_mass=0.9)
    expected = 1 - NormalDist(mu=1.5, sigma=sqrt(1.25)).cdf(2.0)
    assert isclose(output.expected_aep, expected, rel_tol=0, abs_tol=5 / (2 * sqrt(800)))
    assert isclose(output.expected_non_exceedance + output.expected_aep, 1.0, abs_tol=1e-12)


@pytest.mark.parametrize("data", [b"Year,AMS\n2001,0\n", b"Year,AMS\n2001,\n", b"Year,AMS\n2001,1e-500\n"])
def test_exact_csv_rejects_unadmitted_records(data: bytes) -> None:
    with pytest.raises(PosteriorBindingError):
        _csv(data)


def test_reject_mismatched_prior_and_unverified_source() -> None:
    model = _model((_row(2000, ObservationKind.EXACT, value=1.0),))
    with pytest.raises(PosteriorBindingError, match="coordinates"):
        replace(model, prior=_prior(estimate_scale=True))
    with pytest.raises(PosteriorBindingError, match="source"):
        replace(model, source_classification="native")
    with pytest.raises(PosteriorBindingError, match="native"):
        replace(model, series=replace(model.series, source_file_version="3.10"))
    with pytest.raises(PosteriorBindingError, match="zero-threshold"):
        replace(model, series=replace(model.series, zero_threshold=0.0))
    with pytest.raises(PosteriorBindingError, match="parameter vector"):
        model.kernel((0.0, 0.0))


def test_duplicate_exposure_and_unsupported_observations_fail() -> None:
    exact = _row(2000, ObservationKind.EXACT, value=2.0)
    with pytest.raises(ValueError, match="overlapping"):
        _model((exact, exact))
    with pytest.raises(PosteriorBindingError, match="systematic"):
        _model((replace(exact, exposure=ExposureKind.HISTORICAL),))
    with pytest.raises(PosteriorBindingError, match="native flags"):
        _model((replace(exact, source_flag="0"),))
    with pytest.raises(PosteriorBindingError, match="strictly positive"):
        _model((replace(exact, value=0.0),))
    with pytest.raises(PosteriorBindingError, match="censor bounds"):
        _model((_row(2001, ObservationKind.LEFT_CENSORED, upper=0.0),))
    with pytest.raises(PosteriorBindingError, match="at least one"):
        _model((_row(2001, ObservationKind.MISSING),))
