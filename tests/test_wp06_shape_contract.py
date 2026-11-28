"""Prior normalization and the boundary between mathematical existence and numerical admission."""

import json
from dataclasses import replace
from decimal import Context, Decimal, localcontext
from hashlib import sha256
from math import exp, fsum, inf, isclose, log, nextafter, pi, sqrt
from pathlib import Path
from typing import cast

import pytest
from scipy.integrate import quad

from ffa_methods.bayesian_outputs import predictive_quantile
from ffa_methods.contracts import ResultMeaning
from ffa_methods.deterministic_workflow import calculate_from_bytes
from ffa_methods.family_posterior import PosteriorFamily
from ffa_methods.lognormal_posterior import PosteriorBindingError
from ffa_methods.reporting import report_document
from ffa_methods.shape_posterior import GEV_LATENT, execute_shape_posterior
from tests.test_wp06_shape_reference import CASES, SAMPLES, synthetic_shape_model


def test_correlated_physical_prior_and_jacobian() -> None:
    model = synthetic_shape_model(*CASES[1])
    prior = model.prior
    a, b = prior.bounds
    u = (0.4, -0.7, 0.9)
    z = 0.2 * u[0] - 0.15 * u[1] + 0.6 * u[2]
    t = 1 / (1 + exp(-z))
    x = (0.25 * u[0], 0.08 * u[0] + 0.14 * u[1], a + (b - a) * t)
    expected = -0.5 * sum(v * v for v in u) - 1.5 * log(2 * pi) - log(0.25 * 0.14 * 0.6 * (b - a) * t * (1 - t))
    assert isclose(prior.logpdf(x), expected, abs_tol=2e-14, rel_tol=0)
    for observed, ref in zip(prior.innovations(x), u, strict=True):
        assert isclose(observed, ref, abs_tol=1e-14, rel_tol=0)
    assert prior.logpdf((0, 0, a)) == -inf
    assert prior.logpdf((0, 0, b)) == -inf
    # Integrate conditional skew at fixed first two coordinates; divide by
    # their independent bivariate marginal density at this same location.
    marginal = exp(-0.5 * (u[0] ** 2 + u[1] ** 2)) / (2 * pi * 0.25 * 0.14)
    mass, error = quad(lambda s: exp(prior.logpdf((x[0], x[1], s))) / marginal, a, b, epsabs=1e-10, epsrel=1e-10)
    assert abs(mass - 1) < 1e-10
    assert error < 1e-10


def test_scaled_proposal_weights_and_original_coordinate_custody() -> None:
    model = synthetic_shape_model(*CASES[0])
    sampling = replace(SAMPLES, draw_count=72, minimum_effective_size=2, maximum_single_weight=0.7)
    gamma = 1.35
    result = execute_shape_posterior(model, gamma=gamma, sampling=sampling)
    assert result == execute_shape_posterior(model, gamma=gamma, sampling=sampling)
    particles = result.particles
    assert particles.coordinates == model.coordinates
    for theta, lp, lq in zip(particles.draws, particles.log_prior, particles.log_proposal, strict=True):
        u = model.prior.innovations(theta)
        ref = 3 * log(gamma) - 0.5 * sum(v * v for v in u) * (1 - 1 / gamma**2)
        assert isclose(lp - lq, ref, abs_tol=2e-14, rel_tol=0)
    with pytest.raises(PosteriorBindingError, match="second-weight-moment"):
        execute_shape_posterior(model, gamma=0.7, sampling=sampling)


def test_support_rejections_and_tied_years() -> None:
    gev = synthetic_shape_model(*CASES[0])
    lp3 = synthetic_shape_model(*CASES[1])
    for bounds in ((-1.0, 0.0), (-2.0, 2.0)):
        with pytest.raises(PosteriorBindingError, match="above -1"):
            replace(gev, prior=replace(gev.prior, bounds=bounds))
    for bounds in ((-0.2, 0.3), (0.0, 1.0), (-2.0, -0.5), (0.2, 2.0), (1e-6, 1.25)):
        with pytest.raises(PosteriorBindingError, match="near-normal"):
            replace(lp3, prior=replace(lp3.prior, bounds=bounds))
    for factor in (
        ((1.0, 0.0, 0.01), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    ):
        with pytest.raises(PosteriorBindingError, match="lower triangular"):
            replace(gev.prior, lower_factor=factor)
    with pytest.raises(PosteriorBindingError, match="representable open"):
        replace(gev.prior, bounds=(0.5, nextafter(0.5, inf)))
    with pytest.raises(PosteriorBindingError, match="unresolved"):
        gev.prior.physical((0.0, 0.0, 3000.0))
    first = gev.series.annual[0]
    tied = replace(gev, series=replace(gev.series, annual=(first, replace(first, year=2010))))
    assert first.value is not None
    q = first.value
    # Upper endpoint mu+scale/abs(xi) coincides with a tied discharge at xi=-.25.
    mu = q - 4
    assert tied.log_likelihood((mu, 0.0, -0.30)) == -inf
    for signed_shape in (-0.24, 0.12):
        assert tied.log_likelihood((mu, 0.0, signed_shape)) == 2 * tied.kernel((mu, 0.0, signed_shape)).logpdf(q)
    interval = gev.series.annual[3]
    assert interval.upper_bound is not None
    assert interval.lower_bound is not None
    just_interval = replace(gev, series=replace(gev.series, annual=(interval,)))
    # The upper endpoint is inside the interval; its mass equals the lower survival.
    mu = 3.0 - 4
    kernel = gev.kernel((mu, 0.0, -0.25))
    assert kernel.sf(interval.upper_bound) == 0
    assert isclose(just_interval.log_likelihood((mu, 0.0, -0.25)), kernel.logsf(interval.lower_bound), abs_tol=1e-12)


def test_shape_zero_limits_and_lp3_moments() -> None:
    gev = synthetic_shape_model(*CASES[0])
    cdf = exp(-exp(-0.8))
    for xi in (-1e-9, 0.0, 1e-9):
        assert abs(gev.kernel((3.0, 0.0, xi)).cdf(3.8) - cdf) < 2e-10
    for family, bounds in CASES[1:]:
        model = synthetic_shape_model(family, bounds)
        assert model.raw_mean_undefined_reason(0.99)
        assert model.raw_mean_undefined_reason(0.10) is None
        # First skew correction at the standardized mean is g*phi(0)/6.
        small = 1.600001e-5 if bounds[0] > 0 else -1.600001e-5
        small_bounds = (1.6e-5, 0.1) if small > 0 else (-0.1, -1.6e-5)
        near = replace(model, prior=replace(model.prior, bounds=small_bounds))
        assert abs(near.kernel((0.0, 0.0, small)).cdf(2.0) - (0.5 + small / (6 * sqrt(2 * pi)))) < 2e-10


@pytest.mark.parametrize("family", [PosteriorFamily.GEV, PosteriorFamily.LP3])
def test_installed_shape_report_contract(family: PosteriorFamily) -> None:
    root = Path(__file__).resolve().parents[1] / "docs" / "examples"
    settings = json.loads((root / "bayesian-shape-settings.json").read_text(encoding="utf8"))
    raw = (root / "bayesian-shape-ams.csv").read_bytes()
    settings["sampling"].update(draw_count=72, minimum_effective_size=2, maximum_single_weight=0.7)
    settings["outputs"]["non_exceedance_probabilities"] = [0.99]
    if family == PosteriorFamily.GEV:
        settings.update(estimator="gev_bayesian", reference_discharge=None)
        settings["prior"].update(means=[3, 0, 0], coordinates=list(GEV_LATENT), shape_bounds=[-0.35, 0.35])
    encoded = json.dumps(settings).encode()
    result, meta = calculate_from_bytes(raw, encoded)
    doc = report_document(result, settings=meta)
    assert result.provenance.settings_sha256 == sha256(encoded).hexdigest()
    assert result.provenance.input_sha256 == sha256(raw).hexdigest()
    rows = cast(list[dict[str, object]], doc["results"])
    assert all(row["precision_state"] == "not_assessed" for row in rows)
    meanings = {row["meaning"] for row in rows}
    assert ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS in meanings
    assert ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE in meanings
    assert (ResultMeaning.POSTERIOR_MEAN_QUANTILE in meanings) == (family == PosteriorFamily.GEV)
    assert (ResultMeaning.LOG10_QUANTILE_POSTERIOR_MOMENT in meanings) == (family == PosteriorFamily.LP3)
    if family == PosteriorFamily.LP3:
        assert any("raw LP3 design-quantile mean diverges" in warning for warning in result.warnings)
    settings["precision"] = {"mode": "audit"}
    with pytest.raises(ValueError, match="not_assessed"):
        calculate_from_bytes(raw, json.dumps(settings).encode())


def test_nearly_degenerate_logit_density() -> None:
    model = synthetic_shape_model(*CASES[0])
    prior = replace(
        model.prior,
        means=(0.0, 0.0, log(3.0)),
        lower_factor=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1e-16)),
        bounds=(0.0, 1.0),
    )
    with localcontext(Context(prec=110)):
        expected = float((Decimal(3).ln() - Decimal(log(3.0))) / Decimal(prior.lower_factor[2][2]))
    assert prior.innovations((0.0, 0.0, 0.75))[2] == expected
    assert isclose(prior.logpdf((0.0, 0.0, 0.75)), 35.34708015423365, abs_tol=1e-13, rel_tol=0)


def test_unrepresentable_likelihood_and_sample_covariance() -> None:
    model = synthetic_shape_model(*CASES[0])
    extreme = replace(model, prior=replace(model.prior, bounds=(0.0, 1.2e308)))
    kernel = extreme.kernel((0.0, 0.0, 1e308))
    with pytest.raises(PosteriorBindingError, match="shape times"):
        kernel.logcdf(2.0)
    lp3 = synthetic_shape_model(*CASES[1])
    factor = ((1e-200, 0.0, 0.0), *lp3.prior.lower_factor[1:])
    with pytest.raises(PosteriorBindingError, match="variance is unresolved"):
        execute_shape_posterior(
            replace(lp3, prior=replace(lp3.prior, lower_factor=factor)),
            gamma=1.0,
            sampling=replace(SAMPLES, draw_count=32, minimum_effective_size=1, maximum_single_weight=1),
        )


@pytest.mark.parametrize("token", [b"1e-1000", b"1e-" + b"9" * 32])
def test_retained_excluded_underflow(token: bytes) -> None:
    root = Path(__file__).resolve().parents[1] / "docs" / "examples"
    raw = (root / "bayesian-shape-ams.csv").read_bytes().replace(b"excluded,0,", b"excluded," + token + b",")
    with pytest.raises(ValueError, match="underflows"):
        calculate_from_bytes(raw, (root / "bayesian-shape-settings.json").read_bytes())


@pytest.mark.parametrize(("family", "bounds"), CASES[:2])
def test_rare_predictive_inverse_uses_direct_survival(family: PosteriorFamily, bounds: tuple[float, float]) -> None:
    model = synthetic_shape_model(family, bounds)
    run = execute_shape_posterior(
        model, gamma=1.0, sampling=replace(SAMPLES, draw_count=32, minimum_effective_size=1, maximum_single_weight=1)
    )
    p = nextafter(1.0, 0.0)
    q = predictive_quantile(run.particles, p, model.kernel, direct_upper_tail=True)

    def survival(x: float) -> float:
        return fsum(
            w * model.kernel(theta).sf(x) for theta, w in zip(run.particles.draws, run.particles.weights, strict=True)
        )

    assert survival(q) <= 1 - p
    assert survival(nextafter(q, -inf)) > 1 - p
