"""Analytic, endpoint, and coordinate checks independent of the kernel implementation."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol, SupportsFloat, TypeAlias, cast

import pytest
from scipy.special import gammainc, gammaincc, ndtr, ndtri
from scipy.stats import norm

from ffa_methods.distributions import (
    GEV,
    GeneralizedPareto,
    Gumbel,
    LogCoordinates,
    Lognormal,
    LogPearsonIII,
    gev_scipy_c_to_xi,
    gev_xi_to_scipy_c,
)


class _ApproxCallable(Protocol):
    """Narrow typed boundary around the third-party pytest approximate matcher."""

    def __call__(self, expected: SupportsFloat | tuple[SupportsFloat, ...], **tolerances: float | None) -> object: ...


# pytest 9 leaves approx parameter annotations partially unknown; ignore only this typed cast.
approx_typed = cast(_ApproxCallable, pytest.approx)  # pyright: ignore[reportUnknownMemberType]


Kernel: TypeAlias = Lognormal | LogPearsonIII | Gumbel | GEV | GeneralizedPareto


def _models() -> list[Kernel]:
    return [
        Lognormal(LogCoordinates(2.0, -0.2, reference_discharge=10.0)),
        LogPearsonIII(LogCoordinates(2.0, -0.2, reference_discharge=10.0), 0.8),
        LogPearsonIII(LogCoordinates(2.0, -0.2, reference_discharge=10.0), -0.8),
        Gumbel(3.0, math.log(2.0)),
        GEV(3.0, math.log(2.0), 0.3),
        GEV(3.0, math.log(2.0), -0.3),
        GeneralizedPareto(2.0, math.log(3.0), 0.3),
        GeneralizedPareto(2.0, math.log(3.0), -0.3),
    ]


@pytest.mark.parametrize("model", _models())
def test_probability_and_survival_are_consistent(model: Kernel) -> None:
    for p in (1e-10, 0.0001, 0.1, 0.5, 0.9, 1 - 1e-9):
        q = model.ppf(p)
        assert model.cdf(q) == approx_typed(p, rel=2e-9, abs=2e-12)
        assert model.sf(q) == approx_typed(1 - p, rel=2e-9, abs=2e-12)
        if 1e-6 < p < 1 - 1e-6:
            assert model.isf(1 - p) == approx_typed(q, rel=5e-9, abs=5e-10)
        assert model.logcdf(q) == approx_typed(math.log(p), rel=2e-8, abs=5e-10)


@pytest.mark.parametrize("model", _models())
def test_endpoints_remain_valid_predictions(model: Kernel) -> None:
    lower, upper = model.support()
    for q in (-math.inf, math.inf):
        if q == -math.inf:
            assert model.cdf(q) == 0.0
            assert model.sf(q) == 1.0
            assert model.logpdf(q) == -math.inf
        else:
            assert model.cdf(q) == 1.0
            assert model.sf(q) == 0.0
            assert model.logpdf(q) == -math.inf
    if math.isfinite(lower):
        q = lower - 1.0
        assert model.cdf(q) == 0.0
        assert model.logcdf(q) == -math.inf
        assert model.logsf(q) == 0.0
        assert model.logpdf(q) == -math.inf
    if math.isfinite(upper):
        q = upper + 1.0
        assert model.cdf(q) == 1.0
        assert model.logcdf(q) == 0.0
        assert model.logsf(q) == -math.inf
        assert model.logpdf(q) == -math.inf


def test_lognormal_normal_identity_and_discharge_jacobian() -> None:
    mu, sd, ref, q = 1.5, 0.8, 100.0, 290.0
    model = Lognormal(LogCoordinates(mu, math.log(sd), reference_discharge=ref))
    z = (math.log(q / ref) - mu) / sd
    expected_pdf = math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * sd * q)
    assert math.exp(model.logpdf(q)) == approx_typed(expected_pdf)
    assert model.cdf(q) == approx_typed(ndtr(z))
    for p in (1e-12, 0.5, 1 - 1e-10):
        assert model.ppf(p) == approx_typed(ref * math.exp(mu + sd * ndtri(p)))
    assert model.cdf(0.0) == 0.0
    assert model.logpdf(0.0) == -math.inf
    assert model.logcdf(-1.0) == -math.inf
    assert model.logsf(-1.0) == 0.0


@pytest.mark.parametrize("skew", [0.5, 1.0, -0.5, -1.0])
def test_lp3_matches_independent_gamma_transform(skew: float) -> None:
    mu, sd, ref = 2.0, 1.4, 20.0
    model = LogPearsonIII(LogCoordinates(mu, math.log(sd), reference_discharge=ref), skew)
    shape = 4.0 / (skew * skew)
    edge = mu - 2 * sd / skew
    gamma_scale = sd * abs(skew) / 2
    q = ref * math.exp(mu + 0.3 * sd)
    g = abs((math.log(q / ref) - edge) / gamma_scale)
    expected_cdf = gammainc(shape, g) if skew > 0 else gammaincc(shape, g)
    expected_logpdf = (shape - 1) * math.log(g) - g - math.lgamma(shape) - math.log(gamma_scale) - math.log(q)
    assert model.cdf(q) == approx_typed(expected_cdf, abs=3e-14)
    assert model.logpdf(q) == approx_typed(expected_logpdf, abs=2e-13)
    lower, upper = model.support()
    if skew > 0:
        assert lower == approx_typed(ref * math.exp(edge))
        assert upper == math.inf
    else:
        assert lower == 0.0
        assert upper == approx_typed(ref * math.exp(edge))


def test_lp3_zero_skew_is_log_normal_and_small_skew_limit() -> None:
    log = LogCoordinates(1.2, math.log(0.9), reference_discharge=100.0)
    ln = Lognormal(log)
    for skew in (0.0, 1e-5, -1e-5):
        lp3 = LogPearsonIII(log, skew)
        for q in (50.0, 100.0, 200.0, 400.0):
            assert lp3.cdf(q) == approx_typed(ln.cdf(q), abs=3e-6)
            assert lp3.logpdf(q) == approx_typed(ln.logpdf(q), abs=3e-5)


def test_gumbel_closed_form() -> None:
    mu, sd, q = 5.0, 2.0, 8.0
    model = Gumbel(mu, math.log(sd))
    z = (q - mu) / sd
    assert model.cdf(q) == approx_typed(math.exp(-math.exp(-z)), abs=1e-15)
    assert model.logpdf(q) == approx_typed(-z - math.exp(-z) - math.log(sd), abs=1e-15)
    for p in (0.01, 0.5, 0.99):
        assert model.ppf(p) == approx_typed(mu - sd * math.log(-math.log(p)))


@pytest.mark.parametrize("xi", [-0.4, 0.4])
def test_gev_closed_form_and_shape_sign(xi: float) -> None:
    mu, sd, q = 4.0, 2.0, 5.0
    model = GEV(mu, math.log(sd), xi)
    z = (q - mu) / sd
    t = (1 + xi * z) ** (-1 / xi)
    assert model.cdf(q) == approx_typed(math.exp(-t), abs=2e-15)
    assert model.logpdf(q) == approx_typed(-t - (1 / xi + 1) * math.log(1 + xi * z) - math.log(sd))
    for p in (0.01, 0.3, 0.97):
        expected = mu + sd * math.expm1(-xi * math.log(-math.log(p))) / xi
        assert model.ppf(p) == approx_typed(expected, rel=1e-13)
    assert gev_xi_to_scipy_c(xi) == -xi
    assert gev_scipy_c_to_xi(-xi) == xi
    if xi < 0:
        assert model.support()[1] == approx_typed(mu - sd / xi)
    else:
        assert model.support()[0] == approx_typed(mu - sd / xi)


def test_gev_gumbel_limit() -> None:
    baseline = Gumbel(4.0, math.log(1.5))
    for xi in (-1e-7, 0.0, 1e-7):
        gev = GEV(4.0, math.log(1.5), xi)
        for q in (2.0, 4.0, 8.0):
            assert gev.cdf(q) == approx_typed(baseline.cdf(q), abs=2e-7)
            assert gev.logpdf(q) == approx_typed(baseline.logpdf(q), abs=2e-6)


@pytest.mark.parametrize("xi", [-0.5, 0.0, 0.5])
def test_gp_closed_form_and_exponential_limit(xi: float) -> None:
    mu, sd, q = 3.0, 4.0, 6.0
    model = GeneralizedPareto(mu, math.log(sd), xi)
    z = (q - mu) / sd
    expected_sf = math.exp(-z) if xi == 0 else (1 + xi * z) ** (-1 / xi)
    assert model.sf(q) == approx_typed(expected_sf, abs=2e-15)
    expected_logpdf = -math.log(sd) - z if xi == 0 else -math.log(sd) - (1 + 1 / xi) * math.log(1 + xi * z)
    assert model.logpdf(q) == approx_typed(expected_logpdf, abs=2e-15)
    for s in (1e-6, 0.02, 0.5):
        expected_isf = mu - sd * math.log(s) if xi == 0 else mu + sd * math.expm1(-xi * math.log(s)) / xi
        assert model.isf(s) == approx_typed(expected_isf)
    assert model.support()[0] == mu
    if xi < 0:
        assert model.support()[1] == approx_typed(mu - sd / xi)


def test_logbase_reference_and_unit_conversion_is_reversible() -> None:
    source = LogCoordinates(2.0, math.log(0.7), base=10.0, reference_discharge=10.0)
    natural = source.with_base(math.e)
    assert natural.mean == approx_typed(2 * math.log(10.0))
    assert natural.log_sd == approx_typed(math.log(0.7 * math.log(10.0)))
    assert natural.with_base(10.0).mean == approx_typed(source.mean)
    assert natural.with_base(10.0).log_sd == approx_typed(source.log_sd)
    assert source.log_abs_det_natural_wrt_source() == approx_typed(math.log(math.log(10.0)))
    assert source.log_abs_det_sd_wrt_log_sd == source.log_sd

    shifted = source.with_reference(1000.0)
    assert shifted.mean == approx_typed(source.mean - 2.0)
    assert shifted.with_reference(10.0).mean == approx_typed(source.mean)
    original = LogPearsonIII(source, -0.6)
    changed_base = LogPearsonIII(natural, -0.6)
    changed_reference = LogPearsonIII(shifted, -0.6)
    for q in (80.0, 200.0, 700.0):
        assert changed_base.cdf(q) == approx_typed(original.cdf(q), abs=3e-14)
        assert changed_reference.cdf(q) == approx_typed(original.cdf(q), abs=3e-14)
        assert changed_base.logpdf(q) == approx_typed(original.logpdf(q), abs=3e-14)
        assert changed_reference.logpdf(q) == approx_typed(original.logpdf(q), abs=3e-14)

    factor = 1000.0
    metric = Lognormal(source)
    rescaled = Lognormal(source.in_rescaled_discharge_units(factor))
    q = 200.0
    assert rescaled.cdf(factor * q) == approx_typed(metric.cdf(q))
    assert rescaled.logpdf(factor * q) == approx_typed(metric.logpdf(q) - math.log(factor))
    assert rescaled.ppf(0.95) == approx_typed(factor * metric.ppf(0.95))


@pytest.mark.parametrize("model", _models())
def test_extreme_survival_is_computed_without_one_minus_cdf(model: Kernel) -> None:
    s = 1e-25
    q = model.isf(s)
    assert math.isfinite(q)
    assert model.sf(q) == approx_typed(s, rel=1e-7)
    assert model.logsf(q) == approx_typed(math.log(s), abs=1e-7)


@pytest.mark.parametrize(
    "create",
    [
        lambda: LogCoordinates(0.0, math.nan),
        lambda: LogCoordinates(0.0, 0.0, base=1.0),
        lambda: LogCoordinates(0.0, 0.0, reference_discharge=-1),
        lambda: LogCoordinates(0.0, 1000.0),
        lambda: LogPearsonIII(LogCoordinates(0.0, 0.0), math.nan),
        lambda: Gumbel(0.0, -1000.0),
        lambda: GEV(math.inf, 0.0, 0.0),
        lambda: GEV(0.0, 0.0, math.nan),
        lambda: GeneralizedPareto(0.0, math.inf, 0.0),
        lambda: GeneralizedPareto(0.0, 0.0, math.inf),
    ],
)
def test_invalid_parameters_are_rejected(create: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match=r"must|unrepresentable"):
        create()


@pytest.mark.parametrize("model", _models())
def test_invalid_probabilities_and_nan_predictions_are_rejected(model: Kernel) -> None:
    for p in (-0.01, 1.01, math.nan, math.inf):
        with pytest.raises(ValueError, match="probability must"):
            model.ppf(p)
        with pytest.raises(ValueError, match="probability must"):
            model.isf(p)
    for method in (model.logpdf, model.cdf, model.sf, model.logcdf, model.logsf):
        with pytest.raises(ValueError, match="must not be NaN"):
            method(math.nan)


def test_gp_negative_shape_is_uniform_at_minus_one() -> None:
    model = GeneralizedPareto(2.0, math.log(4.0), -1.0)
    assert model.support() == approx_typed((2.0, 6.0))
    for q in (2.0, 3.0, 4.0, 5.0):
        assert model.cdf(q) == approx_typed((q - 2.0) / 4.0)
        assert model.logpdf(q) == approx_typed(-math.log(4.0))


def test_lp3_gamma_log_tails_remain_finite_after_probability_underflow() -> None:
    positive = LogPearsonIII(LogCoordinates(0.0, math.log(0.1)), 2.0)
    negative = LogPearsonIII(LogCoordinates(0.0, math.log(0.1)), -2.0)
    assert positive.sf(math.exp(100.0)) == 0.0
    assert negative.cdf(math.exp(-100.0)) == 0.0
    # Shape one gamma tail is exactly exp(-x), with x=1001 in both cases.
    assert math.isclose(positive.logsf(math.exp(100.0)), -1001.0, abs_tol=2e-12)
    assert math.isclose(negative.logcdf(math.exp(-100.0)), -1001.0, abs_tol=2e-12)


def test_lp3_underflowed_gamma_lower_tail_matches_poisson_identity() -> None:
    # For integer gamma shape a, P(a,x) equals Pr[Poisson(x) >= a].
    shape, x = 400, 10.0
    term, series = 1.0, 1.0
    for j in range(1, 60):
        term *= x / (shape + j)
        series += term
    expected = shape * math.log(x) - x - math.lgamma(shape + 1) + math.log(series)

    positive = LogPearsonIII(LogCoordinates(0.0, 0.0), 0.1)
    negative = LogPearsonIII(LogCoordinates(0.0, 0.0), -0.1)
    assert positive.cdf(math.exp(-19.5)) == 0.0
    assert negative.sf(math.exp(19.5)) == 0.0
    assert math.isclose(positive.logcdf(math.exp(-19.5)), expected, abs_tol=1e-11)
    assert math.isclose(negative.logsf(math.exp(19.5)), expected, abs_tol=1e-11)


def test_lp3_large_scale_does_not_overflow_intermediate_gamma_coordinates() -> None:
    log_sd = 709.5
    positive = LogPearsonIII(LogCoordinates(0.0, log_sd), 2.0)
    negative = LogPearsonIII(LogCoordinates(0.0, log_sd), -2.0)
    assert math.isclose(positive.cdf(1.0), -math.expm1(-1.0), abs_tol=1e-14)
    assert math.isclose(negative.cdf(1.0), math.exp(-1.0), abs_tol=1e-14)
    assert math.isclose(positive.logpdf(1.0), -1.0 - log_sd, abs_tol=1e-12)
    assert math.isclose(negative.logpdf(1.0), -1.0 - log_sd, abs_tol=1e-12)


def test_lp3_unrepresentable_derived_gamma_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="gamma shape"):
        LogPearsonIII(LogCoordinates(0.0, 0.0), 1e200)
    with pytest.raises(ValueError, match="support endpoint"):
        LogPearsonIII(LogCoordinates(0.0, 709.5), 1.0)


@pytest.mark.parametrize(
    ("skew", "expected_lower", "expected_upper"),
    [
        (0.01, -930.3503719533658756, -711.7462970045258524),
        (0.001, -815.4376996432623760, -794.0992697843885181),
        (0.0003, -807.8229077423288419, -801.4227769801055585),
        (0.0001, -805.6767099980118645, -803.5433740384410887),
        (0.00002, -804.8218391182044722, -804.3951729093332381),
        (0.000016, -804.7791494514121107, -804.4378164961116057),
    ],
)
def test_lp3_small_skew_rare_tails_match_independent_quadrature(
    skew: float, expected_lower: float, expected_upper: float
) -> None:
    """80-digit direct endpoint quadrature, not SciPy or a second gamma wrapper.

    See docs/distribution-kernels.md for the transformed integral, environment,
    published mathematical basis, and source precision. The negative skew
    equalities follow by reflecting the Pearson III log variable.
    """
    positive = LogPearsonIII(LogCoordinates(0.0, 0.0), skew)
    negative = LogPearsonIII(LogCoordinates(0.0, 0.0), -skew)
    low, high = math.exp(-40.0), math.exp(40.0)
    # The lower tails underflow. For skew=0.01 the upper tail is
    # subnormal, not zero, so it must not be asserted to underflow.
    assert positive.cdf(low) == 0.0
    assert negative.sf(high) == 0.0
    for actual, expected in (
        (positive.logcdf(low), expected_lower),
        (negative.logsf(high), expected_lower),
        (positive.logsf(high), expected_upper),
        (negative.logcdf(low), expected_upper),
    ):
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-9)


@pytest.mark.parametrize(
    ("z", "expected"),
    [
        (-37.0, -689.8759739832407111),
        (-39.0, -766.0732534457897997),
    ],
)
def test_lp3_scipy_to_log_fallback_transition_matches_quadrature(z: float, expected: float) -> None:
    # At the first point SciPy returns a finite but inaccurate log CDF for
    # this shape; the second point underflows entirely. Both need correction.
    model = LogPearsonIII(LogCoordinates(0.0, 0.0), 0.0001)
    assert math.isclose(model.logcdf(math.exp(z)), expected, rel_tol=0.0, abs_tol=2e-9)


@pytest.mark.parametrize("skew", [-1.599e-5, -1.601e-5, 0.0, 1.599e-5, 1.601e-5])
def test_lp3_normal_to_gamma_transition_for_regular_quantiles(skew: float) -> None:
    # The documented normal approximation is used only below |skew|=1.6e-5.
    # Ordinary quantiles remain close, but the approximation is not a
    # uniform relative-error guarantee for probabilities at deep endpoints.
    model = LogPearsonIII(LogCoordinates(0.0, 0.0), skew)
    baseline = Lognormal(LogCoordinates(0.0, 0.0))
    for z in (-3.0, 0.0, 3.0):
        q = math.exp(z)
        assert math.isclose(model.logcdf(q), baseline.logcdf(q), rel_tol=0.0, abs_tol=2e-4)
        assert math.isclose(model.logsf(q), baseline.logsf(q), rel_tol=0.0, abs_tol=2e-4)


def test_lp3_external_uniform_law_quantile_and_interior_density() -> None:
    # Authoritative 0BSD Numerics v2.2.0 candidate:
    # reference/public-candidates/numerics-vectors/lp3-uniform-law.json
    # at fdf2893a53777b07fc8a583387cb7449cbdf77c3 (PR #22).
    # Source log-SD coordinate is ln(1)=0, not 1. Original tolerances retained.
    model = LogPearsonIII(LogCoordinates(0.0, 0.0), -2.0)
    assert math.isclose(model.logpdf(1.0), -1.0, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(model.ppf(1e-20), math.e * 1e-20, rel_tol=0.0, abs_tol=2e-31)
    for nonexceedance in (0.25, 0.5, 0.9):
        q = math.e * nonexceedance
        assert math.isclose(model.cdf(q), nonexceedance, rel_tol=0.0, abs_tol=2e-14)
        assert math.isclose(model.logpdf(q), -1.0, rel_tol=0.0, abs_tol=2e-14)


@pytest.mark.parametrize(
    ("skew", "z", "expected"),
    [
        (0.001, -4.5, -12.607538446692049825),
        (0.001, -5.0, -15.085783561700149176),
        (0.001, -10.0, -53.398531216857613564),
        (0.0001, -4.5, -12.593929294762035682),
        (0.0001, -5.0, -15.067073386389282083),
        (0.0001, -10.0, -53.247953257024787171),
        (0.00002, -4.5, -12.592721606492261616),
        (0.00002, -5.0, -15.065413329936143478),
        (0.00002, -10.0, -53.234617771306689341),
    ],
)
def test_lp3_large_shape_finite_lower_tails_match_direct_quadrature(skew: float, z: float, expected: float) -> None:
    # SciPy gamma logcdf may be finite yet wrong by >2 log units at z=-5.
    positive = LogPearsonIII(LogCoordinates(0.0, 0.0), skew)
    negative = LogPearsonIII(LogCoordinates(0.0, 0.0), -skew)
    assert math.isclose(positive.logcdf(math.exp(z)), expected, rel_tol=0.0, abs_tol=2e-9)
    assert math.isclose(negative.logsf(math.exp(-z)), expected, rel_tol=0.0, abs_tol=2e-9)
    # Probability-space methods must agree with the corrected log tails.
    if expected > -700.0:
        assert math.isclose(positive.cdf(math.exp(z)), math.exp(expected), rel_tol=2e-9)
        assert math.isclose(negative.sf(math.exp(-z)), math.exp(expected), rel_tol=2e-9)


@pytest.mark.parametrize(
    ("skew", "z", "expected_logpdf"),
    [
        (0.001, -5.0, -8.437307897941330832),
        (0.0001, -10.0, -40.935111327539032557),
        (0.00002, 0.0, -0.918938533213006075),
        (0.00002, -5.0, -8.419305214255256100),
    ],
)
def test_lp3_large_shape_log_density_matches_independent_gamma_equation(
    skew: float, z: float, expected_logpdf: float
) -> None:
    # 80-digit log-Gamma density: (a-1)*log(x)-x-loggamma(a)-log(scale)-z.
    # No SciPy gamma.logpdf or FFA density was used to derive these targets.
    model = LogPearsonIII(LogCoordinates(0.0, 0.0), skew)
    assert math.isclose(model.logpdf(math.exp(z)), expected_logpdf, rel_tol=0.0, abs_tol=2e-9)


def test_scalar_lognormal_repeated_evaluation_preserves_scipy_density_and_tails() -> None:
    log_flow = LogCoordinates(0.3, -0.2, reference_discharge=2.5)
    kernel = Lognormal(log_flow)
    frozen = norm(loc=log_flow.mean, scale=math.exp(log_flow.log_sd))
    for flow in (1e-200, 0.125, 1.0, 2.5, 17.0, 1e200):
        y = math.log(flow) - math.log(log_flow.reference_discharge)
        assert kernel.cdf(flow) == float(frozen.cdf(y))
        assert kernel.sf(flow) == float(frozen.sf(y))
        assert kernel.logcdf(flow) == float(frozen.logcdf(y))
        assert kernel.logsf(flow) == float(frozen.logsf(y))
        assert kernel.logpdf(flow) == float(frozen.logpdf(y)) - math.log(flow)
