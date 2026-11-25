"""Independent population-law and adversarial tests for WP04 H=0 family fits."""

from decimal import Decimal, localcontext
from fractions import Fraction
from math import erf, exp, isclose, isfinite, log, pi, sqrt

import pytest
from scipy.stats import gamma, genpareto, gumbel_r, norm

from ffa_methods.distributions import GeneralizedPareto, Gumbel, LogCoordinates, Lognormal, LogPearsonIII
from ffa_methods.estimators import ordinary_lmoments
from ffa_methods.estimators.ordinary_lmoments import (
    fit_gp_lmoments,
    fit_gumbel_lmoments,
    fit_lognormal_lmoments,
    fit_lp3_lmoments,
)


def _near(actual: float, expected: float, *, absolute: float) -> None:
    assert isclose(actual, expected, rel_tol=0.0, abs_tol=absolute)


def _grid(count: int) -> list[float]:
    """Deterministic midpoint quadrature nodes, not iid simulation."""
    return [(i + 0.5) / count for i in range(count)]


def test_gumbel_exact_arithmetic_series_moments() -> None:
    fit = fit_gumbel_lmoments([1.0, 2.0, 3.0, 4.0, 5.0])
    assert isinstance(fit.distribution, Gumbel)
    assert fit.family == "gumbel"
    assert fit.moment_coordinates == "discharge"
    assert fit.sample_moments == (3.0, 1.0)
    _near(exp(fit.distribution.log_scale), 1.0 / log(2.0), absolute=1e-14)
    _near(fit.distribution.location, 3.0 - 0.5772156649015329 / log(2.0), absolute=1e-14)
    _near(fit.quantile(0.9), float(gumbel_r.ppf(0.9, loc=fit.distribution.location, scale=1 / log(2))), absolute=1e-13)


@pytest.mark.parametrize(("location", "scale"), [(12.0, 3.5), (-4.0, 1.0)])
def test_gumbel_independent_midpoint_quantile_grid(location: float, scale: float) -> None:
    samples = [location - scale * log(-log(probability)) for probability in _grid(4000)]
    fit = fit_gumbel_lmoments(samples)
    assert isinstance(fit.distribution, Gumbel)
    _near(fit.distribution.location, location, absolute=0.02)
    _near(exp(fit.distribution.log_scale), scale, absolute=0.02)
    _near(fit.quantile(0.99), float(gumbel_r.ppf(0.99, loc=location, scale=scale)), absolute=0.1)
    transformed = fit_gumbel_lmoments([1000.0 + 4 * value for value in reversed(samples)])
    _near(transformed.distribution.location, 1000.0 + 4 * fit.distribution.location, absolute=2e-11)


def test_lognormal_direct_l_moments_and_raw_vs_log_coordinate() -> None:
    reference, mu, sd = 10.0, 0.7, 0.6
    sample = [reference * exp(mu + sd * norm.ppf(f)) for f in _grid(5000)]
    fit = fit_lognormal_lmoments(sample, reference_discharge=reference)
    assert isinstance(fit.distribution, Lognormal)
    assert fit.family == "lognormal"
    assert fit.moment_coordinates == "discharge"
    # Independently: E(Q)=ref*exp(mu+sd^2/2), L2=E(Q)*erf(sd/2).
    population_l1 = reference * exp(mu + 0.5 * sd * sd)
    population_l2 = population_l1 * erf(sd / 2)
    _near(fit.sample_moments[0], population_l1, absolute=0.02)
    _near(fit.sample_moments[1], population_l2, absolute=0.02)
    _near(fit.distribution.log.natural_mean, mu, absolute=0.006)
    _near(exp(fit.distribution.log.natural_log_sd), sd, absolute=0.006)
    _near(fit.quantile(0.95), reference * exp(mu + sd * norm.ppf(0.95)), absolute=0.08)
    rescaled = fit_lognormal_lmoments([12.0 * x for x in sample], reference_discharge=120.0)
    _near(rescaled.distribution.log.natural_mean, fit.distribution.log.natural_mean, absolute=2e-13)
    _near(rescaled.quantile(0.95), fit.quantile(0.95) * 12.0, absolute=1e-9)


def test_lognormal_simple_sample_and_invalid_range() -> None:
    fit = fit_lognormal_lmoments([1.0, 2.0, 3.0])
    assert isinstance(fit.distribution, Lognormal)
    _near(fit.sample_moments[0], 2.0, absolute=1e-14)
    _near(fit.sample_moments[1], 2.0 / 3.0, absolute=1e-14)
    with pytest.raises(ValueError, match="strictly positive"):
        fit_lognormal_lmoments([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="reference_discharge"):
        fit_lognormal_lmoments([1.0, 2.0, 3.0], reference_discharge=0.0)


@pytest.mark.parametrize("skew", [-2.0, -1.0, 1.0, 2.0])
def test_lp3_inverts_independent_gamma_population_quantile_grid(skew: float) -> None:
    reference, mu, sd = 10.0, 0.8, 0.65
    shape = 4.0 / (skew * skew)
    signed_scale = sd / sqrt(shape) * (1.0 if skew > 0.0 else -1.0)
    logs = [mu + signed_scale * (float(gamma.ppf(f, a=shape)) - shape) for f in _grid(6000)]
    sample = [reference * exp(value) for value in logs]
    fit = fit_lp3_lmoments(sample, reference_discharge=reference)
    assert isinstance(fit.distribution, LogPearsonIII)
    assert fit.family == "lp3"
    assert fit.moment_coordinates == "ln(Q/reference_discharge)"
    _near(fit.distribution.log.natural_mean, mu, absolute=0.01)
    _near(exp(fit.distribution.log.natural_log_sd), sd, absolute=0.01)
    _near(fit.distribution.skew, skew, absolute=0.02)
    expected_median = reference * exp(mu + signed_scale * (float(gamma.ppf(0.5, a=shape)) - shape))
    _near(fit.quantile(0.5), expected_median, absolute=0.15)
    rescaled = fit_lp3_lmoments([12 * q for q in sample], reference_discharge=120.0)
    _near(rescaled.distribution.skew, fit.distribution.skew, absolute=1e-11)
    _near(rescaled.quantile(0.5), 12.0 * fit.quantile(0.5), absolute=1e-8)


def test_lp3_normal_limit_exactly_symmetric_log_observations() -> None:
    fit = fit_lp3_lmoments([0.25, 1.0, 4.0])
    assert isinstance(fit.distribution, LogPearsonIII)
    assert fit.distribution.skew == 0.0
    _near(fit.distribution.log.natural_mean, 0.0, absolute=1e-14)
    _near(exp(fit.distribution.log.natural_log_sd), sqrt(pi) * log(4.0) * 2.0 / 3.0, absolute=1e-13)
    _near(fit.quantile(0.5), 1.0, absolute=1e-13)


@pytest.mark.parametrize("factor", [1.0, 2.0, 3.0, 10.0])
@pytest.mark.parametrize("reference_factor", [1.0, 2.0])
def test_lp3_symmetric_log_law_is_discharge_unit_invariant(factor: float, reference_factor: float) -> None:
    """Exact geometric ratios must not create a spurious nonzero L-skew."""
    source = [0.25, 1.0, 4.0]
    baseline = fit_lp3_lmoments(source, reference_discharge=reference_factor)
    scaled = fit_lp3_lmoments(
        [factor * discharge for discharge in source],
        reference_discharge=factor * reference_factor,
    )
    assert isinstance(scaled.distribution, LogPearsonIII)
    assert baseline.distribution.skew == scaled.distribution.skew == 0.0
    assert baseline.sample_moments[2] == scaled.sample_moments[2] == 0.0
    _near(scaled.sample_moments[1], baseline.sample_moments[1], absolute=2e-15)
    _near(scaled.sample_moments[0], baseline.sample_moments[0], absolute=2e-15)
    _near(scaled.quantile(0.5), factor * baseline.quantile(0.5), absolute=5e-14)
    _near(scaled.quantile(0.9), factor * baseline.quantile(0.9), absolute=5e-13)


def test_lp3_geometric_overflow_ratios_preserve_zero_lskew_under_unit_change() -> None:
    """The largest/smallest ratios exceed binary64 even though Q is finite."""
    sample = [2.0**-1073, 2.0**-49, 2.0**975]
    original = fit_lp3_lmoments(sample)
    scaled = fit_lp3_lmoments([value / 2.0 for value in sample])
    assert original.distribution.skew == scaled.distribution.skew == 0.0
    assert original.sample_moments[2] == scaled.sample_moments[2] == 0.0
    assert ordinary_lmoments.stable_log_ratio(sample[0], sample[1]) == -ordinary_lmoments.stable_log_ratio(
        sample[1], sample[0]
    )
    _near(scaled.quantile(0.5), original.quantile(0.5) / 2.0, absolute=2e-28)


def test_lp3_false_zero_skew_is_rejected_without_ulp_range_floor() -> None:
    """An exact-binary64 non-geometric triple must not fit the Normal limit."""
    sample = [1.0, 1.1, 1.2100000000000002]
    with localcontext() as context:
        context.prec = 100
        low, mid, high = [Decimal(value) for value in sample]
        log_low, log_mid, log_high = low.ln(), mid.ln(), high.ln()
        skew = (log_low - 2 * log_mid + log_high) / (log_high - log_low)
        assert abs(skew) > Decimal("1e-18")
        assert low * high != mid * mid
    # The stabilized log1p path may resolve genuinely nonzero skew; it
    # must still reject this sample because its inferred gamma shape is
    # outside the declared numerical admission interval, not force a
    # zero-skew Normal fit.
    with pytest.raises(ValueError, match="L-skew"):
        fit_lp3_lmoments(sample)


@pytest.mark.parametrize(
    "sample",
    [
        [0.25, 1.0, 4.0],
        [1.0, 7.0, 49.0],
        [1.0, 10.0, 100.0],
        [2.0**-1073, 2.0**-49, 2.0**975],
    ],
)
def test_lp3_exact_three_point_geometric_normal_limit_is_admitted(sample: list[float]) -> None:
    """The exact-rational geometric identity remains supported at all scales."""
    low, middle, high = sorted(Fraction(value) for value in sample)
    assert low * high == middle * middle
    fitted = fit_lp3_lmoments(sample)
    assert fitted.distribution.skew == 0.0
    assert fitted.sample_moments[2] == 0.0


@pytest.mark.parametrize(
    "sample",
    [
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        [1.0, 2.0, 4.0, 8.0],
        [1.0, 2.0, 3.0, 6.0],
        [1.0, 1.0, 4.0, 4.0],
    ],
)
@pytest.mark.parametrize("unit_factor", [1.0, 2.0, 4.0])
def test_lp3_exact_higher_n_log_mirror_symmetry_is_normal(sample: list[float], unit_factor: float) -> None:
    """Exact rational pair-products independently prove log L3=0."""
    source = [Fraction(x) for x in sample]
    product = source[0] * source[-1]
    assert all(source[i] * source[-1 - i] == product for i in range(len(source) // 2))
    fitted = fit_lp3_lmoments([unit_factor * value for value in sample], reference_discharge=unit_factor)
    assert fitted.distribution.skew == 0.0
    assert fitted.sample_moments[2] == 0.0
    baseline = fit_lp3_lmoments(sample)
    _near(fitted.quantile(0.5), unit_factor * baseline.quantile(0.5), absolute=5e-13)
    _near(fitted.quantile(0.9), unit_factor * baseline.quantile(0.9), absolute=3e-12)


def test_lp3_nonmirrored_six_point_sample_never_forces_zero_skew() -> None:
    sample = [1.0, 2.0, 4.0, 8.0, 16.0, 33.0]
    ordered = [Fraction(value) for value in sample]
    assert ordered[0] * ordered[-1] != ordered[1] * ordered[-2]
    try:
        fit = fit_lp3_lmoments(sample)
    except ValueError as exc:
        if "L-skew" not in str(exc) and "fitted support" not in str(exc):
            raise
    else:
        assert fit.distribution.skew != 0.0


def test_lp3_four_point_false_zero_skew_is_not_admitted_as_normal() -> None:
    """The exact source floats do not satisfy the four-point L3 identity."""
    sample = [1.0, 1.1, 1.1, 1.2100000000000002]
    left = Fraction(sample[0]) * Fraction(sample[3])
    right = Fraction(sample[1]) * Fraction(sample[2])
    assert left != right
    with localcontext() as context:
        context.prec = 100
        logs = [Decimal(value).ln() for value in sample]
        source_l3 = (logs[0] - logs[1] - logs[2] + logs[3]) / 4
        assert abs(source_l3) > Decimal("1e-19")
    with pytest.raises(ValueError, match="L-skew"):
        fit_lp3_lmoments(sample)


def test_lp3_five_point_genuine_zero_l3_without_mirror_symmetry() -> None:
    """A nonsymmetric source record may still have exactly zero log L3."""
    sample = [1.0, 2.0, 4.0, 32.0, 32.0]
    # n=5 signed integer sample-L3 weights [6,-3,-6,-3,+6].
    values = [Fraction(value) for value in sample]
    assert values[0] * values[4] != values[1] * values[3]
    assert values[0] ** 6 * values[4] ** 6 == values[1] ** 3 * values[2] ** 6 * values[3] ** 3
    fit = fit_lp3_lmoments(sample)
    assert fit.distribution.skew == 0.0
    assert fit.sample_moments[2] == 0.0


def test_lp3_one_ulp_series_rejects_unresolvable_exact_log_skew() -> None:
    """The exact binary64 inputs are not geometrically spaced in log flow."""
    sample = [1.5, 1.5000000000000002, 1.5000000000000004]
    with localcontext() as context:
        context.prec = 100
        low, mid, high = (Decimal(value) for value in sample)
        negative_gap = -(mid / low).ln()
        positive_gap = (high / mid).ln()
        exact_tau3 = (positive_gap + negative_gap) / (positive_gap - negative_gap)
        assert abs(exact_tau3) > Decimal("1e-18")
    with pytest.raises(ValueError, match="L-skew"):
        fit_lp3_lmoments(sample)


def test_lp3_near_unity_log_ratio_matches_high_precision_oracle() -> None:
    low, high = 1.5, 1.5000000000000002
    with localcontext() as context:
        context.prec = 100
        expected = float((Decimal(high) / Decimal(low)).ln())
    actual = ordinary_lmoments.stable_log_ratio(high, low)
    _near(actual, expected, absolute=4e-32)
    assert actual == -ordinary_lmoments.stable_log_ratio(low, high)


@pytest.mark.parametrize("ratio", [3.0, 7.0, 10.0])
def test_lp3_log_ratios_preserve_exact_reciprocal_symmetry(ratio: float) -> None:
    positive = ordinary_lmoments.stable_log_ratio(ratio, 1.0)
    negative = ordinary_lmoments.stable_log_ratio(1.0, ratio)
    assert positive + negative == 0.0


@pytest.mark.parametrize("ratio", [7.0, 10.0])
@pytest.mark.parametrize("unit_factor", [1.0, 2.0, 3.0, 10.0])
def test_lp3_nonbinary_geometric_sample_is_unit_invariant(ratio: float, unit_factor: float) -> None:
    """A geometric series has exact log-L-skew zero at every physical scale."""
    source = [1.0, ratio, ratio * ratio]
    baseline = fit_lp3_lmoments(source)
    scaled = fit_lp3_lmoments([unit_factor * value for value in source], reference_discharge=unit_factor)
    assert baseline.distribution.skew == scaled.distribution.skew == 0.0
    assert baseline.sample_moments[2] == scaled.sample_moments[2] == 0.0
    _near(scaled.sample_moments[0], baseline.sample_moments[0], absolute=3e-15)
    _near(scaled.sample_moments[1], baseline.sample_moments[1], absolute=3e-15)
    _near(scaled.quantile(0.5), unit_factor * baseline.quantile(0.5), absolute=2e-12)
    _near(scaled.quantile(0.9), unit_factor * baseline.quantile(0.9), absolute=2e-11)


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [
        (1e-20, 1.7e303),  # positive subnormal ratio rounds to 5e-324
        (1e-320, 1e-10),  # positive ratio in the subnormal region
        (1e300, 1e-20),  # direct floating-point division overflows
    ],
)
def test_lp3_stable_log_ratio_matches_independent_decimal_oracle(numerator: float, denominator: float) -> None:
    """Compare to 80-digit logarithms of the actual input binary64 values."""
    with localcontext() as context:
        context.prec = 80
        expected = float(Decimal(numerator).ln() - Decimal(denominator).ln())
    _near(ordinary_lmoments.stable_log_ratio(numerator, denominator), expected, absolute=3e-13)


def test_lp3_extreme_reference_avoids_subnormal_log_ratio_quantization() -> None:
    sample = [2.5e-21, 1e-20, 4e-20]
    ordinary = fit_lp3_lmoments(sample, reference_discharge=1.0)
    extreme = fit_lp3_lmoments(sample, reference_discharge=1.7e303)
    assert ordinary.distribution.skew == extreme.distribution.skew == 0.0
    assert ordinary.sample_moments[2] == extreme.sample_moments[2] == 0.0
    _near(extreme.sample_moments[1], ordinary.sample_moments[1], absolute=2e-15)
    _near(extreme.quantile(0.5), ordinary.quantile(0.5), absolute=4e-33)
    _near(extreme.quantile(0.9), ordinary.quantile(0.9), absolute=2e-32)


def test_lp3_rejects_unresolved_shape_and_nonpositive_data() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        fit_lp3_lmoments([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="admitted exact gamma-shape"):
        fit_lp3_lmoments([1.0] * 9 + [1e100])
    with pytest.raises(ValueError, match="at least 3"):
        fit_lp3_lmoments([1.0, 2.0])


def test_gp_uniform_exact_rational_three_point_oracle() -> None:
    # Three evenly spaced order statistics have L1=2, L2=2/3 and tau3=0.
    # Hence GP kappa=1, EVT xi=-1, alpha=4, location=0: Uniform(0,4).
    for fit in (fit_gp_lmoments([1.0, 2.0, 3.0]), fit_gp_lmoments([1.0, 2.0, 3.0], location=0.0)):
        assert isinstance(fit.distribution, GeneralizedPareto)
        _near(fit.distribution.location, 0.0, absolute=1e-14)
        _near(exp(fit.distribution.log_scale), 4.0, absolute=1e-14)
        _near(fit.distribution.xi, -1.0, absolute=1e-14)
        _near(fit.quantile(0.25), 1.0, absolute=1e-13)
        _near(fit.quantile(0.75), 3.0, absolute=1e-13)


@pytest.mark.parametrize("xi", [-0.8, -0.2, 0.0, 0.3])
def test_gp_free_location_independent_quantile_grid(xi: float) -> None:
    sample = [float(genpareto.ppf(f, c=xi, loc=8.0, scale=4.0)) for f in _grid(8000)]
    fit = fit_gp_lmoments(sample)
    assert isinstance(fit.distribution, GeneralizedPareto)
    assert len(fit.sample_moments) == 3
    _near(fit.distribution.xi, xi, absolute=0.012)
    _near(fit.distribution.location, 8.0, absolute=0.025)
    _near(exp(fit.distribution.log_scale), 4.0, absolute=0.03)
    _near(fit.quantile(0.99), float(genpareto.ppf(0.99, c=xi, loc=8.0, scale=4.0)), absolute=0.45)
    transformed = fit_gp_lmoments([100.0 + 2.0 * q for q in reversed(sample)])
    _near(transformed.distribution.location, 100.0 + 2.0 * fit.distribution.location, absolute=1e-9)


@pytest.mark.parametrize("xi", [-0.4, 0.0, 0.4])
def test_gp_explicit_fixed_location_uses_two_l_moments(xi: float) -> None:
    sample = [float(genpareto.ppf(f, c=xi, loc=8.0, scale=4.0)) for f in _grid(8000)]
    fit = fit_gp_lmoments(sample, location=8.0)
    assert len(fit.sample_moments) == 2
    _near(fit.distribution.location, 8.0, absolute=1e-14)
    _near(fit.distribution.xi, xi, absolute=0.015)
    _near(exp(fit.distribution.log_scale), 4.0, absolute=0.04)


def test_gp_support_failures_are_explicit() -> None:
    with pytest.raises(ValueError, match="below its lower support"):
        fit_gp_lmoments([1.0, 2.0, 3.0], location=1.1)
    with pytest.raises(ValueError, match="strictly between -1 and 1"):
        fit_gp_lmoments([1.0] * 9 + [100.0])
    with pytest.raises(ValueError, match="at least 3"):
        fit_gp_lmoments([1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        fit_gp_lmoments([1.0, 2.0, float("inf")])


@pytest.mark.parametrize("family", ["gumbel", "lognormal", "lp3", "gp"])
def test_fit_is_finite_and_quantile_probability_is_nonexceedance(family: str) -> None:
    sample = [1.0, 2.0, 3.0, 5.0, 8.0, 11.0, 15.0, 22.0, 29.0]
    if family == "gumbel":
        fit = fit_gumbel_lmoments(sample)
    elif family == "lognormal":
        fit = fit_lognormal_lmoments(sample)
    elif family == "lp3":
        fit = fit_lp3_lmoments(sample)
    else:
        fit = fit_gp_lmoments(sample)
    for p in [0.1, 0.5, 0.9, 0.99]:
        assert isfinite(fit.quantile(p))
    with pytest.raises(ValueError, match="strictly between"):
        fit.quantile(1.0)
    with pytest.raises(ValueError, match="strictly between"):
        fit.quantile(0.0)
    with pytest.raises(ValueError, match="strictly between"):
        fit.quantile(float("nan"))


def test_all_four_reject_constant_or_nonfinite_samples() -> None:
    for fit in (fit_gumbel_lmoments, fit_lognormal_lmoments, fit_lp3_lmoments, fit_gp_lmoments):
        with pytest.raises(ValueError, match="positive second L-moment"):
            fit([7.0] * 10)
        with pytest.raises(ValueError, match="finite"):
            fit([1.0, 2.0, float("nan"), 4.0])


@pytest.mark.parametrize("reference", [1.0, 2.0, 10.0])
def test_lp3_zero_lskew_preserved_under_reference_changes(reference: float) -> None:
    fit = fit_lp3_lmoments([0.25, 1.0, 4.0], reference_discharge=reference)
    assert isinstance(fit.distribution, LogPearsonIII)
    assert fit.distribution.skew == 0.0
    assert fit.sample_moments[2] == 0.0
    _near(fit.sample_moments[0], -log(reference), absolute=4e-15)
    _near(fit.quantile(0.5), 1.0, absolute=4e-15)


@pytest.mark.parametrize("shape", [10_000.0, 100_000.0, 1e8])
@pytest.mark.parametrize("skew_sign", [-1.0, 1.0])
def test_lp3_gamma_sd_ratio_agrees_with_80_digit_reference(shape: float, skew_sign: float) -> None:
    # mpmath 1.3 at 80 decimal places: sqrt(pi*a) Gamma(a)/Gamma(a+1/2).
    targets = {
        10_000.0: 1.7724760067171166486771191561073551,
        100_000.0: 1.772456066474214380109650362732253,
        1e8: 1.7724538531210833423147920798292291,
    }
    source_l2 = 0.7
    expected_sd = source_l2 * targets[shape]
    sd = source_l2 * ordinary_lmoments.gamma_sd_l2_multiplier(shape)
    _near(sd, expected_sd, absolute=4e-15)
    kernel = LogPearsonIII(LogCoordinates(0.2, log(sd)), skew_sign * 2.0 / sqrt(shape))
    _near(exp(kernel.log.natural_log_sd), expected_sd, absolute=4e-15)
    _near(sd / targets[shape], source_l2, absolute=4e-15)


def _exact_gp_three_point(values: list[float], *, fixed_location: float | None) -> tuple[float, float, float]:
    """Exact Fraction order-statistic moments then algebraic GP inversion."""
    low, middle, high = sorted(Fraction(value) for value in values)
    first = (low + middle + high) / 3
    second = (high - low) / 3
    third = (low - 2 * middle + high) / 3
    if fixed_location is None:
        kappa = (second - 3 * third) / (second + third)
        location = first - second * (2 + kappa)
        scale = second * (1 + kappa) * (2 + kappa)
    else:
        location = Fraction(fixed_location)
        excess = first - location
        kappa = excess / second - 2
        scale = excess * (1 + kappa)
    return float(location), float(scale), float(-kappa)


@pytest.mark.parametrize("fixed_location", [None, 0.0])
def test_gp_mean_boundary_matches_independent_exact_rational_parameters(fixed_location: float | None) -> None:
    sample = [0.0, 2e-16, 1.0]
    location, scale, xi = _exact_gp_three_point(sample, fixed_location=fixed_location)
    fit = fit_gp_lmoments(sample, location=fixed_location)
    _near(fit.distribution.location, location, absolute=2e-30)
    _near(exp(fit.distribution.log_scale), scale, absolute=2e-30)
    _near(fit.distribution.xi, xi, absolute=2e-15)
    assert fit.distribution.xi < 1.0


def test_gp_final_logscale_support_rejects_unrepresentable_upper_endpoint() -> None:
    # A rounded pre-log scale had an upper endpoint exactly at the sample
    # maximum; reconstructing the stored log-scale moved it below that flow.
    with pytest.raises(ValueError, match="above its upper support"):
        fit_gp_lmoments([0.0, 1.0 - 1e-8, 1.0])


@pytest.mark.parametrize("xi", [-0.8, -0.2, 0.0, 0.3])
def test_all_accepted_gp_fits_contain_input_under_final_kernel(xi: float) -> None:
    sample = [float(genpareto.ppf(f, c=xi, loc=8.0, scale=4.0)) for f in _grid(600)]
    for fit in (fit_gp_lmoments(sample), fit_gp_lmoments(sample, location=8.0)):
        lower, upper = fit.distribution.support()
        assert lower <= min(sample)
        assert max(sample) <= upper
        if fit.distribution.xi < 0.0:
            standardized = (max(sample) - fit.distribution.location) / exp(fit.distribution.log_scale)
            assert standardized <= -1.0 / fit.distribution.xi


@pytest.mark.parametrize(
    ("sample", "fixed_location"),
    [
        ([0.0, 2e-16, 1.0], 0.0),
        ([-1.0, -1.0 + 2e-16, 0.0], -1.0),
    ],
)
def test_gp_fixed_location_retains_fraction_scale_below_anchor_ulp(sample: list[float], fixed_location: float) -> None:
    expected_location, expected_scale, expected_xi = _exact_gp_three_point(sample, fixed_location=fixed_location)
    fit = fit_gp_lmoments(sample, location=fixed_location)
    assert fit.distribution.location == expected_location
    _near(exp(fit.distribution.log_scale), expected_scale, absolute=2e-30)
    _near(fit.distribution.xi, expected_xi, absolute=2e-15)
    assert fit.distribution.xi < 1.0
