"""Independent combinatorial and analytic tests for fixed-H GEV LH fitting."""

from collections.abc import Iterable
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
from math import comb, exp, expm1, inf, isclose, isfinite, log, nextafter
from struct import pack

import pytest
from scipy.stats import genextreme

from ffa_methods.estimators import gev_lh as gev_lh_module
from ffa_methods.estimators.gev_lh import (
    bootstrap_fixed_gev_lh,
    fit_gev_lh,
    sample_lh_moments,
    theoretical_gev_lh_moments,
)


def _near(actual: float, expected: float, *, tolerance: float = 1e-12) -> bool:
    """Assert numeric agreement without partially typed pytest.approx."""
    return isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def _near_each(actual: tuple[float, ...], expected: tuple[float, ...], *, tolerance: float = 1e-12) -> bool:
    return len(actual) == len(expected) and all(
        _near(left, right, tolerance=tolerance) for left, right in zip(actual, expected, strict=True)
    )


def _subset_estimator(values: list[float], *, shift: int, order: int) -> float:
    """Independent U-statistic from the defining LH ordered-subset expectation."""
    subsets = list(combinations(sorted(values), shift + order))
    return sum(
        sum((-1) ** j * comb(order - 1, j) * group[-1 - j] for j in range(order)) / order for group in subsets
    ) / len(subsets)


@pytest.mark.parametrize("shift", [0, 1, 2, 3, 4])
def test_sample_estimators_equal_direct_order_statistic_expectations(shift: int) -> None:
    values = [1.0, 2.0, 2.0, 4.0, 7.0, 8.0, 13.0, 15.0]
    for order, actual in enumerate(sample_lh_moments(values, shift=shift), start=1):
        assert _near(actual, _subset_estimator(values, shift=shift, order=order), tolerance=2e-13)


def test_hosking_lmoments_of_arithmetic_series() -> None:
    assert _near_each(sample_lh_moments([1, 2, 3, 4, 5]), (3.0, 1.0, 0.0, 0.0), tolerance=1e-14)


@pytest.mark.parametrize("shift", [0, 1, 2, 3, 4])
def test_shift_and_positive_scale_invariance(shift: int) -> None:
    values = [float(i * i) for i in range(1, 20)]
    original = sample_lh_moments(values, shift=shift)
    scaled = sample_lh_moments([20 + 3 * value for value in values], shift=shift)
    assert _near(scaled[0], 20 + 3 * original[0], tolerance=5e-12)
    assert _near_each(scaled[1:], tuple(3 * val for val in original[1:]), tolerance=5e-12)


def test_gumbel_population_lmoments_have_published_ratios() -> None:
    first, second, third, fourth = theoretical_gev_lh_moments(10, 2, 0)
    assert _near(first, 10 + 2 * 0.5772156649015329)
    assert _near(second, 2 * log(2))
    assert _near(third / second, 2 * log(3) / log(2) - 3)
    assert _near(fourth / second, 0.1503749927884382)


@pytest.mark.parametrize("shift", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("shape", [-0.6, -0.2, 0.0, 0.3, 1.0])
def test_gev_lh_fit_recovers_discretised_population_quantiles(shift: int, shape: float) -> None:
    location, scale = 12.0, 3.5
    if shape == 0.0:
        sample = [location - scale * log(-log((i + 0.5) / 2000)) for i in range(2000)]
    else:
        sample = [location - scale * expm1(shape * log(-log((i + 0.5) / 2000))) / shape for i in range(2000)]
    actual = fit_gev_lh(sample, shift=shift)
    assert _near(actual.location, location, tolerance=0.2)
    assert _near(actual.scale, scale, tolerance=0.3)
    assert _near(actual.shape, shape, tolerance=0.05)
    assert _near_each(actual.sample_moments[:3], actual.fitted_moments[:3], tolerance=1e-9)
    assert actual.lh_kurtosis_residual is not None


def test_minimum_sample_has_no_fourth_moment_diagnostic() -> None:
    result = fit_gev_lh([1, 2, 3], shift=0)
    assert len(result.sample_moments) == 3
    assert result.lh_kurtosis_residual is None


@pytest.mark.parametrize("shape", [-0.4, 0.0, 0.3])
def test_gev_quantile_matches_distribution_inversion(shape: float) -> None:
    fit = replace(
        fit_gev_lh([20, 22, 23, 23, 29, 30, 40, 51, 60, 81], shift=2),
        location=12.0,
        scale=3.5,
        shape=shape,
    )
    for probability in [0.01, 0.5, 0.9, 0.99, 0.9999]:
        q = fit.quantile(probability)
        if shape:
            restored = exp(-((1 - shape * (q - fit.location) / fit.scale) ** (1 / shape)))
        else:
            restored = exp(-exp(-(q - fit.location) / fit.scale))
        assert _near(restored, probability, tolerance=2e-13)


def test_bootstrap_is_seeded_records_failures_and_is_not_bayesian() -> None:
    values = [10.0 + i**1.5 for i in range(1, 50)]
    actual = bootstrap_fixed_gev_lh(values, shift=1, probabilities=(0.5, 0.99), replications=25, seed=2026)
    assert actual == bootstrap_fixed_gev_lh(values, shift=1, probabilities=(0.5, 0.99), replications=25, seed=2026)
    assert actual.requested == actual.successful + actual.failed
    assert actual.successful > 0
    assert actual.confidence == 0.9
    assert all(low <= high for low, high in zip(actual.lower_limits, actual.upper_limits, strict=True))
    assert actual != bootstrap_fixed_gev_lh(values, shift=1, probabilities=(0.5, 0.99), replications=25, seed=1)


@pytest.mark.parametrize("shift", [-1, 5, True])
def test_shift_is_strictly_restricted_to_legacy_range(shift: int) -> None:
    with pytest.raises(ValueError, match="shift"):
        sample_lh_moments([1, 2, 3, 4, 5], shift=shift)


def test_failures_are_explicit_and_do_not_silently_drop_observations() -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_gev_lh([1, 2, float("nan"), 4])
    with pytest.raises(ValueError, match="nonconstant"):
        fit_gev_lh([4] * 20, shift=4)
    with pytest.raises(ValueError, match="at least 7"):
        fit_gev_lh([1, 2, 3, 4, 5, 6], shift=4)
    with pytest.raises(ValueError, match="strictly between"):
        fit_gev_lh([1, 2, 3, 4, 5]).quantile(1.0)
    with pytest.raises(ValueError, match="positive"):
        theoretical_gev_lh_moments(0, 0, 0)
    with pytest.raises(ValueError, match="greater than -1"):
        theoretical_gev_lh_moments(0, 1, -1)
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_fixed_gev_lh([1, 2, 3, 4], shift=0, probabilities=[0.9], replications=0, seed=1)


def _source_sample_sha256(values: Iterable[float]) -> str:
    """Match the source fixture's original-order binary64 byte convention."""
    return sha256(b"".join(pack("<d", float(value)) for value in values)).hexdigest()


def _exact_subset_moment(values: list[float], *, shift: int, order: int) -> Fraction:
    """An independent rational U-statistic, not a reuse of production weights."""
    ordered = tuple(Fraction(value) for value in sorted(values))
    subset_count = comb(len(values), shift + order)
    numerator = sum(
        (
            sum(((-1) ** j * comb(order - 1, j) * group[-1 - j] for j in range(order)), Fraction(0))
            for group in combinations(ordered, shift + order)
        ),
        Fraction(0),
    )
    return numerator / (order * subset_count)


@pytest.mark.parametrize("shift", [0, 1, 2, 3, 4])
def test_constant_series_has_exact_zero_lh_dispersion(shift: int) -> None:
    assert sample_lh_moments([10.0] * 10, shift=shift) == (10.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="nonconstant"):
        fit_gev_lh([10.0] * 10, shift=shift)


@pytest.mark.parametrize("shift", [1, 2, 3, 4])
@pytest.mark.parametrize(("offset", "factor"), [(0.0, 1.0), (1000.0, 4.0), (-10.0, 0.5)])
def test_tied_low_outlier_has_exact_zero_lh_scale(shift: int, offset: float, factor: float) -> None:
    """One low point is absent from all H>=1 upper ordered-subset contrasts."""
    values = [offset] + [offset + 10.0 * factor] * 9
    moments = sample_lh_moments(values, shift=shift)
    assert _near(moments[0], offset + 10.0 * factor)
    assert moments[1:] == (0.0, 0.0, 0.0)
    assert _exact_subset_moment(values, shift=shift, order=2) == 0
    with pytest.raises(ValueError, match="positive sample LH scale"):
        fit_gev_lh(values, shift=shift)


@pytest.mark.parametrize("shift", [0, 1, 2, 3, 4])
def test_one_ulp_near_degenerate_dispersion_is_not_rejected(shift: int) -> None:
    """Exact rational subsets establish genuine sub-1e-13 LH dispersion."""
    ulp = nextafter(10.0, inf) - 10.0
    values = [10.0 + i * i * ulp for i in range(10)]
    moments = sample_lh_moments(values, shift=shift)
    for order, actual in enumerate(moments, start=1):
        exact = _exact_subset_moment(values, shift=shift, order=order)
        tolerance = 3e-14 if order == 1 else 1e-27
        assert _near(actual, float(exact), tolerance=tolerance)
    assert moments[1] > 0.0
    fitted = fit_gev_lh(values, shift=shift)
    assert fitted.scale > 0.0
    assert fitted.sample_moments[1] == moments[1]


@pytest.mark.parametrize("shift", [0, 1, 2, 3, 4])
def test_small_exact_binary_affine_dispersion_remains_fittable(shift: int) -> None:
    values = [float(i * i) for i in range(1, 20)]
    baseline = fit_gev_lh(values, shift=shift)
    offset, factor = 2**20, 2**-8
    transformed = fit_gev_lh([offset + factor * value for value in values], shift=shift)
    assert _near(transformed.location, offset + factor * baseline.location, tolerance=3e-10)
    assert _near(transformed.scale, factor * baseline.scale, tolerance=1e-10)
    assert _near(transformed.shape, baseline.shape, tolerance=2e-10)


@pytest.mark.parametrize(
    ("count", "raw", "ratios", "source_hash"),
    [
        (
            1292,
            (522512.5024550116, 261084.88504577902, 52098.516265937826, -5.756515452627805),
            (0.19954627498563463, -2.204844394426912e-05),
            "b4be8dd14ffc675934a6159566d1bc2cab30dd57615ff75d64da5bf203253b17",
        ),
        (
            1293,
            (522161.99051382445, 261046.5194639539, 52195.02623431416, -0.9911925792957503),
            (0.19994530607607433, -3.796995958157631e-06),
            "ad2cbc046ccc3598660650bab54039d6c617481555bbf280143f66335e3dffb6",
        ),
        (
            1460,
            (522816.5913848459, 261222.97693894972, 52165.08917698612, 36.627630512131084),
            (0.19969563852408587, 0.00014021596010174597),
            "ad60fc261b4f524b82454a9bdad9d51b5ae261bab7dbbff2b088dacbc8c7a8f4",
        ),
    ],
)
def test_external_exact_rational_large_l_moments(
    count: int, raw: tuple[float, float, float, float], ratios: tuple[float, float], source_hash: str
) -> None:
    """Generator and targets: 80-digit Fraction reference, not a port."""
    values = [((i * 7919) % 10007) ** 2 / 64 for i in range(1, count + 1)]
    assert _source_sample_sha256(values) == source_hash
    actual = sample_lh_moments(values, shift=0)
    for received, expected in zip(actual, raw, strict=True):
        assert abs(received - expected) <= max(abs(expected) * 1e-12, 1e-13)
    for received, expected in zip((actual[2] / actual[1], actual[3] / actual[1]), ratios, strict=True):
        assert abs(received - expected) <= max(abs(expected) * 1e-12, 1e-13)


def test_bootstrap_counts_actual_degenerate_lh_refit_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A controlled refit path exercises the real zero-LH-scale rejection."""
    original_fit = gev_lh_module.fit_gev_lh
    calls = 0

    def refit_with_two_tied_failures(values: Iterable[float], *, shift: int = 0) -> gev_lh_module.GEVLHFit:
        nonlocal calls
        calls += 1
        if calls in (3, 6):
            # The original algorithm incorrectly admitted this H=1 series.
            return original_fit([0.0] + [10.0] * 9, shift=shift)
        return original_fit(values, shift=shift)

    monkeypatch.setattr(gev_lh_module, "fit_gev_lh", refit_with_two_tied_failures)
    result = bootstrap_fixed_gev_lh(
        [3.0 + 1.1**i for i in range(15)],
        shift=1,
        probabilities=(0.5, 0.95),
        replications=10,
        seed=2026,
    )
    assert calls == 11  # Initial fit plus exactly ten attempted refits.
    assert (result.requested, result.successful, result.failed) == (10, 8, 2)
    assert len(result.failure_reasons) == 2
    assert all("positive sample LH scale" in reason for reason in result.failure_reasons)
    assert len(result.lower_limits) == len(result.upper_limits) == 2


def test_bootstrap_nonfinite_refit_quantiles_count_as_failures() -> None:
    """Codex P2: finite initial Q(.99), but one refitted Q(.99) overflows."""
    sample = [1e307 * i for i in range(1, 13)]
    result = bootstrap_fixed_gev_lh(sample, shift=0, probabilities=(0.5, 0.99), replications=10, seed=1)
    assert (result.requested, result.successful, result.failed) == (10, 9, 1)
    assert len(result.failure_reasons) == 1
    assert "bootstrap refit GEV LH quantiles" in result.failure_reasons[0]
    assert all(isfinite(value) for value in (*result.estimates, *result.lower_limits, *result.upper_limits))


def test_bootstrap_rejects_unrepresentable_initial_quantile() -> None:
    sample = [1e307 * i for i in range(1, 13)]
    with pytest.raises(ValueError, match="initial GEV LH quantiles cannot be represented"):
        bootstrap_fixed_gev_lh(sample, shift=0, probabilities=(0.99999,), replications=10, seed=1)


def test_bootstrap_failure_is_atomic_when_later_probability_overflows() -> None:
    """No first-probability partial append when a later requested Q fails."""
    sample = [1e307 * i for i in range(1, 13)]
    result = bootstrap_fixed_gev_lh(sample, shift=0, probabilities=(0.5, 0.95, 0.99), replications=10, seed=1)
    assert result.successful + result.failed == result.requested
    assert result.failed >= 1
    assert all(isfinite(value) for value in (*result.lower_limits, *result.upper_limits))


@pytest.mark.parametrize("shift", [0, 1, 2, 3, 4])
def test_generated_sample_moments_match_exact_subsets(shift: int) -> None:
    sample = [float(i * i + 2 * i + 1) for i in range(10)]
    actual = sample_lh_moments(sample, shift=shift)
    expected = tuple(float(_exact_subset_moment(sample, shift=shift, order=j)) for j in range(1, 5))
    assert _near_each(actual, expected, tolerance=5e-11)


def test_generated_gev_fit_preserves_exact_skew_and_shape_convention() -> None:
    fitted = fit_gev_lh([float(i * i + 2 * i + 1) for i in range(20)])
    assert _near(
        fitted.fitted_moments[2] / fitted.fitted_moments[1],
        fitted.sample_moments[2] / fitted.sample_moments[1],
        tolerance=1e-12,
    )
    expected = float(genextreme.ppf(0.99, c=fitted.shape, loc=fitted.location, scale=fitted.scale))
    assert _near(fitted.quantile(0.99), expected, tolerance=1e-10)
