"""Independent algebra and policy checks, not fitted-GEV calibration admission."""

from math import inf, nan, sqrt

import pytest

from ffa_methods.estimators.gev_lh_diagnostics import (
    LHConditionalVariance,
    conditional_lh_variance,
    select_lowest_passing_h,
    standardized_lh_z,
)


def test_conditional_variance_exact_rational_reference() -> None:
    # Means zero; sums of squares 1 and 1/2, cross product 1/2.
    pairs = ((-0.5, -0.5), (-0.5, 0.0), (0.5, 0.0), (0.5, 0.5))
    result = conditional_lh_variance(pairs)
    assert result.count == 4
    assert result.skew_variance == pytest.approx(1 / 3, rel=1e-14)
    assert result.kurtosis_variance == pytest.approx(1 / 6, rel=1e-14)
    assert result.covariance == pytest.approx(1 / 6, rel=1e-14)
    assert result.conditional_variance == pytest.approx(1 / 12, rel=1e-14)
    # Independent affine transformation: scale kurtosis by 2, skew by -1.
    transformed = conditional_lh_variance((-x + 0.25, 2 * y - 0.25) for x, y in pairs)
    assert transformed.conditional_variance == pytest.approx(1 / 3, rel=1e-14)
    assert standardized_lh_z(0.25, 0.125, result) == pytest.approx(sqrt(12) / 8, rel=1e-14)
    assert standardized_lh_z(0.125, 0.25, result) == pytest.approx(-sqrt(12) / 8, rel=1e-14)


@pytest.mark.parametrize(
    "pairs",
    [
        (),
        ((0.0, 0.1), (0.1, 0.0)),
        ((0.0, 0.1), (0.0, 0.2), (0.0, 0.3)),
        ((-0.5, -0.5), (0.0, 0.0), (0.5, 0.5)),
        ((-0.5, 0.0), (0.0, 0.0), (0.5, 0.0)),
        ((0.0, 0.0), (0.1, nan), (0.2, 0.1)),
        ((inf, 0.0), (0.1, 0.2), (0.2, 0.1)),
    ],
)
def test_conditional_variance_rejects_unusable_calibration(pairs: tuple[tuple[float, float], ...]) -> None:
    with pytest.raises(ValueError):
        conditional_lh_variance(pairs)


@pytest.mark.parametrize("variance", [0.0, -1.0, inf, nan])
def test_z_rejects_invalid_variance(variance: float) -> None:
    with pytest.raises(ValueError):
        standardized_lh_z(0.25, 0.125, LHConditionalVariance(4, 1 / 3, 1 / 6, 1 / 6, variance))


def test_lowest_passing_h_not_largest_or_smallest_residual() -> None:
    assert select_lowest_passing_h((1.5, 0.1, 0.0, -0.1, -0.5), critical_z=2.0) == 0
    assert select_lowest_passing_h((-3.0, 2.0, 0.0, -0.1, -0.5), critical_z=2.0) == 1
    assert select_lowest_passing_h((-3.0, -2.1, 2.1, -3.0, -2.0), critical_z=2.0) == 4
    assert select_lowest_passing_h((-3.0, -2.1, 2.1, -3.0, 3.0), critical_z=2.0) is None


@pytest.mark.parametrize("critical_z", [0.0, -1.0, inf, nan])
def test_selection_rejects_invalid_cutoff(critical_z: float) -> None:
    with pytest.raises(ValueError):
        select_lowest_passing_h((0.0, 1.0, 2.0, 3.0, 4.0), critical_z=critical_z)


@pytest.mark.parametrize("z_by_h", [(0.0, 1.0), (0.0, 1.0, 2.0, 3.0, nan), (0.0, 1.0, 2.0, inf, 4.0)])
def test_selection_rejects_missing_or_invalid_candidates(z_by_h: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        select_lowest_passing_h(z_by_h, critical_z=2.0)
