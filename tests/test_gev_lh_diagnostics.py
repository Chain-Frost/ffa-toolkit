"""Independent algebra and policy checks, not fitted-GEV calibration admission."""

from math import inf, isclose, nan, sqrt

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
    assert isclose(result.skew_variance, 1 / 3, rel_tol=1e-14, abs_tol=1e-12)
    assert isclose(result.kurtosis_variance, 1 / 6, rel_tol=1e-14, abs_tol=1e-12)
    assert isclose(result.covariance, 1 / 6, rel_tol=1e-14, abs_tol=1e-12)
    assert isclose(result.conditional_variance, 1 / 12, rel_tol=1e-14, abs_tol=1e-12)
    # Independent affine transformation: scale kurtosis by 2, skew by -1.
    transformed = conditional_lh_variance((-x + 0.25, 2 * y - 0.25) for x, y in pairs)
    assert isclose(transformed.conditional_variance, 1 / 3, rel_tol=1e-14, abs_tol=1e-12)
    assert isclose(standardized_lh_z(0.25, 0.125, result), sqrt(12) / 8, rel_tol=1e-14, abs_tol=1e-12)
    assert isclose(standardized_lh_z(0.125, 0.25, result), -sqrt(12) / 8, rel_tol=1e-14, abs_tol=1e-12)


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
    with pytest.raises(ValueError, match="conditional LH"):
        conditional_lh_variance(pairs)


@pytest.mark.parametrize("variance", [0.0, -1.0, inf, nan])
def test_z_rejects_invalid_variance(variance: float) -> None:
    with pytest.raises(ValueError, match="finite kurtoses"):
        standardized_lh_z(0.25, 0.125, LHConditionalVariance(4, 1 / 3, 1 / 6, 1 / 6, variance))


def test_lowest_passing_h_not_largest_or_smallest_residual() -> None:
    assert select_lowest_passing_h((1.5, 0.1, 0.0, -0.1, -0.5), critical_z=2.0) == 0
    assert select_lowest_passing_h((-3.0, 2.0, 0.0, -0.1, -0.5), critical_z=2.0) == 1
    assert select_lowest_passing_h((-3.0, -2.1, 2.1, -3.0, -2.0), critical_z=2.0) == 4
    assert select_lowest_passing_h((-3.0, -2.1, 2.1, -3.0, 3.0), critical_z=2.0) is None


@pytest.mark.parametrize("critical_z", [0.0, -1.0, inf, nan])
def test_selection_rejects_invalid_cutoff(critical_z: float) -> None:
    with pytest.raises(ValueError, match="positive explicit critical z"):
        select_lowest_passing_h((0.0, 1.0, 2.0, 3.0, 4.0), critical_z=critical_z)


@pytest.mark.parametrize("z_by_h", [(0.0, 1.0), (0.0, 1.0, 2.0, 3.0, nan), (0.0, 1.0, 2.0, inf, 4.0)])
def test_selection_rejects_missing_or_invalid_candidates(z_by_h: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="all five finite diagnostics"):
        select_lowest_passing_h(z_by_h, critical_z=2.0)
