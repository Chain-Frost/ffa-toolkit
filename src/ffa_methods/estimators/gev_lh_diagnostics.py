"""Internal Wang (1998) LH diagnostic algebra, not an admitted calibration or CLI.

No published coefficient table is embedded. A caller must separately establish
that paired ratios are from appropriate GEV simulations at the same H, n and
population LH skew, and qualify simulation precision before using the result.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import fsum, isfinite, sqrt


@dataclass(frozen=True, slots=True)
class LHConditionalVariance:
    """Empirical covariance components; count is simulations, not record length."""

    count: int
    skew_variance: float
    kurtosis_variance: float
    covariance: float
    conditional_variance: float


def conditional_lh_variance(pairs: Iterable[tuple[float, float]]) -> LHConditionalVariance:
    """Estimate Wang equation (3) squared from paired sample skew/kurtosis ratios.

    Use the sample covariance convention (denominator B-1 for B simulations).
    The centered regression residual sum avoids subtracting two nearly equal
    covariance terms. This is the joint-normal approximation, not an estimate
    of the exact conditional law or a numerical precision admission decision.
    """
    values = tuple(pairs)
    count = len(values)
    if count < 3 or any(not isfinite(x) or not isfinite(y) for x, y in values):
        msg = "conditional LH variance requires at least three finite paired ratios"
        raise ValueError(msg)
    try:
        mean3 = fsum(x for x, _ in values) / count
        mean4 = fsum(y for _, y in values) / count
        centered = tuple((x - mean3, y - mean4) for x, y in values)
        ss3 = fsum(x * x for x, _ in centered)
        ss4 = fsum(y * y for _, y in centered)
        cross = fsum(x * y for x, y in centered)
        if not all(isfinite(value) for value in (ss3, ss4, cross)) or ss3 <= 0 or ss4 <= 0:
            msg = "LH ratio covariance must be finite with positive marginal variances"
            raise ValueError(msg)
        slope = cross / ss3
        residuals = tuple(y - slope * x for x, y in centered)
        variance = fsum(residual * residual for residual in residuals) / (count - 1)
    except (OverflowError, ValueError) as exc:
        msg = "conditional LH covariance cannot be evaluated from these ratios"
        raise ValueError(msg) from exc
    skew_variance = ss3 / (count - 1)
    kurtosis_variance = ss4 / (count - 1)
    covariance = cross / (count - 1)
    if not all(isfinite(value) for value in (variance, skew_variance, kurtosis_variance, covariance)) or min(
        variance, skew_variance, kurtosis_variance
    ) <= 0:
        msg = "conditional LH variance must be finite and positive"
        raise ValueError(msg)
    return LHConditionalVariance(count, skew_variance, kurtosis_variance, covariance, variance)


def standardized_lh_z(sample_kurtosis: float, fitted_kurtosis: float, calibration: LHConditionalVariance) -> float:
    """Wang equation (4); caller must qualify calibration for this candidate."""
    variance = calibration.conditional_variance
    if not all(isfinite(value) for value in (sample_kurtosis, fitted_kurtosis, variance)) or variance <= 0:
        msg = "standardized LH diagnostic requires finite kurtoses and positive conditional variance"
        raise ValueError(msg)
    z = (sample_kurtosis - fitted_kurtosis) / sqrt(variance)
    if not isfinite(z):
        msg = "standardized LH diagnostic is not representable"
        raise ValueError(msg)
    return z


def select_lowest_passing_h(z_by_h: tuple[float, ...], *, critical_z: float) -> int | None:
    """Explicit two-sided, lowest-H policy for all five evaluated candidates.

    Input order is H=0,1,2,3,4. Equality passes; no pass returns None.
    Incomplete or nonfinite diagnostics invalidate the entire selection, even
    if another candidate would pass. No default critical value is provided.
    """
    if not isfinite(critical_z) or critical_z <= 0:
        msg = "H selection requires a finite positive explicit critical z"
        raise ValueError(msg)
    if len(z_by_h) != 5 or any(not isfinite(z) for z in z_by_h):
        msg = "H selection requires all five finite diagnostics in H=0..4 order"
        raise ValueError(msg)
    return next((shift for shift, z in enumerate(z_by_h) if abs(z) <= critical_z), None)
