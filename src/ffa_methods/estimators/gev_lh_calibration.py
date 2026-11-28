"""Coefficient-free finite-record simulation of Wang's LH ratio covariance."""

from dataclasses import dataclass
from math import comb, fsum, isfinite, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.special import stdtrit

from ffa_methods.estimators.gev_lh import GEVLHFit, fit_gev_lh, lh_gap_weight
from ffa_methods.estimators.gev_lh_diagnostics import (
    LHConditionalVariance,
    require_resolved_lh_residual,
    select_lowest_passing_h,
    standardized_lh_z,
)

_BATCHES = 20
_RECORDS_PER_BATCH = 10_000


class LHDomainError(ValueError):
    """The complete candidate set is outside the calibration domain."""


class LHPrecisionError(ValueError):
    """Simulation precision is insufficient to admit calibration or selection."""


class LHNumericalError(ValueError):
    """A candidate fit or calibration calculation failed."""


def _gap_weights(count: int, shift: int) -> NDArray[np.float64]:
    weights: list[list[float]] = []
    for order in (2, 3, 4):
        size = shift + order
        denominator = comb(count, size)
        weights.append([lh_gap_weight(count, size, order, i, denominator) for i in range(1, count)])
    return np.asarray(weights, dtype=np.float64).T


def _simulate_pairs(
    rng: np.random.Generator, count: int, shape: float, weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    uniforms = rng.random((_RECORDS_PER_BATCH, count))
    reduced = np.log(-np.log(np.maximum(uniforms, 2.0**-53)))
    peaks = -reduced if shape == 0 else -np.expm1(shape * reduced) / shape
    peaks.sort(axis=1)
    moments = (peaks[:, 1:] - peaks[:, :-1]) @ weights
    if not np.isfinite(moments).all() or (moments[:, 0] <= 0).any():
        msg = "calibration generated nonfinite moments or nonpositive LH scale"
        raise ValueError(msg)
    return moments[:, 1:] / moments[:, :1]


def _covariance(ratios: NDArray[np.float64]) -> LHConditionalVariance:
    """Centered regression for bounded simulated ratios, including the pooled ensemble."""
    count = len(ratios)
    skew = ratios[:, 0] - ratios[0, 0]
    kurtosis = ratios[:, 1] - ratios[0, 1]
    skew -= skew.mean()
    kurtosis -= kurtosis.mean()
    ss3 = float((skew * skew).sum())
    ss4 = float((kurtosis * kurtosis).sum())
    cross = float((skew * kurtosis).sum())
    if not all(isfinite(value) for value in (ss3, ss4, cross)) or min(ss3, ss4) <= 0:
        msg = "simulated LH pairs have invalid marginal covariance"
        raise ValueError(msg)
    residual = kurtosis - (cross / ss3) * skew
    variance = float((residual * residual).sum()) / (count - 1)
    require_resolved_lh_residual(variance, ss4 / (count - 1))
    return LHConditionalVariance(count, ss3 / (count - 1), ss4 / (count - 1), cross / (count - 1), variance)


@dataclass(frozen=True, slots=True)
class FittedLHCalibration:
    covariance: LHConditionalVariance
    variance_lower: float
    variance_upper: float
    seed: int


def calibrate_fitted_lh(count: int, shape: float, shift: int, *, seed: int) -> FittedLHCalibration:
    """Calibrate at fitted shape using standardized GEV records and independent batches."""
    if not 20 <= count <= 60 or not -0.8 <= shape <= 0.6 or shift not in range(5):
        msg = "provisional LH calibration requires n=20..60, ARR shape k in [-0.8, 0.6] and H=0..4"
        raise LHDomainError(msg)
    rng = np.random.Generator(np.random.PCG64(seed))
    weights = _gap_weights(count, shift)
    batches = np.empty((_BATCHES, _RECORDS_PER_BATCH, 2), dtype=np.float64)
    for batch in batches:
        batch[:] = _simulate_pairs(rng, count, shape, weights)
    variances = [_covariance(batch) for batch in batches]
    covariance = _covariance(batches.reshape(-1, 2))
    mean = fsum(estimate.conditional_variance for estimate in variances) / _BATCHES
    se = sqrt(fsum((estimate.conditional_variance - mean) ** 2 for estimate in variances) / (_BATCHES * (_BATCHES - 1)))
    # Allocate 1% across five candidates for the approximate batch-means intervals.
    half_width = float(stdtrit(_BATCHES - 1, 1 - 0.01 / 10)) * se + abs(mean - covariance.conditional_variance)
    lower = covariance.conditional_variance - half_width
    upper = covariance.conditional_variance + half_width
    relative_half_width = half_width / covariance.conditional_variance
    if not 0 < lower < upper or not isfinite(upper) or relative_half_width > 0.02:
        msg = f"LH calibration failed the 2% relative variance precision target (half-width {relative_half_width:.3%})"
        raise LHPrecisionError(msg)
    return FittedLHCalibration(covariance, lower, upper, seed)


@dataclass(frozen=True, slots=True)
class LHCandidate:
    fit: GEVLHFit
    calibration: FittedLHCalibration
    z: float


def evaluate_candidate_h(
    values: tuple[float, ...], *, critical_z: float, seed: int
) -> tuple[tuple[LHCandidate, ...], int | None]:
    """Evaluate all five fitted nulls and reject a precision-sensitive decision."""
    if not isfinite(critical_z) or critical_z <= 0 or type(seed) is not int or seed < 0:
        msg = "selection requires a finite positive critical z and a nonnegative integer seed"
        raise ValueError(msg)
    if not 20 <= len(values) <= 60:
        msg = "selected-H domain refusal: calibration requires 20 to 60 exact annual maxima"
        raise LHDomainError(msg)
    fits: list[GEVLHFit] = []
    for shift in range(5):
        try:
            fits.append(fit_gev_lh(values, shift=shift))
        except (ValueError, ArithmeticError) as exc:
            msg = f"H={shift}: selected-H numerical failure during fit: {exc}"
            raise LHNumericalError(msg) from exc
    outside = ", ".join(f"H={fit.shift} (k={fit.shape:.17g})" for fit in fits if not -0.8 <= fit.shape <= 0.6)
    if outside:
        msg = f"selected-H domain refusal: all five fitted shapes must be in [-0.8, 0.6]; outside: {outside}"
        raise LHDomainError(msg)
    candidates: list[LHCandidate] = []
    for fit in fits:
        try:
            calibration = calibrate_fitted_lh(len(values), fit.shape, fit.shift, seed=seed + fit.shift)
            sample_kurtosis = fit.sample_moments[3] / fit.sample_moments[1]
            fitted_kurtosis = fit.fitted_moments[3] / fit.fitted_moments[1]
            z = standardized_lh_z(sample_kurtosis, fitted_kurtosis, calibration.covariance)
        except LHPrecisionError as exc:
            msg = f"H={fit.shift}: selected-H calibration precision refusal: {exc}"
            raise LHPrecisionError(msg) from exc
        except (ValueError, ArithmeticError) as exc:
            msg = f"H={fit.shift}: selected-H numerical failure during calibration: {exc}"
            raise LHNumericalError(msg) from exc
        residual = abs(sample_kurtosis - fitted_kurtosis)
        if residual / sqrt(calibration.variance_upper) <= critical_z < residual / sqrt(calibration.variance_lower):
            msg = f"H={fit.shift}: selected-H calibration precision refusal: uncertainty straddles the critical z"
            raise LHPrecisionError(msg)
        candidates.append(LHCandidate(fit, calibration, z))
    return tuple(candidates), select_lowest_passing_h(
        tuple(candidate.z for candidate in candidates), critical_z=critical_z
    )
