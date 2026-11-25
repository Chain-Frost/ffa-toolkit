"""Independent GEV L/LH moment estimators for WP04 (not legacy-qualified).

ARR Book 3, Chapter 2, equations 3.2.69-3.2.77 and Table 3.2.4;
Wang (1997). Only complete uncensored annual-maximum records are admitted.
This internal module does not implement Flike's optimized-H choice.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import comb, exp, expm1, fsum, isfinite, lgamma, log, pi
from random import Random

# Euler-Mascheroni constant, not Euler's number.
_EULER_GAMMA = 0.5772156649015329
_MIN_SHAPE = -0.999999
_MAX_SHAPE = 8.0


def _comb(n: int, k: int) -> int:
    """The zero convention for inadmissible combinatorial indices."""
    return comb(n, k) if 0 <= k <= n else 0


def _validate_shift(shift: int) -> None:
    if type(shift) is not int or shift not in range(5):
        msg = "GEV LH shift must be one of H=0, 1, 2, 3, 4"
        raise ValueError(msg)


def _sorted_sample(values: Iterable[float]) -> list[float]:
    sample = sorted(float(value) for value in values)
    if not sample:
        msg = "at least one exact annual maximum is required"
        raise ValueError(msg)
    if not all(isfinite(value) for value in sample):
        msg = "exact annual maxima must be finite"
        raise ValueError(msg)
    return sample


def sample_lh_moments(values: Iterable[float], *, shift: int = 0, order: int = 4) -> tuple[float, ...]:
    """Unbiased Wang (1997) order-statistic LH estimators, 1 <= order <= 4.

    The H=0 case is the usual Hosking sample L-moments. The input must contain
    complete, exact, equally weighted observations; no flags are discarded.
    """
    _validate_shift(shift)
    if type(order) is not int or not 1 <= order <= 4:
        msg = "LH moment order must be an integer from 1 to 4"
        raise ValueError(msg)
    sample = _sorted_sample(values)
    n = len(sample)
    if n < shift + order:
        msg = f"at least {shift + order} exact observations are required for H={shift}, order={order}"
        raise ValueError(msg)
    # Expand each order statistic about the smallest observation and express
    # all higher LH moments in adjacent sorted spacings.  The signed binomial
    # counts below are exact integers; only the final weighted gaps are floats.
    # For r=2 every weight is nonnegative, so a mathematically zero LH scale
    # remains exactly zero even for tied, nonconstant records.
    gaps = [sample[i] - sample[i - 1] for i in range(1, n)]
    result: list[float] = []
    for r in range(1, order + 1):
        subset_size = shift + r
        denominator = _comb(n, subset_size)
        if r == 1:
            anchor = sample[0]
            result.append(
                anchor
                + fsum(
                    (value - anchor) * (_comb(i, shift) / denominator) for i, value in enumerate(sample) if i >= shift
                )
            )
            continue
        # A gap after i observations contributes only when 1..r-1 of the
        # subset's observations lie above the gap.  The inner sum is the exact
        # partial alternating-binomial coefficient of that event.
        terms: list[float] = []
        for i, gap in enumerate(gaps, start=1):
            if gap == 0:
                continue
            signed_count = sum(
                (-1) ** (k - 1) * _comb(r - 2, k - 1) * _comb(n - i, k) * _comb(i, subset_size - k) for k in range(1, r)
            )
            terms.append(gap * (signed_count / (r * denominator)))
        result.append(fsum(terms))
    if not all(isfinite(value) for value in result):
        msg = "sample LH moments are not numerically representable"
        raise ValueError(msg)
    return tuple(result)


def _shape_transform(k: float, count: int) -> float:
    """Compute [1 - count**(-k)] / k with a stable Gumbel limit."""
    l_count = log(count)
    return -expm1(-k * l_count) / k if k else l_count


def _unit_gev_lh(shape: float, shift: int) -> tuple[float, float, float, float]:
    """First four GEV LH moments at location=0, scale=1 (ARR Table 3.2.4)."""
    if not isfinite(shape) or shape <= -1:
        msg = "GEV shape must be finite and greater than -1 for finite LH moments"
        raise ValueError(msg)
    try:
        gamma_value = exp(lgamma(1 + shape))
    except OverflowError as exc:
        msg = "GEV shape makes the population LH moments numerically unrepresentable"
        raise ValueError(msg) from exc
    f1, f2, f3, f4 = (_shape_transform(shape, shift + j) for j in range(1, 5))
    # The expansion avoids catastrophic cancellation in (1-Gamma(1+k))/k.
    if abs(shape) < 1e-6:
        u = _EULER_GAMMA + log(shift + 1)
        first = u - shape * (u * u / 2 + pi * pi / 12)
    else:
        first = -expm1(lgamma(1 + shape)) / shape + gamma_value * f1
    second = gamma_value * (shift + 2) / 2 * (f2 - f1)
    third = gamma_value * (shift + 3) / 6 * ((shift + 4) * f3 - 2 * (shift + 3) * f2 + (shift + 2) * f1)
    fourth = (
        gamma_value
        * (shift + 4)
        / 24
        * (
            (shift + 6) * (shift + 5) * f4
            - 3 * (shift + 5) * (shift + 4) * f3
            + 3 * (shift + 4) * (shift + 3) * f2
            - (shift + 3) * (shift + 2) * f1
        )
    )
    if not all(isfinite(value) for value in (first, second, third, fourth)) or second <= 0:
        msg = "GEV population LH moments are not numerically representable"
        raise ValueError(msg)
    return first, second, third, fourth


def theoretical_gev_lh_moments(
    location: float, scale: float, shape: float, *, shift: int = 0
) -> tuple[float, float, float, float]:
    """ARR GEV parameter convention: q(F)=u+a/k*(1-(-ln F)**k)."""
    _validate_shift(shift)
    if not isfinite(location) or not isfinite(scale) or scale <= 0:
        msg = "GEV location must be finite and scale must be finite and positive"
        raise ValueError(msg)
    first, second, third, fourth = _unit_gev_lh(shape, shift)
    return location + scale * first, scale * second, scale * third, scale * fourth


def _shape_from_lh_skew(skew: float, shift: int) -> float:
    """Invert the exact population LH-skew ratio without an approximate polynomial."""
    if not isfinite(skew):
        msg = "sample LH skewness must be finite"
        raise ValueError(msg)

    def ratio(k: float) -> float:
        _, second, third, _ = _unit_gev_lh(k, shift)
        return third / second

    lo, hi = _MIN_SHAPE, _MAX_SHAPE
    lo_ratio, hi_ratio = ratio(lo), ratio(hi)
    if not hi_ratio <= skew <= lo_ratio:
        msg = (
            f"sample LH skewness {skew:.8g} is outside the numerically admitted GEV H={shift} "
            f"shape interval ({_MIN_SHAPE}, {_MAX_SHAPE}); range [{hi_ratio:.8g}, {lo_ratio:.8g}]"
        )
        raise ValueError(msg)
    for _ in range(80):
        mid = (lo + hi) / 2
        if ratio(mid) > skew:
            lo = mid
        else:
            hi = mid
    shape = (lo + hi) / 2
    if abs(ratio(shape) - skew) > 5e-11:
        msg = "GEV LH shape inversion did not attain the required residual"
        raise ValueError(msg)
    return shape


@dataclass(frozen=True, slots=True)
class GEVLHFit:
    """Fixed-shift deterministic estimate; no claim of legacy output equivalence."""

    shift: int
    location: float
    scale: float
    shape: float
    sample_moments: tuple[float, ...]
    fitted_moments: tuple[float, float, float, float]
    lh_kurtosis_residual: float | None

    def quantile(self, non_exceedance_probability: float) -> float:
        """Discharge at non-exceedance F (AEP = 1-F), including the Gumbel limit."""
        probability = non_exceedance_probability
        if not isfinite(probability) or not 0 < probability < 1:
            msg = "non-exceedance probability must be finite and strictly between zero and one"
            raise ValueError(msg)
        log_reduced_variate = log(-log(probability))
        if self.shape == 0:
            return self.location - self.scale * log_reduced_variate
        return self.location - self.scale * expm1(self.shape * log_reduced_variate) / self.shape


def fit_gev_lh(values: Iterable[float], *, shift: int = 0) -> GEVLHFit:
    """Match the first three sample LH moments to GEV; retain fourth-moment misfit.

    Does not perform optimized-H selection, censor treatment or uncertainty fitting.
    """
    _validate_shift(shift)
    sample = _sorted_sample(values)
    if sample[0] == sample[-1]:
        msg = "GEV LH fit requires nonconstant exact observations"
        raise ValueError(msg)
    if len(sample) < shift + 3:
        msg = f"at least {shift + 3} exact observations are required to fit GEV at H={shift}"
        raise ValueError(msg)
    order = 4 if len(sample) >= shift + 4 else 3
    moments = sample_lh_moments(sample, shift=shift, order=order)
    first, second, third = moments[:3]
    if second <= 0:
        msg = "GEV LH fit requires positive sample LH scale (nonconstant observations)"
        raise ValueError(msg)
    shape = _shape_from_lh_skew(third / second, shift)
    unit_first, unit_second, _, _ = _unit_gev_lh(shape, shift)
    scale = second / unit_second
    location = first - scale * unit_first
    if not isfinite(location) or not isfinite(scale) or scale <= 0:
        msg = "GEV LH fit produced nonfinite or nonpositive location/scale"
        raise ValueError(msg)
    fitted = theoretical_gev_lh_moments(location, scale, shape, shift=shift)
    residual = moments[3] / second - fitted[3] / fitted[1] if order == 4 else None
    return GEVLHFit(shift, location, scale, shape, moments, fitted, residual)


def _linear_quantile(sorted_values: list[float], probability: float) -> float:
    """Quantile type 7 interpolation; not yet a verified Flike output rule."""
    index = (len(sorted_values) - 1) * probability
    lower = int(index)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[min(lower + 1, len(sorted_values) - 1)] * weight


def _finite_gev_quantiles(fitted: GEVLHFit, probabilities: tuple[float, ...], *, context: str) -> tuple[float, ...]:
    """Reject nonrepresentable draws before publishing any of a replicate."""
    try:
        quantiles = tuple(fitted.quantile(probability) for probability in probabilities)
    except OverflowError as exc:
        msg = f"{context} quantiles cannot be represented as finite discharges"
        raise ValueError(msg) from exc
    if any(not isfinite(value) for value in quantiles):
        msg = f"{context} quantiles cannot be represented as finite discharges"
        raise ValueError(msg)
    return quantiles


def _finite_bootstrap_sample(fitted: GEVLHFit, count: int, rng: Random) -> list[float]:
    """Generate one complete, finite trial before attempting a refit."""
    try:
        synthetic = [fitted.quantile(max(rng.random(), 2**-53)) for _ in range(count)]
    except OverflowError as exc:
        msg = "bootstrap synthetic quantile cannot be represented as a finite discharge"
        raise ValueError(msg) from exc
    if any(not isfinite(value) for value in synthetic):
        msg = "bootstrap synthetic quantile cannot be represented as a finite discharge"
        raise ValueError(msg)
    return synthetic


@dataclass(frozen=True, slots=True)
class FixedHBootstrap:
    """Independent parametric-bootstrap confidence limits, NOT Bayesian posterior."""

    shift: int
    seed: int
    requested: int
    successful: int
    failed: int
    confidence: float
    probabilities: tuple[float, ...]
    estimates: tuple[float, ...]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    failure_reasons: tuple[str, ...]


def bootstrap_fixed_gev_lh(
    values: Iterable[float],
    *,
    shift: int,
    probabilities: Iterable[float],
    replications: int,
    seed: int,
    confidence: float = 0.9,
) -> FixedHBootstrap:
    """ARR §2.6.4.6 complete-record bootstrap, holding H fixed in every refit.

    One attempted refit per generated sample; failures count and are not replaced.
    The interval uses type-7 empirical quantiles; no posterior probabilities.
    """
    if type(replications) is not int or replications < 1:
        msg = "bootstrap replications must be a positive integer"
        raise ValueError(msg)
    if type(seed) is not int:
        msg = "bootstrap seed must be an integer"
        raise ValueError(msg)
    if not isfinite(confidence) or not 0 < confidence < 1:
        msg = "bootstrap confidence must lie strictly between zero and one"
        raise ValueError(msg)
    p = tuple(float(value) for value in probabilities)
    if not p or any(not isfinite(value) or not 0 < value < 1 for value in p):
        msg = "bootstrap probabilities must contain finite non-exceedance fractions in (0,1)"
        raise ValueError(msg)
    sample = _sorted_sample(values)
    fitted = fit_gev_lh(sample, shift=shift)
    rng = Random(seed)  # noqa: S311  # Reproducible simulation; not a security RNG.
    estimates = _finite_gev_quantiles(fitted, p, context="initial GEV LH")
    draws: list[list[float]] = [[] for _ in p]
    failed_reasons: list[str] = []
    for _ in range(replications):
        try:
            synthetic = _finite_bootstrap_sample(fitted, len(sample), rng)
            refitted = fit_gev_lh(synthetic, shift=shift)
            replicate_quantiles = _finite_gev_quantiles(refitted, p, context="bootstrap refit GEV LH")
        except (ValueError, OverflowError) as exc:
            failed_reasons.append(str(exc))
            continue
        # Commit the complete replicate atomically; never append a partial
        # or infinite confidence-limit draw if one requested quantile fails.
        for bucket, quantile in zip(draws, replicate_quantiles, strict=True):
            bucket.append(quantile)
    succeeded = replications - len(failed_reasons)
    if not succeeded:
        msg = "all bootstrap refits failed; no confidence limits can be reported"
        raise ValueError(msg)
    lower_p = (1 - confidence) / 2
    upper_p = (1 + confidence) / 2
    lows = tuple(_linear_quantile(sorted(bucket), lower_p) for bucket in draws)
    highs = tuple(_linear_quantile(sorted(bucket), upper_p) for bucket in draws)
    return FixedHBootstrap(
        shift,
        seed,
        replications,
        succeeded,
        len(failed_reasons),
        confidence,
        p,
        estimates,
        lows,
        highs,
        tuple(failed_reasons),
    )
