"""Independent H=0 L-moment fits for the remaining four continuous families.

ARR Book 3 (2019), section 2.6.4, Table 3.2.3; Hosking (1990);
Hosking and Wallis (1987, 1997) for GPD and Pearson III. These are
complete, equally weighted, uncensored mathematical fits, not target-build
Flike defaults, supported estimator combinations or a POT workflow.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import comb, e, exp, frexp, fsum, isfinite, lgamma, log, log1p, pi, sqrt
from typing import Generic, Literal, TypeAlias, TypeVar

from scipy.optimize import brentq
from scipy.special import betainc, erfinv

from ffa_methods.distributions import GeneralizedPareto, Gumbel, LogCoordinates, Lognormal, LogPearsonIII
from ffa_methods.estimators.gev_lh import sample_lh_moments

_EULER_GAMMA = 0.5772156649015329
_MIN_GAMMA_SHAPE = 0.01
_MAX_GAMMA_SHAPE = 1e8

OrdinaryFamily: TypeAlias = Literal["lognormal", "lp3", "gumbel", "generalized-pareto"]
OrdinaryKernel: TypeAlias = Lognormal | LogPearsonIII | Gumbel | GeneralizedPareto
KernelT = TypeVar("KernelT", bound=OrdinaryKernel)


@dataclass(frozen=True, slots=True)
class OrdinaryLMomentFit(Generic[KernelT]):
    """An internal mathematical H=0 fit, with the fitting coordinate labelled."""

    family: OrdinaryFamily
    moment_coordinates: Literal["discharge", "ln(Q/reference_discharge)"]
    sample_moments: tuple[float, ...]
    distribution: KernelT

    def quantile(self, non_exceedance_probability: float) -> float:
        """Finite discharge at 0<F<1; AEP is 1-F."""
        probability = non_exceedance_probability
        if not isfinite(probability) or not 0.0 < probability < 1.0:
            msg = "non-exceedance probability must be finite and strictly between zero and one"
            raise ValueError(msg)
        try:
            result = float(self.distribution.ppf(probability))
        except OverflowError as exc:
            msg = "fitted quantile is not a representable finite discharge"
            raise ValueError(msg) from exc
        if not isfinite(result) or (self.family in ("lognormal", "lp3") and result <= 0):
            msg = "fitted quantile is not a representable finite discharge"
            raise ValueError(msg)
        return result


def _sample(values: Iterable[float], *, positive: bool, minimum_count: int) -> list[float]:
    try:
        sample = [float(value) for value in values]
    except OverflowError as exc:
        msg = "observations are not representable as finite binary64 values"
        raise ValueError(msg) from exc
    if len(sample) < minimum_count:
        msg = f"at least {minimum_count} exact observations are required"
        raise ValueError(msg)
    if not all(isfinite(value) for value in sample):
        msg = "observations must be finite"
        raise ValueError(msg)
    if positive and any(value <= 0.0 for value in sample):
        msg = "log-flow fitting requires strictly positive exact discharges"
        raise ValueError(msg)
    return sample


def _reference(value: float) -> float:
    if not isfinite(value) or value <= 0.0:
        msg = "reference_discharge must be finite and positive"
        raise ValueError(msg)
    return value


def _lmoments(sample: list[float], order: int) -> tuple[float, ...]:
    moments = sample_lh_moments(sample, shift=0, order=order)
    if moments[1] <= 0.0:
        msg = "ordinary L-moment fit requires positive second L-moment"
        raise ValueError(msg)
    return moments


def fit_gumbel_lmoments(values: Iterable[float]) -> OrdinaryLMomentFit[Gumbel]:
    """H=0 Gumbel: L1=u+gamma*a and L2=a*ln(2)."""
    sample = _sample(values, positive=False, minimum_count=2)
    moments = _lmoments(sample, 2)
    scale = moments[1] / log(2.0)
    location = moments[0] - _EULER_GAMMA * scale
    if not isfinite(scale) or not isfinite(location) or scale <= 0.0:
        msg = "Gumbel L-moment parameters are not representable"
        raise ValueError(msg)
    kernel = Gumbel(location, log(scale))
    return OrdinaryLMomentFit("gumbel", "discharge", moments, kernel)


def fit_lognormal_lmoments(
    values: Iterable[float], *, reference_discharge: float = 1.0
) -> OrdinaryLMomentFit[Lognormal]:
    """Zero-location LN2, matching L1 and L2 of *discharge*, not log discharge.

    For a lognormal law, L2/L1 = erf(sigma_ln/2). No shift is fitted.
    """
    reference = _reference(reference_discharge)
    sample = _sample(values, positive=True, minimum_count=2)
    moments = _lmoments(sample, 2)
    first, second = moments
    ratio = second / first
    if not isfinite(ratio) or not 0.0 < ratio < 1.0:
        msg = "positive sample has no representable two-parameter lognormal L-moment fit"
        raise ValueError(msg)
    log_sd = 2.0 * float(erfinv(ratio))
    log_mean = log(first) - log(reference) - log_sd * log_sd / 2.0
    if not isfinite(log_mean) or not isfinite(log_sd) or log_sd <= 0.0:
        msg = "lognormal L-moment parameters are not representable"
        raise ValueError(msg)
    kernel = Lognormal(LogCoordinates(log_mean, log(log_sd), base=e, reference_discharge=reference))
    return OrdinaryLMomentFit("lognormal", "discharge", moments, kernel)


def _gamma_lskew(shape: float) -> float:
    """Exact Pearson III positive-skew L3/L2 from the regularized beta law."""
    return 6.0 * float(betainc(shape, 2.0 * shape, 1.0 / 3.0)) - 3.0


def gamma_sd_l2_multiplier(shape: float) -> float:
    """Stable Pearson III SD/L2, including the large-shape gamma ratio.

    sqrt(pi*a)*Gamma(a)/Gamma(a+1/2) uses an asymptotic expansion when
    lgamma(a)-lgamma(a+1/2) would cancel most significant digits.
    """
    if shape >= 10_000.0:
        inverse = 1.0 / shape
        correction = 1.0 + inverse / 8.0 + inverse**2 / 128.0 - 5.0 * inverse**3 / 1024.0
        return sqrt(pi) * correction
    return exp(0.5 * log(pi) + 0.5 * log(shape) + lgamma(shape) - lgamma(shape + 0.5))


def _pearson_shape_from_lskew(tau3: float) -> float:
    """Bounded exact gamma-shape root; no polynomial surrogate or clipping."""
    absolute = abs(tau3)
    smallest, largest = _gamma_lskew(_MAX_GAMMA_SHAPE), _gamma_lskew(_MIN_GAMMA_SHAPE)
    if not smallest <= absolute <= largest:
        msg = (
            "LP3 sample log-space L-skew is outside the admitted exact gamma-shape "
            f"interval ({_MIN_GAMMA_SHAPE}, {_MAX_GAMMA_SHAPE}); "
            "no numerical or legacy fallback is asserted"
        )
        raise ValueError(msg)
    root = brentq(
        lambda log_shape: _gamma_lskew(exp(log_shape)) - absolute,
        log(_MIN_GAMMA_SHAPE),
        log(_MAX_GAMMA_SHAPE),
        xtol=1e-12,
        rtol=1e-12,
        maxiter=150,
    )
    shape = exp(float(root))
    if not isfinite(shape) or abs(_gamma_lskew(shape) - absolute) > 1e-11:
        msg = "LP3 L-skew inversion did not attain a representable residual"
        raise ValueError(msg)
    return shape


def stable_log_ratio(numerator: float, denominator: float) -> float:
    """Return ln(numerator/denominator) without cancellation or lost symmetry.

    Both operands must be finite and positive. For the larger/smaller
    quotient use log1p of its *positive difference* near unity, ordinary
    log for normal ratios, and frexp exponent decomposition if division
    overflows. Apply the sign last, so a reciprocal pair is antisymmetric.
    """
    if numerator == denominator:
        return 0.0
    upper, lower = (numerator, denominator) if numerator > denominator else (denominator, numerator)
    difference = upper - lower
    if difference <= lower:
        magnitude = log1p(difference / lower)
    else:
        quotient = upper / lower
        if isfinite(quotient):
            magnitude = log(quotient)
        else:
            upper_mantissa, upper_exponent = frexp(upper)
            lower_mantissa, lower_exponent = frexp(lower)
            magnitude = (upper_exponent - lower_exponent) * log(2.0) + log(upper_mantissa / lower_mantissa)
    return magnitude if numerator > denominator else -magnitude


def _exact_log_mirror_symmetry(values: list[float]) -> bool:
    """Prove zero odd log L-moments for a mirror-symmetric exact sample.

    If every opposite order-statistic pair has the same positive product,
    the logarithms are exactly symmetric around half the log product. For
    odd n the central order statistic must square to that product. At n=3
    this identity is also necessary for the third L-moment to be zero.
    Use exact binary-rational source values; never clip a small L-skew.
    """
    ordered = sorted(Fraction(value) for value in values)
    count = len(ordered)
    pair_product = ordered[0] * ordered[-1]
    return all(ordered[i] * ordered[-1 - i] == pair_product for i in range(1, count // 2)) and (
        count % 2 == 0 or ordered[count // 2] ** 2 == pair_product
    )


def _exact_log_l3_zero_for_small_sample(values: list[float]) -> bool:
    """Prove sample log L3=0 using the exact order-statistic coefficients.

    For 3 <= n <= 16, the unnormalised coefficient of sorted log(Q_i) is
    C(i,2) - 2*i*(n-1-i) + C(n-1-i,2). Consequently L3(log Q)=0
    precisely when the products of positive/negative source powers match.
    The bounded sample size prevents huge intermediate rational powers.
    """
    ordered = sorted(Fraction(value) for value in values)
    if not 3 <= len(ordered) <= 16:
        msg = "exact general log L3 identity is limited to 3..16 observations"
        raise ValueError(msg)
    size = len(ordered)
    positive, negative = Fraction(1), Fraction(1)
    for index, value in enumerate(ordered):
        weight = comb(index, 2) - 2 * index * (size - 1 - index) + comb(size - 1 - index, 2)
        if weight > 0:
            positive *= value**weight
        elif weight < 0:
            negative *= value ** (-weight)
    return positive == negative


def fit_lp3_lmoments(values: Iterable[float], *, reference_discharge: float = 1.0) -> OrdinaryLMomentFit[LogPearsonIII]:
    """H=0 Pearson III L1-L3 of ln(Q/reference), then log-space LP3 kernel.

    At exactly zero sample L-skew the Pearson III law is its Normal limit.
    For a nonzero L-skew an exact incomplete-beta root yields the gamma shape.
    """
    reference = _reference(reference_discharge)
    sample = _sample(values, positive=True, minimum_count=3)
    # Center the logarithms on an observed median discharge BEFORE
    # evaluating L2/L3. Taking absolute logs first and then subtracting
    # a reference can introduce artificial L-skew under a change of units
    # (e.g. [.25, 1, 4] versus [.5, 2, 8]). Positive ratios preserve
    # reciprocal geometric pairs exactly when their inputs allow it.
    anchor = sorted(sample)[len(sample) // 2]
    log_values = [stable_log_ratio(value, anchor) for value in sample]
    centered_mean, second, third = _lmoments(log_values, 3)
    # A source series symmetric in *exact binary-rational log space* has
    # theoretical L3=0. Floating-point weighted sums can leave a tiny
    # nonzero residual, especially for even-sized series. Force zero only
    # when this independent exact identity proves it, not by threshold.
    exact_log_symmetry = _exact_log_mirror_symmetry(sample)
    exact_zero_l3 = exact_log_symmetry or (len(sample) <= 16 and _exact_log_l3_zero_for_small_sample(sample))
    if exact_zero_l3:
        third = 0.0
    reference_offset = stable_log_ratio(anchor, reference)
    first = fsum((centered_mean, reference_offset))
    if not isfinite(first):
        msg = "LP3 log-space mean is not numerically representable"
        raise ValueError(msg)
    moments = (first, second, third)
    tau3 = third / second
    if not isfinite(tau3):
        msg = "LP3 sample L-skew is not representable"
        raise ValueError(msg)
    if tau3 == 0.0:
        if not exact_zero_l3:
            msg = "LP3 sample cannot resolve L-skew at binary64 precision: computed zero is unverified"
            raise ValueError(msg)
        sd = second * sqrt(pi)
        skew = 0.0
    else:
        gamma_shape = _pearson_shape_from_lskew(tau3)
        # L2 = |beta| Gamma(a+1/2) / (sqrt(pi) Gamma(a)).
        sd = second * gamma_sd_l2_multiplier(gamma_shape)
        skew = (2.0 if tau3 > 0.0 else -2.0) / sqrt(gamma_shape)
    if not isfinite(sd) or sd <= 0.0 or not isfinite(skew):
        msg = "LP3 L-moment parameters are not representable"
        raise ValueError(msg)
    if skew != 0.0:
        edge = first - 2.0 * sd / skew
        if not isfinite(edge):
            msg = "LP3 L-moment support endpoint is not representable"
            raise ValueError(msg)
        # Compare in the same centered coordinates used for L2/L3.
        # Do not reconstruct an absolute-log endpoint and lose precision.
        centered_edge = centered_mean - 2.0 * sd / skew
        if (skew > 0.0 and min(log_values) < centered_edge) or (skew < 0.0 and max(log_values) > centered_edge):
            msg = "LP3 L-moment fit has observations outside its fitted support"
            raise ValueError(msg)
    kernel = LogPearsonIII(LogCoordinates(first, log(sd), reference_discharge=reference), skew)
    return OrdinaryLMomentFit("lp3", "ln(Q/reference_discharge)", moments, kernel)


def _gp_stable_moment_combinations(sample: list[float]) -> tuple[float, float, float, float]:
    """Gap sums of L2-L3, L2+L3 and the offset in L1-L2 without cancellation.

    The exact integer weights on the positive adjacent sorted gaps are
    respectively 2*i*(n-i)*(n-i-1), 2*i*(n-i)*(i-1) and
    (n-i)*(n-i-1), with the corresponding n-dependent denominators.
    """
    ordered = sorted(sample)
    n = len(ordered)
    denominator = n * (n - 1) * (n - 2)
    differences: list[float] = []
    sums: list[float] = []
    location_terms: list[float] = []
    for i in range(1, n):
        gap = ordered[i] - ordered[i - 1]
        if gap == 0.0:
            continue
        common = 2 * i * (n - i)
        differences.append(gap * (common * (n - i - 1) / denominator))
        sums.append(gap * (common * (i - 1) / denominator))
        location_terms.append(gap * ((n - i) * (n - i - 1) / (n * (n - 1))))
    l2_minus_l3 = fsum(differences)
    l2_plus_l3 = fsum(sums)
    l1_minus_l2_gap = fsum(location_terms)
    if not all(isfinite(v) for v in (l2_minus_l3, l2_plus_l3, ordered[0], l1_minus_l2_gap)):
        msg = "GP L-moment gap combinations are not numerically representable"
        raise ValueError(msg)
    # Keep the minimum and its tiny positive gap correction separate.
    # For fixed location==minimum, adding them before subtraction could
    # erase the entire small correction (e.g., -1 + 7.4e-17 rounds -1).
    return l2_minus_l3, l2_plus_l3, ordered[0], l1_minus_l2_gap


def fit_gp_lmoments(values: Iterable[float], *, location: float | None = None) -> OrdinaryLMomentFit[GeneralizedPareto]:
    """H=0 three-parameter GP, or a two-parameter GP with explicit fixed location.

    Hosking's kappa = -EVT xi = -SciPy genpareto.c. A fixed location
    consumes L1/L2; a free location consumes L1/L2/L3.
    """
    if location is not None and not isfinite(location):
        msg = "fixed generalized Pareto location must be finite"
        raise ValueError(msg)
    sample = _sample(values, positive=False, minimum_count=2 if location is not None else 3)
    moments = _lmoments(sample, 2 if location is not None else 3)
    _, second = moments[:2]
    # Near the finite-mean boundary xi=1, L2 and L3 may agree to many
    # digits. Preserve the small L2-L3 gap rather than subtract rounded
    # values or computing 1+kappa after rounding kappa to -1.
    if len(sample) >= 3:
        difference, total, anchor, l1_minus_l2_gap = _gp_stable_moment_combinations(sample)
    else:
        difference, total, anchor, l1_minus_l2_gap = 0.0, 0.0, min(sample), 0.0
    if location is None:
        if difference <= 0.0 or total <= 0.0:
            msg = "GP sample L-skew must be strictly between -1 and 1"
            raise ValueError(msg)
        one_plus_kappa = 2.0 * difference / total
        fitted_location = fsum((anchor, l1_minus_l2_gap, -second * one_plus_kappa))
        scale = second * one_plus_kappa * (1.0 + one_plus_kappa)
    else:
        fitted_location = location
        excess_minus_l2 = fsum((anchor - location, l1_minus_l2_gap))
        one_plus_kappa = excess_minus_l2 / second
        excess = fsum((second, excess_minus_l2))
        scale = excess * one_plus_kappa
    xi = 1.0 - one_plus_kappa
    if not all(isfinite(value) for value in (fitted_location, scale, xi)) or scale <= 0.0 or xi >= 1.0:
        msg = "GP L-moment scale/shape/location is invalid or not representable"
        raise ValueError(msg)
    kernel = GeneralizedPareto(fitted_location, log(scale), xi)
    lower, upper = kernel.support()
    if lower > min(sample):
        msg = "GP L-moment fit has observations below its lower support endpoint"
        raise ValueError(msg)
    if upper < max(sample) or (xi < 0.0 and (max(sample) - fitted_location) / exp(kernel.log_scale) > -1.0 / xi):
        msg = "GP L-moment fit has observations above its upper support endpoint"
        raise ValueError(msg)
    return OrdinaryLMomentFit("generalized-pareto", "discharge", moments, kernel)
