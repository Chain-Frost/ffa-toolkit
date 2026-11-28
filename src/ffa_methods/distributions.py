"""Independent continuous probability kernels for the legacy Flike family set.

This module defines mathematical coordinates, not unverified Flike defaults or fits.
Probability arguments are fractions in [0, 1], not AEP percentages. The SciPy
GEV shape is the negative of the standard extreme-value xi; GP uses the same xi.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Protocol, cast

from scipy.stats import gamma, genextreme, genpareto, gumbel_r, norm


class _FrozenContinuous(Protocol):
    """Only the SciPy frozen-distribution operations needed by the kernels."""

    def support(self) -> tuple[float, float]: ...
    def logpdf(self, value: float) -> float: ...
    def logcdf(self, value: float) -> float: ...
    def logsf(self, value: float) -> float: ...
    def cdf(self, value: float) -> float: ...
    def sf(self, value: float) -> float: ...
    def ppf(self, probability: float) -> float: ...
    def isf(self, probability: float) -> float: ...


def _finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        message = f"{name} must be finite"
        raise ValueError(message)


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        message = f"{name} must be finite and positive"
        raise ValueError(message)


def _scale(log_scale: float) -> float:
    _finite(log_scale, "log_scale")
    try:
        value = math.exp(log_scale)
    except OverflowError as exc:
        message = "log_scale gives an unrepresentable scale"
        raise ValueError(message) from exc
    _positive(value, "scale")
    return value


def _probability(value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        message = "probability must be finite and between 0 and 1"
        raise ValueError(message)
    return value


def _point(value: float) -> float:
    if math.isnan(value):
        message = "prediction or observation must not be NaN"
        raise ValueError(message)
    return value


def _exp_discharge(log_ratio: float, reference: float) -> float:
    if log_ratio == math.inf:
        return math.inf
    if log_ratio == -math.inf:
        return 0.0
    log_discharge = math.log(reference) + log_ratio
    if log_discharge > math.log(sys.float_info.max):
        return math.inf
    if log_discharge < -745.0:
        return 0.0
    return math.exp(log_discharge)


@dataclass(frozen=True, slots=True)
class LogCoordinates:
    """Mean and ln(standard deviation) of log_base(Q / reference).

    reference_discharge is expressed in the same physical units as Q.
    The outer logarithm of the SD is always natural even when base != e.
    Bases > 1 make the log transform monotone increasing.
    """

    mean: float
    log_sd: float
    base: float = math.e
    reference_discharge: float = 1.0

    def __post_init__(self) -> None:
        _finite(self.mean, "mean")
        _positive(self.base, "base")
        if self.base <= 1.0:
            message = "base must exceed one"
            raise ValueError(message)
        _positive(self.reference_discharge, "reference_discharge")
        _scale(self.log_sd)
        # A valid source SD can still become unrepresentable in natural-log space.
        _scale(self.natural_log_sd)
        _finite(self.natural_mean, "natural_mean")

    @property
    def natural_mean(self) -> float:
        return self.mean * math.log(self.base)

    @property
    def natural_log_sd(self) -> float:
        return self.log_sd + math.log(math.log(self.base))

    def with_base(self, base: float) -> LogCoordinates:
        _positive(base, "base")
        if base <= 1.0:
            message = "base must exceed one"
            raise ValueError(message)
        ln_base = math.log(base)
        return LogCoordinates(
            mean=self.natural_mean / ln_base,
            log_sd=self.natural_log_sd - math.log(ln_base),
            base=base,
            reference_discharge=self.reference_discharge,
        )

    def with_reference(self, reference_discharge: float) -> LogCoordinates:
        _positive(reference_discharge, "reference_discharge")
        offset = (math.log(self.reference_discharge) - math.log(reference_discharge)) / math.log(self.base)
        return LogCoordinates(
            mean=self.mean + offset,
            log_sd=self.log_sd,
            base=self.base,
            reference_discharge=reference_discharge,
        )

    def in_rescaled_discharge_units(self, factor: float) -> LogCoordinates:
        """For Q_new = factor * Q_old; the reference scales with Q."""
        _positive(factor, "factor")
        return LogCoordinates(self.mean, self.log_sd, self.base, self.reference_discharge * factor)

    def log_abs_det_natural_wrt_source(self) -> float:
        """Log |d(mu_ln, ln(sd_ln)) / d(mu_base, ln(sd_base))|.

        Use with the *joint* parameter density; skew is unchanged.
        """
        return math.log(math.log(self.base))

    @property
    def log_abs_det_sd_wrt_log_sd(self) -> float:
        """Log |d(sd_base)/d(ln(sd_base))| for prior coordinate changes."""
        return self.log_sd


class _ContinuousKernel:
    """Scalar evaluation contract. Outside-support queries are not parameter errors."""

    def _frozen(self) -> _FrozenContinuous:
        raise NotImplementedError

    def _to_statistic(self, discharge: float) -> float:
        return discharge

    def _from_statistic(self, statistic: float) -> float:
        return statistic

    def _log_jacobian(self, _discharge: float) -> float:
        return 0.0

    @property
    def _is_log_flow(self) -> bool:
        return False

    def support(self) -> tuple[float, float]:
        lower, upper = self._frozen().support()
        return self._from_statistic(float(lower)), self._from_statistic(float(upper))

    def logpdf(self, discharge: float) -> float:
        discharge = _point(discharge)
        if math.isinf(discharge) or (self._is_log_flow and discharge <= 0.0):
            return -math.inf
        return float(self._frozen().logpdf(self._to_statistic(discharge))) + self._log_jacobian(discharge)

    def logcdf(self, discharge: float) -> float:
        discharge = _point(discharge)
        if self._is_log_flow and discharge <= 0.0:
            return -math.inf
        return float(self._frozen().logcdf(self._to_statistic(discharge)))

    def logsf(self, discharge: float) -> float:
        discharge = _point(discharge)
        if self._is_log_flow and discharge <= 0.0:
            return 0.0
        return float(self._frozen().logsf(self._to_statistic(discharge)))

    def cdf(self, discharge: float) -> float:
        discharge = _point(discharge)
        if self._is_log_flow and discharge <= 0.0:
            return 0.0
        return float(self._frozen().cdf(self._to_statistic(discharge)))

    def sf(self, discharge: float) -> float:
        discharge = _point(discharge)
        if self._is_log_flow and discharge <= 0.0:
            return 1.0
        return float(self._frozen().sf(self._to_statistic(discharge)))

    def ppf(self, probability: float) -> float:
        value = float(self._frozen().ppf(_probability(probability)))
        return self._from_statistic(value)

    def isf(self, probability: float) -> float:
        value = float(self._frozen().isf(_probability(probability)))
        return self._from_statistic(value)


class _LogFlowKernel(_ContinuousKernel):
    log: LogCoordinates

    @property
    def _is_log_flow(self) -> bool:
        return True

    def _to_statistic(self, discharge: float) -> float:
        return math.log(discharge) - math.log(self.log.reference_discharge)

    def _from_statistic(self, statistic: float) -> float:
        return _exp_discharge(statistic, self.log.reference_discharge)

    def _log_jacobian(self, discharge: float) -> float:
        # Density with respect to Q rather than natural-log Q.
        return -math.log(discharge)


@dataclass(frozen=True, slots=True)
class Lognormal(_LogFlowKernel):
    log: LogCoordinates
    _normal: _FrozenContinuous = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        # Posterior predictive inversion evaluates each component repeatedly.
        distribution = norm(loc=self.log.natural_mean, scale=math.exp(self.log.natural_log_sd))
        object.__setattr__(self, "_normal", cast(_FrozenContinuous, distribution))

    def _frozen(self) -> _FrozenContinuous:
        return self._normal


def _gamma_upper_fraction(shape: float, x: float) -> float:
    """Continued-fraction correction for the upper incomplete gamma Q."""
    tiny = 1e-300
    b = x + 1.0 - shape
    if abs(b) < tiny:
        b = tiny
    c = 1.0 / tiny
    d = 1.0 / b
    total = d
    for i in range(1, 10_001):
        coeff = -i * (i - shape)
        b += 2.0
        d = coeff * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coeff / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        factor = d * c
        total *= factor
        if abs(factor - 1.0) <= 1e-14:
            return total
    message = "LP3 gamma upper-tail continuation did not converge"
    raise ArithmeticError(message)


def _gamma_lower_series(shape: float, x: float) -> float:
    """Lower gamma series with an explicit bound on the omitted positive terms.

    For x < shape + 1, the term ratios decrease. After term n the omitted
    sum is at most term_n * r_n / (1-r_n), r_n=x/(shape+n+1).
    The initial geometric ratio also gives a finite a priori iteration bound.
    """
    initial_ratio = x / (shape + 1.0)
    if not 0.0 <= initial_ratio < 1.0:
        message = "LP3 gamma lower-tail series requires x < shape + 1"
        raise ArithmeticError(message)
    tolerance = 1e-14
    max_terms = (
        1
        if initial_ratio == 0.0
        else max(1, math.ceil(math.log(tolerance * (1.0 - initial_ratio)) / math.log(initial_ratio)))
    )
    term = 1.0
    total = 1.0
    for i in range(1, max_terms + 2):
        term *= x / (shape + i)
        total += term
        next_ratio = x / (shape + i + 1.0)
        if (term * next_ratio) / (1.0 - next_ratio) <= tolerance * total:
            return total
    message = "LP3 gamma lower-tail series exceeded its geometric error bound"
    raise ArithmeticError(message)


def _log_gamma_prefactor(shape: float, x: float) -> float:
    """Evaluate shape*log(x)-x-lgamma(shape) without large-shape cancellation.

    Stirling's correction is valid at shape >= 10000; close to shape, evaluate
    log1p(delta)-delta by its alternating power series when delta is small.
    """
    if shape < 10_000.0 or not 0.9 * shape < x < 1.1 * shape:
        return shape * math.log(x) - x - math.lgamma(shape)
    delta = (x - shape) / shape
    if abs(delta) < 0.01:
        power = delta * delta
        log1pmx = -power / 2.0
        for order in range(3, 17):
            power *= delta
            log1pmx += power / order if order % 2 else -power / order
    else:
        log1pmx = math.log1p(delta) - delta
    return shape * log1pmx + 0.5 * math.log(shape / (2.0 * math.pi)) - 1.0 / (12.0 * shape) + 1.0 / (360.0 * shape**3)


# 16-point Gauss-Laguerre nodes/weights for the weight exp(-v), from the
# classical L_16 orthogonal polynomial (rounded IEEE-754 values).
# Their mathematical integration rule is independent of SciPy gamma CDF.
_LAGUERRE_16: tuple[tuple[float, float], ...] = (
    (0.087649410478927839, 0.20615171495780069),
    (0.46269632891508083, 0.33105785495088402),
    (1.1410577748312269, 0.26579577764421436),
    (2.129283645098381, 0.13629693429637774),
    (3.4370866338932067, 0.047328928694125222),
    (5.0780186145497677, 0.01129990008033945),
    (7.0703385350482346, 0.0018490709435263094),
    (9.4383143363919384, 0.00020427191530827824),
    (12.214223368866159, 1.4844586873981333e-05),
    (15.441527368781617, 6.828319330871246e-07),
    (19.180156856753136, 1.8810248410796997e-08),
    (23.515905693991908, 2.8623502429738586e-10),
    (28.578729742882139, 2.1270790332240987e-12),
    (34.583398702286622, 6.2979670025178009e-15),
    (41.940452647688332, 5.0504737000355466e-18),
    (51.701160339543321, 4.161462370372804e-22),
)


def _expm1_minus_linear(value: float) -> float:
    """Accurate exp(value)-1-value for the endpoint-integral curvature."""
    if abs(value) >= 0.1:
        return math.expm1(value) - value
    term = value * value / 2.0
    total = term
    for order in range(3, 14):
        term *= value / order
        total += term
    return total


def _gamma_lower_endpoint_quadrature(shape: float, x: float) -> float:
    """Stable large-shape lower-gamma log tail, independent of gamma CDF.

    For d=a-x>0, P(a,x)=exp(prefactor)/d times the integral against exp(-v)
    of exp(-x*(exp(-v/d)-1+v/d)). Sixteen Laguerre nodes are accurate for
    d>=4.4*sqrt(a); independent 80-digit integration verifies the regime.
    """
    delta = shape - x
    integral = math.fsum(weight * math.exp(-x * _expm1_minus_linear(-node / delta)) for node, weight in _LAGUERRE_16)
    return min(0.0, _log_gamma_prefactor(shape, x) - math.log(delta) + math.log(integral))


def _prefer_lower_quadrature(shape: float, x: float) -> bool:
    # SciPy's gamma logcdf can be finite but inaccurate for large shapes
    # below about -4.5 standard deviations; preempt this regime.
    return shape >= 100_000.0 and 0.0 < x < shape - 4.4 * math.sqrt(shape)


def _log_regularized_gamma_tail(shape: float, x: float, *, upper: bool) -> float:
    """Use log-domain incomplete-gamma formulas after a SciPy log-tail underflow.

    P uses a power series; Q uses a continued fraction. The log-prefactor
    preserves a nonzero mathematical tail even when the probability rounds to 0.
    """
    correction = _gamma_upper_fraction(shape, x) if upper else _gamma_lower_series(shape, x)
    if not math.isfinite(correction) or correction <= 0.0:
        message = "LP3 gamma logarithmic tail could not be represented"
        raise ArithmeticError(message)
    log_correction = math.log(correction) if upper else math.log(correction) - math.log(shape)
    log_prefactor = _log_gamma_prefactor(shape, x)
    return min(0.0, log_prefactor + log_correction)


@dataclass(frozen=True, slots=True)
class _GammaPearson3:
    """Gamma-transform implementation to retain both Pearson III extreme tails.

    scipy.stats.pearson3.isf can lose rare survival probabilities in the
    negative-skew case. Gamma PPF/ISF keep the required tail distinct.
    """

    mean: float
    sd: float
    skew: float

    def __post_init__(self) -> None:
        if not math.isfinite(self._edge):
            message = "LP3 gamma support endpoint is not representable"
            raise ValueError(message)
        if not math.isfinite(self._shape) or self._shape <= 0.0:
            message = "LP3 gamma shape is not representable"
            raise ValueError(message)
        _positive(self._gamma_scale, "LP3 gamma scale")

    @property
    def _shape(self) -> float:
        # Squaring skew directly can overflow before its reciprocal is taken.
        return (2.0 / abs(self.skew)) ** 2

    @property
    def _edge(self) -> float:
        return self.mean - self.sd * (2.0 / self.skew)

    @property
    def _gamma_scale(self) -> float:
        return (self.sd / 2.0) * abs(self.skew)

    def _gamma(self) -> _FrozenContinuous:
        return cast(_FrozenContinuous, gamma(self._shape))

    def _arg(self, value: float) -> float:
        sign = 1.0 if self.skew > 0.0 else -1.0
        delta = value - self._edge
        if math.isfinite(delta):
            return sign * (delta / self._gamma_scale)
        # The difference can overflow even though the scaled result is finite.
        return sign * (value / self._gamma_scale - self._edge / self._gamma_scale)

    def _value(self, arg: float) -> float:
        # Centre the transform at its mean. In the high-scale case, the
        # uncentred scale * arg can overflow while the quantile is finite.
        return self.mean + ((self.sd / 2.0) * self.skew) * (arg - self._shape)

    def support(self) -> tuple[float, float]:
        if self.skew > 0.0:
            return self._edge, math.inf
        return -math.inf, self._edge

    def logpdf(self, value: float) -> float:
        x = self._arg(value)
        if math.isfinite(x) and x > 0.0:
            return _log_gamma_prefactor(self._shape, x) - math.log(x) - math.log(self._gamma_scale)
        # Retain SciPy's support/endpoint density convention when x <= 0.
        return float(self._gamma().logpdf(x)) - math.log(self._gamma_scale)

    def logcdf(self, value: float) -> float:
        x = self._arg(value)
        upper = self.skew < 0.0
        if not upper and _prefer_lower_quadrature(self._shape, x):
            return _gamma_lower_endpoint_quadrature(self._shape, x)
        gam = self._gamma()
        result = float(gam.logsf(x) if upper else gam.logcdf(x))
        if not math.isfinite(result) and math.isfinite(x) and x > 0.0:
            return _log_regularized_gamma_tail(self._shape, x, upper=upper)
        return result

    def logsf(self, value: float) -> float:
        x = self._arg(value)
        upper = self.skew > 0.0
        if not upper and _prefer_lower_quadrature(self._shape, x):
            return _gamma_lower_endpoint_quadrature(self._shape, x)
        gam = self._gamma()
        result = float(gam.logsf(x) if upper else gam.logcdf(x))
        if not math.isfinite(result) and math.isfinite(x) and x > 0.0:
            return _log_regularized_gamma_tail(self._shape, x, upper=upper)
        return result

    def cdf(self, value: float) -> float:
        x = self._arg(value)
        if self.skew > 0.0 and _prefer_lower_quadrature(self._shape, x):
            return math.exp(_gamma_lower_endpoint_quadrature(self._shape, x))
        gam = self._gamma()
        return float(gam.cdf(x) if self.skew > 0.0 else gam.sf(x))

    def sf(self, value: float) -> float:
        x = self._arg(value)
        if self.skew < 0.0 and _prefer_lower_quadrature(self._shape, x):
            return math.exp(_gamma_lower_endpoint_quadrature(self._shape, x))
        gam = self._gamma()
        return float(gam.sf(x) if self.skew > 0.0 else gam.cdf(x))

    def ppf(self, probability: float) -> float:
        gam = self._gamma()
        arg = gam.ppf(probability) if self.skew > 0.0 else gam.isf(probability)
        return self._value(float(arg))

    def isf(self, probability: float) -> float:
        gam = self._gamma()
        arg = gam.isf(probability) if self.skew > 0.0 else gam.ppf(probability)
        return self._value(float(arg))


@dataclass(frozen=True, slots=True)
class LogPearsonIII(_LogFlowKernel):
    log: LogCoordinates
    skew: float

    def __post_init__(self) -> None:
        _finite(self.skew, "skew")
        if abs(self.skew) >= 1.6e-5:
            # Validate derived gamma parameters during construction.
            _ = _GammaPearson3(self.log.natural_mean, math.exp(self.log.natural_log_sd), self.skew)

    def _frozen(self) -> _FrozenContinuous:
        mean, sd = self.log.natural_mean, math.exp(self.log.natural_log_sd)
        if abs(self.skew) < 1.6e-5:
            return cast(_FrozenContinuous, norm(loc=mean, scale=sd))
        return _GammaPearson3(mean, sd, self.skew)


@dataclass(frozen=True, slots=True)
class Gumbel(_ContinuousKernel):
    location: float
    log_scale: float

    def __post_init__(self) -> None:
        _finite(self.location, "location")
        _scale(self.log_scale)

    def _frozen(self) -> _FrozenContinuous:
        return cast(_FrozenContinuous, gumbel_r(loc=self.location, scale=math.exp(self.log_scale)))


def gev_xi_to_scipy_c(xi: float) -> float:
    """For F(q)=exp(-(1+xi*(q-mu)/scale)**(-1/xi)), SciPy c=-xi."""
    _finite(xi, "xi")
    return -xi


def gev_scipy_c_to_xi(c: float) -> float:
    _finite(c, "c")
    return -c


@dataclass(frozen=True, slots=True)
class GEV(_ContinuousKernel):
    """Extreme-value xi convention: xi>0 has an unbounded upper tail.

    Flike's reported k sign is NOT yet proven to equal xi or -xi.
    The reported u and ln(a) may be supplied as location and log_scale;
    mapping k must be independently verified before claiming replication.
    """

    location: float
    log_scale: float
    xi: float

    def __post_init__(self) -> None:
        _finite(self.location, "location")
        _scale(self.log_scale)
        _finite(self.xi, "xi")

    def _frozen(self) -> _FrozenContinuous:
        return cast(
            _FrozenContinuous,
            genextreme(gev_xi_to_scipy_c(self.xi), loc=self.location, scale=math.exp(self.log_scale)),
        )


@dataclass(frozen=True, slots=True)
class GeneralizedPareto(_ContinuousKernel):
    """Continuous GP kernel; does not model POT exposure or event arrival."""

    location: float
    log_scale: float
    xi: float

    def __post_init__(self) -> None:
        _finite(self.location, "location")
        _scale(self.log_scale)
        _finite(self.xi, "xi")

    def _frozen(self) -> _FrozenContinuous:
        # Unlike GEV, SciPy's genpareto c has the same sign as xi.
        return cast(_FrozenContinuous, genpareto(self.xi, loc=self.location, scale=math.exp(self.log_scale)))
