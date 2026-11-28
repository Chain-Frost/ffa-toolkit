"""Standardize observations in the estimated-scale family's original mathematical coordinates."""

from decimal import Context, Decimal, localcontext
from fractions import Fraction
from math import inf, isfinite, isnan, log, ulp

from ffa_methods.distributions import GEV, LogCoordinates, LogPearsonIII
from ffa_methods.lognormal_posterior import PosteriorBindingError

Bounds = tuple[Fraction, Fraction]


def _transform_bounds(value: float, *, logarithm: bool) -> Bounds:
    """Enclose each transcendental transform of the exact binary64 input.

    Decimal ln and exp are correctly rounded. Enclose their results by the two
    adjacent decimals, then carry out all endpoint comparisons as rational operations.
    """
    if logarithm and value == 1.0:
        return Fraction(0), Fraction(0)
    if not logarithm and value == 0.0:
        return Fraction(1), Fraction(1)
    with localcontext(Context(prec=80)):
        operand = Decimal(value)
        result = operand.ln() if logarithm else operand.exp()
        return Fraction(result.next_minus()), Fraction(result.next_plus())


class EstimatedScaleKernel:
    """Use WP03 standardized laws for both likelihood terms and predictions."""

    def __init__(self, raw: GEV | LogPearsonIII) -> None:
        self.raw = raw
        if isinstance(raw, GEV):
            self._law = GEV(0.0, 0.0, raw.xi)
            self._reference = None
            self._reference_log = (Fraction(0), Fraction(0))
            self._mean = Fraction(raw.location)
            self._eta = raw.log_scale
            slope = Fraction(raw.xi)
        else:
            unit = LogPearsonIII(LogCoordinates(0.0, 0.0), raw.skew)
            self._law = unit.log_flow_law()
            self._reference = raw.log.reference_discharge
            self._reference_log = _transform_bounds(self._reference, logarithm=True)
            self._mean = Fraction(raw.log.mean)
            self._eta = raw.log.log_sd
            slope = Fraction(raw.skew) / 2
        self._scale = _transform_bounds(self._eta, logarithm=False)
        self._edge = -1 / slope if slope else None
        self._lower_endpoint = slope > 0

    def _statistic(self, flow: float) -> Bounds:
        if self._reference is None:
            return Fraction(flow), Fraction(flow)
        if flow == self._reference:
            return Fraction(0), Fraction(0)
        lower, upper = _transform_bounds(flow, logarithm=True)
        return lower - self._reference_log[1], upper - self._reference_log[0]

    def _standardized(self, flow: float) -> float:
        if isnan(flow):
            msg = "discharge must not be NaN"
            raise PosteriorBindingError(msg)
        if self._reference is not None and flow <= 0:
            return -inf
        if not isfinite(flow):
            return flow
        lower, upper = self._statistic(flow)
        lower -= self._mean
        upper -= self._mean
        low = lower / self._scale[0 if lower < 0 else 1]
        high = upper / self._scale[1 if upper < 0 else 0]
        if self._edge is not None:
            if self._lower_endpoint and high <= self._edge:
                return -inf
            if not self._lower_endpoint and low >= self._edge:
                return inf
        try:
            z = float(low)
            resolved = z == float(high) and isfinite(z)
        except OverflowError as exc:
            msg = "unresolved standardized family coordinate"
            raise PosteriorBindingError(msg) from exc
        if not resolved:
            msg = "unresolved standardized family coordinate"
            raise PosteriorBindingError(msg)
        if self._edge is not None and abs(Fraction(z) - self._edge) <= 32 * Fraction(ulp(z)):
            # Even a correctly rounded z cannot resolve the small interior
            # distance used by the binary64 standardized density and tails.
            msg = "unresolved interior moving-endpoint distance"
            raise PosteriorBindingError(msg)
        return z

    @staticmethod
    def _log_result(value: float) -> float:
        if not isfinite(value):
            msg = "unresolved interior log likelihood"
            raise PosteriorBindingError(msg)
        return value

    def logpdf(self, discharge: float) -> float:
        z = self._standardized(discharge)
        if not isfinite(z):
            return -inf
        jacobian = -log(discharge) if self._reference is not None else 0.0
        return self._log_result(float(self._law.logpdf(z)) - self._eta + jacobian)

    def logcdf(self, discharge: float) -> float:
        z = self._standardized(discharge)
        if not isfinite(z):
            return -inf if z < 0 else 0.0
        return self._log_result(float(self._law.logcdf(z)))

    def logsf(self, discharge: float) -> float:
        z = self._standardized(discharge)
        if not isfinite(z):
            return 0.0 if z < 0 else -inf
        return self._log_result(float(self._law.logsf(z)))

    @staticmethod
    def _probability_result(value: float) -> float:
        if not 0.0 < value <= 1.0:
            msg = "unresolved interior family probability"
            raise PosteriorBindingError(msg)
        return value

    def cdf(self, discharge: float) -> float:
        z = self._standardized(discharge)
        if not isfinite(z):
            return 0.0 if z < 0 else 1.0
        return self._probability_result(float(self._law.cdf(z)))

    def sf(self, discharge: float) -> float:
        z = self._standardized(discharge)
        if not isfinite(z):
            return 1.0 if z < 0 else 0.0
        return self._probability_result(float(self._law.sf(z)))

    def ppf(self, non_exceedance_probability: float) -> float:
        return self.raw.ppf(non_exceedance_probability)

    def isf(self, survival_probability: float) -> float:
        return self.raw.isf(survival_probability)
