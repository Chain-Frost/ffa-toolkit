"""Deterministic fixed-scale Gaussian/censor reference enclosures.

The calculation targets the real-valued model with the supplied binary64
inputs interpreted exactly. It does not change or certify the sampler.
"""

from dataclasses import dataclass
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    Overflow,
)
from enum import StrEnum
from math import isfinite

_D = Decimal


class CensoredReferenceError(ValueError):
    """A reference or resource requirement cannot be established."""


@dataclass(frozen=True, slots=True)
class ReferenceBounds:
    lower: Decimal
    upper: Decimal

    def __post_init__(self) -> None:
        if not self.lower.is_finite() or not self.upper.is_finite() or self.lower > self.upper:
            msg = "reference interval is invalid or nonfinite"
            raise CensoredReferenceError(msg)

    def contains(self, value: Decimal) -> bool:
        return self.lower <= value <= self.upper


def _point(value: int | float | Decimal) -> ReferenceBounds:
    # from_float also bypasses the ambient context's FloatOperation trap.
    number = Decimal.from_float(value) if isinstance(value, float) else Decimal(value)  # noqa: FURB164
    return ReferenceBounds(number, number)


_ZERO = _point(0)
_ONE = _point(1)
_TWO = _point(2)
_HALF = _point(_D("0.5"))
PI_BOUNDS = ReferenceBounds(
    _D("3.14159265358979323846264338327950288419716939937510"),
    _D("3.14159265358979323846264338327950288419716939937511"),
)


class _Arithmetic:
    """Private contexts; all interval arithmetic has explicit rounding."""

    def __init__(self) -> None:
        traps = [InvalidOperation, DivisionByZero, Overflow]
        self.down = Context(prec=64, rounding=ROUND_FLOOR, Emin=-10000, Emax=10000, flags=[], traps=traps)
        self.up = Context(prec=64, rounding=ROUND_CEILING, Emin=-10000, Emax=10000, flags=[], traps=traps)
        self.nearest = Context(prec=64, rounding=ROUND_HALF_EVEN, Emin=-10000, Emax=10000, flags=[], traps=traps)
        self.normal_constant = self.div(_ONE, self.sqrt(self.mul(_TWO, PI_BOUNDS)))
        self.sf_cache: dict[Decimal, ReferenceBounds] = {}

    def add(self, a: ReferenceBounds, b: ReferenceBounds) -> ReferenceBounds:
        return ReferenceBounds(self.down.add(a.lower, b.lower), self.up.add(a.upper, b.upper))

    def sub(self, a: ReferenceBounds, b: ReferenceBounds) -> ReferenceBounds:
        return ReferenceBounds(self.down.subtract(a.lower, b.upper), self.up.subtract(a.upper, b.lower))

    def mul(self, a: ReferenceBounds, b: ReferenceBounds) -> ReferenceBounds:
        corners = ((a.lower, b.lower), (a.lower, b.upper), (a.upper, b.lower), (a.upper, b.upper))
        return ReferenceBounds(
            min(self.down.multiply(x, y) for x, y in corners), max(self.up.multiply(x, y) for x, y in corners)
        )

    def div(self, a: ReferenceBounds, b: ReferenceBounds) -> ReferenceBounds:
        if b.lower <= 0 <= b.upper:
            msg = "reference interval division spans zero"
            raise CensoredReferenceError(msg)
        inverse = ReferenceBounds(self.down.divide(_D(1), b.upper), self.up.divide(_D(1), b.lower))
        return self.mul(a, inverse)

    def square(self, a: ReferenceBounds) -> ReferenceBounds:
        low = _D(0) if a.lower <= 0 <= a.upper else min(a.lower.copy_abs(), a.upper.copy_abs())
        high = max(a.lower.copy_abs(), a.upper.copy_abs())
        return ReferenceBounds(self.down.multiply(low, low), self.up.multiply(high, high))

    def sqrt(self, a: ReferenceBounds) -> ReferenceBounds:
        if a.lower < 0:
            msg = "negative square-root interval"
            raise CensoredReferenceError(msg)
        return ReferenceBounds(
            max(_D(0), self.nearest.next_minus(self.nearest.sqrt(a.lower))),
            self.nearest.next_plus(self.nearest.sqrt(a.upper)),
        )

    def exp(self, a: ReferenceBounds) -> ReferenceBounds:
        return ReferenceBounds(
            max(_D(0), self.nearest.next_minus(self.nearest.exp(a.lower))),
            self.nearest.next_plus(self.nearest.exp(a.upper)),
        )

    def ln(self, a: ReferenceBounds) -> ReferenceBounds:
        if a.lower <= 0:
            msg = "logarithm interval is not strictly positive"
            raise CensoredReferenceError(msg)
        return ReferenceBounds(
            self.nearest.next_minus(self.nearest.ln(a.lower)), self.nearest.next_plus(self.nearest.ln(a.upper))
        )

    def phi(self, z: ReferenceBounds) -> ReferenceBounds:
        return self.mul(self.normal_constant, self.exp(self.mul(_point(_D("-0.5")), self.square(z))))

    def sf(self, z: ReferenceBounds) -> ReferenceBounds:
        # The survival function is decreasing. Evaluate the appropriate endpoints.
        return ReferenceBounds(self._sf_point(z.upper).lower, self._sf_point(z.lower).upper)

    def cdf(self, z: ReferenceBounds) -> ReferenceBounds:
        return self.sf(ReferenceBounds(z.upper.copy_negate(), z.lower.copy_negate()))

    def _sf_point(self, x: Decimal) -> ReferenceBounds:
        if x not in self.sf_cache:
            if x < 0:
                result = self.sub(_ONE, self._sf_point(x.copy_negate()))
            elif x == 0:
                result = _HALF
            elif x <= 8:
                result = self._sf_series(x)
            else:
                result = self._sf_mills(x)
            self.sf_cache[x] = _probability(result)
        return self.sf_cache[x]

    def _sf_series(self, x: Decimal) -> ReferenceBounds:
        # Integral of exp(-t*t/2) on [0,x]. Taylor's signed remainder
        # gives alternating bounds even before the terms decrease.
        # After an even number of terms, sum <= integral <= sum+next_term.
        term = _point(x)
        factor = self.mul(_HALF, self.square(term))
        total = _ZERO
        for k in range(192):
            total = self.sub(total, term) if k % 2 else self.add(total, term)
            term = self.div(self.mul(self.mul(term, factor), _point(2 * k + 1)), _point((k + 1) * (2 * k + 3)))
            if k % 2 and term.upper <= _D("1e-55"):
                break
        integral = ReferenceBounds(max(_D(0), total.lower), self.up.add(total.upper, term.upper))
        return self.sub(_HALF, self.mul(self.normal_constant, integral))

    def _sf_mills(self, x: Decimal) -> ReferenceBounds:
        # Integration by parts: phi(x)*x/(1+x*x) <= S(x) <= phi(x)/x.
        # Beyond 64, monotonicity avoids exponent evaluation at enormous x.
        point = _point(min(x, _D(64)))
        density = self.phi(point)
        upper = self.div(density, point).upper
        lower = self.div(self.mul(density, point), self.add(_ONE, self.square(point))).lower if x <= 64 else _D(0)
        return ReferenceBounds(lower, upper)


def _probability(bounds: ReferenceBounds) -> ReferenceBounds:
    return ReferenceBounds(max(_D(0), bounds.lower), min(_D(1), bounds.upper))


@dataclass(frozen=True, slots=True, kw_only=True)
class CensorEvent:
    """One positive-discharge observation interval; None denotes an open end."""

    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        values = (self.lower, self.upper)
        if all(value is None for value in values) or any(
            value is not None and (not isfinite(value) or value <= 0) for value in values
        ):
            msg = "a censor event requires finite positive bounds on at least one side"
            raise CensoredReferenceError(msg)
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            msg = "censor interval must have positive width"
            raise CensoredReferenceError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class FixedScaleReferenceInputs:
    prior_mean: float
    prior_variance: float
    fixed_log_sd: float
    reference_discharge: float
    exact_flows: tuple[float, ...]
    censor_events: tuple[CensorEvent, ...]
    proposal_mean: float
    proposal_variance: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.prior_mean, self.fixed_log_sd, self.proposal_mean)):
            msg = "reference coordinates must be finite"
            raise CensoredReferenceError(msg)
        if any(
            not isfinite(value) or value <= 0
            for value in (self.prior_variance, self.reference_discharge, self.proposal_variance, *self.exact_flows)
        ):
            msg = "reference variances, reference discharge and exact flows must be finite and positive"
            raise CensoredReferenceError(msg)
        if not self.exact_flows and not self.censor_events:
            msg = "at least one included observation is required"
            raise CensoredReferenceError(msg)


class ReferenceKind(StrEnum):
    MEAN_MU = "mean_mu"
    CENTERED_SECOND_MU = "centered_second_mu"
    EXP_MU = "exp_mu"
    EXPECTED_F = "expected_f"
    EXPECTED_AEP = "expected_aep"
    MU_CDF = "mu_cdf"
    MU_SF = "mu_sf"


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceRequest:
    kind: ReferenceKind
    argument: float | None = None

    def __post_init__(self) -> None:
        if self.kind in (ReferenceKind.MEAN_MU, ReferenceKind.EXP_MU) and self.argument is None:
            return
        if (
            self.kind
            in (
                ReferenceKind.CENTERED_SECOND_MU,
                ReferenceKind.EXPECTED_F,
                ReferenceKind.EXPECTED_AEP,
                ReferenceKind.MU_CDF,
                ReferenceKind.MU_SF,
            )
            and self.argument is not None
            and isfinite(self.argument)
            and (self.kind not in (ReferenceKind.EXPECTED_F, ReferenceKind.EXPECTED_AEP) or self.argument > 0)
        ):
            return
        msg = "reference request has an invalid kind or argument"
        raise CensoredReferenceError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceControls:
    extent: int
    dyadic_level: int
    maximum_cells: int
    maximum_cell_terms: int
    maximum_log_ratio: float

    def __post_init__(self) -> None:
        if type(self.extent) is not int or not 1 <= self.extent <= 16:
            msg = "standardized reference extent must be an integer from 1 to 16"
            raise CensoredReferenceError(msg)
        if type(self.dyadic_level) is not int or not 0 <= self.dyadic_level <= 10:
            msg = "reference subdivision level must be an integer from 0 to 10"
            raise CensoredReferenceError(msg)
        if type(self.maximum_cells) is not int or self.cells > self.maximum_cells:
            msg = "reference partition exceeds the declared cell budget"
            raise CensoredReferenceError(msg)
        if type(self.maximum_cell_terms) is not int or self.maximum_cell_terms <= 0:
            msg = "reference cell-term budget must be a positive integer"
            raise CensoredReferenceError(msg)
        if not isfinite(self.maximum_log_ratio) or self.maximum_log_ratio <= 0:
            msg = "reference ratio ceiling must be finite and positive"
            raise CensoredReferenceError(msg)

    @property
    def panels_per_unit(self) -> int:
        return 2**self.dyadic_level

    @property
    def cells(self) -> int:
        return 2 * self.extent * self.panels_per_unit


@dataclass(frozen=True, slots=True)
class ReferenceValue:
    request: ReferenceRequest
    bounds: ReferenceBounds | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class FixedScaleReference:
    inputs: FixedScaleReferenceInputs
    controls: ReferenceControls
    base_mean: ReferenceBounds
    base_variance: ReferenceBounds
    normalizer: ReferenceBounds
    log_ratio_upper: Decimal | None
    ratio_reason: str | None
    reserved_cell_terms: int
    values: tuple[ReferenceValue, ...]

    @property
    def ratio_eligible(self) -> bool:
        return self.log_ratio_upper is not None and self.ratio_reason is None


@dataclass(frozen=True, slots=True)
class _Base:
    mean: ReferenceBounds
    variance: ReferenceBounds
    sd: ReferenceBounds
    log_flow_sd: ReferenceBounds
    log_reference: ReferenceBounds


def _base(a: _Arithmetic, inputs: FixedScaleReferenceInputs) -> _Base:
    log_reference = a.ln(_point(inputs.reference_discharge))
    sigma = a.exp(_point(inputs.fixed_log_sd))
    prior = _point(inputs.prior_variance)
    if not inputs.exact_flows:
        return _Base(_point(inputs.prior_mean), prior, a.sqrt(prior), sigma, log_reference)
    total = _ZERO
    for flow in inputs.exact_flows:
        total = a.add(total, a.sub(a.ln(_point(flow)), log_reference))
    sigma2 = a.square(sigma)
    variance = a.div(_ONE, a.add(a.div(_ONE, prior), a.div(_point(len(inputs.exact_flows)), sigma2)))
    mean = a.mul(variance, a.add(a.div(_point(inputs.prior_mean), prior), a.div(total, sigma2)))
    return _Base(mean, variance, a.sqrt(variance), sigma, log_reference)


@dataclass(frozen=True, slots=True)
class _Event:
    lower: ReferenceBounds | None
    upper: ReferenceBounds | None


def _log_events(a: _Arithmetic, base: _Base, events: tuple[CensorEvent, ...]) -> tuple[_Event, ...]:
    return tuple(
        _Event(
            None if event.lower is None else a.sub(a.ln(_point(event.lower)), base.log_reference),
            None if event.upper is None else a.sub(a.ln(_point(event.upper)), base.log_reference),
        )
        for event in events
    )


def _event_probability(a: _Arithmetic, base: _Base, event: _Event, mu: ReferenceBounds) -> ReferenceBounds:
    low = None if event.lower is None else a.div(a.sub(event.lower, mu), base.log_flow_sd)
    high = None if event.upper is None else a.div(a.sub(event.upper, mu), base.log_flow_sd)
    if low is None and high is not None:
        return a.cdf(high)
    if high is None and low is not None:
        return a.sf(low)
    if low is None or high is None:
        msg = "missing censor interval bounds"
        raise CensoredReferenceError(msg)
    # Dependency over the whole mu cell can widen this enclosure; intersect
    # with the proven [0,1] support rather than treating a negative lower
    # bound as negative probability. Use direct survival for upper tails.
    if low.lower >= 0:
        return _probability(a.sub(a.sf(low), a.sf(high)))
    return _probability(a.sub(a.cdf(high), a.cdf(low)))


@dataclass(frozen=True, slots=True)
class _Cell:
    mu: ReferenceBounds
    mass: ReferenceBounds


def _partition(
    a: _Arithmetic, base: _Base, events: tuple[_Event, ...], controls: ReferenceControls
) -> tuple[_Cell, ...]:
    cells: list[_Cell] = []
    width = a.div(_ONE, _point(controls.panels_per_unit))
    start = -controls.extent * controls.panels_per_unit
    for index in range(start, -start):
        z = a.mul(ReferenceBounds(_D(index), _D(index + 1)), width)
        mu = a.add(base.mean, a.mul(base.sd, z))
        mass = a.mul(width, a.phi(z))
        for event in events:
            mass = a.mul(mass, _event_probability(a, base, event, mu))
        cells.append(_Cell(mu, mass))
    return tuple(cells)


def _sum_bounds(a: _Arithmetic, values: tuple[ReferenceBounds, ...]) -> ReferenceBounds:
    total = _ZERO
    for value in values:
        total = a.add(total, value)
    return total


def _tail_mass(a: _Arithmetic, extent: int) -> ReferenceBounds:
    upper = a.div(a.mul(_TWO, a.phi(_point(extent))), _point(extent)).upper
    return ReferenceBounds(_D(0), min(_D(1), upper))


def _ratio_upper(
    a: _Arithmetic,
    inputs: FixedScaleReferenceInputs,
    base: _Base,
    normalizer: ReferenceBounds,
    controls: ReferenceControls,
) -> tuple[Decimal | None, str | None]:
    if normalizer.lower <= 0:
        return None, "positive censor normalizer lower bound unresolved"
    if inputs.proposal_variance < inputs.prior_variance:
        return None, "proposal variance below conservative prior-variance criterion"
    if (
        not inputs.exact_flows
        and inputs.proposal_variance == inputs.prior_variance
        and inputs.proposal_mean == inputs.prior_mean
    ):
        log_b = _ZERO
    else:
        proposal = _point(inputs.proposal_variance)
        gap = a.sub(proposal, base.variance)
        if gap.lower <= 0:
            return None, "strict variance gap or identical Gaussian base is not established"
        shift = a.square(a.sub(_point(inputs.proposal_mean), base.mean))
        log_b = a.mul(_HALF, a.add(a.ln(a.div(proposal, base.variance)), a.div(shift, gap)))
    upper = a.sub(log_b, a.ln(normalizer)).upper
    return upper, None if upper <= _D.from_float(controls.maximum_log_ratio) else "normalized ratio ceiling exceeded"


def _argument(request: ReferenceRequest) -> ReferenceBounds:
    if request.argument is None:
        msg = "reference argument is required"
        raise CensoredReferenceError(msg)
    return _point(request.argument)


def _integrand(a: _Arithmetic, base: _Base, request: ReferenceRequest, mu: ReferenceBounds) -> ReferenceBounds:
    if request.kind == ReferenceKind.MEAN_MU:
        return mu
    if request.kind == ReferenceKind.EXP_MU:
        return a.exp(mu)
    if request.kind == ReferenceKind.CENTERED_SECOND_MU:
        return a.square(a.sub(mu, _argument(request)))
    if request.kind in (ReferenceKind.MU_CDF, ReferenceKind.MU_SF):
        threshold = _argument(request).lower
        if mu.upper <= threshold:
            indicator = _ONE
        elif mu.lower > threshold:
            indicator = _ZERO
        else:
            indicator = ReferenceBounds(_D(0), _D(1))
        return indicator if request.kind == ReferenceKind.MU_CDF else a.sub(_ONE, indicator)
    log_flow = a.sub(a.ln(_argument(request)), base.log_reference)
    z = a.div(a.sub(log_flow, mu), base.log_flow_sd)
    return a.sf(z) if request.kind == ReferenceKind.EXPECTED_AEP else a.cdf(z)


def _tail_bound(
    a: _Arithmetic, base: _Base, request: ReferenceRequest, extent: int, tail: ReferenceBounds
) -> ReferenceBounds:
    edge = _point(extent)
    density = a.phi(edge)
    if request.kind == ReferenceKind.MEAN_MU:
        absolute_mean = _point(max(base.mean.lower.copy_abs(), base.mean.upper.copy_abs()))
        upper = a.add(a.mul(absolute_mean, tail), a.mul(base.sd, a.mul(_TWO, density))).upper
        return ReferenceBounds(upper.copy_negate(), upper)
    if request.kind == ReferenceKind.CENTERED_SECOND_MU:
        centred = a.square(a.sub(base.mean, _argument(request)))
        second_tail = a.add(a.mul(_TWO, a.mul(edge, density)), tail)
        bound = a.mul(_TWO, a.add(a.mul(centred, tail), a.mul(base.variance, second_tail)))
        return ReferenceBounds(_D(0), bound.upper)
    if request.kind == ReferenceKind.EXP_MU:
        tilt = a.exp(a.add(base.mean, a.mul(_HALF, base.variance)))
        probability = a.add(a.sf(a.add(edge, base.sd)), a.sf(a.sub(edge, base.sd)))
        return ReferenceBounds(_D(0), a.mul(tilt, probability).upper)
    return tail


def _value(
    a: _Arithmetic,
    base: _Base,
    request: ReferenceRequest,
    cells: tuple[_Cell, ...],
    tail: ReferenceBounds,
    normalizer: ReferenceBounds,
    extent: int,
) -> ReferenceValue:
    if normalizer.lower <= 0:
        return ReferenceValue(request, None, "positive normalizer lower bound unresolved")
    try:
        interior = _sum_bounds(a, tuple(a.mul(cell.mass, _integrand(a, base, request, cell.mu)) for cell in cells))
        bounds = a.div(a.add(interior, _tail_bound(a, base, request, extent, tail)), normalizer)
        if request.kind in (
            ReferenceKind.EXPECTED_AEP,
            ReferenceKind.EXPECTED_F,
            ReferenceKind.MU_CDF,
            ReferenceKind.MU_SF,
        ):
            bounds = _probability(bounds)
    except (DecimalException, CensoredReferenceError) as exc:
        return ReferenceValue(request, None, f"reference output enclosure unresolved: {exc}")
    return ReferenceValue(request, bounds)


def fixed_scale_reference(
    inputs: FixedScaleReferenceInputs, controls: ReferenceControls, requests: tuple[ReferenceRequest, ...]
) -> FixedScaleReference:
    """Enclose the normalized model and scalar functionals, without a stochastic pass.

    Reserve a bound on cell terms before evaluation. Every event/functional
    cell uses at most four point survival evaluations, each with at most
    192 series terms. No adaptive subdivision or random draw is used.
    """
    reserved = (
        controls.cells * (1 + len(inputs.censor_events) + len(requests))
        + len(inputs.exact_flows)
        + 2 * len(inputs.censor_events)
        + 3 * len(requests)
        + 8
    )
    if reserved > controls.maximum_cell_terms:
        msg = "reference calculation exceeds declared cell-term budget"
        raise CensoredReferenceError(msg)
    if len(set(requests)) != len(requests):
        msg = "duplicate reference requests"
        raise CensoredReferenceError(msg)
    try:
        a = _Arithmetic()
        base = _base(a, inputs)
        events = _log_events(a, base, inputs.censor_events)
        cells = _partition(a, base, events, controls)
        tail = _tail_mass(a, controls.extent)
        normalizer = _probability(a.add(_sum_bounds(a, tuple(cell.mass for cell in cells)), tail)) if events else _ONE
    except DecimalException as exc:
        msg = "reference partition or base is not representable"
        raise CensoredReferenceError(msg) from exc
    try:
        upper, reason = _ratio_upper(a, inputs, base, normalizer, controls)
    except (DecimalException, CensoredReferenceError) as exc:
        upper, reason = None, f"normalized ratio enclosure unresolved: {exc}"
    values = tuple(_value(a, base, request, cells, tail, normalizer, controls.extent) for request in requests)
    return FixedScaleReference(inputs, controls, base.mean, base.variance, normalizer, upper, reason, reserved, values)
