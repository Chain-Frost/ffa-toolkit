"""Deterministic scalar enclosures for the explicit estimated-scale lognormal target.

No stochastic admission, inverse acceptance, or change to the existing audit.
"""

from dataclasses import dataclass
from decimal import Decimal, DecimalException
from enum import StrEnum
from fractions import Fraction
from math import isfinite

from ffa_methods.censored_reference import (
    CensoredReferenceError,
    CensorEvent,
    ReferenceBounds,
    _Arithmetic,  # pyright: ignore[reportPrivateUsage] -- shared internal interval arithmetic
    _point,  # pyright: ignore[reportPrivateUsage]
    _probability,  # pyright: ignore[reportPrivateUsage]
)

_ZERO = _point(0)
_ONE = _point(1)
_UNIT = ReferenceBounds(Decimal(0), Decimal(1))


@dataclass(frozen=True, slots=True, kw_only=True)
class EstimatedScaleInputs:
    prior_mean: tuple[float, float]
    prior_covariance: tuple[tuple[float, float], tuple[float, float]]
    reference_discharge: float
    exact_flows: tuple[float, ...]
    censor_events: tuple[CensorEvent, ...]

    def __post_init__(self) -> None:
        if any(not isfinite(v) for v in (*self.prior_mean, *self.prior_covariance[0], *self.prior_covariance[1])):
            msg = "estimated-scale Gaussian inputs must be finite"
            raise CensoredReferenceError(msg)
        _covariance(self)
        if any(not isfinite(v) or v <= 0 for v in (self.reference_discharge, *self.exact_flows)):
            msg = "reference discharge and exact flows must be positive and finite"
            raise CensoredReferenceError(msg)
        if not self.exact_flows and not self.censor_events:
            msg = "at least one included observation is required"
            raise CensoredReferenceError(msg)


def _covariance(inputs: EstimatedScaleInputs) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    a, b = (Fraction(v) for v in inputs.prior_covariance[0])
    c, d = (Fraction(v) for v in inputs.prior_covariance[1])
    determinant = a * d - b * c
    if b != c or a <= 0 or determinant <= 0:
        msg = "exact symmetric positive-definite prior covariance is required"
        raise CensoredReferenceError(msg)
    return a, b, d, determinant


class EstimatedScaleKind(StrEnum):
    MEAN_MU = "mean_mu"
    MEAN_ETA = "mean_eta"
    EXP_MU = "exp_mu"
    EXP_ETA = "exp_eta"
    CENTERED_SECOND_MU = "centered_second_mu"
    CENTERED_SECOND_ETA = "centered_second_eta"
    EXPECTED_F = "expected_f"
    EXPECTED_AEP = "expected_aep"
    MU_CDF = "mu_cdf"
    MU_SF = "mu_sf"


_MEANS = frozenset((EstimatedScaleKind.MEAN_MU, EstimatedScaleKind.MEAN_ETA))
_EXP = frozenset((EstimatedScaleKind.EXP_MU, EstimatedScaleKind.EXP_ETA))
_SECOND = frozenset((EstimatedScaleKind.CENTERED_SECOND_MU, EstimatedScaleKind.CENTERED_SECOND_ETA))
_FLOW = frozenset((EstimatedScaleKind.EXPECTED_F, EstimatedScaleKind.EXPECTED_AEP))
_INDICATORS = frozenset((EstimatedScaleKind.MU_CDF, EstimatedScaleKind.MU_SF))
_ETA = frozenset((EstimatedScaleKind.MEAN_ETA, EstimatedScaleKind.EXP_ETA, EstimatedScaleKind.CENTERED_SECOND_ETA))


@dataclass(frozen=True, slots=True, kw_only=True)
class EstimatedScaleRequest:
    kind: EstimatedScaleKind
    argument: float | None = None

    def __post_init__(self) -> None:
        if self.kind in _MEANS | _EXP and self.argument is None:
            return
        if (
            self.kind in _SECOND | _FLOW | _INDICATORS
            and self.argument is not None
            and isfinite(self.argument)
            and (self.kind not in _FLOW or self.argument > 0)
        ):
            return
        msg = "invalid estimated-scale reference request"
        raise CensoredReferenceError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class EstimatedScaleControls:
    extent: int
    dyadic_level: int
    maximum_cells: int
    maximum_cell_terms: int

    def __post_init__(self) -> None:
        if type(self.extent) is not int or not 1 <= self.extent <= 16:
            msg = "estimated-scale reference extent must be an integer from 1 to 16"
            raise CensoredReferenceError(msg)
        if type(self.dyadic_level) is not int or not 0 <= self.dyadic_level <= 10:
            msg = "estimated-scale reference subdivision level must be an integer from 0 to 10"
            raise CensoredReferenceError(msg)
        if type(self.maximum_cells) is not int or self.cells > self.maximum_cells:
            msg = "two-dimensional reference partition exceeds declared cell budget"
            raise CensoredReferenceError(msg)
        if type(self.maximum_cell_terms) is not int or self.maximum_cell_terms <= 0:
            msg = "reference cell-term budget must be a positive integer"
            raise CensoredReferenceError(msg)

    @property
    def axis_cells(self) -> int:
        return 2 * self.extent * 2**self.dyadic_level

    @property
    def cells(self) -> int:
        return self.axis_cells**2


@dataclass(frozen=True, slots=True)
class EstimatedScaleValue:
    request: EstimatedScaleRequest
    bounds: ReferenceBounds | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class EstimatedScaleReference:
    inputs: EstimatedScaleInputs
    controls: EstimatedScaleControls
    tilted_mean: tuple[ReferenceBounds, ReferenceBounds]
    normalizer: ReferenceBounds
    omitted_mass: ReferenceBounds
    reserved_cell_terms: int
    values: tuple[EstimatedScaleValue, ...]


def _fraction(a: _Arithmetic, value: Fraction) -> ReferenceBounds:
    return a.div(_point(value.numerator), _point(value.denominator))


@dataclass(frozen=True, slots=True)
class _Base:
    mean: tuple[ReferenceBounds, ReferenceBounds]
    variance: tuple[ReferenceBounds, ReferenceBounds]
    l00: ReferenceBounds
    l10: ReferenceBounds
    l11: ReferenceBounds
    log_reference: ReferenceBounds
    exact: tuple[ReferenceBounds, ...]
    events: tuple[tuple[ReferenceBounds | None, ReferenceBounds | None], ...]


def _base(a: _Arithmetic, inputs: EstimatedScaleInputs) -> _Base:
    p00, p01, p11, det = _covariance(inputs)
    n = len(inputs.exact_flows)
    mean = (
        _fraction(a, Fraction(inputs.prior_mean[0]) - n * p01),
        _fraction(a, Fraction(inputs.prior_mean[1]) - n * p11),
    )
    l00 = a.sqrt(_fraction(a, p00))
    ref = a.ln(_point(inputs.reference_discharge))
    events = tuple(
        (
            None if event.lower is None else a.sub(a.ln(_point(event.lower)), ref),
            None if event.upper is None else a.sub(a.ln(_point(event.upper)), ref),
        )
        for event in inputs.censor_events
    )
    return _Base(
        mean,
        (_fraction(a, p00), _fraction(a, p11)),
        l00,
        a.div(_fraction(a, p01), l00),
        a.sqrt(_fraction(a, det / p00)),
        ref,
        tuple(a.sub(a.ln(_point(flow)), ref) for flow in inputs.exact_flows),
        events,
    )


def _panels(a: _Arithmetic, controls: EstimatedScaleControls) -> tuple[tuple[ReferenceBounds, ReferenceBounds], ...]:
    width = a.div(_ONE, _point(2**controls.dyadic_level))
    start = -(controls.axis_cells // 2)
    panels: list[tuple[ReferenceBounds, ReferenceBounds]] = []
    for index in range(start, -start):
        z = a.mul(ReferenceBounds(Decimal(index), Decimal(index + 1)), width)
        # Integrate each independent standard Normal panel using endpoint
        # probabilities; direct survival on the positive half-axis.
        low, high = _point(z.lower), _point(z.upper)
        mass = a.sub(a.sf(low), a.sf(high)) if z.lower >= 0 else a.sub(a.cdf(high), a.cdf(low))
        panels.append((z, _probability(mass)))
    return tuple(panels)


@dataclass(frozen=True, slots=True)
class _Cell:
    mu: ReferenceBounds
    eta: ReferenceBounds
    sigma: ReferenceBounds
    mass: ReferenceBounds


def _event(
    a: _Arithmetic,
    bounds: tuple[ReferenceBounds | None, ReferenceBounds | None],
    mu: ReferenceBounds,
    sigma: ReferenceBounds,
) -> ReferenceBounds:
    low = None if bounds[0] is None else a.div(a.sub(bounds[0], mu), sigma)
    high = None if bounds[1] is None else a.div(a.sub(bounds[1], mu), sigma)
    if low is None and high is not None:
        return a.cdf(high)
    if high is None and low is not None:
        return a.sf(low)
    if low is None or high is None:
        msg = "missing censor event bounds"
        raise CensoredReferenceError(msg)
    return _probability(a.sub(a.sf(low), a.sf(high)) if low.lower >= 0 else a.sub(a.cdf(high), a.cdf(low)))


def _cell(
    a: _Arithmetic,
    base: _Base,
    x: tuple[ReferenceBounds, ReferenceBounds],
    y: tuple[ReferenceBounds, ReferenceBounds],
) -> _Cell:
    mu = a.add(base.mean[0], a.mul(base.l00, x[0]))
    eta = a.add(a.add(base.mean[1], a.mul(base.l10, x[0])), a.mul(base.l11, y[0]))
    sigma = a.exp(eta)
    mass = a.mul(x[1], y[1])
    for log_flow in base.exact:
        residual = a.div(a.sub(log_flow, mu), sigma)
        mass = a.mul(mass, _probability(a.exp(a.mul(_point(-0.5), a.square(residual)))))
    for bounds in base.events:
        mass = a.mul(mass, _event(a, bounds, mu, sigma))
    return _Cell(mu, eta, sigma, mass)


def _argument(request: EstimatedScaleRequest) -> ReferenceBounds:
    if request.argument is None:
        msg = "estimated-scale reference argument is required"
        raise CensoredReferenceError(msg)
    return _point(request.argument)


def _integrand(a: _Arithmetic, base: _Base, request: EstimatedScaleRequest, cell: _Cell) -> ReferenceBounds:
    coord = cell.eta if request.kind in _ETA else cell.mu
    if request.kind in _MEANS:
        return coord
    if request.kind in _EXP:
        return a.exp(coord)
    if request.kind in _SECOND:
        return a.square(a.sub(coord, _argument(request)))
    if request.kind in _INDICATORS:
        threshold = _argument(request).lower
        indicator = _ONE if coord.upper <= threshold else _ZERO if coord.lower > threshold else _UNIT
        return indicator if request.kind == EstimatedScaleKind.MU_CDF else a.sub(_ONE, indicator)
    log_flow = a.sub(a.ln(_argument(request)), base.log_reference)
    z = a.div(a.sub(log_flow, cell.mu), cell.sigma)
    return a.sf(z) if request.kind == EstimatedScaleKind.EXPECTED_AEP else a.cdf(z)


def _tail(a: _Arithmetic, base: _Base, request: EstimatedScaleRequest, mass: ReferenceBounds) -> ReferenceBounds:
    if request.kind in _FLOW | _INDICATORS:
        return mass
    index = 1 if request.kind in _ETA else 0
    mean, variance = base.mean[index], base.variance[index]
    if request.kind in _MEANS:
        second = a.add(a.square(mean), variance)
    elif request.kind in _EXP:
        second = a.exp(a.mul(_point(2), a.add(mean, variance)))
    else:
        centered = a.square(a.sub(mean, _argument(request)))
        second = a.add(
            a.add(a.square(centered), a.mul(_point(6), a.mul(centered, variance))),
            a.mul(_point(3), a.square(variance)),
        )
    upper = a.sqrt(a.mul(second, mass)).upper
    return ReferenceBounds(upper.copy_negate() if request.kind in _MEANS else Decimal(0), upper)


def _finish(
    a: _Arithmetic,
    base: _Base,
    request: EstimatedScaleRequest,
    total: ReferenceBounds,
    reason: str | None,
    normalizer: ReferenceBounds,
    omitted: ReferenceBounds,
) -> EstimatedScaleValue:
    if reason is not None or normalizer.lower <= 0:
        return EstimatedScaleValue(request, None, reason or "positive normalizer lower bound unresolved")
    try:
        value = a.div(a.add(total, _tail(a, base, request, omitted)), normalizer)
        if request.kind in _FLOW | _INDICATORS:
            value = _probability(value)
    except (CensoredReferenceError, DecimalException) as exc:
        return EstimatedScaleValue(request, None, f"reference output enclosure unresolved: {exc}")
    return EstimatedScaleValue(request, value)


def _accumulate(
    a: _Arithmetic,
    base: _Base,
    requests: tuple[EstimatedScaleRequest, ...],
    panels: tuple[tuple[ReferenceBounds, ReferenceBounds], ...],
) -> tuple[ReferenceBounds, list[ReferenceBounds], list[str | None]]:
    normalizer = _ZERO
    totals = [_ZERO for _ in requests]
    reasons: list[str | None] = [None for _ in requests]
    for x in panels:
        for y in panels:
            cell = _cell(a, base, x, y)
            normalizer = a.add(normalizer, cell.mass)
            for i, request in enumerate(requests):
                if reasons[i] is not None:
                    continue
                try:
                    totals[i] = a.add(totals[i], a.mul(cell.mass, _integrand(a, base, request, cell)))
                except (CensoredReferenceError, DecimalException) as exc:
                    reasons[i] = f"reference output enclosure unresolved: {exc}"
        # Bound memoization to a single outer panel rather than every 2D cell.
        a.sf_cache.clear()
    return normalizer, totals, reasons


def estimated_scale_reference(
    inputs: EstimatedScaleInputs,
    controls: EstimatedScaleControls,
    requests: tuple[EstimatedScaleRequest, ...],
) -> EstimatedScaleReference:
    """Integrate over a fixed dyadic box in independent tilted Gaussian coordinates.

    Cell terms are reserved before density or probability evaluation. The
    omitted region is enclosed analytically; never substitute a truncated target.
    """
    reserved = (
        controls.cells * (1 + len(inputs.exact_flows) + len(inputs.censor_events) + len(requests))
        + 2 * controls.axis_cells
        + len(inputs.exact_flows)
        + 2 * len(inputs.censor_events)
        + 8 * len(requests)
        + 64
    )
    if reserved > controls.maximum_cell_terms:
        msg = "estimated-scale reference exceeds declared cell-term budget"
        raise CensoredReferenceError(msg)
    if len(set(requests)) != len(requests):
        msg = "duplicate estimated-scale reference requests"
        raise CensoredReferenceError(msg)
    try:
        a = _Arithmetic()
        base = _base(a, inputs)
        panels = _panels(a, controls)
        interior, totals, reasons = _accumulate(a, base, requests, panels)
        # Two independent standard coordinates; union bound, no correlation assumption on mu/eta.
        omitted = ReferenceBounds(
            Decimal(0), min(Decimal(1), a.mul(_point(4), a.sf(_point(controls.extent))).upper)
        )
        normalizer = _probability(a.add(interior, omitted))
    except DecimalException as exc:
        msg = "estimated-scale partition or base is not representable"
        raise CensoredReferenceError(msg) from exc
    values = tuple(
        _finish(a, base, request, total, reason, normalizer, omitted)
        for request, total, reason in zip(requests, totals, reasons, strict=True)
    )
    return EstimatedScaleReference(inputs, controls, base.mean, normalizer, omitted, reserved, values)
