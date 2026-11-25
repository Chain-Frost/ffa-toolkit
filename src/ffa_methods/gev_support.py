"""Support diagnostic for an already fitted ARR-coordinate GEV, not a fit rule."""

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Literal


@dataclass(frozen=True, slots=True)
class GEVSupportObservation:
    """Identity and actual binary64 discharge consumed by the estimator."""

    year: int
    discharge: float


@dataclass(frozen=True, slots=True)
class GEVFittedSupport:
    """Finite mathematical endpoint (if any), with strict outside and equality distinct."""

    kind: Literal["unbounded", "finite_lower", "finite_upper"]
    endpoint_exact: str | None
    endpoint_approximate: float | None
    observation_count: int
    outside: tuple[GEVSupportObservation, ...]
    at_endpoint: tuple[GEVSupportObservation, ...]
    unit: str


def _approximate(endpoint: Fraction) -> float | None:
    try:
        return float(endpoint)
    except OverflowError:
        return None


def diagnose_gev_support(
    observations: Iterable[GEVSupportObservation], *, location: float, scale: float, shape: float, unit: str
) -> GEVFittedSupport:
    """Compare the admitted binary64 inputs to the exact endpoint of those fitted coordinates.

    ARR k>0: upper endpoint; k<0: lower endpoint; k=0: neither.
    No rounded endpoint, tolerance, log-scale conversion or likelihood is used to classify.
    """
    if not all(isfinite(x) for x in (location, scale, shape)) or scale <= 0:
        msg = "GEV support requires finite coordinates and positive scale"
        raise ValueError(msg)
    records = tuple(observations)
    if any(not isfinite(record.discharge) for record in records):
        msg = "GEV support requires finite fitted observations"
        raise ValueError(msg)
    if shape == 0:
        return GEVFittedSupport("unbounded", None, None, len(records), (), (), unit)
    endpoint = Fraction(location) + Fraction(scale) / Fraction(shape)
    outside: list[GEVSupportObservation] = []
    equal: list[GEVSupportObservation] = []
    for record in records:
        value = Fraction(record.discharge)
        if value == endpoint:
            equal.append(record)
        elif (shape > 0 and value > endpoint) or (shape < 0 and value < endpoint):
            outside.append(record)
    return GEVFittedSupport(
        "finite_upper" if shape > 0 else "finite_lower",
        str(endpoint),
        _approximate(endpoint),
        len(records),
        tuple(outside),
        tuple(equal),
        unit,
    )
