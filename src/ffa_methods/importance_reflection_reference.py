"""Exact reflection admission for selected classified lognormal references."""

from collections import Counter
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal, localcontext
from fractions import Fraction
from math import inf, isfinite, nextafter

from ffa_methods.contracts import ObservationKind
from ffa_methods.lognormal_posterior import ClassifiedLognormalPosterior


def has_reflection_symmetry(model: ClassifiedLognormalPosterior) -> bool:
    """Check the posterior reflection mu -> -mu in exact stored coordinates."""
    if model.prior.means[0] != 0 or (model.fixed_log_sd is None and model.prior.covariance[0][1] != 0):
        return False
    reference_squared = Fraction(model.reference_discharge) ** 2
    events: Counter[tuple[Fraction | None, Fraction | None]] = Counter()
    for row in model.series.annual:
        if row.kind in (ObservationKind.MISSING, ObservationKind.EXCLUDED):
            continue
        lower, upper = (
            (row.value, row.value) if row.kind == ObservationKind.EXACT else (row.lower_bound, row.upper_bound)
        )
        events[(None if lower is None else Fraction(lower), None if upper is None else Fraction(upper))] += 1
    return all(
        count
        == events[
            (None if upper is None else reference_squared / upper, None if lower is None else reference_squared / lower)
        ]
        for (lower, upper), count in events.items()
    )


def reflected_discharge_bounds(reference: float, lower: float, upper: float) -> tuple[float, float]:
    """Enclose R exp(mu) without a rounded log(R) round trip."""
    with localcontext(Context(prec=64, rounding=ROUND_FLOOR)):
        low = Decimal(reference) * Decimal(lower).exp().next_minus()
        left = nextafter(float(low), -inf)
    with localcontext(Context(prec=64, rounding=ROUND_CEILING)):
        high = Decimal(reference) * Decimal(upper).exp().next_plus()
        right = nextafter(float(high), inf)
    if not isfinite(right) or left <= 0:
        msg = "reflected discharge interval is not representable"
        raise ArithmeticError(msg)
    return left, right


def reflected_discharge(reference: float, mean: float) -> float:
    """Evaluate the conditional median directly in discharge units."""
    with localcontext(Context(prec=64)):
        return float(Decimal(reference) * Decimal(mean).exp())
