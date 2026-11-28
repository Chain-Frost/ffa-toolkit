"""Exact reflection admission for selected classified lognormal references."""

from collections import Counter
from fractions import Fraction

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
