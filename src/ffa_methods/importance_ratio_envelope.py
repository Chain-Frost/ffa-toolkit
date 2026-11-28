"""Outward-rounded sufficient normalized ratio bound for classified lognormal targets."""

from decimal import ROUND_CEILING, Context, Decimal, localcontext
from fractions import Fraction
from math import inf, isfinite, nextafter

from ffa_methods.contracts import ObservationKind
from ffa_methods.importance_sampling import ExplicitNormalProposal
from ffa_methods.lognormal_posterior import ClassifiedLognormalPosterior


class RatioEnvelopeError(ValueError):
    """The sufficient normalized ratio bound could not be established."""


def _matrix(rows: tuple[tuple[float, ...], ...]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(Fraction(value) for value in row) for row in rows)


def _determinant(matrix: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    a = matrix[0][0]
    if len(matrix) == 1:
        return a
    b, c = matrix[0][1], matrix[1][0]
    if b != c:
        return Fraction(0)
    return a * matrix[1][1] - b * c


def _gaussian_ratio(
    model: ClassifiedLognormalPosterior, proposal: ExplicitNormalProposal
) -> tuple[tuple[Fraction, ...], Fraction, Fraction]:
    prior = _matrix(model.prior.covariance)
    covariance = _matrix(proposal.covariance)
    count = sum(row.kind == ObservationKind.EXACT for row in model.series.annual)
    centre = tuple(
        Fraction(mean) - (count * row[1] if model.fixed_log_sd is None else 0)
        for mean, row in zip(model.prior.means, prior, strict=True)
    )
    delta = tuple(Fraction(mean) - value for mean, value in zip(proposal.means, centre, strict=True))
    prior_det = _determinant(prior)
    proposal_det = _determinant(covariance)
    if min(prior_det, proposal_det, prior[0][0], covariance[0][0]) <= 0:
        msg = "exact Gaussian covariance positivity is not established"
        raise RatioEnvelopeError(msg)
    if covariance == prior and not any(delta):
        return centre, Fraction(1), Fraction(0)
    gap = tuple(tuple(g - p for g, p in zip(gr, pr, strict=True)) for gr, pr in zip(covariance, prior, strict=True))
    gap_det = _determinant(gap)
    if gap[0][0] <= 0 or gap_det <= 0:
        msg = "strict Gaussian ratio covariance envelope is not established"
        raise RatioEnvelopeError(msg)
    if len(centre) == 1:
        quadratic = delta[0] ** 2 / gap_det
    else:
        x, y = delta
        quadratic = (gap[1][1] * x**2 - 2 * gap[0][1] * x * y + gap[0][0] * y**2) / gap_det
    return centre, proposal_det / prior_det, quadratic


def _upper(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _log_bounds(value: Decimal) -> tuple[Decimal, Decimal]:
    mid = value.ln()
    return mid.next_minus(), mid.next_plus()


def _relative_log_bounds(value: float, reference: tuple[Decimal, Decimal]) -> tuple[Decimal, Decimal]:
    lower, upper = _log_bounds(Decimal(value))
    # Under the upper-rounding context, negate an upper bound on the negative.
    return (reference[1] - lower).copy_negate(), upper - reference[0]


def _box(centre: Fraction, variance: float) -> tuple[Decimal, Decimal]:
    radius = Decimal(variance).sqrt().next_plus()
    return (radius + _upper(-centre)).copy_negate(), _upper(centre) + radius


def _residual_penalty(
    flow: tuple[Decimal, Decimal], location: tuple[Decimal, Decimal], inverse_variance: Decimal
) -> Decimal:
    distance = max(flow[1] - location[0], location[1] - flow[0])
    return distance * distance * inverse_variance / 2


def _negative_log_mass(model: ClassifiedLognormalPosterior, centre: tuple[Fraction, ...]) -> Decimal:
    # A standard Normal puts at least 2/3 of its mass within one SD.
    # The marginal union bound gives at least 1/3 for either dimension.
    location = _box(centre[0], model.prior.covariance[0][0])
    if model.fixed_log_sd is None:
        scale = _box(centre[1], model.prior.covariance[1][1])
    else:
        scale = (Decimal(model.fixed_log_sd),) * 2
    inverse_variance = (Decimal(-2) * scale[0]).exp().next_plus()
    reference = _log_bounds(Decimal(model.reference_discharge))
    penalty = _log_bounds(Decimal(3))[1]
    for row in model.series.annual:
        if row.kind in (ObservationKind.MISSING, ObservationKind.EXCLUDED):
            continue
        if row.kind == ObservationKind.EXACT and row.value is not None:
            flow = _relative_log_bounds(row.value, reference)
            penalty += _residual_penalty(flow, location, inverse_variance)
            continue
        lower = None if row.lower_bound is None else _relative_log_bounds(row.lower_bound, reference)[1]
        upper = None if row.upper_bound is None else _relative_log_bounds(row.upper_bound, reference)[0]
        if lower is None and upper is not None:
            lower = (Decimal(1) - upper).copy_negate()
        if upper is None and lower is not None:
            upper = (Decimal(-1) - lower).copy_negate()
        if lower is None or upper is None:
            msg = "censor event lacks a finite interior subinterval"
            raise RatioEnvelopeError(msg)
        width = (lower - upper).copy_negate()
        if width <= 0:
            msg = "censor subinterval width is unresolved"
            raise RatioEnvelopeError(msg)
        # sqrt(2*pi) < 3. Integrate the minimum density over this interior interval.
        penalty += _log_bounds(Decimal(3))[1] + scale[1] - _log_bounds(width)[0]
        penalty += _residual_penalty((lower, upper), location, inverse_variance)
    return penalty


def normalized_log_ratio_bound(model: ClassifiedLognormalPosterior, proposal: ExplicitNormalProposal) -> float:
    """Upper bound log(sup posterior/proposal) for the stored model parameters."""
    if model.coordinates != proposal.coordinates:
        msg = "proposal and target coordinates differ"
        raise ValueError(msg)
    centre, determinant_ratio, quadratic = _gaussian_ratio(model, proposal)
    try:
        with localcontext(Context(prec=64, rounding=ROUND_CEILING)):
            gaussian = (_log_bounds(_upper(determinant_ratio))[1] + _upper(quadratic)) / 2
            bound = nextafter(float(gaussian + _negative_log_mass(model, centre)), inf)
    except ArithmeticError as exc:
        msg = "normalized ratio bound is not numerically resolved"
        raise RatioEnvelopeError(msg) from exc
    if not isfinite(bound):
        msg = "normalized ratio bound is not finite"
        raise RatioEnvelopeError(msg)
    return bound
