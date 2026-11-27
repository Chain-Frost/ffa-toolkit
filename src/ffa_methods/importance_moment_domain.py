"""Conservative mathematical moment eligibility for explicit lognormal precision.

Not an audit, a proposal selector, a normalized ratio ceiling or CLI admission.
See docs/wp06-censored-scale-precision.md.
"""

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from ffa_methods.contracts import ResultMeaning
from ffa_methods.importance_precision import PrecisionRequest
from ffa_methods.importance_sampling import ExplicitNormalProposal
from ffa_methods.lognormal_posterior import ClassifiedLognormalPosterior


class MomentState(StrEnum):
    FINITE = "finite"
    DIVERGENT = "divergent"
    UNESTABLISHED = "unestablished"


@dataclass(frozen=True, slots=True)
class MomentDomain:
    """Existence in real arithmetic, not representability or precision acceptance."""

    posterior: MomentState
    scalar_second_moments: MomentState
    reason: str


def _positive_definite(matrix: tuple[tuple[Fraction, ...], ...]) -> bool:
    # Exact signs for the stored binary64 entries. Reject approximate
    # symmetry rather than treating a rounded eigenvalue as a proof.
    if len(matrix) == 1:
        return matrix[0][0] > 0
    if len(matrix) != 2:
        return False
    a, b = matrix[0]
    c, d = matrix[1]
    return b == c and a > 0 and a * d > b * c


def _second_moment_envelope(prior: tuple[tuple[float, ...], ...], proposal: tuple[tuple[float, ...], ...]) -> bool:
    p = tuple(tuple(Fraction(v) for v in row) for row in prior)
    g = tuple(tuple(Fraction(v) for v in row) for row in proposal)
    difference = tuple(
        tuple(2 * gv - pv for gv, pv in zip(gr, pr, strict=True)) for gr, pr in zip(g, p, strict=True)
    )
    return _positive_definite(p) and _positive_definite(g) and _positive_definite(difference)


def assess_lognormal_moments(
    model: ClassifiedLognormalPosterior, proposal: ExplicitNormalProposal, request: PrecisionRequest
) -> MomentDomain:
    """Assess sufficient existence conditions for the existing request's scalar route.

    For inverses this concerns fixed-point CDF/SF or indicator integrands,
    not moments of the inverse itself. For a quantile at expected parameters
    this concerns coordinate means, not their nonlinear error propagation.
    """
    if model.coordinates != proposal.coordinates:
        msg = "proposal and target coordinates differ"
        raise ValueError(msg)
    if (
        model.fixed_log_sd is None
        and request.meaning == ResultMeaning.POSTERIOR_MEAN_QUANTILE
        and request.non_exceedance is not None
        and request.non_exceedance > 0.5
    ):
        return MomentDomain(
            MomentState.DIVERGENT,
            MomentState.DIVERGENT,
            "upper design-flood mean diverges under the untruncated Gaussian log-SD prior",
        )
    if not _second_moment_envelope(model.prior.covariance, proposal.covariance):
        return MomentDomain(
            MomentState.FINITE,
            MomentState.UNESTABLISHED,
            "strict Gaussian second-moment covariance envelope is not established; this is not a divergence proof",
        )
    return MomentDomain(
        MomentState.FINITE,
        MomentState.FINITE,
        "2 * proposal covariance - prior covariance is strictly positive definite; "
        "exact/censor likelihood and scalar integrand growth bounds establish the required second moments",
    )
