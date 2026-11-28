"""Bind deterministic censored reference evidence to the existing typed model.

This is not an additional sampler, a CLI workflow or a precision-pass gate.
"""

from dataclasses import dataclass

from ffa_methods.censored_reference import (
    CensoredReferenceError,
    CensorEvent,
    FixedScaleReference,
    FixedScaleReferenceInputs,
    ReferenceControls,
    ReferenceRequest,
    fixed_scale_reference,
)
from ffa_methods.contracts import ObservationKind
from ffa_methods.importance_sampling import ExplicitNormalProposal
from ffa_methods.lognormal_posterior import ClassifiedLognormalPosterior


@dataclass(frozen=True, slots=True)
class BoundFixedScaleReference:
    """Retain the complete input identity alongside the mathematical enclosure."""

    model: ClassifiedLognormalPosterior
    proposal: ExplicitNormalProposal
    reference: FixedScaleReference


def bound_fixed_scale_reference(
    *,
    model: ClassifiedLognormalPosterior,
    proposal: ExplicitNormalProposal,
    controls: ReferenceControls,
    requests: tuple[ReferenceRequest, ...],
) -> BoundFixedScaleReference:
    """Extract only included records from the admitted fixed-scale source binding."""
    fixed_log_sd = model.fixed_log_sd
    if fixed_log_sd is None:
        msg = "fixed-scale reference does not admit estimated log-SD"
        raise CensoredReferenceError(msg)
    if model.coordinates != proposal.coordinates:
        msg = "reference proposal and model coordinates differ"
        raise CensoredReferenceError(msg)
    exact = tuple(
        row.value for row in model.series.annual if row.kind == ObservationKind.EXACT and row.value is not None
    )
    events = tuple(
        CensorEvent(lower=row.lower_bound, upper=row.upper_bound)
        for row in model.series.annual
        if row.kind in (ObservationKind.LEFT_CENSORED, ObservationKind.RIGHT_CENSORED, ObservationKind.INTERVAL_CENSORED)
    )
    inputs = FixedScaleReferenceInputs(
        prior_mean=model.prior.means[0],
        prior_variance=model.prior.covariance[0][0],
        fixed_log_sd=fixed_log_sd,
        reference_discharge=model.reference_discharge,
        exact_flows=exact,
        censor_events=events,
        proposal_mean=proposal.means[0],
        proposal_variance=proposal.covariance[0][0],
    )
    return BoundFixedScaleReference(model, proposal, fixed_scale_reference(inputs, controls, requests))
