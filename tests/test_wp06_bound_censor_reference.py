"""Source binding and proposal identity for fixed-scale reference evidence."""

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256

import pytest

from ffa_methods.censored_reference import CensoredReferenceError, ReferenceControls, ReferenceKind, ReferenceRequest
from ffa_methods.contracts import AnnualObservation, ExposureKind, ObservationKind, ObservationSeries
from ffa_methods.importance_censor_reference import bound_fixed_scale_reference
from ffa_methods.importance_sampling import ExplicitNormalProposal
from ffa_methods.lognormal_posterior import (
    LOG_SD_COORDINATE,
    MEAN_COORDINATE,
    SOURCE_CLASSIFICATION,
    ClassifiedLognormalPosterior,
)
from ffa_methods.priors import JointGaussianPrior, LegacyPriorMode


def _model() -> ClassifiedLognormalPosterior:
    return ClassifiedLognormalPosterior(
        series=ObservationSeries(
            discharge_unit="m3/s",
            year_definition="calendar year",
            annual=(
                AnnualObservation(
                    year=2000, kind=ObservationKind.LEFT_CENSORED, exposure=ExposureKind.SYSTEMATIC, upper_bound=2.0
                ),
                AnnualObservation(year=2001, kind=ObservationKind.MISSING, exposure=ExposureKind.SYSTEMATIC),
                AnnualObservation(
                    year=2002, kind=ObservationKind.EXCLUDED, exposure=ExposureKind.SYSTEMATIC, value=0.0
                ),
            ),
        ),
        source="synthetic explicitly classified annual records",
        source_sha256=sha256(b"synthetic all-censored source identity").hexdigest(),
        source_classification=SOURCE_CLASSIFICATION,
        reference_discharge=2.0,
        fixed_log_sd=0.0,
        prior=JointGaussianPrior(
            coordinates=(MEAN_COORDINATE,),
            means=(0.0,),
            covariance=((1.0,),),
            source="synthetic explicit Gaussian mean prior",
            mode=LegacyPriorMode.SUPPLIED,
        ),
    )


def test_reference_preserves_bound_identity_and_missing_excluded_rows() -> None:
    model = _model()
    proposal = ExplicitNormalProposal(
        coordinates=model.coordinates, means=(0.0,), covariance=((1.0,),), source="synthetic prior proposal"
    )
    controls = ReferenceControls(
        extent=8, dyadic_level=2, maximum_cells=128, maximum_cell_terms=1000, maximum_log_ratio=2.0
    )
    requests = (ReferenceRequest(kind=ReferenceKind.EXPECTED_F, argument=2.0),)
    result = bound_fixed_scale_reference(model=model, proposal=proposal, controls=controls, requests=requests)
    assert result.model is model and result.proposal is proposal
    assert len(result.model.series.annual) == 3
    assert not result.reference.inputs.exact_flows and len(result.reference.inputs.censor_events) == 1
    assert result.reference.inputs.censor_events[0].upper == 2.0
    assert result.reference.normalizer.contains(Decimal("0.5"))
    bounds = result.reference.values[0].bounds
    assert bounds is not None
    assert bounds.contains(Decimal(2) / Decimal(3))
    assert result.reference.ratio_eligible


def test_reference_rejects_free_scale_and_mismatched_proposal() -> None:
    model = _model()
    controls = ReferenceControls(
        extent=4, dyadic_level=1, maximum_cells=32, maximum_cell_terms=1000, maximum_log_ratio=2.0
    )
    proposal = ExplicitNormalProposal(
        coordinates=("wrong",), means=(0.0,), covariance=((1.0,),), source="synthetic wrong coordinate"
    )
    with pytest.raises(CensoredReferenceError, match="coordinates"):
        bound_fixed_scale_reference(model=model, proposal=proposal, controls=controls, requests=())
    free = replace(
        model,
        fixed_log_sd=None,
        prior=JointGaussianPrior(
            coordinates=(MEAN_COORDINATE, LOG_SD_COORDINATE),
            means=(0.0, 0.0),
            covariance=((1.0, 0.0), (0.0, 1.0)),
            source="synthetic explicit joint Gaussian prior",
            mode=LegacyPriorMode.SUPPLIED,
        ),
    )
    with pytest.raises(CensoredReferenceError, match="estimated log-SD"):
        bound_fixed_scale_reference(model=free, proposal=proposal, controls=controls, requests=())
