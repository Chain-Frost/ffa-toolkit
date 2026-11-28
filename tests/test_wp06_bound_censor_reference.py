"""Source binding and proposal identity for fixed-scale reference evidence."""

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256

import pytest

from ffa_methods import importance_precision
from ffa_methods.censored_reference import (
    CensoredReferenceError,
    ReferenceBounds,
    ReferenceControls,
    ReferenceKind,
    ReferenceRequest,
)
from ffa_methods.contracts import (
    AnnualObservation,
    EvidenceStatus,
    ExposureKind,
    ObservationKind,
    ObservationSeries,
    ResultMeaning,
    RunProvenance,
)
from ffa_methods.importance_censor_reference import bound_fixed_scale_reference
from ffa_methods.importance_precision import Bounds, PrecisionPlan, PrecisionRequest, PrecisionState
from ffa_methods.importance_sampling import (
    ExplicitNormalProposal,
    ExplicitPosteriorTarget,
    ExplicitSamplingControls,
    IndependentSamplingResult,
)
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
    assert result.model is model
    assert result.proposal is proposal
    assert len(result.model.series.annual) == 3
    assert not result.reference.inputs.exact_flows
    assert len(result.reference.inputs.censor_events) == 1
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


def _audit_inputs() -> tuple[ClassifiedLognormalPosterior, ExplicitNormalProposal, RunProvenance, PrecisionPlan]:
    model = _model()
    proposal = ExplicitNormalProposal(
        coordinates=model.coordinates, means=(0.0,), covariance=((1.0,),), source="synthetic prior proposal"
    )
    provenance = RunProvenance(
        compatibility_profile="synthetic fixed-scale left censoring",
        method_id="importance-sampling-reference",
        method_version="test",
        package_version="test",
        evidence_status=EvidenceStatus.OBSERVED_ARTIFACT,
        input_sha256=model.source_sha256,
    )
    plan = PrecisionPlan(
        primary=ExplicitSamplingControls(
            draw_count=1800, random_seed=51, minimum_effective_size=600, maximum_single_weight=0.05
        ),
        audit_seeds=(52, 53),
        maximum_total_draws=5400,
        maximum_scalar_evaluations=1_500_000,
        alpha=0.05,
        minimum_side_count=8,
        minimum_side_ess=8.0,
        maximum_log_ratio=1.2,
        criterion_source="synthetic single-event Beta check",
        censored_reference_controls=ReferenceControls(
            extent=8, dyadic_level=8, maximum_cells=4096, maximum_cell_terms=21_000, maximum_log_ratio=2.0
        ),
    )
    return model, proposal, provenance, plan


def test_fixed_budget_censored_audit_uses_complete_reference_enclosures() -> None:
    model, proposal, provenance, plan = _audit_inputs()
    requests = (
        PrecisionRequest(meaning="parameter_mean", absolute_tolerance=0.16),
        PrecisionRequest(
            meaning=ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY, discharge=2.0, absolute_tolerance=0.09
        ),
        PrecisionRequest(
            meaning=ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY, discharge=2.0, absolute_tolerance=0.09
        ),
        PrecisionRequest(
            meaning=ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE, non_exceedance=0.9, absolute_tolerance=1.0
        ),
    )
    result = importance_precision.audit_lognormal_precision(
        model=model, proposal=proposal, provenance=provenance, plan=plan, requests=requests
    )
    assert result.completed_count == 3
    assert result.states == (PrecisionState.CONDITIONAL_PASS,) * 3 + (PrecisionState.INDETERMINATE,)
    assert result.censored_reference is not None
    assert result.censored_reference.reference.reserved_cell_terms == 5 * 4096 + 19
    for index, oracle in enumerate((Decimal("-0.56418958354775628695"), Decimal(2) / 3, Decimal(1) / 3)):
        for attempt in result.attempts:
            output = attempt.outputs[index]
            assert output.reference is None
            reference, window = output.reference_interval, output.window
            assert reference is not None
            assert reference.contains(oracle)
            assert window is not None
            assert window.lower <= float(reference.lower) <= float(reference.upper) <= window.upper
            if attempt.seed != plan.primary.random_seed:
                (check,) = output.scalar_checks
                assert check.reference == reference
                assert check.interval is not None
                assert check.interval.lower <= float(reference.lower)
                assert float(reference.upper) <= check.interval.upper


def test_censored_gate_accounts_for_budget_and_uses_stricter_ratio_ceiling() -> None:
    model, proposal, provenance, plan = _audit_inputs()
    controls = plan.censored_reference_controls
    assert controls is not None
    request = PrecisionRequest(meaning="parameter_mean", absolute_tolerance=0.16)
    for variant, expected in (
        (replace(plan, maximum_log_ratio=0.5), "normalized ratio ceiling exceeded"),
        (replace(plan, censored_reference_controls=replace(controls, maximum_cell_terms=1)), "cell-term budget"),
    ):
        result = importance_precision.audit_lognormal_precision(
            model=model, proposal=proposal, provenance=provenance, plan=variant, requests=(request,)
        )
        assert result.attempted_count == 0
        assert result.states == (PrecisionState.INDETERMINATE,)
        assert tuple(attempt.seed for attempt in result.attempts) == plan.seeds
        assert all(attempt.reason is not None and expected in attempt.reason for attempt in result.attempts)


def test_reference_overlap_cannot_pass_on_midpoint_alone() -> None:
    reference = ReferenceBounds(Decimal("0.2"), Decimal("0.8"))
    assert (
        importance_precision._reference_containment(Bounds(0.3, 0.7), reference)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        == PrecisionState.INDETERMINATE
    )


def test_censored_sampler_failure_preserves_schedule_and_unsupported_state(monkeypatch: pytest.MonkeyPatch) -> None:
    model, proposal, provenance, plan = _audit_inputs()
    controls = plan.censored_reference_controls
    assert controls is not None
    plan = replace(plan, censored_reference_controls=replace(controls, dyadic_level=2))
    run_sampler = importance_precision.independent_importance_sample

    def fail_one_stream(
        *,
        proposal: ExplicitNormalProposal,
        target: ExplicitPosteriorTarget,
        controls: ExplicitSamplingControls,
        provenance: RunProvenance,
    ) -> IndependentSamplingResult:
        if controls.random_seed == 52:
            msg = "synthetic scheduled failure"
            raise ValueError(msg)
        return run_sampler(proposal=proposal, target=target, controls=controls, provenance=provenance)

    monkeypatch.setattr(importance_precision, "independent_importance_sample", fail_one_stream)
    requests = (
        PrecisionRequest(meaning="parameter_mean", absolute_tolerance=0.16),
        PrecisionRequest(meaning=ResultMeaning.POSTERIOR_MEAN_QUANTILE, non_exceedance=0.9, absolute_tolerance=6.0),
    )
    result = importance_precision.audit_lognormal_precision(
        model=model, proposal=proposal, provenance=provenance, plan=plan, requests=requests
    )
    assert tuple(attempt.seed for attempt in result.attempts) == plan.seeds
    assert result.attempted_count == 3
    assert result.completed_count == 2
    assert result.failed_count == 1
    assert result.states == (PrecisionState.EXECUTION_FAILED, PrecisionState.INDETERMINATE)
    assert result.attempts[1].reason == "synthetic scheduled failure"
