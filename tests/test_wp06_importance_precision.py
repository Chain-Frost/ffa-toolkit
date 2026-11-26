"""Independent conjugate reference and fixed-budget precision failure tests."""

from dataclasses import replace
from math import exp, fsum, isclose, log, sqrt
from statistics import NormalDist

import pytest

from ffa_methods import importance_precision
from ffa_methods.contracts import EvidenceStatus, ResultMeaning, RunProvenance
from ffa_methods.importance_precision import PrecisionPlan, PrecisionRequest, PrecisionState, audit_lognormal_precision
from ffa_methods.importance_sampling import (
    ExplicitNormalProposal,
    ExplicitPosteriorTarget,
    ExplicitSamplingControls,
    IndependentSamplingResult,
    independent_importance_sample,
)
from ffa_methods.lognormal_posterior import (
    LOG_SD_COORDINATE,
    MEAN_COORDINATE,
    SOURCE_CLASSIFICATION,
    ClassifiedLognormalPosterior,
    bind_exact_lognormal_csv,
)
from ffa_methods.priors import JointGaussianPrior, LegacyPriorMode


def _model(*, free: bool = False) -> ClassifiedLognormalPosterior:
    data = ("Year,AMS\n" + "".join(f"{2000 + i},{exp(i):.17g}\n" for i in (1, 2, 3))).encode()
    return bind_exact_lognormal_csv(
        data,
        source="synthetic exact lognormal observations",
        discharge_unit="m3/s",
        year_definition="calendar year",
        source_classification=SOURCE_CLASSIFICATION,
        reference_discharge=1.0,
        fixed_log_sd=None if free else 0.0,
        prior=JointGaussianPrior(
            coordinates=(MEAN_COORDINATE, LOG_SD_COORDINATE) if free else (MEAN_COORDINATE,),
            means=(0.0, 0.0) if free else (0.0,),
            covariance=((1.0, 0.0), (0.0, 1.0)) if free else ((1.0,),),
            source="synthetic supplied Gaussian prior",
            mode=LegacyPriorMode.SUPPLIED,
        ),
    )


def _proposal(*, variance: float = 1.0, free: bool = False) -> ExplicitNormalProposal:
    return ExplicitNormalProposal(
        coordinates=(MEAN_COORDINATE, LOG_SD_COORDINATE) if free else (MEAN_COORDINATE,),
        means=(1.5, 0.0) if free else (1.5,),
        covariance=((variance, 0.0), (0.0, 1.0)) if free else ((variance,),),
        source="explicit synthetic broader proposal",
    )


def _provenance(model: ClassifiedLognormalPosterior) -> RunProvenance:
    return RunProvenance(
        compatibility_profile="independent synthetic model",
        method_id="importance-sampling-reference",
        method_version="test",
        package_version="test",
        evidence_status=EvidenceStatus.OBSERVED_ARTIFACT,
        input_sha256=model.source_sha256,
    )


def _plan(*, draws: int = 1800) -> PrecisionPlan:
    return PrecisionPlan(
        primary=ExplicitSamplingControls(
            draw_count=draws, random_seed=51, minimum_effective_size=draws / 3, maximum_single_weight=0.05
        ),
        audit_seeds=(52, 53),
        maximum_total_draws=3 * draws,
        maximum_scalar_evaluations=3 * draws * 400,
        alpha=0.05,
        minimum_side_count=8,
        minimum_side_ess=8.0,
        maximum_log_ratio=1.2,
        criterion_source="predeclared synthetic conjugate test only",
    )


def _requests() -> tuple[PrecisionRequest, ...]:
    p, x = 0.9, exp(2)
    return (
        PrecisionRequest(meaning="parameter_mean", absolute_tolerance=0.16),
        PrecisionRequest(meaning=ResultMeaning.POSTERIOR_MEAN_QUANTILE, non_exceedance=p, absolute_tolerance=6.0),
        PrecisionRequest(meaning=ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY, discharge=x, absolute_tolerance=0.09),
        PrecisionRequest(meaning=ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY, discharge=x, absolute_tolerance=0.09),
        PrecisionRequest(meaning=ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP, discharge=x, absolute_tolerance=1.0),
        PrecisionRequest(
            meaning=ResultMeaning.LOG10_QUANTILE_POSTERIOR_MOMENT,
            non_exceedance=p,
            log_moment="mean",
            absolute_tolerance=0.09,
        ),
        PrecisionRequest(meaning=ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE, non_exceedance=p, absolute_tolerance=9.0),
        PrecisionRequest(meaning=ResultMeaning.POSTERIOR_MEDIAN_QUANTILE, non_exceedance=p, absolute_tolerance=7.0),
        PrecisionRequest(
            meaning=ResultMeaning.QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT,
            non_exceedance=p,
            interval_mass=0.8,
            side="lower",
            absolute_tolerance=2.5,
        ),
        PrecisionRequest(
            meaning=ResultMeaning.QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT,
            non_exceedance=p,
            interval_mass=0.8,
            side="upper",
            absolute_tolerance=8.0,
        ),
        PrecisionRequest(
            meaning=ResultMeaning.RETURN_PERIOD_MONTE_CARLO_PROBABILITY_LIMIT,
            discharge=x,
            interval_mass=0.8,
            side="upper",
            absolute_tolerance=3.0,
        ),
    )


def test_independent_fixed_budget_all_admitted_outputs_against_conjugate_oracle() -> None:
    model, proposal, plan, requests = _model(), _proposal(), _plan(), _requests()
    result = audit_lognormal_precision(
        model=model, proposal=proposal, provenance=_provenance(model), plan=plan, requests=requests
    )
    # Completing the square for log observations 1,2,3, sigma=1 and N(0,1)
    # prior gives posterior N(1.5,1/4) and predictive N(1.5,5/4).
    normal = NormalDist()
    z = normal.inv_cdf(0.9)
    f = normal.cdf(0.5 / sqrt(1.25))
    expected = (
        1.5,
        exp(1.5 + z + 0.125),
        f,
        1 - f,
        1 / (1 - f),
        (1.5 + z) / log(10),
        exp(1.5 + sqrt(1.25) * z),
        exp(1.5 + z),
        exp(1.5 + z + 0.5 * normal.inv_cdf(0.1)),
        exp(1.5 + z + 0.5 * z),
        1 / normal.cdf(1.5 - 0.5 * z - 2),
    )
    assert result.admission is not None
    assert isclose(result.admission.mean, 1.5, abs_tol=1e-14)
    assert isclose(result.admission.variance, 0.25, abs_tol=1e-14)
    assert isclose(result.admission.log_ratio_bound, log(2), abs_tol=1e-14)
    assert tuple(attempt.seed for attempt in result.attempts) == plan.seeds
    assert all(attempt.attempted and attempt.completed_draws == 1800 for attempt in result.attempts)
    assert result.state == PrecisionState.CONDITIONAL_PASS, result.states
    assert result.critical_value == -normal.inv_cdf(plan.alpha / (2 * 2 * len(requests)))
    for attempt in result.attempts:
        for output, reference in zip(attempt.outputs, expected, strict=True):
            assert output.reference is not None and isclose(output.reference, reference, rel_tol=1e-12)
            assert output.estimate is not None and abs(output.estimate - reference) < output.request.absolute_tolerance
            assert output.window is not None and output.window.contains(reference)
            for check in output.scalar_checks:
                assert check.approximate_mcse is not None and check.approximate_mcse > 0
                assert check.positive.count >= plan.minimum_side_count
                assert check.negative.effective_size >= plan.minimum_side_ess
    # The first audit stream uses the primary window, not its own estimate.
    assert tuple(item.window for item in result.attempts[1].outputs) == tuple(
        item.window for item in result.attempts[0].outputs
    )
    # The empirical variance of the scalar parameter mean should agree with
    # direct weighted arithmetic on replayed original sampler draws.
    replay = independent_importance_sample(
        target=model.target(),
        proposal=proposal,
        controls=replace(plan.primary, random_seed=52),
        provenance=_provenance(model),
    )
    mean = fsum(w * draw[0] for w, draw in zip(replay.particles.weights, replay.particles.draws, strict=True))
    expected_mcse = sqrt(
        fsum((w * (draw[0] - mean)) ** 2 for w, draw in zip(replay.particles.weights, replay.particles.draws, strict=True))
    )
    diagnostic = result.attempts[1].outputs[0].scalar_checks[0]
    assert diagnostic.approximate_mcse is not None
    assert isclose(diagnostic.approximate_mcse, expected_mcse, rel_tol=1e-12)
    # With proposal variance s²=1 and posterior variance v=1/4,
    # pi²/g is a constant times N(m, u); u=v*s²/(2*s²-v).
    ratio_second = 1 / sqrt(0.25 * 1.75)
    tilted_variance = 0.25 / 1.75
    assert isclose(diagnostic.approximate_mcse, sqrt(ratio_second * tilted_variance / 1800), rel_tol=0.15)


def test_explicit_preflight_seed_and_budget_validation() -> None:
    plan = _plan(draws=300)
    with pytest.raises(ValueError, match="distinct"):
        replace(plan, audit_seeds=(51, 53))
    with pytest.raises(ValueError, match="resource"):
        replace(plan, maximum_total_draws=899)
    with pytest.raises(ValueError, match="evaluations"):
        replace(plan, maximum_scalar_evaluations=1).preflight(_requests())
    with pytest.raises(ValueError, match="strictly between"):
        PrecisionRequest(
            meaning=ResultMeaning.POSTERIOR_MEAN_QUANTILE, non_exceedance=1.0, absolute_tolerance=1.0
        )
    with pytest.raises(ValueError, match="side"):
        PrecisionRequest(
            meaning=ResultMeaning.QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT,
            non_exceedance=0.9,
            interval_mass=0.9,
            absolute_tolerance=1.0,
        )


def test_unadmitted_proposal_and_divergent_free_scale_mean() -> None:
    requests = (
        PrecisionRequest(meaning=ResultMeaning.POSTERIOR_MEAN_QUANTILE, non_exceedance=0.99, absolute_tolerance=1),
        PrecisionRequest(meaning=ResultMeaning.POSTERIOR_MEDIAN_QUANTILE, non_exceedance=0.99, absolute_tolerance=1),
    )
    fixed = _model()
    for proposal in (_proposal(variance=0.125), replace(_proposal(), means=(8.0,))):
        result = audit_lognormal_precision(
            model=fixed, proposal=proposal, provenance=_provenance(fixed), plan=_plan(draws=100), requests=requests
        )
        assert result.states == (PrecisionState.INDETERMINATE,) * 2
        assert len(result.attempts) == 3
        assert all(not attempt.attempted and attempt.completed_draws is None for attempt in result.attempts)
    free = _model(free=True)
    result = audit_lognormal_precision(
        model=free, proposal=_proposal(free=True), provenance=_provenance(free), plan=_plan(draws=100), requests=requests
    )
    assert result.states == (PrecisionState.UNDEFINED, PrecisionState.INDETERMINATE)
    assert all(not attempt.attempted and attempt.outputs[0].estimate is None for attempt in result.attempts)


def test_failure_retention_and_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    model, proposal, plan = _model(), _proposal(), _plan(draws=300)
    seen: list[int] = []

    def fail_one_stream(
        *,
        proposal: ExplicitNormalProposal,
        target: ExplicitPosteriorTarget,
        controls: ExplicitSamplingControls,
        provenance: RunProvenance,
    ) -> IndependentSamplingResult:
        seen.append(controls.random_seed)
        if controls.random_seed == 52:
            msg = "synthetic density failure"
            raise ValueError(msg)
        return independent_importance_sample(proposal=proposal, target=target, controls=controls, provenance=provenance)

    monkeypatch.setattr(importance_precision, "independent_importance_sample", fail_one_stream)
    requests = (PrecisionRequest(meaning="parameter_mean", absolute_tolerance=0.5),)
    first = audit_lognormal_precision(
        model=model, proposal=proposal, provenance=_provenance(model), plan=plan, requests=requests
    )
    second = audit_lognormal_precision(
        model=model, proposal=proposal, provenance=_provenance(model), plan=plan, requests=requests
    )
    assert first == second
    assert seen == [*plan.seeds, *plan.seeds]
    assert first.state == PrecisionState.EXECUTION_FAILED
    assert first.attempts[1].completed_draws is None and first.attempts[1].reason == "synthetic density failure"
    assert first.attempts[2].completed_draws == 300
    assert first.attempts[2].outputs[0].estimate is not None


def test_tight_tolerance_and_unimplemented_propagation_do_not_pass() -> None:
    model = _model()
    requests = (
        PrecisionRequest(meaning="parameter_mean", absolute_tolerance=1e-7),
        PrecisionRequest(
            meaning=ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS, non_exceedance=0.9, absolute_tolerance=100
        ),
        PrecisionRequest(
            meaning=ResultMeaning.LOG10_QUANTILE_POSTERIOR_MOMENT,
            non_exceedance=0.9,
            log_moment="standard_deviation",
            absolute_tolerance=100,
        ),
    )
    result = audit_lognormal_precision(
        model=model, proposal=_proposal(), provenance=_provenance(model), plan=_plan(draws=400), requests=requests
    )
    assert result.states == (
        PrecisionState.CRITERIA_NOT_MET,
        PrecisionState.INDETERMINATE,
        PrecisionState.INDETERMINATE,
    )
    assert all(item.estimate is not None for item in result.attempts[0].outputs)


def test_rare_direct_sf_and_unresolved_probability_do_not_silently_pass() -> None:
    model = _model()
    requests = (
        PrecisionRequest(
            meaning=ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY, discharge=exp(12), absolute_tolerance=1e-21
        ),
        PrecisionRequest(
            meaning=ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY, discharge=exp(12), absolute_tolerance=0.01
        ),
        PrecisionRequest(
            meaning=ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP, discharge=exp(40), absolute_tolerance=1e100
        ),
    )
    result = audit_lognormal_precision(
        model=model, proposal=_proposal(), provenance=_provenance(model), plan=_plan(draws=600), requests=requests
    )
    rare, saturated, unresolved = result.attempts[0].outputs
    assert rare.estimate is not None and 0 < rare.estimate < 1e-15
    assert saturated.state == PrecisionState.INDETERMINATE
    assert unresolved.state in (PrecisionState.INDETERMINATE, PrecisionState.EXECUTION_FAILED)
    assert result.state != PrecisionState.CONDITIONAL_PASS
