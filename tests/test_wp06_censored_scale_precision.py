"""Independent censored/free-scale numerical and production admission checks."""

from dataclasses import replace
from hashlib import sha256
from math import erfc, exp, fsum, isclose, log, sqrt
from statistics import NormalDist

import pytest

from ffa_methods import importance_precision
from ffa_methods.contracts import (
    AnnualObservation,
    EvidenceStatus,
    ExposureKind,
    ObservationKind,
    ObservationSeries,
    ResultMeaning,
    RunProvenance,
)
from ffa_methods.importance_moment_domain import MomentState, assess_lognormal_moments
from ffa_methods.importance_precision import PrecisionPlan, PrecisionRequest, PrecisionState, audit_lognormal_precision
from ffa_methods.importance_ratio_envelope import RatioEnvelopeError, normalized_log_ratio_bound
from ffa_methods.importance_reflection_reference import reflected_discharge, reflected_discharge_bounds
from ffa_methods.importance_sampling import (
    ExplicitNormalProposal,
    ExplicitSamplingControls,
    independent_importance_sample,
)
from ffa_methods.lognormal_posterior import (
    LOG_SD_COORDINATE,
    MEAN_COORDINATE,
    SOURCE_CLASSIFICATION,
    ClassifiedLognormalPosterior,
)
from ffa_methods.priors import JointGaussianPrior, LegacyPriorMode

_NORMAL = NormalDist()
_UPPER_Z = _NORMAL.inv_cdf(0.9)
_LOWER_Z = _NORMAL.inv_cdf(0.2)
_DRAWS = 4000
_SEEDS = (51, 52, 53)


def _row(
    year: int,
    kind: ObservationKind,
    *,
    value: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> AnnualObservation:
    return AnnualObservation(
        year=year, kind=kind, exposure=ExposureKind.SYSTEMATIC, value=value, lower_bound=lower, upper_bound=upper
    )


def _model(*, free: bool) -> ClassifiedLognormalPosterior:
    return ClassifiedLognormalPosterior(
        series=ObservationSeries(
            discharge_unit="m3/s",
            year_definition="calendar year",
            annual=(
                _row(2000, ObservationKind.EXACT, value=exp(-0.4)),
                _row(2001, ObservationKind.EXACT, value=exp(0.5)),
                _row(2002, ObservationKind.LEFT_CENSORED, upper=exp(-0.2)),
                _row(2003, ObservationKind.RIGHT_CENSORED, lower=exp(0.1)),
                _row(2004, ObservationKind.INTERVAL_CENSORED, lower=exp(-0.5), upper=exp(0.5)),
                _row(2005, ObservationKind.MISSING),
                _row(2006, ObservationKind.EXCLUDED, value=0.0),
            ),
        ),
        source="original synthetic mixed systematic records",
        source_sha256=sha256(b"original synthetic mixed systematic records").hexdigest(),
        source_classification=SOURCE_CLASSIFICATION,
        reference_discharge=1.0,
        fixed_log_sd=None if free else 0.0,
        prior=JointGaussianPrior(
            coordinates=(MEAN_COORDINATE, LOG_SD_COORDINATE) if free else (MEAN_COORDINATE,),
            means=(0.0, 0.0) if free else (0.0,),
            covariance=((0.16, 0.0), (0.0, 0.0144)) if free else ((0.16,),),
            source="explicit synthetic Gaussian prior",
            mode=LegacyPriorMode.SUPPLIED,
        ),
    )


def _proposal(model: ClassifiedLognormalPosterior) -> ExplicitNormalProposal:
    # Tilt the Gaussian prior by exp(-2 eta) for the two exact observations.
    return ExplicitNormalProposal(
        coordinates=model.coordinates,
        means=(0.0, -0.0288) if model.fixed_log_sd is None else (0.0,),
        covariance=model.prior.covariance,
        source="explicit Gaussian reference proposal, no curvature default",
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


def _plan() -> PrecisionPlan:
    return PrecisionPlan(
        primary=ExplicitSamplingControls(
            draw_count=_DRAWS, random_seed=_SEEDS[0], minimum_effective_size=3000.0, maximum_single_weight=0.01
        ),
        audit_seeds=_SEEDS[1:],
        maximum_total_draws=len(_SEEDS) * _DRAWS,
        maximum_scalar_evaluations=len(_SEEDS) * _DRAWS * 100,
        alpha=0.05,
        minimum_side_count=8,
        minimum_side_ess=8.0,
        maximum_log_ratio=10.0,
        criterion_source="synthetic scalar regression only; normalized ratio not admitted",
    )


def _sf(z: float) -> float:
    return 0.5 * erfc(z / sqrt(2))


def _cdf(z: float) -> float:
    return _sf(-z)


def _weight(mu: float, eta: float) -> float:
    # Independent log-space likelihood with constant exact-flow Jacobians
    # and exp(-2 eta) already absorbed into the proposal.
    sd = exp(eta)
    a, b = (-0.5 - mu) / sd, (0.5 - mu) / sd
    interval = _sf(a) - _sf(b) if a > 0 else _cdf(b) - _cdf(a)
    return (
        exp(-0.5 * (((-0.4 - mu) / sd) ** 2 + ((0.5 - mu) / sd) ** 2))
        * _cdf((-0.2 - mu) / sd)
        * _sf((0.1 - mu) / sd)
        * interval
    )


def _independent_values(mu: float, eta: float, *, free: bool) -> tuple[float, ...]:
    sd = exp(eta)
    log_q = (mu + _UPPER_Z * sd) / log(10)
    common = (
        mu,
        _sf((0.35 - mu) / sd),
        _cdf((0.35 - mu) / sd),
        _sf((4.0 - mu) / sd),
        log_q,
        log_q**2,
        exp(mu + _LOWER_Z * sd),
        _sf((1.1 - mu) / sd),
        _sf((1.5 - mu) / sd),
    )
    return (*common, eta) if free else (*common, exp(mu + _UPPER_Z))


def _reference(
    points: int, *, free: bool, indicator_bound: float | None = None
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    # Integrate against g over nine marginal SDs. For indicators split
    # the mu integral at the exact threshold instead of integrating a jump.
    step = 18 / points
    standard = tuple(-9 + (i + 0.5) * step for i in range(points))
    etas = tuple((-0.0288 + 0.12 * v, _NORMAL.pdf(v) * step) for v in standard) if free else ((0.0, 1.0),)
    cells: list[tuple[float, float, tuple[float, ...]]] = []
    for eta, eta_mass in etas:
        cut = 0.0 if indicator_bound is None else max(-3.6, min(3.6, indicator_bound - _UPPER_Z * exp(eta)))
        ranges = ((-3.6, 3.6),) if indicator_bound is None else ((-3.6, cut), (cut, 3.6))
        for low, high in ranges:
            width = (high - low) / points
            for i in range(points):
                mu = low + (i + 0.5) * width
                mass = eta_mass * _NORMAL.pdf(mu / 0.4) * width / 0.4
                values = _independent_values(mu, eta, free=free) if indicator_bound is None else (float(mu > cut),)
                cells.append((mass, _weight(mu, eta), values))
    normalizer = fsum(mass * weight for mass, weight, _ in cells)
    means = tuple(
        fsum(mass * weight * values[k] for mass, weight, values in cells) / normalizer for k in range(len(cells[0][2]))
    )
    variances = tuple(
        fsum(mass * weight**2 * (values[k] - mean) ** 2 for mass, weight, values in cells) / normalizer**2
        for k, mean in enumerate(means)
    )
    return means, variances


def _actual_values(model: ClassifiedLognormalPosterior, draw: tuple[float, ...]) -> tuple[float, ...]:
    kernel = model.kernel(draw)
    upper_q = kernel.ppf(0.9)
    log_q = log(upper_q) / log(10)
    common = (
        draw[0],
        kernel.sf(exp(0.35)),
        kernel.cdf(exp(0.35)),
        kernel.sf(exp(4.0)),
        log_q,
        log_q**2,
        kernel.ppf(0.2),
        kernel.sf(exp(1.1)),
        kernel.sf(exp(1.5)),
    )
    return (*common, draw[1]) if model.fixed_log_sd is None else (*common, upper_q)


def _references(*, free: bool) -> tuple[tuple[float, ...], tuple[float, ...]]:
    coarse, coarse_v = _reference(81, free=free)
    means, variances = _reference(161, free=free)
    for a, b in zip((*coarse, *coarse_v), (*means, *variances), strict=True):
        assert isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)
    # Fixed-point survival brackets, not a density-based quantile MCSE.
    assert means[7] > 0.1 > means[8]
    for bound in (1.1, 1.5):
        a, av = _reference(161, free=free, indicator_bound=bound)
        b, bv = _reference(321, free=free, indicator_bound=bound)
        assert isclose(a[0], b[0], rel_tol=0, abs_tol=1e-4)
        assert isclose(av[0], bv[0], rel_tol=0, abs_tol=1e-4)
        means = (*means, *b)
        variances = (*variances, *bv)
    assert means[-2] > 0.5 > means[-1]
    return means, variances


@pytest.mark.parametrize("free", [False, True])
def test_mixed_censor_scalar_precision_against_independent_quadrature(*, free: bool) -> None:
    model = _model(free=free)
    proposal = _proposal(model)
    plan = _plan()
    means, variances = _references(free=free)
    # Exercise the real scalar diagnostic with actual generated particles.
    # This test-only call intentionally bypasses production model admission.
    scalar_check = vars(importance_precision)["_scalar_check"]
    for seed in _SEEDS:
        run = independent_importance_sample(
            proposal=proposal,
            target=model.target(),
            controls=replace(plan.primary, random_seed=seed),
            provenance=_provenance(model),
        )
        rows = tuple(_actual_values(model, draw) for draw in run.particles.draws)
        quantiles = tuple(model.kernel(draw).ppf(0.9) for draw in run.particles.draws)
        columns = (*zip(*rows, strict=True), *(tuple(float(q > exp(b)) for q in quantiles) for b in (1.1, 1.5)))
        for values, expected, coefficient in zip(columns, means, variances, strict=True):
            assert coefficient > 0
            reference_se = sqrt(coefficient / _DRAWS)
            check = scalar_check(run.particles, values, expected, plan, 6.0)
            assert check.approximate_mcse is not None
            assert check.interval is not None
            assert abs(check.estimate - expected) <= 6 * reference_se
            # Wide regression guard for these synthetic cases, not an
            # acceptance threshold or a proof of finite-sample coverage.
            assert 0.5 * reference_se < check.approximate_mcse < 2 * reference_se


@pytest.mark.parametrize("free", [False, True])
def test_moment_eligibility_does_not_admit_runtime_precision(*, free: bool) -> None:
    model = _model(free=free)
    proposal = _proposal(model)
    aep = PrecisionRequest(
        meaning=ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY, discharge=exp(4.0), absolute_tolerance=1e-5
    )
    upper = PrecisionRequest(meaning=ResultMeaning.POSTERIOR_MEAN_QUANTILE, non_exceedance=0.9, absolute_tolerance=0.1)
    lower = replace(upper, non_exceedance=0.2)
    for request in (aep, lower):
        result = assess_lognormal_moments(model, proposal, request)
        assert result.posterior == result.scalar_second_moments == MomentState.FINITE
    expected = MomentState.DIVERGENT if free else MomentState.FINITE
    result = assess_lognormal_moments(model, proposal, upper)
    assert result.posterior == result.scalar_second_moments == expected
    # At the strict half-prior boundary the sufficient condition fails.
    narrow = replace(proposal, covariance=tuple(tuple(v / 2 for v in row) for row in proposal.covariance))
    result = assess_lognormal_moments(model, narrow, aep)
    assert result.posterior == MomentState.FINITE
    assert result.scalar_second_moments == MomentState.UNESTABLISHED
    audit = audit_lognormal_precision(
        model=model, proposal=proposal, provenance=_provenance(model), plan=_plan(), requests=(aep, upper)
    )
    assert audit.attempted_count == 0
    assert audit.states == (
        PrecisionState.INDETERMINATE,
        PrecisionState.UNDEFINED if free else PrecisionState.INDETERMINATE,
    )


def test_correlated_covariance_requires_matrix_not_marginal_dominance() -> None:
    model = _model(free=True)
    model = replace(model, prior=replace(model.prior, covariance=((1.0, 0.75), (0.75, 1.0))))
    proposal = replace(_proposal(model), covariance=((0.75, -0.5), (-0.5, 0.75)))
    request = PrecisionRequest(meaning="parameter_mean", absolute_tolerance=0.1)
    # Both proposal marginals exceed half the prior marginals, but the
    # difference has a negative eigenvalue.
    assert assess_lognormal_moments(model, proposal, request).scalar_second_moments == MomentState.UNESTABLISHED
    proposal = replace(proposal, covariance=((1.0, 0.75), (0.75, 1.0)))
    assert assess_lognormal_moments(model, proposal, request).scalar_second_moments == MomentState.FINITE


@pytest.mark.parametrize("free", [False, True])
def test_reflection_admits_only_independently_known_outputs(*, free: bool) -> None:
    model = _model(free=free)
    model = replace(
        model,
        series=replace(
            model.series,
            annual=(
                _row(2000, ObservationKind.LEFT_CENSORED, upper=1.0),
                _row(2001, ObservationKind.RIGHT_CENSORED, lower=1.0),
            ),
        ),
    )
    proposal = replace(_proposal(model), means=model.prior.means)
    mean = PrecisionRequest(meaning="parameter_mean", absolute_tolerance=0.1)
    aep = PrecisionRequest(
        meaning=ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY, discharge=1.0, absolute_tolerance=0.02
    )
    period = replace(aep, meaning=ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP, absolute_tolerance=0.1)
    at_mean = PrecisionRequest(
        meaning=ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS, non_exceedance=0.5, absolute_tolerance=0.1
    )
    unsupported = replace(aep, discharge=1.25)
    upper = replace(at_mean, meaning=ResultMeaning.POSTERIOR_MEAN_QUANTILE, non_exceedance=0.9)
    median = replace(at_mean, meaning=ResultMeaning.POSTERIOR_MEDIAN_QUANTILE)
    log_mean = replace(at_mean, meaning=ResultMeaning.LOG10_QUANTILE_POSTERIOR_MOMENT, log_moment="mean")
    requests = (mean, aep, period, at_mean, log_mean, unsupported, upper, median)
    plan = _plan()
    audit = audit_lognormal_precision(
        model=model, proposal=proposal, provenance=_provenance(model), plan=plan, requests=requests
    )
    assert audit.attempted_count == audit.completed_count == len(plan.seeds)
    assert audit.states == (
        *((PrecisionState.CONDITIONAL_PASS,) * 5),
        PrecisionState.INDETERMINATE,
        PrecisionState.UNDEFINED if free else PrecisionState.INDETERMINATE,
        PrecisionState.INDETERMINATE,
    )
    for attempt in audit.attempts[1:]:
        assert tuple(output.reference for output in attempt.outputs[:5]) == (0.0, 0.5, 2.0, 1.0, 0.0)
        assert all(output.integration_interval is not None for output in attempt.outputs[:5])
        inverse = attempt.outputs[-1]
        assert inverse.reference == 1.0
        assert len(inverse.scalar_checks) == 2
        assert all(check.reference is None for check in inverse.scalar_checks)
        assert inverse.reason == "independent inverse endpoint probability bounds unavailable"

    rejected = audit_lognormal_precision(
        model=model,
        proposal=proposal,
        provenance=_provenance(model),
        plan=replace(plan, maximum_log_ratio=0.01),
        requests=requests,
    )
    assert rejected.attempted_count == 0
    assert rejected.attempts[0].reason == "normalized ratio envelope exceeds the explicit ceiling"


@pytest.mark.parametrize(("free", "ideal_cap"), [(False, 6.67354), (True, 8.79755)])
def test_outward_ratio_bound_and_exact_covariance_gate(*, free: bool, ideal_cap: float) -> None:
    model = _model(free=free)
    proposal = _proposal(model)
    cap = normalized_log_ratio_bound(model, proposal)
    # Independently computed box using the sharper Gaussian mass and density
    # constants in the worked example. The implemented upper bound is looser.
    assert ideal_cap < cap < 11.0
    displaced = replace(proposal, means=(0.01, *proposal.means[1:]))
    with pytest.raises(RatioEnvelopeError, match="strict Gaussian ratio"):
        normalized_log_ratio_bound(model, displaced)
    wider = replace(displaced, covariance=tuple(tuple(2 * v for v in row) for row in proposal.covariance))
    assert normalized_log_ratio_bound(model, wider) > cap


@pytest.mark.parametrize("reference", [1e-300, 1e300])
def test_reflection_discharge_encloses_exact_identity_at_zero(reference: float) -> None:
    assert reflected_discharge(reference, 0.0) == reference
    lower, upper = reflected_discharge_bounds(reference, 0.0, 0.0)
    assert lower < reference < upper
