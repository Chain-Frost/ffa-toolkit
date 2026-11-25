"""Independent analytic and source-coordinated WP06 particle generation tests."""

import math
from collections.abc import Callable
from random import getstate

import pytest
from scipy.stats import multivariate_normal

from ffa_methods.contracts import (
    AnnualObservation,
    EvidenceStatus,
    ExposureKind,
    ObservationKind,
    ObservationSeries,
    RunProvenance,
)
from ffa_methods.distributions import LogCoordinates, Lognormal
from ffa_methods.importance_sampling import (
    ExplicitNormalProposal,
    ExplicitPosteriorTarget,
    ExplicitSamplingControls,
    IndependentSamplingResult,
    SamplingError,
    independent_importance_sample,
    standardized_independent_replication_difference,
)
from ffa_methods.likelihood import log_likelihood
from ffa_methods.likelihood_integration import _as_shared_kernel  # pyright: ignore[reportPrivateUsage]  # test-only
from ffa_methods.priors import JointGaussianPrior, LegacyPriorMode


def _provenance(status: EvidenceStatus = EvidenceStatus.OBSERVED_ARTIFACT) -> RunProvenance:
    return RunProvenance(
        compatibility_profile="tuflow-bmt-flike-5.0.306.0-file3.10",
        method_id="quantile-estimands",
        method_version="independent-gaussian-proposal-v1",
        package_version="0.0.0.dev0",
        evidence_status=status,
    )


def _proposal(mean: float = 6 / 7, variance: float = 4 / 7) -> ExplicitNormalProposal:
    return ExplicitNormalProposal(
        coordinates=("synthetic.log_mean",),
        means=(mean,),
        covariance=((variance,),),
        source="synthetic: explicit specified full Normal proposal",
    )


def _target(
    likelihood: Callable[[tuple[float, ...]], float] | None = None,
    prior: Callable[[tuple[float, ...]], float] | None = None,
) -> ExplicitPosteriorTarget:
    supplied_prior = JointGaussianPrior(
        coordinates=("synthetic.log_mean",),
        means=(0.0,),
        covariance=((1.0,),),
        source="synthetic: specified mean prior",
        mode=LegacyPriorMode.SUPPLIED,
    )

    def normal_likelihood(theta: tuple[float, ...]) -> float:
        return -1.5 * math.log(8.0 * math.pi) - sum((x - theta[0]) ** 2 for x in (1.0, 2.0, 3.0)) / 8.0

    return ExplicitPosteriorTarget(
        coordinates=("synthetic.log_mean",),
        log_likelihood=likelihood if likelihood is not None else normal_likelihood,
        log_prior=prior if prior is not None else supplied_prior.logpdf,
        source="synthetic: Normal known-sigma observations with Normal source prior",
    )


def _controls(
    *,
    seed: int = 20261124,
    draws: int = 12000,
    minimum_ess: float = 8000.0,
    max_weight: float = 0.001,
) -> ExplicitSamplingControls:
    return ExplicitSamplingControls(
        draw_count=draws,
        random_seed=seed,
        minimum_effective_size=minimum_ess,
        maximum_single_weight=max_weight,
    )


def test_full_covariance_proposal_logpdf_against_independent_scipy() -> None:
    proposal = ExplicitNormalProposal(
        coordinates=("source.mean", "source.log_sd"),
        means=(1.0, -0.2),
        covariance=((4.0, 1.5), (1.5, 9.0)),
        source="synthetic: full Gaussian proposal",
    )
    x = (2.5, 1.2)
    target = float(multivariate_normal.logpdf(x, mean=[1.0, -0.2], cov=[[4.0, 1.5], [1.5, 9.0]]))
    assert math.isclose(proposal.logpdf(x), target, rel_tol=0.0, abs_tol=1e-14)


def test_real_random_draws_reproduce_independent_full_gaussian_moments() -> None:
    proposal = ExplicitNormalProposal(
        coordinates=("source.mean", "source.log_sd"),
        means=(0.5, -0.4),
        covariance=((1.0, 0.75), (0.75, 2.25)),
        source="synthetic: full covariance proposal supplied by caller",
    )
    target = ExplicitPosteriorTarget(
        coordinates=proposal.coordinates,
        log_likelihood=lambda _: 0.0,
        log_prior=proposal.logpdf,
        source="synthetic: target exactly equals the declared proposal",
    )
    controls = _controls(draws=14000, minimum_ess=13990.0, max_weight=0.0001)
    result = independent_importance_sample(
        proposal=proposal, target=target, controls=controls, provenance=_provenance()
    )
    assert len(result.particles.draws) == 14000
    assert math.isclose(result.moments.effective_sample_size, 14000.0, rel_tol=0.0, abs_tol=1e-7)
    assert math.isclose(result.moments.covariance[0][0], 1.0, rel_tol=0.0, abs_tol=0.07)
    assert math.isclose(result.moments.covariance[1][1], 2.25, rel_tol=0.0, abs_tol=0.15)
    assert math.isclose(result.moments.covariance[0][1], 0.75, rel_tol=0.0, abs_tol=0.10)
    assert math.isclose(result.moments.parameters[0].mean, 0.5, rel_tol=0.0, abs_tol=0.06)
    assert math.isclose(result.moments.parameters[1].mean, -0.4, rel_tol=0.0, abs_tol=0.08)


def test_actual_sampler_matches_independent_normal_normal_posterior() -> None:
    # x=[1,2,3], obs SD=2, source mean prior N(0,1):
    # precision=1+3/4=7/4, mean=(6/4)/(7/4)=6/7, variance=4/7.
    sampler = independent_importance_sample(
        proposal=_proposal(),
        target=_target(),
        controls=_controls(),
        provenance=_provenance(),
    )
    sample_mean = sampler.moments.parameters[0].mean
    sample_variance = sampler.moments.covariance[0][0]
    assert math.isclose(sample_mean, 6 / 7, rel_tol=0.0, abs_tol=0.03)
    assert math.isclose(sample_variance, 4 / 7, rel_tol=0.0, abs_tol=0.04)
    assert sampler.moments.effective_fraction > 0.97
    assert sampler.approximate_mean_mcse[0] > 0
    assert sampler.approximate_mean_mcse[0] < 0.01
    assert "not native Flike RNG" in sampler.particles.random_stream_description


def test_actual_wp03_wp05_log_likelihood_and_gaussian_prior_bind_in_source_coordinates() -> None:
    # Known log(Q) observation SD=1, real WP03 lognormal kernel and real WP05
    # exact-record likelihood. Three log discharges 1,2,3 and prior N(0,1):
    # posterior mean=1.5 and variance=1/4 (discharge Jacobians constant in theta).
    series = ObservationSeries(
        discharge_unit="m3/s",
        annual=tuple(
            AnnualObservation(
                year=2000 + i,
                kind=ObservationKind.EXACT,
                exposure=ExposureKind.SYSTEMATIC,
                value=math.exp(float(i)),
            )
            for i in (1, 2, 3)
        ),
    )
    prior = JointGaussianPrior(
        coordinates=("synthetic.log_mean",),
        means=(0.0,),
        covariance=((1.0,),),
        source="synthetic: source-coordinates full Gaussian",
        mode=LegacyPriorMode.SUPPLIED,
    )
    target = ExplicitPosteriorTarget(
        coordinates=("synthetic.log_mean",),
        log_likelihood=lambda theta: (
            log_likelihood(_as_shared_kernel(Lognormal(LogCoordinates(theta[0], 0.0))), series).log_value
        ),
        log_prior=prior.logpdf,
        source="synthetic: three exact discharge observations, explicitly classified",
    )
    result = independent_importance_sample(
        proposal=_proposal(mean=1.5, variance=0.25),
        target=target,
        controls=_controls(draws=10000, minimum_ess=9000, max_weight=0.001),
        provenance=_provenance(),
    )
    assert math.isclose(result.moments.parameters[0].mean, 1.5, rel_tol=0.0, abs_tol=0.03)
    assert math.isclose(result.moments.covariance[0][0], 0.25, rel_tol=0.0, abs_tol=0.03)
    assert result.moments.effective_fraction > 0.95


@pytest.mark.parametrize("mean", [1e20, -1e20])
def test_reject_gaussian_proposal_that_loses_coordinate_variation(mean: float) -> None:
    # One source-coordinate standard deviation (1) is smaller than the
    # representable spacing at this mean; stored iid particles would coincide.
    with pytest.raises(SamplingError, match="resolution"):
        ExplicitNormalProposal(
            coordinates=("source.large",),
            means=(mean,),
            covariance=((1.0,),),
            source="synthetic: binary64 resolution regression",
        )


def test_subnormal_variance_has_nonzero_mean_mcse_and_replicate_scale() -> None:
    # With covariance 1e-320, weighted residuals are ~1e-163, whose
    # squares underflow. The correct norm is still ~3e-162 at 1000 draws.
    proposal = ExplicitNormalProposal(
        coordinates=("source.tiny",),
        means=(0.0,),
        covariance=((1e-320,),),
        source="synthetic: retained subnormal covariance",
    )
    target = ExplicitPosteriorTarget(
        coordinates=proposal.coordinates,
        log_likelihood=lambda _: 0.0,
        log_prior=proposal.logpdf,
        source="synthetic: matched Gaussian target",
    )

    def sample(seed: int) -> IndependentSamplingResult:
        return independent_importance_sample(
            proposal=proposal,
            target=target,
            controls=_controls(seed=seed, draws=1000, minimum_ess=990.0, max_weight=0.002),
            provenance=_provenance(),
        )

    first = sample(1)
    other = sample(2)
    first_mcse = first.approximate_mean_mcse[0]
    other_mcse = other.approximate_mean_mcse[0]
    assert 1e-163 < first_mcse < 1e-161
    assert 1e-163 < other_mcse < 1e-161
    assert all(math.isfinite(score) for score in standardized_independent_replication_difference(first, other))


def test_seeded_replay_is_exact_and_does_not_mutate_global_random_state() -> None:
    before = getstate()
    proposal, target = _proposal(), _target()
    controls = _controls(draws=350, minimum_ess=300, max_weight=0.01)
    provenance = _provenance()
    first = independent_importance_sample(proposal=proposal, target=target, controls=controls, provenance=provenance)
    second = independent_importance_sample(proposal=proposal, target=target, controls=controls, provenance=provenance)
    assert first.particles.draws == second.particles.draws
    assert first.particles.log_proposal == second.particles.log_proposal
    assert first.particles.weights == second.particles.weights
    assert first.moments == second.moments
    assert getstate() == before


def test_independent_seed_replicates_report_named_mean_uncertainty() -> None:
    first = independent_importance_sample(
        proposal=_proposal(),
        target=_target(),
        controls=_controls(draws=3500, seed=443, minimum_ess=2500, max_weight=0.005),
        provenance=_provenance(),
    )
    other = independent_importance_sample(
        proposal=_proposal(),
        target=_target(),
        controls=_controls(draws=3500, seed=444, minimum_ess=2500, max_weight=0.005),
        provenance=_provenance(),
    )
    diagnostic = standardized_independent_replication_difference(first, other)
    assert len(diagnostic) == 1
    assert math.isfinite(diagnostic[0])
    assert diagnostic[0] < 5.0
    with pytest.raises(SamplingError, match="distinct seeds"):
        standardized_independent_replication_difference(first, first)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coordinates": ("x", "x"), "means": (0.0, 0.0), "covariance": ((1.0, 0.0), (0.0, 1.0)), "source": "synthetic"},
        {"coordinates": ("x",), "means": (0.0,), "covariance": ((0.0,),), "source": "synthetic"},
        {"coordinates": ("x",), "means": (0.0,), "covariance": ((-1.0,),), "source": "synthetic"},
        {"coordinates": ("x", "y"), "means": (0.0, 0.0), "covariance": ((1.0, 2.0), (0.0, 1.0)), "source": "synthetic"},
        {"coordinates": ("x", "y"), "means": (0.0, 0.0), "covariance": ((1.0, 2.0), (2.0, 1.0)), "source": "synthetic"},
        {"coordinates": ("x",), "means": (math.inf,), "covariance": ((1.0,),), "source": "synthetic"},
    ],
)
def test_invalid_proposals_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(SamplingError):
        ExplicitNormalProposal(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("draw_count", "seed", "min_ess", "max_weight"),
    [
        (1, 1, 1.0, 1.0),
        (True, 1, 1.0, 1.0),
        (100, -1, 1.0, 1.0),
        (100, 1.1, 1.0, 1.0),
        (100, 1, math.nan, 1.0),
        (100, 1, 101.0, 1.0),
        (100, 1, 1.0, 0.001),
    ],
)
def test_invalid_controls_are_rejected(draw_count: object, seed: object, min_ess: float, max_weight: float) -> None:
    with pytest.raises(SamplingError):
        ExplicitSamplingControls(
            draw_count=draw_count,  # type: ignore[arg-type]
            random_seed=seed,  # type: ignore[arg-type]
            minimum_effective_size=min_ess,
            maximum_single_weight=max_weight,
        )


def test_mismatched_source_coordinate_order_and_unsupported_native_qualification_fail_closed() -> None:
    target = ExplicitPosteriorTarget(
        coordinates=("other.coordinate",),
        log_likelihood=lambda _: 0.0,
        log_prior=lambda _: 0.0,
        source="synthetic: mismatched names",
    )
    with pytest.raises(SamplingError, match="names and order"):
        independent_importance_sample(
            proposal=_proposal(), target=target, controls=_controls(), provenance=_provenance()
        )
    with pytest.raises(SamplingError, match="cannot assert native qualification"):
        independent_importance_sample(
            proposal=_proposal(),
            target=_target(),
            controls=_controls(),
            provenance=_provenance(EvidenceStatus.QUALIFIED_REFERENCE),
        )


def _invalid_nan(_: tuple[float, ...]) -> float:
    return math.nan


def _invalid_infinity(_: tuple[float, ...]) -> float:
    return math.inf


def _invalid_exception(_: tuple[float, ...]) -> float:
    msg = "undefined model"
    raise ValueError(msg)


@pytest.mark.parametrize("likelihood", [_invalid_nan, _invalid_infinity, _invalid_exception])
def test_invalid_target_density_aborts_instead_of_dropping_proposal_draw(
    likelihood: Callable[[tuple[float, ...]], float],
) -> None:
    with pytest.raises(SamplingError):
        independent_importance_sample(
            proposal=_proposal(),
            target=_target(likelihood=likelihood),
            controls=_controls(draws=20, minimum_ess=1, max_weight=1.0),
            provenance=_provenance(),
        )


def test_outside_support_retains_zero_weight_particle_and_all_zero_rejected() -> None:
    sampler = independent_importance_sample(
        proposal=_proposal(mean=0.0, variance=1.0),
        target=_target(likelihood=lambda theta: 0.0 if theta[0] >= 0 else -math.inf),
        controls=_controls(draws=600, minimum_ess=100, max_weight=0.02),
        provenance=_provenance(),
    )
    assert 0 < sampler.moments.zero_weight_count < 600
    assert all(
        weight == 0.0
        for theta, weight in zip(sampler.particles.draws, sampler.particles.weights, strict=True)
        if theta[0] < 0
    )
    with pytest.raises(SamplingError, match="weight adequacy"):
        independent_importance_sample(
            proposal=_proposal(),
            target=_target(likelihood=lambda _: -math.inf),
            controls=_controls(draws=20, minimum_ess=1, max_weight=1.0),
            provenance=_provenance(),
        )


def test_concentrated_weights_trigger_explicit_quality_gate() -> None:
    with pytest.raises(SamplingError, match="weight adequacy"):
        independent_importance_sample(
            proposal=_proposal(mean=0.0, variance=1.0),
            target=_target(likelihood=lambda theta: -0.5 * ((theta[0] - 5.0) / 0.5) ** 2),
            controls=_controls(draws=800, minimum_ess=500, max_weight=0.02),
            provenance=_provenance(),
        )
