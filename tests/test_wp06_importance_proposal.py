"""Independent analytic curvature and real builder-to-sampler acceptance cases."""

from dataclasses import replace
from math import isclose, log, sqrt

import pytest

from ffa_methods.bayesian_outputs import weighted_quantile
from ffa_methods.contracts import EvidenceStatus, RunProvenance
from ffa_methods.importance_proposal import CurvatureControls, ProposalConstructionError, build_curvature_proposal
from ffa_methods.importance_sampling import (
    ExplicitPosteriorTarget,
    ExplicitSamplingControls,
    LogDensity,
    independent_importance_sample,
)


def _assert_vector_close(
    actual: tuple[float, ...], expected: tuple[float, ...], *, atol: float = 0.0, rtol: float = 0.0
) -> None:
    for value, reference in zip(actual, expected, strict=True):
        assert isclose(value, reference, abs_tol=atol, rel_tol=rtol), (value, reference)


def _controls(scales: tuple[float, ...] = (1.0, 1.0)) -> CurvatureControls:
    return CurvatureControls(
        scales=scales,
        initial_step=0.01,
        max_refinements=8,
        max_iterations=30,
        gradient_tolerance=1e-5,
        mode_tolerance=1e-5,
        curvature_tolerance=1e-4,
        max_condition=1e5,
    )


def _likelihood(theta: tuple[float, ...]) -> float:
    a, b = theta
    return -0.5 * ((a + b) ** 2 + 2 * (a - 3.5) ** 2)


def _prior(theta: tuple[float, ...]) -> float:
    a, b = theta
    return -0.5 * (a**2 + b**2)


def _target() -> ExplicitPosteriorTarget:
    return ExplicitPosteriorTarget(
        coordinates=("a", "b"),
        log_likelihood=_likelihood,
        log_prior=_prior,
        source="independent synthetic linear Gaussian observations and unit Gaussian prior",
    )


def _one_dimensional(density: LogDensity) -> ExplicitPosteriorTarget:
    return ExplicitPosteriorTarget(
        coordinates=("x",),
        log_likelihood=density,
        log_prior=lambda _: 0.0,
        source="synthetic numerical target",
    )


def test_independent_correlated_posterior() -> None:
    # Expanding the independent likelihood/prior gives precision [[4,1],[1,2]]
    # and linear coefficient (7,0); solve to obtain mode (2,-1) and determinant 7.
    result = build_curvature_proposal(target=_target(), starting_point=(("a", 0.0), ("b", 0.0)), controls=_controls())
    assert result.proposal.coordinates == ("a", "b")
    _assert_vector_close(result.proposal.means, (2.0, -1.0), atol=1e-6)
    for actual, expected in zip(result.negative_log_hessian, ((4.0, 1.0), (1.0, 2.0)), strict=True):
        _assert_vector_close(actual, expected, atol=1e-7)
    for actual, expected in zip(result.proposal.covariance, ((2 / 7, -1 / 7), (-1 / 7, 4 / 7)), strict=True):
        _assert_vector_close(actual, expected, atol=1e-7)
    assert isclose(result.log_posterior, -5.25, abs_tol=1e-10, rel_tol=0.0)
    assert result.scaled_gradient_norm + result.scaled_gradient_error <= result.controls.gradient_tolerance
    assert result.iterations > 0
    assert result.evaluations > 0
    assert result.scaled_step <= result.controls.initial_step / 2
    assert "local Newton/adaptive inverse Hessian" in result.proposal.source


def test_affine_scale_translation_and_coordinate_order() -> None:
    # v = b/4 - 2, u = 10*a + 3. Density is with respect to dv du,
    # so the prior includes the constant inverse-Jacobian factor 1/2.5.
    def original(theta: tuple[float, ...]) -> tuple[float, float]:
        v, u = theta
        return (u - 3) / 10, (v + 2) * 4

    target = ExplicitPosteriorTarget(
        coordinates=("v", "u"),
        log_likelihood=lambda theta: _likelihood(original(theta)),
        log_prior=lambda theta: _prior(original(theta)) - log(2.5),
        source="independent affine reordered Gaussian with explicit density Jacobian",
    )
    result = build_curvature_proposal(
        target=target, starting_point=(("v", -2.0), ("u", 3.0)), controls=_controls((0.25, 10.0))
    )
    assert result.proposal.coordinates == ("v", "u")
    _assert_vector_close(result.proposal.means, (-2.25, 23.0), atol=1e-5)
    for actual, expected in zip(result.negative_log_hessian, ((32.0, 0.4), (0.4, 0.04)), strict=True):
        _assert_vector_close(actual, expected, rtol=1e-6)
    expected_covariance = ((0.25 / 7, -2.5 / 7), (-2.5 / 7, 200 / 7))
    for actual, expected in zip(result.proposal.covariance, expected_covariance, strict=True):
        _assert_vector_close(actual, expected, rtol=1e-6)


def test_nonquadratic_local_mode_and_curvature() -> None:
    # Two equal modes at +/-1, negative log Hessian 8 at either mode.
    target = _one_dimensional(lambda t: -((t[0] ** 2 - 1) ** 2))
    result = build_curvature_proposal(target=target, starting_point=(("x", 1.2),), controls=_controls((1.0,)))
    _assert_vector_close(result.proposal.means, (1.0,), atol=1e-5)
    assert isclose(result.negative_log_hessian[0][0], 8.0, rel_tol=1e-4)
    assert isclose(result.proposal.covariance[0][0], 0.125, rel_tol=1e-4)


def test_negligible_newton_component_may_round_to_zero() -> None:
    # Analytic mode (1, 0), precision diag(1, 2); a tiny numerical x update
    # must not discard a well-resolved y update from the initial point (1, 1).
    target = ExplicitPosteriorTarget(
        coordinates=("x", "y"),
        log_likelihood=lambda t: -0.5 * (t[0] - 1) ** 2 - t[1] ** 4 - t[1] ** 2,
        log_prior=lambda _: 0.0,
        source="independent smooth convex negative log posterior",
    )
    result = build_curvature_proposal(target=target, starting_point=(("x", 1.0), ("y", 1.0)), controls=_controls())
    _assert_vector_close(result.proposal.means, (1.0, 0.0), atol=1e-5)
    _assert_vector_close(result.negative_log_hessian[0], (1.0, 0.0), atol=1e-4)
    _assert_vector_close(result.negative_log_hessian[1], (0.0, 2.0), atol=1e-4)
    _assert_vector_close(result.proposal.covariance[0], (1.0, 0.0), atol=1e-4)
    _assert_vector_close(result.proposal.covariance[1], (0.0, 0.5), atol=1e-4)


def test_generated_proposal_drives_real_sampler_and_weighted_outputs() -> None:
    target = _target()
    result = build_curvature_proposal(target=target, starting_point=(("a", 0.0), ("b", 0.0)), controls=_controls())
    count = 16000
    run = independent_importance_sample(
        target=target,
        proposal=result.proposal,
        controls=ExplicitSamplingControls(
            draw_count=count, random_seed=47, minimum_effective_size=count * 0.99, maximum_single_weight=0.0001
        ),
        provenance=RunProvenance(
            compatibility_profile="tuflow-bmt-flike-5.0.306.0-file3.10",
            method_id="quantile-estimands",
            method_version="synthetic-curvature-proposal-v1",
            package_version="0.0.0.dev0",
            evidence_status=EvidenceStatus.OBSERVED_ARTIFACT,
        ),
    )
    assert len(run.particles.draws) == count
    assert run.moments.effective_sample_size > count * 0.999
    # Five independent Normal standard errors of the mean; separately test covariance
    # with approximately six Gaussian covariance standard errors.
    for moment, expected, variance in zip(run.moments.parameters, (2.0, -1.0), (2 / 7, 4 / 7), strict=True):
        assert isclose(moment.mean, expected, abs_tol=5 * sqrt(variance / count), rel_tol=0.0)
    for actual, expected in zip(run.moments.covariance, ((2 / 7, -1 / 7), (-1 / 7, 4 / 7)), strict=True):
        _assert_vector_close(actual, expected, atol=0.04)
    # a + b is exactly N(1, 4/7); independent known 95th percentile.
    sums = tuple(a + b for a, b in run.particles.draws)
    q95 = weighted_quantile(sums, run.particles.weights, 0.95)
    assert isclose(q95, 1 + 1.6448536269514722 * sqrt(4 / 7), abs_tol=0.08, rel_tol=0.0)
    assert all(value > 0 for value in run.approximate_mean_mcse)
    assert run.particles.proposal_description == result.proposal.source


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_target_rejected(value: float) -> None:
    with pytest.raises(ProposalConstructionError):
        build_curvature_proposal(
            target=_one_dimensional(lambda _: value), starting_point=(("x", 0.0),), controls=_controls((1.0,))
        )


@pytest.mark.parametrize("start", [(("b", 0.0), ("a", 0.0)), (("a", float("nan")), ("b", 0.0))])
def test_invalid_start_rejected(start: tuple[tuple[str, float], ...]) -> None:
    with pytest.raises(ProposalConstructionError, match="start coordinates"):
        build_curvature_proposal(target=_target(), starting_point=start, controls=_controls())


def test_boundary_mode_is_not_admitted() -> None:
    target = _one_dimensional(lambda t: -((t[0] + 1) ** 2) if t[0] >= 0 else -float("inf"))
    with pytest.raises(ProposalConstructionError, match="boundary"):
        build_curvature_proposal(target=target, starting_point=(("x", 0.1),), controls=_controls((1.0,)))


@pytest.mark.parametrize("kind", ["singular", "saddle", "flat"])
def test_invalid_curvature_rejected(kind: str) -> None:
    densities: dict[str, LogDensity] = {
        "singular": lambda t: -(t[0] ** 2),
        "saddle": lambda t: t[0] ** 2 - t[1] ** 2 - t[0] ** 4,
        "flat": lambda t: -(t[0] ** 4 + t[1] ** 4),
    }
    density = densities[kind]
    target = ExplicitPosteriorTarget(
        coordinates=("a", "b"), log_likelihood=density, log_prior=lambda _: 0.0, source="synthetic failure target"
    )
    with pytest.raises(ProposalConstructionError):
        build_curvature_proposal(target=target, starting_point=(("a", 0.0), ("b", 0.0)), controls=_controls())


def test_condition_limit_is_enforced() -> None:
    target = ExplicitPosteriorTarget(
        coordinates=("a", "b"),
        log_likelihood=lambda t: -(100 * t[0] ** 2 + t[1] ** 2) / 2,
        log_prior=lambda _: 0.0,
        source="synthetic anisotropic quadratic",
    )
    with pytest.raises(ProposalConstructionError, match="ill-conditioned"):
        build_curvature_proposal(
            target=target,
            starting_point=(("a", 0.0), ("b", 0.0)),
            controls=replace(_controls(), max_condition=10.0),
        )


def test_iteration_limit_rejects_unconverged_mode() -> None:
    target = _one_dimensional(lambda t: -(t[0] ** 4 + t[0] ** 2))
    with pytest.raises(ProposalConstructionError, match="iteration limit"):
        build_curvature_proposal(
            target=target,
            starting_point=(("x", 1.0),),
            controls=replace(_controls((1.0,)), max_iterations=1),
        )


def test_unresolved_coordinates_and_objective_roundoff() -> None:
    with pytest.raises(ProposalConstructionError, match="perturbation"):
        build_curvature_proposal(
            target=_one_dimensional(lambda t: -((t[0] - 1e16) ** 2)),
            starting_point=(("x", 1e16),),
            controls=_controls((1.0,)),
        )
    with pytest.raises(ProposalConstructionError, match="adaptive differences"):
        build_curvature_proposal(
            target=_one_dimensional(lambda t: 1e20 - t[0] ** 2),
            starting_point=(("x", 0.0),),
            controls=_controls((1.0,)),
        )
