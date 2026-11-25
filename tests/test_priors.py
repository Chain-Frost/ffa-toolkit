"""Hand-computed correlated Gaussian and nonlinear-density Jacobian tests."""

from math import exp, isfinite, log, pi

import pytest

from ffa_methods.contracts import ParameterCoordinate
from ffa_methods.priors import (
    CoordinateTransform,
    JointGaussianPrior,
    LegacyPriorMode,
    LegacyPriorSelection,
    PriorEvidenceError,
)
from tests.numeric_assertions import near


def prior(*, rho: float = 0.6) -> JointGaussianPrior:
    return JointGaussianPrior(
        coordinates=(ParameterCoordinate.LP3_MEAN_LOGE_FLOW, ParameterCoordinate.LP3_LOGE_SD_LOGE_FLOW),
        means=(0.0, 0.0),
        covariance=((1.0, rho), (rho, 1.0)),
        mode=LegacyPriorMode.REGIONAL,
        source="fully specified mathematical fixture, not legacy Flike import",
        site_in_regional_estimate=False,
    )


def test_full_covariance_normal_density_matches_closed_form() -> None:
    p = prior(rho=0.6)
    x, y, rho = 1.0, -0.5, 0.6
    quadratic = (x * x - 2 * rho * x * y + y * y) / (1 - rho * rho)
    expected = -log(2 * pi) - 0.5 * log(1 - rho * rho) - 0.5 * quadratic
    assert near(p.logpdf((x, y)), expected, abs_tol=1e-13)
    assert not near(p.logpdf((x, y)), prior(rho=0).logpdf((x, y)))


def test_raw_standard_deviation_transform_retains_jacobian() -> None:
    p = prior()
    sd = exp(0.7)
    actual = p.logpdf_in_evaluator_coordinates(
        (1.2, sd),
        transformations=(CoordinateTransform.IDENTITY, CoordinateTransform.RAW_POSITIVE_TO_LOG),
    )
    assert near(actual, p.logpdf((1.2, 0.7)) - log(sd), abs_tol=1e-13)
    assert not near(actual, p.logpdf((1.2, 0.7)))


def test_logit_transform_uses_both_endpoint_terms() -> None:
    p = JointGaussianPrior(
        coordinates=(ParameterCoordinate.ZERO_THRESHOLD_LOGIT_PROBABILITY,),
        means=(0.0,),
        covariance=((4.0,),),
        mode=LegacyPriorMode.SUPPLIED,
        source="analytic fixture",
    )
    probability = 0.8
    result = p.logpdf_in_evaluator_coordinates(
        (probability,), transformations=(CoordinateTransform.PROBABILITY_TO_LOGIT,)
    )
    assert near(result, p.logpdf((log(4.0),)) - log(0.8) - log(0.2), abs_tol=1e-13)


@pytest.mark.parametrize(
    "matrix",
    [((1.0, 1.0), (1.0, 1.0)), ((1.0, 2.0), (0.0, 1.0)), ((-1.0, 0.0), (0.0, 1.0))],
)
def test_non_positive_definite_or_asymmetric_covariance_rejected(matrix: tuple[tuple[float, ...], ...]) -> None:
    with pytest.raises(PriorEvidenceError):
        JointGaussianPrior(
            coordinates=("m", "eta"),
            means=(0.0, 0.0),
            covariance=matrix,
            mode=LegacyPriorMode.SUPPLIED,
            source="synthetic",
        )


def test_unverified_no_prior_and_default_fail_closed() -> None:
    for mode in (LegacyPriorMode.NO_PRIOR, LegacyPriorMode.DEFAULT):
        spec = LegacyPriorSelection(mode=mode)
        with pytest.raises(PriorEvidenceError, match="unverified"):
            spec.logpdf((0.0,))


def test_supplied_prior_cannot_be_used_as_a_different_mode() -> None:
    with pytest.raises(PriorEvidenceError, match="must match"):
        LegacyPriorSelection(mode=LegacyPriorMode.SUPPLIED, supplied=prior())


def test_invalid_transforms_rejected_without_fallback() -> None:
    p = prior()
    with pytest.raises(PriorEvidenceError):
        p.logpdf_in_evaluator_coordinates(
            (1.0, 0.0), transformations=(CoordinateTransform.IDENTITY, CoordinateTransform.RAW_POSITIVE_TO_LOG)
        )
    with pytest.raises(PriorEvidenceError):
        p.logpdf_in_evaluator_coordinates(
            (1.0, 1.0), transformations=(CoordinateTransform.IDENTITY, CoordinateTransform.PROBABILITY_TO_LOGIT)
        )
    with pytest.raises(PriorEvidenceError):
        p.logpdf((1.0,))
    assert isfinite(p.logpdf((1.0, 2.0)))
