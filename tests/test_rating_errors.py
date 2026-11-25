"""Shared latent error tests: integrate *jointly*, with density Jacobian."""

from math import exp, log

import pytest

from ffa_methods.contracts import AnnualObservation, ExposureKind, ObservationKind, ObservationSeries
from ffa_methods.rating_errors import (
    JointRatingScenario,
    LogMultiplicativeKernel,
    RatingEpoch,
    RatingEvidenceError,
    marginal_shared_rating_log_likelihood,
)
from tests.analytic_kernels import ExponentialKernel
from tests.numeric_assertions import near


def epoch(name: str, years: tuple[int, ...], flow: float = 1.0, *, unit: str = "m3/s") -> RatingEpoch:
    series = ObservationSeries(
        discharge_unit=unit,
        annual=tuple(
            AnnualObservation(year=year, kind=ObservationKind.EXACT, exposure=ExposureKind.SYSTEMATIC, value=flow)
            for year in years
        ),
    )
    return RatingEpoch(identifier=name, series=series)


def test_fixed_rating_shift_preserves_exact_density_jacobian() -> None:
    log_error = 0.5
    kernel = LogMultiplicativeKernel(ExponentialKernel(), log_error)
    assert near(kernel.logpdf(2), -2 * exp(-log_error) - log_error)
    assert near(kernel.logsf(2), -2 * exp(-log_error))
    assert near(kernel.ppf(0.5), log(2) * exp(log_error))


def test_joint_shared_error_is_not_an_independent_error_per_record() -> None:
    x = 2.0
    shifts = (log(0.8), log(1.2))
    scenarios = tuple(JointRatingScenario(weight=0.5, log_error_by_epoch=(("old", shift),)) for shift in shifts)
    result = marginal_shared_rating_log_likelihood(ExponentialKernel(), (epoch("old", (2000, 2001), x),), scenarios)
    single = [exp(-x * exp(-shift) - shift) for shift in shifts]
    shared = 0.5 * (single[0] ** 2 + single[1] ** 2)
    independent = ((single[0] + single[1]) / 2) ** 2
    assert near(exp(result), shared, abs_tol=1e-14)
    assert not near(exp(result), independent, abs_tol=1e-9)


def test_joint_vector_carries_correlation_across_different_epochs() -> None:
    shifts = (log(0.8), log(1.2))
    scenarios = (
        JointRatingScenario(weight=0.5, log_error_by_epoch=(("A", shifts[0]), ("B", shifts[0]))),
        JointRatingScenario(weight=0.5, log_error_by_epoch=(("A", shifts[1]), ("B", shifts[1]))),
    )
    result = marginal_shared_rating_log_likelihood(
        ExponentialKernel(), (epoch("A", (2000,)), epoch("B", (2001,))), scenarios
    )
    values = [exp(-exp(-shift) - shift) for shift in shifts]
    assert near(exp(result), sum(value * value for value in values) / 2)


def test_identity_scenario_matches_uncertain_rating_disabled() -> None:
    scenario = JointRatingScenario(weight=1, log_error_by_epoch=(("A", 0),))
    result = marginal_shared_rating_log_likelihood(ExponentialKernel(), (epoch("A", (2001,)),), (scenario,))
    assert near(result, -1)


@pytest.mark.parametrize(
    "scenarios",
    [
        (JointRatingScenario(weight=0.9, log_error_by_epoch=(("A", 0),)),),
        (JointRatingScenario(weight=1.0, log_error_by_epoch=(("WRONG", 0),)),),
        (JointRatingScenario(weight=1.0, log_error_by_epoch=(("A", 0), ("A", 0))),),
        (JointRatingScenario(weight=0, log_error_by_epoch=(("A", 0),)),),
    ],
)
def test_invalid_joint_rating_integration_is_rejected(scenarios: tuple[JointRatingScenario, ...]) -> None:
    with pytest.raises(RatingEvidenceError):
        marginal_shared_rating_log_likelihood(ExponentialKernel(), (epoch("A", (2000,)),), scenarios)


def test_year_cannot_be_in_two_rating_groups() -> None:
    scenarios = (JointRatingScenario(weight=1.0, log_error_by_epoch=(("A", 0), ("B", 0))),)
    with pytest.raises(RatingEvidenceError, match="multiple rating"):
        marginal_shared_rating_log_likelihood(
            ExponentialKernel(), (epoch("A", (2000,)), epoch("B", (2000,))), scenarios
        )


def test_rating_epochs_reject_mixed_discharge_units_before_joint_integration() -> None:
    scenario = JointRatingScenario(weight=1.0, log_error_by_epoch=(("A", 0.0), ("B", 0.0)))
    first = epoch("A", (2000,), flow=1.0, unit="m3/s")
    same_unit = epoch("B", (2001,), flow=1.0, unit="m3/s")
    different_unit = epoch("B", (2001,), flow=1.0, unit="ft3/s")
    assert near(
        marginal_shared_rating_log_likelihood(ExponentialKernel(), (first, same_unit), (scenario,)),
        -2.0,
        abs_tol=1e-14,
        rel_tol=0.0,
    )
    with pytest.raises(RatingEvidenceError, match="single recorded discharge unit"):
        marginal_shared_rating_log_likelihood(ExponentialKernel(), (first, different_unit), (scenario,))
