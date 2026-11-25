"""Explicit joint shared-rating-error likelihood; no assumed Flike controls.

An error scenario contains one *joint* vector of shared log-discharge errors.
The model is an independently testable composition kernel, NOT a claim that
Flike 5.0.306.0 uses this distribution or quadrature rule.
"""

from dataclasses import dataclass
from itertools import pairwise
from math import exp, fsum, isclose, isfinite, log

from ffa_methods.contracts import DistributionKernel, ObservationSeries
from ffa_methods.likelihood import HistoricalPeriod, LikelihoodEvidenceError, exposure_intervals, log_likelihood


class RatingEvidenceError(ValueError):
    """The measurement model or joint rating evidence is ambiguous."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RatingEpoch:
    """Disjoint observations that share one rating-curve error component."""

    identifier: str
    series: ObservationSeries
    historical_periods: tuple[HistoricalPeriod, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class JointRatingScenario:
    """One joint log-error vector and its probability/integration weight.

    `measured_discharge = true_discharge * exp(log_error)`. All events in an
    epoch share that same offset. Correlation across epochs is represented by
    their *joint* scenario weights, not by multiplying marginal integrations.
    """

    weight: float
    log_error_by_epoch: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class LogMultiplicativeKernel:
    """Distribution of a measurement under a fixed shared multiplicative error."""

    underlying: DistributionKernel
    log_error: float

    def __post_init__(self) -> None:
        if not isfinite(self.log_error):
            msg = "rating log error must be finite"
            raise RatingEvidenceError(msg)

    def _true_discharge(self, measured: float) -> float:
        try:
            true_discharge = measured * exp(-self.log_error)
        except OverflowError as exc:
            msg = "rating error transformation exceeds floating point range"
            raise RatingEvidenceError(msg) from exc
        if not isfinite(true_discharge):
            msg = "rating error transformation exceeds floating point range"
            raise RatingEvidenceError(msg)
        return true_discharge

    def _measured_discharge(self, true_discharge: float) -> float:
        try:
            measured = true_discharge * exp(self.log_error)
        except OverflowError as exc:
            msg = "rating error transformation exceeds floating point range"
            raise RatingEvidenceError(msg) from exc
        return measured

    def logpdf(self, discharge: float) -> float:
        return self.underlying.logpdf(self._true_discharge(discharge)) - self.log_error

    def logcdf(self, discharge: float) -> float:
        return self.underlying.logcdf(self._true_discharge(discharge))

    def logsf(self, discharge: float) -> float:
        return self.underlying.logsf(self._true_discharge(discharge))

    def cdf(self, discharge: float) -> float:
        return self.underlying.cdf(self._true_discharge(discharge))

    def sf(self, discharge: float) -> float:
        return self.underlying.sf(self._true_discharge(discharge))

    def ppf(self, non_exceedance_probability: float) -> float:
        return self._measured_discharge(self.underlying.ppf(non_exceedance_probability))

    def isf(self, survival_probability: float) -> float:
        return self._measured_discharge(self.underlying.isf(survival_probability))


def _validate_epoch_exposure(epochs: tuple[RatingEpoch, ...]) -> tuple[str, ...]:
    identifiers = tuple(epoch.identifier for epoch in epochs)
    if any(not name.strip() for name in identifiers) or len(set(identifiers)) != len(identifiers):
        msg = "rating epoch identifiers must be nonempty and unique"
        raise RatingEvidenceError(msg)
    units = {epoch.series.discharge_unit.strip() for epoch in epochs}
    if len(units) != 1:
        msg = "rating epochs must use a single recorded discharge unit; conversion is not implicit"
        raise RatingEvidenceError(msg)
    intervals: list[tuple[int, int]] = []
    for epoch in epochs:
        try:
            intervals.extend(exposure_intervals(epoch.series, epoch.historical_periods))
        except LikelihoodEvidenceError as exc:
            msg = "invalid exposure within rating epoch"
            raise RatingEvidenceError(msg) from exc
    intervals.sort()
    if any(current[0] <= previous[1] for previous, current in pairwise(intervals)):
        msg = "an observation year belongs to multiple rating epochs"
        raise RatingEvidenceError(msg)
    return identifiers


def _validate_scenario_weights(scenarios: tuple[JointRatingScenario, ...]) -> None:
    if any(not isfinite(s.weight) or s.weight <= 0 for s in scenarios):
        msg = "all joint rating integration weights must be positive and finite"
        raise RatingEvidenceError(msg)
    if not isclose(fsum(s.weight for s in scenarios), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        msg = "joint rating integration weights must sum to one"
        raise RatingEvidenceError(msg)


def _scenario_log_component(
    kernel: DistributionKernel,
    epochs: tuple[RatingEpoch, ...],
    identifiers: tuple[str, ...],
    scenario: JointRatingScenario,
) -> float:
    offsets = dict(scenario.log_error_by_epoch)
    if len(offsets) != len(scenario.log_error_by_epoch) or set(offsets) != set(identifiers):
        msg = "each joint scenario must name every epoch exactly once"
        raise RatingEvidenceError(msg)
    if not all(isfinite(value) for value in offsets.values()):
        msg = "nonfinite rating error offset"
        raise RatingEvidenceError(msg)
    conditional_log = fsum(
        log_likelihood(
            LogMultiplicativeKernel(kernel, offsets[epoch.identifier]),
            epoch.series,
            historical_periods=epoch.historical_periods,
        ).log_value
        for epoch in epochs
    )
    return log(scenario.weight) + conditional_log


def marginal_shared_rating_log_likelihood(
    kernel: DistributionKernel,
    epochs: tuple[RatingEpoch, ...],
    scenarios: tuple[JointRatingScenario, ...],
) -> float:
    """Integrate joint shared errors, without inventing a legacy rating-error law."""
    if not epochs or not scenarios:
        msg = "rating integration requires epochs and explicit joint scenarios"
        raise RatingEvidenceError(msg)
    identifiers = _validate_epoch_exposure(epochs)
    _validate_scenario_weights(scenarios)
    components = tuple(_scenario_log_component(kernel, epochs, identifiers, scenario) for scenario in scenarios)
    largest = max(components)
    if largest == float("-inf"):
        return largest
    return largest + log(fsum(exp(value - largest) for value in components))
