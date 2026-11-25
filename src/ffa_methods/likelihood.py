"""Auditable continuous-data likelihood pieces for the legacy Flike workstream.

These equations have independent mathematical tests. They are not yet qualified
against Flike 5.0.306.0, nor do they infer the program's zero/censor defaults.
"""

from dataclasses import dataclass
from itertools import pairwise
from math import exp, expm1, fsum, isfinite, isnan, lgamma, log

from ffa_methods.contracts import (
    AnnualObservation,
    CensorGroup,
    DistributionKernel,
    ExposureKind,
    ObservationKind,
    ObservationSeries,
)


class LikelihoodEvidenceError(ValueError):
    """Available records do not define a unique, supported likelihood."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoricalPeriod:
    """A complete detection period, with year-labelled magnitudes and unmeasured exceedances.

    An above-threshold flood with a known *year* contributes its unconditional
    density, not an additional independent binomial event. The remaining
    unmeasured exceedances are exchangeable among the remaining years.
    """

    start_year: int
    end_year: int
    threshold: float
    known_exceedances: tuple[tuple[int, float], ...] = ()
    unmeasured_exceedances: int = 0
    source_record: str | None = None

    def __post_init__(self) -> None:
        """Reject impossible counts, coverage and known events."""
        n = self.end_year - self.start_year + 1
        if self.start_year <= 0 or n <= 0 or not isfinite(self.threshold):
            msg = "historical period must have valid years and finite threshold"
            raise LikelihoodEvidenceError(msg)
        if self.unmeasured_exceedances < 0 or self.unmeasured_exceedances + len(self.known_exceedances) > n:
            msg = "historical exceedance counts exceed observation opportunities"
            raise LikelihoodEvidenceError(msg)
        years: set[int] = set()
        for year, flow in self.known_exceedances:
            if year in years or year < self.start_year or year > self.end_year:
                msg = "historical exceedance years must be unique and within the period"
                raise LikelihoodEvidenceError(msg)
            if not isfinite(flow) or flow <= self.threshold:
                msg = "known historical exceedances must be finite and above threshold"
                raise LikelihoodEvidenceError(msg)
            years.add(year)


@dataclass(frozen=True, slots=True)
class LikelihoodTerm:
    """One independently inspectable density/probability contribution."""

    source: str
    log_value: float
    included: bool = True


@dataclass(frozen=True, slots=True)
class LikelihoodResult:
    """Joint log likelihood with all skipped observations shown explicitly."""

    log_value: float
    terms: tuple[LikelihoodTerm, ...]


def _log_difference(log_larger: float, log_smaller: float) -> float:
    """log(exp(a)-exp(b)), including exact zero probability."""
    if log_larger < log_smaller:
        msg = "invalid probability ordering returned by distribution kernel"
        raise ValueError(msg)
    if log_larger == log_smaller:
        return float("-inf")
    return log_larger + log(-expm1(log_smaller - log_larger))


def _coalesced_interval_from_density(kernel: DistributionKernel, lower: float, upper: float) -> float:
    """Recover a narrow probability by integrating the continuous density.

    This fallback is only used when *both* log-CDF and log-survival differences
    have rounded to zero. Composite midpoint estimates at successively doubled
    resolutions must agree twice at log-probability precision; unresolved or
    non-integrable densities fail closed instead of inventing a zero probability.
    """
    width = upper - lower
    if not isfinite(width):
        msg = "interval width cannot be represented for density integration"
        raise LikelihoodEvidenceError(msg)

    previous: float | None = None
    stable_steps = 0
    for bins in (8, 16, 32, 64, 128, 256, 512, 1024):
        logs: list[float] = []
        for index in range(bins):
            point = lower + width * ((index + 0.5) / bins)
            value = kernel.logpdf(point)
            if isnan(value) or value == float("inf"):
                msg = "invalid log density during narrow-interval integration"
                raise ValueError(msg)
            if isfinite(value):
                logs.append(value)
        if not logs:
            previous = None
            stable_steps = 0
            continue

        peak = max(logs)
        log_mass = log(width) - log(bins) + peak + log(fsum(exp(value - peak) for value in logs))
        if previous is not None and abs(log_mass - previous) <= 2e-14:
            stable_steps += 1
            if stable_steps == 2:
                return log_mass
        else:
            stable_steps = 0
        previous = log_mass

    msg = "interval mass is unresolved: rounded tails and density integration did not converge"
    raise LikelihoodEvidenceError(msg)


def _interval_log_probability(kernel: DistributionKernel, lower: float, upper: float) -> float:
    """Use stable tail subtraction, then a checked density integral if needed."""
    if lower >= upper:
        return float("-inf")
    left_cdf = kernel.logcdf(lower)
    right_cdf = kernel.logcdf(upper)
    left_sf = kernel.logsf(lower)
    right_sf = kernel.logsf(upper)
    if any(isnan(value) or value > 0.0 for value in (left_cdf, right_cdf, left_sf, right_sf)):
        msg = "invalid log tail probability returned by distribution kernel"
        raise ValueError(msg)
    if right_cdf < left_cdf or left_sf < right_sf:
        msg = "invalid probability ordering returned by distribution kernel"
        raise ValueError(msg)

    cdf_resolvable = right_cdf > left_cdf
    sf_resolvable = left_sf > right_sf
    if cdf_resolvable and (not sf_resolvable or right_cdf <= left_sf):
        return _log_difference(right_cdf, left_cdf)
    if sf_resolvable:
        return _log_difference(left_sf, right_sf)
    if cdf_resolvable:
        return _log_difference(right_cdf, left_cdf)
    if left_cdf == float("-inf") or left_sf == float("-inf"):
        return float("-inf")
    return _coalesced_interval_from_density(kernel, lower, upper)


def _log_power(log_probability: float, count: int) -> float:
    """Treat probability raised to zero as one even at support endpoints."""
    return 0.0 if count == 0 else count * log_probability


def _log_choose(n: int, k: int) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def exposure_intervals(
    series: ObservationSeries, historical_periods: tuple[HistoricalPeriod, ...] = ()
) -> tuple[tuple[int, int], ...]:
    """Check disjoint exposure without materialising large historical year sets."""
    intervals = [(row.year, row.year) for row in series.annual]
    for group in series.groups:
        if (
            group.start_year is None
            or group.end_year is None
            or group.start_year <= 0
            or group.end_year < group.start_year
        ):
            msg = "grouped thresholds require a complete positive year interval"
            raise LikelihoodEvidenceError(msg)
        if group.count_above is None or group.count_at_or_below is None:
            msg = "both grouped counts are required to identify exposure"
            raise LikelihoodEvidenceError(msg)
        if group.count_above + group.count_at_or_below != group.end_year - group.start_year + 1:
            msg = "grouped counts do not equal the stated full exposure"
            raise LikelihoodEvidenceError(msg)
        intervals.append((group.start_year, group.end_year))
    intervals.extend((period.start_year, period.end_year) for period in historical_periods)
    intervals.sort()
    if any(current[0] <= preceding[1] for preceding, current in pairwise(intervals)):
        msg = "overlapping annual, grouped or historical exposure"
        raise LikelihoodEvidenceError(msg)
    return tuple(intervals)


def _annual_observation_log_value(kernel: DistributionKernel, row: AnnualObservation) -> float:
    """Evaluate a classified continuous observation, without implicit censoring."""
    if row.kind == ObservationKind.EXACT:
        if row.value is None:
            msg = "exact observation is missing its value"
            raise LikelihoodEvidenceError(msg)
        return kernel.logpdf(row.value)
    if row.kind == ObservationKind.LEFT_CENSORED:
        if row.upper_bound is None:
            msg = "left-censored observation has no upper bound"
            raise LikelihoodEvidenceError(msg)
        return kernel.logcdf(row.upper_bound)
    if row.kind == ObservationKind.RIGHT_CENSORED:
        if row.lower_bound is None:
            msg = "right-censored observation has no lower bound"
            raise LikelihoodEvidenceError(msg)
        return kernel.logsf(row.lower_bound)
    if row.kind == ObservationKind.INTERVAL_CENSORED:
        if row.lower_bound is None or row.upper_bound is None:
            msg = "interval-censored observation is missing a bound"
            raise LikelihoodEvidenceError(msg)
        return _interval_log_probability(kernel, row.lower_bound, row.upper_bound)
    msg = f"unsupported observation kind {row.kind}"
    raise LikelihoodEvidenceError(msg)


def _annual_log_term(kernel: DistributionKernel, row: AnnualObservation) -> LikelihoodTerm:
    """Keep exposure classification separate from continuous density logic."""
    name = f"annual:{row.year}:{row.kind.value}"
    if row.kind in (ObservationKind.MISSING, ObservationKind.EXCLUDED):
        return LikelihoodTerm(name, 0.0, included=False)
    if row.exposure == ExposureKind.UNRESOLVED:
        msg = f"unresolved exposure for {name}"
        raise LikelihoodEvidenceError(msg)
    if row.kind == ObservationKind.AT_OR_BELOW_ZERO_THRESHOLD:
        msg = "legacy zero-threshold mass/likelihood is not established (WP07)"
        raise LikelihoodEvidenceError(msg)
    return LikelihoodTerm(name, _annual_observation_log_value(kernel, row))


def _group_log_term(kernel: DistributionKernel, group: CensorGroup, index: int) -> LikelihoodTerm:
    if group.exposure == ExposureKind.UNRESOLVED:
        msg = "grouped exposure classification is unresolved"
        raise LikelihoodEvidenceError(msg)
    if group.count_above is None or group.count_at_or_below is None:
        msg = "grouped counts are incomplete"
        raise LikelihoodEvidenceError(msg)
    n = group.count_above + group.count_at_or_below
    value = (
        _log_choose(n, group.count_above)
        + _log_power(kernel.logsf(group.threshold), group.count_above)
        + _log_power(kernel.logcdf(group.threshold), group.count_at_or_below)
    )
    return LikelihoodTerm(f"group:{index}", value)


def _historical_log_term(kernel: DistributionKernel, period: HistoricalPeriod, index: int) -> LikelihoodTerm:
    n_remain = period.end_year - period.start_year + 1 - len(period.known_exceedances)
    n_below = n_remain - period.unmeasured_exceedances
    exact = fsum(kernel.logpdf(flow) for _, flow in period.known_exceedances)
    value = (
        exact
        + _log_choose(n_remain, period.unmeasured_exceedances)
        + _log_power(kernel.logsf(period.threshold), period.unmeasured_exceedances)
        + _log_power(kernel.logcdf(period.threshold), n_below)
    )
    return LikelihoodTerm(f"historical:{index}", value)


def log_likelihood(
    kernel: DistributionKernel,
    series: ObservationSeries,
    *,
    historical_periods: tuple[HistoricalPeriod, ...] = (),
) -> LikelihoodResult:
    """Evaluate explicitly classified evidence; never invent legacy censor controls."""
    exposure_intervals(series, historical_periods)
    terms = (
        tuple(_annual_log_term(kernel, row) for row in series.annual)
        + tuple(_group_log_term(kernel, group, i) for i, group in enumerate(series.groups))
        + tuple(_historical_log_term(kernel, period, i) for i, period in enumerate(historical_periods))
    )
    if any(isnan(term.log_value) or term.log_value == float("inf") for term in terms):
        msg = "distribution kernel returned invalid likelihood values"
        raise ValueError(msg)
    return LikelihoodResult(fsum(term.log_value for term in terms), terms)
