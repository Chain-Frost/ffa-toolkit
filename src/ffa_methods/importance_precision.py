"""Fixed-budget output precision for a bounded, analytically admitted lognormal model.

See docs/wp06-output-precision.md. Conditional numerical integration audit,
not posterior uncertainty, general convergence certification or native defaults.
"""

from dataclasses import dataclass, replace
from enum import StrEnum
from math import erfc, exp, fsum, hypot, inf, isfinite, log, log10, nextafter, sqrt
from statistics import NormalDist
from typing import Literal

from ffa_methods.bayesian_outputs import ImportanceParticles, predictive_quantile, weighted_quantile
from ffa_methods.contracts import EvidenceStatus, ObservationKind, ResultMeaning, RunProvenance
from ffa_methods.importance_sampling import (
    ExplicitNormalProposal,
    ExplicitSamplingControls,
    independent_importance_sample,
)
from ffa_methods.lognormal_posterior import ClassifiedLognormalPosterior

_NORMAL = NormalDist()
_MEAN = ResultMeaning.POSTERIOR_MEAN_QUANTILE
_MEDIAN = ResultMeaning.POSTERIOR_MEDIAN_QUANTILE
_PREDICTIVE = ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE
_QUANTILE_LIMIT = ResultMeaning.QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT
_PERIOD_LIMIT = ResultMeaning.RETURN_PERIOD_MONTE_CARLO_PROBABILITY_LIMIT
_PERIOD = ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP
_AEP = ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY
_CDF = ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY
_LOG = ResultMeaning.LOG10_QUANTILE_POSTERIOR_MOMENT
_AT_MEAN = ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS
_QUANTILES = frozenset((_MEAN, _MEDIAN, _PREDICTIVE, _QUANTILE_LIMIT, _LOG, _AT_MEAN))
_FLOWS = frozenset((_AEP, _CDF, _PERIOD, _PERIOD_LIMIT))
_LIMITS = frozenset((_QUANTILE_LIMIT, _PERIOD_LIMIT))
_INVERSES = frozenset((_MEDIAN, _PREDICTIVE, _QUANTILE_LIMIT, _PERIOD_LIMIT))


class PrecisionState(StrEnum):
    CONDITIONAL_PASS = "conditional_pass"
    CRITERIA_NOT_MET = "criteria_not_met"
    INDETERMINATE = "indeterminate"
    UNDEFINED = "undefined"
    EXECUTION_FAILED = "execution_failed"


class PrecisionUnavailable(ValueError):
    """Unsupported mathematical admission or insufficient numerical resolution."""


def _level(value: float | None) -> float:
    if value is None or not isfinite(value) or not 0 < value < 1:
        msg = "precision probability must be strictly between zero and one"
        raise ValueError(msg)
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class PrecisionRequest:
    """Complete output identity; levels are non-exceedance fractions, not AEP."""

    meaning: ResultMeaning | Literal["parameter_mean"]
    absolute_tolerance: float
    non_exceedance: float | None = None
    discharge: float | None = None
    interval_mass: float | None = None
    side: Literal["lower", "upper"] | None = None
    log_moment: Literal["mean", "standard_deviation"] | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.absolute_tolerance) or self.absolute_tolerance <= 0:
            msg = "precision tolerance must be finite and positive in the output unit"
            raise ValueError(msg)
        self._check_axes()
        self._check_summary()

    def _check_axes(self) -> None:
        if self.meaning in _QUANTILES:
            _level(self.non_exceedance)
            if self.discharge is None:
                return
        elif self.meaning in _FLOWS:
            if (
                self.non_exceedance is None
                and self.discharge is not None
                and isfinite(self.discharge)
                and self.discharge > 0
            ):
                return
        elif self.meaning == "parameter_mean" and self.non_exceedance is None and self.discharge is None:
            return
        msg = "unsupported output meaning or mismatched probability/discharge axis"
        raise ValueError(msg)

    def _check_summary(self) -> None:
        if self.meaning in _LIMITS:
            _level(self.interval_mass)
            if self.side not in ("lower", "upper"):
                msg = "a posterior probability limit needs an explicit side"
                raise ValueError(msg)
            _level(self.inverse_level)
        elif self.interval_mass is not None or self.side is not None:
            msg = "interval mass/side apply only to posterior limits"
            raise ValueError(msg)
        if self.meaning == _LOG:
            if self.log_moment not in ("mean", "standard_deviation"):
                msg = "a log10 moment needs its explicit kind"
                raise ValueError(msg)
        elif self.log_moment is not None:
            msg = "log10 moment kind applies only to logarithmic moments"
            raise ValueError(msg)

    @property
    def inverse_level(self) -> float:
        if self.meaning == _PREDICTIVE:
            return _level(self.non_exceedance)
        if self.meaning == _MEDIAN:
            return 0.5
        tail = (1 - _level(self.interval_mass)) / 2
        return _level(tail if self.side == "lower" else 1 - tail)

    @property
    def unit_is_discharge(self) -> bool:
        return self.meaning in _QUANTILES and self.meaning != _LOG

    @property
    def supported(self) -> bool:
        # The nonlinear expected-parameter and log-SD precision routes remain separate.
        return self.meaning != _AT_MEAN and self.log_moment != "standard_deviation"


@dataclass(frozen=True, slots=True, kw_only=True)
class PrecisionPlan:
    """Predeclared primary and audit streams, resource and scientific policy."""

    primary: ExplicitSamplingControls
    audit_seeds: tuple[int, ...]
    maximum_total_draws: int
    maximum_scalar_evaluations: int
    alpha: float
    minimum_side_count: int
    minimum_side_ess: float
    maximum_log_ratio: float
    criterion_source: str

    def __post_init__(self) -> None:
        if (
            len(self.audit_seeds) < 2
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**64 for seed in self.seeds)
        ):
            msg = "one primary and at least two distinct uint64 audit seeds are required"
            raise ValueError(msg)
        if (
            type(self.maximum_total_draws) is not int
            or type(self.maximum_scalar_evaluations) is not int
            or not 0 < self.total_draws <= self.maximum_total_draws
            or self.maximum_scalar_evaluations <= 0
        ):
            msg = "fixed stream schedule exceeds the declared resource budget"
            raise ValueError(msg)
        _level(self.alpha)
        if (
            type(self.minimum_side_count) is not int
            or not 0 < self.minimum_side_count <= self.primary.draw_count // 2
            or not isfinite(self.minimum_side_ess)
            or not 0 < self.minimum_side_ess <= self.primary.draw_count // 2
            or not isfinite(self.maximum_log_ratio)
            or self.maximum_log_ratio <= 0
            or not self.criterion_source.strip()
        ):
            msg = "explicit feasible side-resolution criteria and log-ratio ceiling are required"
            raise ValueError(msg)

    @property
    def seeds(self) -> tuple[int, ...]:
        return (self.primary.random_seed, *self.audit_seeds)

    @property
    def total_draws(self) -> int:
        return len(self.seeds) * self.primary.draw_count

    def preflight(self, requests: tuple[PrecisionRequest, ...]) -> tuple[int, float]:
        if not requests or len(set(requests)) != len(requests):
            msg = "a nonempty unique precision request set is required"
            raise ValueError(msg)
        # Conservative per-draw reservation: existing binary64 predictive inverse
        # uses at most 2*64 bracket expansions + 64 search evaluations, plus
        # component PPFs and two fixed-point audit evaluations.
        evaluations = self.total_draws * sum(260 if item.meaning == _PREDICTIVE else 4 for item in requests)
        if evaluations > self.maximum_scalar_evaluations:
            msg = "precision output evaluations exceed the declared budget"
            raise ValueError(msg)
        # Keep slots for unsupported and failed outputs; never recycle alpha.
        bounds = len(self.audit_seeds) * sum(4 if item.log_moment == "standard_deviation" else 2 for item in requests)
        tail = self.alpha / bounds
        if not 0 < tail < 0.5:
            msg = "simultaneous one-sided normal critical value is not representable"
            raise ValueError(msg)
        critical = -_NORMAL.inv_cdf(tail)
        if not isfinite(critical) or critical <= 0:
            msg = "simultaneous one-sided normal critical value is not finite and positive"
            raise ValueError(msg)
        return evaluations, critical


@dataclass(frozen=True, slots=True)
class Bounds:
    lower: float
    upper: float

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


def _window(estimate: float, tolerance: float) -> Bounds:
    # One inward ulp is conservative for a single rounded addition/subtraction.
    lower = nextafter(estimate - tolerance, inf)
    upper = nextafter(estimate + tolerance, -inf)
    if not isfinite(estimate - tolerance) or not isfinite(estimate + tolerance) or not lower < estimate < upper:
        msg = "requested precision window cannot be represented"
        raise PrecisionUnavailable(msg)
    return Bounds(lower, upper)


def _interval(estimate: float, mcse: float, critical: float) -> Bounds:
    half = critical * mcse
    if not isfinite(half) or half <= 0:
        msg = "stochastic interval half-width is not positive and finite"
        raise PrecisionUnavailable(msg)
    half = nextafter(half, inf)
    lower, upper = estimate - half, estimate + half
    if not lower < estimate < upper or not isfinite(lower) or not isfinite(upper):
        msg = "stochastic interval cannot be represented"
        raise PrecisionUnavailable(msg)
    result = Bounds(nextafter(lower, -inf), nextafter(upper, inf))
    if not isfinite(result.lower) or not isfinite(result.upper):
        msg = "outward-rounded stochastic interval is not finite"
        raise PrecisionUnavailable(msg)
    return result


@dataclass(frozen=True, slots=True)
class SideResolution:
    count: int
    mass: float
    effective_size: float


def _side(terms: list[tuple[float, float]]) -> SideResolution:
    if not terms:
        return SideResolution(0, 0.0, 0.0)
    largest = max(value for value, _ in terms)
    scaled = tuple(value / largest for value, _ in terms)
    return SideResolution(len(terms), fsum(weight for _, weight in terms), fsum(scaled) ** 2 / fsum(x * x for x in scaled))


@dataclass(frozen=True, slots=True)
class ScalarCheck:
    estimate: float
    approximate_mcse: float | None
    interval: Bounds | None
    positive: SideResolution
    negative: SideResolution
    reference: float
    reason: str | None = None


def _weighted_mean(particles: ImportanceParticles, values: tuple[float, ...]) -> float:
    if len(values) != len(particles.weights) or any(not isfinite(value) for value in values):
        msg = "invalid or nonfinite scalar integrand"
        raise PrecisionUnavailable(msg)
    # This admitted Normal target is everywhere positive. A stored zero weight
    # cannot represent genuine out-of-support mass in this component.
    if any(weight <= 0 for weight in particles.weights):
        msg = "positive target weight lost numerical resolution"
        raise PrecisionUnavailable(msg)
    products = tuple(w * x for w, x in zip(particles.weights, values, strict=True))
    if any(product == 0 and value != 0 for product, value in zip(products, values, strict=True)):
        msg = "weighted integrand underflow"
        raise PrecisionUnavailable(msg)
    estimate = fsum(products)
    if not isfinite(estimate):
        msg = "weighted expectation is not finite"
        raise PrecisionUnavailable(msg)
    return estimate


def _scalar_check(
    particles: ImportanceParticles,
    values: tuple[float, ...],
    reference: float,
    plan: PrecisionPlan,
    critical: float,
) -> ScalarCheck:
    estimate = _weighted_mean(particles, values)
    positive: list[tuple[float, float]] = []
    negative: list[tuple[float, float]] = []
    for weight, value in zip(particles.weights, values, strict=True):
        residual = value - estimate
        product = weight * residual if isfinite(residual) else weight * value - weight * estimate
        if not isfinite(product) or (product == 0 and value != estimate):
            msg = "weighted centered residual is not representable"
            raise PrecisionUnavailable(msg)
        if product > 0:
            positive.append((product, weight))
        elif product < 0:
            negative.append((-product, weight))
    pos, neg = _side(positive), _side(negative)
    if any(side.count < plan.minimum_side_count or side.effective_size < plan.minimum_side_ess for side in (pos, neg)):
        return ScalarCheck(estimate, None, None, pos, neg, reference, "insufficient centered-contribution resolution")
    # hypot uses scaled Euclidean norm, avoiding premature square underflow.
    mcse = hypot(*(value for value, _ in (*positive, *negative)))
    try:
        interval = _interval(estimate, mcse, critical)
    except PrecisionUnavailable as exc:
        return ScalarCheck(estimate, None, None, pos, neg, reference, str(exc))
    return ScalarCheck(estimate, mcse, interval, pos, neg, reference)


def _normal_probability(z: float, *, survival: bool) -> float:
    value = 0.5 * erfc(z / sqrt(2) if survival else -z / sqrt(2))
    if not 0 < value < 1:
        msg = "analytic probability has saturated or underflowed"
        raise PrecisionUnavailable(msg)
    return value


def _positive(value: float) -> float:
    if not isfinite(value) or value <= 0:
        msg = "positive finite quantity is not representable"
        raise PrecisionUnavailable(msg)
    return value


def _discharge(request: PrecisionRequest) -> float:
    if request.discharge is None:
        msg = "missing discharge for flow-axis output"
        raise ValueError(msg)
    return request.discharge


@dataclass(frozen=True, slots=True)
class NormalAdmission:
    mean: float
    variance: float
    log_flow_sd: float
    log_reference: float
    log_ratio_bound: float

    @property
    def posterior_sd(self) -> float:
        return sqrt(self.variance)

    @property
    def predictive_sd(self) -> float:
        return hypot(self.posterior_sd, self.log_flow_sd)

    def log_design_offset(self, request: PrecisionRequest) -> float:
        return self.log_reference + self.log_flow_sd * _NORMAL.inv_cdf(_level(request.non_exceedance))

    def probability(self, discharge: float, *, survival: bool) -> float:
        return _normal_probability(
            (log(discharge) - self.log_reference - self.mean) / self.predictive_sd, survival=survival
        )

    def inverse_probability(self, request: PrecisionRequest, bound: float, *, survival: bool) -> float:
        if request.meaning == _PREDICTIVE:
            return self.probability(bound, survival=survival)
        if request.meaning == _PERIOD_LIMIT:
            if bound <= 1:
                return 1.0 if survival else 0.0
            threshold = log(_discharge(request)) - self.log_reference
            threshold += self.log_flow_sd * _NORMAL.inv_cdf(_level(1 / bound))
            return _normal_probability((threshold - self.mean) / self.posterior_sd, survival=not survival)
        return _normal_probability(
            (log(bound) - self.log_design_offset(request) - self.mean) / self.posterior_sd, survival=survival
        )


def _admit(
    model: ClassifiedLognormalPosterior, proposal: ExplicitNormalProposal, plan: PrecisionPlan
) -> NormalAdmission:
    if model.fixed_log_sd is None or any(
        row.kind not in (ObservationKind.EXACT, ObservationKind.MISSING, ObservationKind.EXCLUDED)
        for row in model.series.annual
    ):
        msg = "precision admission currently requires known log-SD and exact included observations"
        raise PrecisionUnavailable(msg)
    if model.coordinates != proposal.coordinates:
        msg = "proposal and target coordinates differ"
        raise ValueError(msg)
    sigma = _positive(exp(model.fixed_log_sd))
    prior_variance = model.prior.covariance[0][0]
    proposal_variance = proposal.covariance[0][0]
    # Exact observations strictly reduce prior variance. This conservative
    # sufficient condition avoids a nearly equal variance comparison being
    # mistaken for proof of a globally bounded normalized importance ratio.
    if proposal_variance < prior_variance:
        msg = "bounded-ratio admission requires proposal variance at least the supplied prior variance"
        raise PrecisionUnavailable(msg)
    reference = log(model.reference_discharge)
    logs = tuple(
        log(row.value) - reference
        for row in model.series.annual
        if row.kind == ObservationKind.EXACT and row.value is not None
    )
    sigma2 = _positive(sigma * sigma)
    variance = _positive(1 / (1 / prior_variance + len(logs) / sigma2))
    mean = variance * fsum((model.prior.means[0] / prior_variance, fsum(logs) / sigma2))
    gap = _positive(proposal_variance - variance)
    displacement = (proposal.means[0] - mean) / sqrt(gap)
    log_bound = 0.5 * fsum((log(proposal_variance), -log(variance), displacement * displacement))
    if not isfinite(mean) or not isfinite(log_bound) or log_bound > plan.maximum_log_ratio:
        msg = "analytical ratio envelope exceeds the explicit ceiling or is not resolved"
        raise PrecisionUnavailable(msg)
    return NormalAdmission(mean, variance, sigma, reference, log_bound)


def _reference_value(request: PrecisionRequest, admission: NormalAdmission) -> float:
    if request.meaning in (_CDF, _AEP, _PERIOD):
        probability = admission.probability(_discharge(request), survival=request.meaning != _CDF)
        return _positive(1 / probability) if request.meaning == _PERIOD else probability
    if request.meaning == _PERIOD_LIMIT:
        mu = admission.mean - admission.posterior_sd * _NORMAL.inv_cdf(request.inverse_level)
        z = (log(_discharge(request)) - admission.log_reference - mu) / admission.log_flow_sd
        return _positive(1 / _normal_probability(z, survival=True))
    if request.meaning == "parameter_mean":
        return admission.mean
    if request.meaning == _PREDICTIVE:
        z = _NORMAL.inv_cdf(request.inverse_level)
        return _positive(exp(admission.log_reference + admission.mean + admission.predictive_sd * z))
    centre = admission.mean + admission.log_design_offset(request)
    if request.meaning == _LOG:
        return admission.posterior_sd / log(10) if request.log_moment == "standard_deviation" else centre / log(10)
    if request.meaning == _MEAN:
        centre += admission.variance / 2
    elif request.meaning in (_QUANTILE_LIMIT, _MEDIAN):
        centre += admission.posterior_sd * _NORMAL.inv_cdf(request.inverse_level)
    return _positive(exp(centre))


def _probability_values(
    model: ClassifiedLognormalPosterior, particles: ImportanceParticles, discharge: float, *, survival: bool
) -> tuple[float, ...]:
    kernels = tuple(model.kernel(draw) for draw in particles.draws)
    values = tuple(kernel.sf(discharge) if survival else kernel.cdf(discharge) for kernel in kernels)
    # For strictly positive finite discharge, this model has 0 < F,S < 1.
    # Never substitute 1-CDF for rare SF or accept rounded zero as true support.
    if any(not 0 < value < 1 for value in values):
        msg = "component probability has saturated or underflowed"
        raise PrecisionUnavailable(msg)
    return values


@dataclass(frozen=True, slots=True)
class SampleOutput:
    estimate: float
    values: tuple[float, ...]


def _sample_output(
    model: ClassifiedLognormalPosterior, particles: ImportanceParticles, request: PrecisionRequest
) -> SampleOutput:
    if any(weight <= 0 for weight in particles.weights):
        msg = "positive target weight lost numerical resolution"
        raise PrecisionUnavailable(msg)
    if request.meaning == _PREDICTIVE:
        return SampleOutput(_positive(predictive_quantile(particles, request.inverse_level, model.kernel)), ())
    if request.meaning in _FLOWS:
        values = _probability_values(model, particles, _discharge(request), survival=request.meaning != _CDF)
        if request.meaning == _PERIOD_LIMIT:
            periods = tuple(_positive(1 / value) for value in values)
            return SampleOutput(weighted_quantile(periods, particles.weights, request.inverse_level), periods)
        mean = _weighted_mean(particles, values)
        if not 0 < mean < 1:
            msg = "weighted probability has saturated or underflowed"
            raise PrecisionUnavailable(msg)
        return SampleOutput(_positive(1 / mean) if request.meaning == _PERIOD else mean, values)
    return _sample_quantile_output(model, particles, request)


def _sample_quantile_output(
    model: ClassifiedLognormalPosterior, particles: ImportanceParticles, request: PrecisionRequest
) -> SampleOutput:
    if request.meaning == "parameter_mean":
        values = tuple(draw[0] for draw in particles.draws)
    elif request.meaning == _AT_MEAN:
        mean = _weighted_mean(particles, tuple(draw[0] for draw in particles.draws))
        return SampleOutput(_positive(model.kernel((mean,)).ppf(_level(request.non_exceedance))), ())
    else:
        values = tuple(_positive(model.kernel(draw).ppf(_level(request.non_exceedance))) for draw in particles.draws)
    if request.meaning in (_MEDIAN, _QUANTILE_LIMIT):
        return SampleOutput(weighted_quantile(values, particles.weights, request.inverse_level), values)
    if request.meaning == _LOG:
        values = tuple(log10(value) for value in values)
    estimate = _weighted_mean(particles, values)
    if request.log_moment == "standard_deviation":
        estimate = sqrt(_weighted_mean(particles, tuple((value - estimate) ** 2 for value in values)))
    return SampleOutput(estimate, values)


@dataclass(frozen=True, slots=True)
class OutputCheck:
    request: PrecisionRequest
    state: PrecisionState
    estimate: float | None
    reference: float | None
    window: Bounds | None
    scalar_checks: tuple[ScalarCheck, ...] = ()
    reason: str | None = None


def _bracket_check(
    model: ClassifiedLognormalPosterior,
    particles: ImportanceParticles,
    request: PrecisionRequest,
    sample: SampleOutput,
    window: Bounds,
    admission: NormalAdmission,
    plan: PrecisionPlan,
    critical: float,
) -> tuple[tuple[ScalarCheck, ...], bool]:
    probability = request.inverse_level
    survival = probability > 0.5
    target = 1 - probability if survival else probability
    checks: list[ScalarCheck] = []
    decisions: list[bool] = []
    for bound, is_lower in ((window.lower, True), (window.upper, False)):
        if request.meaning == _PREDICTIVE:
            if bound <= 0:
                msg = "predictive audit window must have positive finite endpoints"
                raise PrecisionUnavailable(msg)
            values = _probability_values(model, particles, bound, survival=survival)
        else:
            values = tuple(float(value > bound if survival else value <= bound) for value in sample.values)
        if request.meaning != _PERIOD_LIMIT and bound <= 0:
            msg = "design-quantile audit window must have positive finite endpoints"
            raise PrecisionUnavailable(msg)
        reference = admission.inverse_probability(request, bound, survival=survival)
        check = _scalar_check(particles, values, reference, plan, critical)
        checks.append(check)
        if check.interval is None:
            decisions.append(False)
            continue
        # CDF: H(L)<b<H(U); SF: K(L)>1-b>K(U).
        # Only the required one-sided bound is counted. The independently
        # derived probability is an additional model-specific check.
        if is_lower != survival:
            decisions.append(reference <= check.interval.upper < target)
        else:
            decisions.append(reference >= check.interval.lower > target)
    return tuple(checks), all(decisions)


def _expectation_check(
    particles: ImportanceParticles,
    request: PrecisionRequest,
    sample: SampleOutput,
    reference: float,
    window: Bounds,
    plan: PrecisionPlan,
    critical: float,
) -> tuple[tuple[ScalarCheck, ...], bool]:
    scalar_reference = _positive(1 / reference) if request.meaning == _PERIOD else reference
    check = _scalar_check(particles, sample.values, scalar_reference, plan, critical)
    if check.interval is None:
        return (check,), False
    interval = check.interval
    if not interval.contains(scalar_reference):
        return (check,), False
    if request.meaning == _PERIOD:
        if interval.lower <= 0:
            return (check,), False
        lower = nextafter(1 / min(interval.upper, 1.0), -inf)
        upper = nextafter(1 / interval.lower, inf)
        if not isfinite(lower) or not isfinite(upper):
            return (check,), False
        interval = Bounds(lower, upper)
    return (check,), window.contains(interval.lower) and window.contains(interval.upper)


def _audit_output(
    model: ClassifiedLognormalPosterior,
    particles: ImportanceParticles,
    request: PrecisionRequest,
    admission: NormalAdmission,
    plan: PrecisionPlan,
    critical: float,
    primary: OutputCheck | None,
    *,
    is_primary: bool,
) -> OutputCheck:
    reference: float | None = None
    sample: SampleOutput | None = None
    window: Bounds | None = None
    try:
        reference = _reference_value(request, admission)
        sample = _sample_output(model, particles, request)
        if not request.supported:
            msg = "precision propagation for this output is not admitted"
            raise PrecisionUnavailable(msg)
        if is_primary:
            window = _window(sample.estimate, request.absolute_tolerance)
            state = PrecisionState.CONDITIONAL_PASS if window.contains(reference) else PrecisionState.CRITERIA_NOT_MET
            return OutputCheck(request, state, sample.estimate, reference, window)
        if primary is None or primary.window is None:
            msg = "primary precision window unavailable"
            raise PrecisionUnavailable(msg)
        window = primary.window
        if request.meaning in _INVERSES:
            checks, accepted = _bracket_check(model, particles, request, sample, window, admission, plan, critical)
        else:
            checks, accepted = _expectation_check(particles, request, sample, reference, window, plan, critical)
        if any(check.interval is None for check in checks):
            state = PrecisionState.INDETERMINATE
        elif accepted and window.contains(reference) and window.contains(sample.estimate):
            state = PrecisionState.CONDITIONAL_PASS
        else:
            state = PrecisionState.CRITERIA_NOT_MET
        return OutputCheck(request, state, sample.estimate, reference, window, checks)
    except PrecisionUnavailable as exc:
        return OutputCheck(
            request,
            PrecisionState.INDETERMINATE,
            None if sample is None else sample.estimate,
            reference,
            window,
            reason=str(exc),
        )
    except (ArithmeticError, ValueError) as exc:
        return OutputCheck(request, PrecisionState.EXECUTION_FAILED, None, reference, window, reason=str(exc))


@dataclass(frozen=True, slots=True)
class Attempt:
    seed: int
    requested_draws: int
    completed_draws: int | None
    attempted: bool
    effective_size: float | None
    largest_weight: float | None
    stream: str | None
    outputs: tuple[OutputCheck, ...]
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PrecisionAudit:
    """All fixed-budget attempts and complete model, proposal and output identities."""

    model: ClassifiedLognormalPosterior
    proposal: ExplicitNormalProposal
    provenance: RunProvenance
    plan: PrecisionPlan
    requests: tuple[PrecisionRequest, ...]
    admission: NormalAdmission | None
    reserved_evaluations: int
    critical_value: float
    attempts: tuple[Attempt, ...]

    @property
    def states(self) -> tuple[PrecisionState, ...]:
        return tuple(
            _aggregate(tuple(attempt.outputs[i].state for attempt in self.attempts)) for i in range(len(self.requests))
        )

    @property
    def state(self) -> PrecisionState:
        return _aggregate(self.states)


def _aggregate(states: tuple[PrecisionState, ...]) -> PrecisionState:
    for state in (
        PrecisionState.UNDEFINED,
        PrecisionState.EXECUTION_FAILED,
        PrecisionState.INDETERMINATE,
        PrecisionState.CRITERIA_NOT_MET,
    ):
        if state in states:
            return state
    return PrecisionState.CONDITIONAL_PASS


def _unavailable(request: PrecisionRequest, model: ClassifiedLognormalPosterior, reason: str) -> OutputCheck:
    if model.fixed_log_sd is None and request.meaning == _MEAN and _level(request.non_exceedance) > 0.5:
        return OutputCheck(
            request,
            PrecisionState.UNDEFINED,
            None,
            None,
            None,
            reason="upper design-flood mean diverges for this untruncated Gaussian log-SD model",
        )
    return OutputCheck(request, PrecisionState.INDETERMINATE, None, None, None, reason=reason)


def audit_lognormal_precision(
    *,
    model: ClassifiedLognormalPosterior,
    proposal: ExplicitNormalProposal,
    provenance: RunProvenance,
    plan: PrecisionPlan,
    requests: tuple[PrecisionRequest, ...],
) -> PrecisionAudit:
    """Execute one fixed primary and every scheduled audit stream; no adaptive stopping.

    Only the known-scale exact-record model receives the analytical moment/tail
    admission implemented here. Unsupported models do not acquire precision
    merely because the sampled weights look healthy.
    """
    reserved, critical = plan.preflight(requests)
    if (
        proposal.coordinates != model.coordinates
        or provenance.input_sha256 != model.source_sha256
        or provenance.evidence_status == EvidenceStatus.QUALIFIED_REFERENCE
    ):
        msg = "audit provenance or proposal does not identify the bound source model"
        raise ValueError(msg)
    admission: NormalAdmission | None = None
    try:
        admission = _admit(model, proposal, plan)
    except (PrecisionUnavailable, ArithmeticError) as exc:
        outputs = tuple(_unavailable(request, model, str(exc)) for request in requests)
        attempts = tuple(
            Attempt(seed, plan.primary.draw_count, None, False, None, None, None, outputs, str(exc)) for seed in plan.seeds
        )
    else:
        attempts = _run_attempts(model, proposal, provenance, plan, requests, admission, critical)
    return PrecisionAudit(model, proposal, provenance, plan, requests, admission, reserved, critical, attempts)


def _run_attempts(
    model: ClassifiedLognormalPosterior,
    proposal: ExplicitNormalProposal,
    provenance: RunProvenance,
    plan: PrecisionPlan,
    requests: tuple[PrecisionRequest, ...],
    admission: NormalAdmission,
    critical: float,
) -> tuple[Attempt, ...]:
    attempts: list[Attempt] = []
    target = model.target()
    for seed in plan.seeds:
        controls = replace(plan.primary, random_seed=seed)
        try:
            run = independent_importance_sample(proposal=proposal, target=target, controls=controls, provenance=provenance)
        except (ArithmeticError, ValueError) as exc:
            failed = tuple(
                OutputCheck(request, PrecisionState.EXECUTION_FAILED, None, None, None, reason=str(exc))
                for request in requests
            )
            attempts.append(Attempt(seed, controls.draw_count, None, True, None, None, None, failed, str(exc)))
            continue
        outputs = tuple(
            _audit_output(
                model,
                run.particles,
                request,
                admission,
                plan,
                critical,
                attempts[0].outputs[index] if attempts else None,
                is_primary=not attempts,
            )
            for index, request in enumerate(requests)
        )
        attempts.append(
            Attempt(
                seed,
                controls.draw_count,
                len(run.particles.draws),
                True,
                run.moments.effective_sample_size,
                run.moments.largest_weight,
                run.particles.random_stream_description,
                outputs,
            )
        )
    return tuple(attempts)
