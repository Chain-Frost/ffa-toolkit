"""Complete-record percentile bootstrap with numerical endpoint checks."""

from bisect import bisect_left
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import Literal

from scipy.special import bdtr

from ffa_methods.estimators.gev_lh import linear_sample_quantile

Quantile = Callable[[float], float]


@dataclass(frozen=True, slots=True)
class BootstrapSettings:
    replications: int
    seed: int
    confidence: float
    maximum_failure_fraction: float
    endpoint_tolerance: float


@dataclass(frozen=True, slots=True)
class BootstrapEndpoint:
    estimate: float
    mc_lower: float
    mc_upper: float


@dataclass(frozen=True, slots=True)
class BootstrapRefit:
    quantile: Quantile
    selected_h: int


@dataclass(frozen=True, slots=True)
class BootstrapAttempt:
    selected_h: int | None
    failure_stage: Literal["generation", "refit", "quantiles"] | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    endpoints: tuple[tuple[BootstrapEndpoint, BootstrapEndpoint], ...]
    failures: tuple[tuple[str, int], ...]
    attempts: tuple[BootstrapAttempt, ...]


class BootstrapRefusalError(ValueError):
    """An interval gate failed after recording the complete attempt stream."""

    def __init__(
        self,
        code: Literal["failure_fraction", "endpoint_count", "endpoint_precision"],
        message: str,
        attempts: tuple[BootstrapAttempt, ...],
        failures: tuple[tuple[str, int], ...],
    ) -> None:
        super().__init__(message)
        self.code = code
        self.attempts = attempts
        self.failures = failures


def _endpoint_ranks(size: int, probability: float, alpha: float) -> tuple[int, int]:
    indices = range(size + 1)

    def cdf(k: int) -> float:
        return float(bdtr(k, size, probability))

    left = bisect_left(indices, alpha, key=cdf) - 1
    right = bisect_left(indices, 1 - alpha, key=cdf)
    if left < 0 or right >= size:
        msg = "too few successful bootstrap replications to bound endpoint Monte Carlo error"
        raise ValueError(msg)
    return left, right


def _run_attempt(
    generate: Quantile,
    refit: Callable[[tuple[float, ...]], Quantile | BootstrapRefit],
    uniforms: tuple[float, ...],
    probabilities: tuple[float, ...],
) -> tuple[BootstrapAttempt, tuple[float, ...] | None]:
    stage: Literal["generation", "refit", "quantiles"] = "generation"
    selected_h = None
    try:
        record = tuple(generate(uniform) for uniform in uniforms)
        if not all(isfinite(value) for value in record):
            return BootstrapAttempt(None, stage, "nonfinite simulated observations"), None
        stage = "refit"
        estimate = refit(record)
        if isinstance(estimate, BootstrapRefit):
            selected_h = estimate.selected_h
            estimate = estimate.quantile
        stage = "quantiles"
        quantiles = tuple(estimate(probability) for probability in probabilities)
        if not all(isfinite(value) for value in quantiles):
            return BootstrapAttempt(selected_h, stage, "nonfinite replicate quantiles"), None
    except (ValueError, OverflowError) as exc:
        return BootstrapAttempt(selected_h, stage, str(exc)), None
    return BootstrapAttempt(selected_h), quantiles


def parametric_bootstrap(
    generate: Quantile,
    refit: Callable[[tuple[float, ...]], Quantile | BootstrapRefit],
    *,
    count: int,
    probabilities: tuple[float, ...],
    settings: BootstrapSettings,
) -> BootstrapResult:
    """Refit every generated record and commit only complete finite replicate vectors."""
    rng = Random(settings.seed)  # noqa: S311  # Reproducible, non-security simulation.
    attempts: list[BootstrapAttempt] = []
    draws: list[tuple[float, ...]] = []
    for _ in range(settings.replications):
        uniforms = tuple(max(rng.random(), 2**-53) for _ in range(count))
        attempt, quantiles = _run_attempt(generate, refit, uniforms, probabilities)
        attempts.append(attempt)
        if quantiles is not None:
            draws.append(quantiles)
    ledger = tuple(attempts)
    failures = Counter(attempt.failure_reason for attempt in ledger if attempt.failure_reason is not None)
    reasons = tuple(sorted(failures.items()))
    failed = failures.total()
    if not draws or failed / settings.replications > settings.maximum_failure_fraction:
        msg = f"bootstrap failure criterion exceeded ({failed}/{settings.replications}): {dict(failures)}"
        code = "failure_fraction"
        raise BootstrapRefusalError(code, msg, ledger, reasons)
    tail = (1 - settings.confidence) / 2
    # Allocate 5% across the two tails of each endpoint at every requested F.
    alpha = 0.05 / (4 * len(probabilities))
    try:
        ranks = [_endpoint_ranks(len(draws), p, alpha) for p in (tail, 1 - tail)]
    except ValueError as exc:
        msg = str(exc)
        code = "endpoint_count"
        raise BootstrapRefusalError(code, msg, ledger, reasons) from exc
    endpoints: list[tuple[BootstrapEndpoint, BootstrapEndpoint]] = []
    for probability, column in zip(probabilities, zip(*draws, strict=True), strict=True):
        ordered = sorted(column)
        lo, hi = (
            BootstrapEndpoint(linear_sample_quantile(ordered, p), ordered[left], ordered[right])
            for p, (left, right) in zip((tail, 1 - tail), ranks, strict=True)
        )
        if (
            max(endpoint.mc_upper - endpoint.estimate for endpoint in (lo, hi)) > settings.endpoint_tolerance
            or max(endpoint.estimate - endpoint.mc_lower for endpoint in (lo, hi)) > settings.endpoint_tolerance
        ):
            msg = f"bootstrap endpoint Monte Carlo precision failed at non-exceedance probability {probability}"
            code = "endpoint_precision"
            raise BootstrapRefusalError(code, msg, ledger, reasons)
        endpoints.append((lo, hi))
    return BootstrapResult(tuple(endpoints), reasons, ledger)
