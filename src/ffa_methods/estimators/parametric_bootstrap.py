"""Complete-record percentile bootstrap with numerical endpoint checks."""

from bisect import bisect_left
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from random import Random

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
class BootstrapResult:
    endpoints: tuple[tuple[BootstrapEndpoint, BootstrapEndpoint], ...]
    failures: tuple[tuple[str, int], ...]


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


def parametric_bootstrap(
    generate: Quantile,
    refit: Callable[[tuple[float, ...]], Quantile],
    *,
    count: int,
    probabilities: tuple[float, ...],
    settings: BootstrapSettings,
) -> BootstrapResult:
    """Refit every generated record and commit only complete finite replicate vectors."""
    rng = Random(settings.seed)  # noqa: S311  # Reproducible, non-security simulation.
    failures: Counter[str] = Counter()
    draws: list[tuple[float, ...]] = []
    for _ in range(settings.replications):
        try:
            record = tuple(generate(max(rng.random(), 2**-53)) for _ in range(count))
            if not all(isfinite(value) for value in record):
                failures["nonfinite simulated observations"] += 1
                continue
            estimate = refit(record)
            quantiles = tuple(estimate(probability) for probability in probabilities)
        except (ValueError, OverflowError) as exc:
            failures[str(exc)] += 1
            continue
        if not all(isfinite(value) for value in quantiles):
            failures["nonfinite replicate quantiles"] += 1
            continue
        draws.append(quantiles)
    failed = failures.total()
    if not draws or failed / settings.replications > settings.maximum_failure_fraction:
        msg = f"bootstrap failure criterion exceeded ({failed}/{settings.replications}): {dict(failures)}"
        raise ValueError(msg)
    tail = (1 - settings.confidence) / 2
    # Allocate 5% across the two tails of each endpoint at every requested F.
    alpha = 0.05 / (4 * len(probabilities))
    ranks = [_endpoint_ranks(len(draws), p, alpha) for p in (tail, 1 - tail)]
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
            raise ValueError(msg)
        endpoints.append((lo, hi))
    return BootstrapResult(tuple(endpoints), tuple(sorted(failures.items())))
