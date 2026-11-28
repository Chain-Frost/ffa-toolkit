"""WP06: weighted Bayesian outputs, without an unverified legacy proposal algorithm.

Inputs must already be in the source parameter coordinates. This is an internal
composition component, not a complete FLIKE-compatible fitting workflow.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import exp, fsum, inf, isfinite, log, log10, sqrt
from struct import pack, unpack
from sys import float_info
from typing import TypeAlias

from .contracts import DistributionKernel, ParameterSummary, RunProvenance

KernelFactory: TypeAlias = Callable[[tuple[float, ...]], DistributionKernel]


def _density_log_ratio(log_likelihood: float, log_prior: float, log_proposal: float) -> float:
    """Preserve finite large ratios without overflowing intermediate sums.

    math.fsum itself can overflow before cancellation, so try each possible
    pair first. A representable sum of three finite terms has at least one
    ordering without an overflowing intermediate.
    """
    for terms in (
        (log_likelihood, log_prior, -log_proposal),
        (log_likelihood, -log_proposal, log_prior),
        (log_prior, -log_proposal, log_likelihood),
    ):
        try:
            return fsum(terms)
        except OverflowError:
            continue
    msg = "importance density ratio is not finite"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class ImportanceParticles:
    """Proposal draws and unaltered density terms, with explicit run custody.

    Values must be densities in *the same named coordinates*. WP05 is responsible
    for the source prior and all transformation Jacobians. No proposal or default
    density is manufactured by this class.
    """

    coordinates: tuple[str, ...]
    draws: tuple[tuple[float, ...], ...]
    log_likelihood: tuple[float, ...]
    log_prior: tuple[float, ...]
    log_proposal: tuple[float, ...]
    provenance: RunProvenance
    proposal_description: str
    random_stream_description: str
    weights: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate density and shape without dropping invalid samples silently."""
        n = len(self.draws)
        if not self.coordinates or len(set(self.coordinates)) != len(self.coordinates):
            msg = "parameter coordinates must be nonempty and unique"
            raise ValueError(msg)
        if not n or any(len(a) != n for a in (self.log_likelihood, self.log_prior, self.log_proposal)):
            msg = "density terms must have one entry per nonempty draw set"
            raise ValueError(msg)
        if any(len(draw) != len(self.coordinates) or any(not isfinite(x) for x in draw) for draw in self.draws):
            msg = "each proposal draw must have finite coordinates in the declared order"
            raise ValueError(msg)
        if not self.proposal_description.strip() or not self.random_stream_description.strip():
            msg = "proposal and random-stream provenance must be explicit"
            raise ValueError(msg)
        if any(not isfinite(x) for x in self.log_proposal):
            msg = "the proposal log density must be finite for each sampled draw"
            raise ValueError(msg)
        if any(x != x or x == inf for x in (*self.log_likelihood, *self.log_prior)):
            msg = "target log densities must be finite or negative infinity"
            raise ValueError(msg)
        log_weights = tuple(
            _density_log_ratio(a, b, c)
            for a, b, c in zip(self.log_likelihood, self.log_prior, self.log_proposal, strict=True)
        )
        if any(w != w or w == inf for w in log_weights):
            msg = "importance density ratio is not finite"
            raise ValueError(msg)
        greatest = max(log_weights)
        if greatest == -inf:
            msg = "all importance weights are zero; posterior is not represented"
            raise ValueError(msg)
        scaled = tuple(exp(w - greatest) for w in log_weights)
        total = fsum(scaled)
        weights = tuple(w / total for w in scaled)
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True, slots=True, kw_only=True)
class PosteriorMoments:
    """Population moments in original coordinates, not rescaled library coordinates."""

    parameters: tuple[ParameterSummary, ...]
    covariance: tuple[tuple[float, ...], ...]
    correlation: tuple[tuple[float | None, ...], ...]
    effective_sample_size: float
    effective_fraction: float
    largest_weight: float
    zero_weight_count: int
    entropy_effective_size: float


@dataclass(frozen=True, slots=True, kw_only=True)
class QuantileOutputs:
    """Different quantile estimands; empirical interval interpolation is provisional."""

    non_exceedance: float
    at_expected_parameters: float
    mean_of_quantiles: float | None
    median_of_quantiles: float
    predictive_quantile: float
    probability_limits: tuple[float, float]
    mean_log10_quantile: float | None
    standard_deviation_log10_quantile: float | None
    nonpositive_quantile_weight: float
    interval_mass: float
    lower_limit_effective_size: float
    upper_limit_effective_size: float
    mean_undefined_reason: str | None = None
    log_undefined_reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbabilityOutputs:
    """The legacy *Expected probability* column corresponds to F, not AEP."""

    discharge: float
    expected_non_exceedance: float
    expected_aep: float
    return_period: float
    return_period_limits: tuple[float, float]
    interval_mass: float
    zero_survival_weight: float


def _level(probability: float) -> None:
    if not isfinite(probability) or not 0.0 < probability < 1.0:
        msg = "probability must be finite and strictly between zero and one"
        raise ValueError(msg)


def _interval_mass(interval_mass: float) -> None:
    _level(interval_mass)


def weighted_quantile(values: Sequence[float], weights: Sequence[float], probability: float) -> float:
    """Left-continuous inverse weighted ECDF; NOT YET verified as FLIKE's rule.

    Values may include +inf (an unbounded return period) but never NaN/-inf.
    This step-function convention is explicit to avoid an arbitrary interpolation.
    """
    if not 0.0 <= probability <= 1.0 or not isfinite(probability):
        msg = "weighted quantile probability must be within [0, 1]"
        raise ValueError(msg)
    if not values or len(values) != len(weights):
        msg = "values and weights must be nonempty and equally sized"
        raise ValueError(msg)
    if any(x != x or x == -inf for x in values):
        msg = "quantile values must not include NaN or negative infinity"
        raise ValueError(msg)
    if any(not isfinite(w) or w < 0 for w in weights):
        msg = "weights must be finite and nonnegative"
        raise ValueError(msg)
    scale = max(weights)
    if scale == 0.0:
        msg = "positive finite total weight is required"
        raise ValueError(msg)
    # Finite unnormalised weights may overflow when summed; normalise first.
    scaled = tuple(w / scale for w in weights)
    total = fsum(scaled)
    ordered = sorted(zip(values, scaled, strict=True), key=lambda pair: pair[0])
    positive = [(value, weight) for value, weight in ordered if weight > 0]
    if probability == 0:
        return positive[0][0]
    if probability == 1:
        return positive[-1][0]
    target = total * probability
    cumulative = 0.0
    for value, weight in positive:
        cumulative += weight
        if cumulative >= target:
            return value
    return positive[-1][0]  # finite-precision cumulative rounding at p=1


def posterior_moments(particles: ImportanceParticles) -> PosteriorMoments:
    """Weighted covariance uses posterior-population (not unbiased sample) units."""
    weights = particles.weights
    dim = len(particles.coordinates)
    means = tuple(fsum(w * draw[i] for w, draw in zip(weights, particles.draws, strict=True)) for i in range(dim))
    covariance = tuple(
        tuple(
            fsum(
                w * (draw[i] - means[i]) * (draw[j] - means[j])
                for w, draw in zip(weights, particles.draws, strict=True)
            )
            for j in range(dim)
        )
        for i in range(dim)
    )
    deviations = tuple(sqrt(covariance[i][i]) for i in range(dim))
    if any(not isfinite(x) for x in (*means, *deviations, *(x for row in covariance for x in row))):
        msg = "posterior moments overflow; parameter scaling or proposal is inadequate"
        raise ValueError(msg)
    correlations = tuple(
        tuple(
            max(-1.0, min(1.0, covariance[i][j] / (deviations[i] * deviations[j])))
            if deviations[i] > 0 and deviations[j] > 0
            else None
            for j in range(dim)
        )
        for i in range(dim)
    )
    ess = 1.0 / fsum(w * w for w in weights)
    entropy = exp(-fsum(w * log(w) for w in weights if w > 0))
    return PosteriorMoments(
        parameters=tuple(
            ParameterSummary(name=name, coordinate=name, mean=mean, standard_deviation=std)
            for name, mean, std in zip(particles.coordinates, means, deviations, strict=True)
        ),
        covariance=covariance,
        correlation=correlations,
        effective_sample_size=ess,
        effective_fraction=ess / len(weights),
        largest_weight=max(weights),
        zero_weight_count=sum(w == 0.0 for w in weights),
        entropy_effective_size=entropy,
    )


def require_weight_adequacy(
    particles: ImportanceParticles, *, minimum_effective_size: float, maximum_single_weight: float
) -> PosteriorMoments:
    """Fail on caller-defined proposal-quality limits; there is no legacy default.

    These global weight checks do NOT certify rare-quantile tail precision.
    """
    if not isfinite(minimum_effective_size) or minimum_effective_size <= 0:
        msg = "minimum effective sample size must be finite and positive"
        raise ValueError(msg)
    if not isfinite(maximum_single_weight) or not 0 < maximum_single_weight <= 1:
        msg = "maximum single weight must be in (0, 1]"
        raise ValueError(msg)
    summary = posterior_moments(particles)
    if summary.effective_sample_size < minimum_effective_size or summary.largest_weight > maximum_single_weight:
        msg = "importance proposal fails the explicitly requested weight adequacy limits"
        raise ValueError(msg)
    return summary


def tail_effective_sample_size(
    weights: Sequence[float], quantiles: Sequence[float], bound: float, *, upper: bool
) -> float:
    """Weight-only tail resolution diagnostic, not Monte Carlo standard error."""
    if not weights or len(weights) != len(quantiles):
        msg = "tail weights and quantiles must be nonempty and equally sized"
        raise ValueError(msg)
    if any(not isfinite(w) or w < 0 for w in weights):
        msg = "tail weights must be finite and nonnegative"
        raise ValueError(msg)
    if bound != bound or bound == -inf or any(q != q or q == -inf for q in quantiles):
        msg = "tail quantiles and bound must not contain NaN or negative infinity"
        raise ValueError(msg)
    tail_weights = tuple(
        w for w, q in zip(weights, quantiles, strict=True) if (q >= bound if upper else q <= bound) and w > 0
    )
    if not tail_weights:
        return 0.0
    scale = max(tail_weights)
    relative = tuple(weight / scale for weight in tail_weights)
    total = fsum(relative)
    return total * total / fsum(weight * weight for weight in relative)


def _kernels(particles: ImportanceParticles, factory: KernelFactory) -> tuple[DistributionKernel, ...]:
    """Do not omit failed posterior components or change distributions."""
    return tuple(factory(draw) for draw in particles.draws)


def _mixture_probability(
    discharge: float, kernels: Sequence[DistributionKernel], weights: Sequence[float], *, survival: bool
) -> float:
    probabilities = tuple(kernel.sf(discharge) if survival else kernel.cdf(discharge) for kernel in kernels)
    if any(not isfinite(p) or p < 0 or p > 1 for p in probabilities):
        kind = "survival" if survival else "non-exceedance"
        msg = f"a posterior component returned an invalid {kind} probability"
        raise ValueError(msg)
    return fsum(w * p for w, p in zip(weights, probabilities, strict=True))


_FLOAT_SIGN = 1 << 63
_FLOAT_MASK = (1 << 64) - 1


def _ordered_float_key(value: float) -> int:
    """Monotonic binary64 integer key for finite values, including negatives."""
    bits = int.from_bytes(pack(">d", value), "big")
    return (~bits & _FLOAT_MASK) if bits & _FLOAT_SIGN else (bits | _FLOAT_SIGN)


def _float_from_ordered_key(key: int) -> float:
    """Inverse of the sortable binary64 representation."""
    bits = (~key & _FLOAT_MASK) if key < _FLOAT_SIGN else (key & ~_FLOAT_SIGN)
    return unpack(">d", bits.to_bytes(8, "big"))[0]


def _predictive_bracket(
    low: float,
    high: float,
    reached: Callable[[float], bool],
) -> tuple[int, int]:
    """Repair finite PPF rounding using exponentially widened binary64 keys.

    A component PPF may round down: its CDF need not reach the exact target.
    Widen until the *evaluated* CDF brackets the target without assuming a
    vendor-specific interpolation tolerance or changing the mixture law.
    """
    low_key, high_key = _ordered_float_key(low), _ordered_float_key(high)
    minimum = _ordered_float_key(-float_info.max)
    maximum = _ordered_float_key(float_info.max)
    stride = 1
    while low_key > minimum and reached(_float_from_ordered_key(low_key)):
        low_key = max(minimum, low_key - stride)
        stride *= 2
    if reached(_float_from_ordered_key(low_key)):
        return low_key, low_key  # The lowest representable discharge already reaches F.
    stride = 1
    while high_key < maximum and not reached(_float_from_ordered_key(high_key)):
        high_key = min(maximum, high_key + stride)
        stride *= 2
    if not reached(_float_from_ordered_key(high_key)):
        msg = "posterior-predictive quantile cannot be represented by a finite discharge"
        raise ValueError(msg)
    return low_key, high_key


def predictive_quantile(
    particles: ImportanceParticles, non_exceedance: float, factory: KernelFactory, *, direct_upper_tail: bool = False
) -> float:
    """Generalized inverse of the posterior-predictive CDF, including atoms.

    The per-component PPFs bracket the mixture quantile for 0<F<1 when
    correctly implemented. The search locates the smallest representable
    discharge meeting the CDF target, even across extreme mixed scales and jumps.
    """
    _level(non_exceedance)
    kernels = _kernels(particles, factory)
    brackets = tuple(kernel.ppf(non_exceedance) for kernel in kernels)
    if any(not isfinite(x) for x in brackets):
        msg = "cannot bracket the predictive quantile: non-finite component quantile"
        raise ValueError(msg)
    survival = direct_upper_tail and non_exceedance > 0.5

    def reached(discharge: float) -> bool:
        tail = _mixture_probability(discharge, kernels, particles.weights, survival=survival)
        return tail <= 1 - non_exceedance if survival else tail >= non_exceedance

    low_key, high_key = _predictive_bracket(min(brackets), max(brackets), reached)
    # Interior binary64 search takes at most 64 evaluations, regardless of
    # component scale separation. The bracketing helper handles PPF roundoff.
    while low_key + 1 < high_key:
        mid_key = (low_key + high_key) // 2
        mid = _float_from_ordered_key(mid_key)
        if reached(mid):
            high_key = mid_key
        else:
            low_key = mid_key
    return _float_from_ordered_key(high_key)


def quantile_outputs(
    particles: ImportanceParticles,
    non_exceedance: float,
    factory: KernelFactory,
    *,
    interval_mass: float,
    mean_undefined_reason: str | None = None,
    log_undefined_reason: str | None = None,
    direct_upper_tail: bool = False,
) -> QuantileOutputs:
    """Compute separate posterior quantile estimands and explicit log10 warnings."""
    _level(non_exceedance)
    _interval_mass(interval_mass)
    weights = particles.weights
    kernels = _kernels(particles, factory)
    quantiles = tuple(k.ppf(non_exceedance) for k in kernels)
    if any(not isfinite(q) for q in quantiles):
        msg = "a posterior component produced a non-finite quantile"
        raise ValueError(msg)
    means = tuple(p.mean for p in posterior_moments(particles).parameters)
    at_expected = factory(means).ppf(non_exceedance)
    if not isfinite(at_expected):
        msg = "expected-parameter quantile is not finite"
        raise ValueError(msg)
    tail = (1.0 - interval_mass) / 2.0
    lower_limit = weighted_quantile(quantiles, weights, tail)
    upper_limit = weighted_quantile(quantiles, weights, 1.0 - tail)
    nonpositive = fsum(w for w, q in zip(weights, quantiles, strict=True) if q <= 0)
    if nonpositive or log_undefined_reason:
        mean_log = sd_log = None
    else:
        log_quantiles = tuple(log10(q) if w > 0 else 0.0 for w, q in zip(weights, quantiles, strict=True))
        mean_log = fsum(w * x for w, x in zip(weights, log_quantiles, strict=True))
        sd_log = sqrt(fsum(w * (x - mean_log) ** 2 for w, x in zip(weights, log_quantiles, strict=True)))
    return QuantileOutputs(
        non_exceedance=non_exceedance,
        at_expected_parameters=at_expected,
        mean_of_quantiles=(
            fsum(w * q for w, q in zip(weights, quantiles, strict=True)) if mean_undefined_reason is None else None
        ),
        median_of_quantiles=weighted_quantile(quantiles, weights, 0.5),
        predictive_quantile=predictive_quantile(
            particles, non_exceedance, factory, direct_upper_tail=direct_upper_tail
        ),
        probability_limits=(lower_limit, upper_limit),
        mean_log10_quantile=mean_log,
        standard_deviation_log10_quantile=sd_log,
        nonpositive_quantile_weight=nonpositive,
        interval_mass=interval_mass,
        lower_limit_effective_size=tail_effective_sample_size(weights, quantiles, lower_limit, upper=False),
        upper_limit_effective_size=tail_effective_sample_size(weights, quantiles, upper_limit, upper=True),
        mean_undefined_reason=mean_undefined_reason,
        log_undefined_reason=log_undefined_reason,
    )


def probability_outputs(
    particles: ImportanceParticles, discharge: float, factory: KernelFactory, *, interval_mass: float
) -> ProbabilityOutputs:
    """Expected F and AEP use separate stable component CDF and survival calls."""
    if not isfinite(discharge):
        msg = "fixed discharge must be finite"
        raise ValueError(msg)
    _interval_mass(interval_mass)
    kernels = _kernels(particles, factory)
    non_exceedances = tuple(kernel.cdf(discharge) for kernel in kernels)
    survivals = tuple(kernel.sf(discharge) for kernel in kernels)
    if any(not isfinite(p) or not 0.0 <= p <= 1.0 for p in (*non_exceedances, *survivals)):
        msg = "posterior component probability is outside [0, 1]"
        raise ValueError(msg)
    expected_f = fsum(w * p for w, p in zip(particles.weights, non_exceedances, strict=True))
    expected_aep = fsum(w * p for w, p in zip(particles.weights, survivals, strict=True))
    periods = tuple(inf if p == 0 else 1.0 / p for p in survivals)
    tail = (1.0 - interval_mass) / 2.0
    return ProbabilityOutputs(
        discharge=discharge,
        expected_non_exceedance=expected_f,
        expected_aep=expected_aep,
        return_period=inf if expected_aep == 0 else 1.0 / expected_aep,
        return_period_limits=(
            weighted_quantile(periods, particles.weights, tail),
            weighted_quantile(periods, particles.weights, 1.0 - tail),
        ),
        interval_mass=interval_mass,
        zero_survival_weight=fsum(w for w, s in zip(particles.weights, survivals, strict=True) if s == 0),
    )
