"""External USGS MGBT 1.0.7 conditional rank-p-value comparator.

The deterministic integral evaluates the pinned mathematical approximation
without adopting its stochastic or silent-failure fallbacks. It does not
identify Flike 5.0.306.0 defaults, perform a native fit or change source records.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite, sqrt
from sys import float_info
from warnings import catch_warnings, simplefilter

from scipy.integrate import IntegrationWarning, quad
from scipy.special import gammaln
from scipy.stats import beta as beta_distribution
from scipy.stats import nct, norm

from .low_flow import MgbDecision, MgbSweepSettings, select_mgb_candidates
from .mgbt_statistic import (
    MgbRankStatistics,
    MgbStatisticConvention,
    calculate_mgb_rank_statistics,
    verify_mgb_cohort,
)

_USGS_1_0_7_COMMIT = "aa4356208f05d6036f3f425ec37c656da345bba0"
_USGS_LOG_FLOOR = sqrt(float_info.epsilon)
_USGS_QUAD_LOWER = sqrt(float_info.epsilon)
_USGS_QUAD_UPPER = 1.0 - _USGS_QUAD_LOWER


class MgbPvalueNumericalError(ValueError):
    """Cannot certify the source-identified conditional p-value numerically."""


@dataclass(frozen=True, slots=True, kw_only=True)
class MgbReferencePvalues:
    """A source-labelled mathematical result, not Flike native qualification."""

    statistics: MgbRankStatistics
    rank_pvalues: tuple[float, ...]
    absolute_error_estimates: tuple[float, ...]
    decision: MgbDecision

    @property
    def source_revision(self) -> str:
        return _USGS_1_0_7_COMMIT


def _positive(name: str, value: float) -> float:
    if not isfinite(value) or value <= 0.0:
        msg = f"conditional MGBT {name} is non-finite or nonpositive"
        raise MgbPvalueNumericalError(msg)
    return value


def conditional_rank_integrand(u: float, n: int, rank: int, eta: float) -> float:
    """Evaluate the pinned Cohn/USGS conditional noncentral-t survival kernel."""
    b = float(beta_distribution.ppf(u, rank, n + 1 - rank))
    if not 0.0 < b < 1.0:
        msg = "Beta order-statistic transform is outside its finite support"
        raise MgbPvalueNumericalError(msg)
    z = float(norm.ppf(b))
    hazard = exp(float(norm.logpdf(z) - norm.logsf(z)))
    _positive("truncated-normal hazard", hazard)

    # First four raw moments of a standard Normal truncated above z.
    m1 = hazard
    m2 = 1.0 + z * hazard
    m3 = 2.0 * m1 + z * z * hazard
    m4 = 3.0 * m2 + z * z * z * hazard
    v = _positive("truncated variance", m2 - m1 * m1)
    central3 = m3 - 3.0 * m2 * m1 + 2.0 * m1**3
    central4 = m4 - 4.0 * m3 * m1 + 6.0 * m2 * m1 * m1 - 3.0 * m1**4

    k = n - rank
    variance_mean = v / k
    variance_s2 = _positive(
        "variance of upper-sample variance",
        (central4 - v * v) / k + 2.0 * v * v / (k * (k - 1)),
    )
    alpha = _positive("gamma shape", v * v / variance_s2)
    scale = _positive("gamma scale", variance_s2 / v)
    expected_s = _positive("expected upper-sample SD", sqrt(scale) * exp(float(gammaln(alpha + 0.5) - gammaln(alpha))))
    covariance_mean_s2 = central3 / sqrt(k * (k - 1))
    covariance_mean_s = covariance_mean_s2 / (2.0 * expected_s)
    variance_s = _positive("upper-sample SD variance", v - expected_s * expected_s)
    regression = covariance_mean_s / variance_s

    conditional_mean = m1 - regression * expected_s
    conditional_variance = _positive(
        "conditional upper-sample mean variance",
        variance_mean - covariance_mean_s * covariance_mean_s / variance_s,
    )
    degrees_freedom = _positive("noncentral-t degrees of freedom", 2.0 * alpha)
    noncentrality = (conditional_mean - z) / sqrt(conditional_variance)
    quantile = -sqrt(v / conditional_variance) * (eta + regression)
    if not isfinite(noncentrality) or not isfinite(quantile):
        msg = "conditional noncentral-t parameters are not finite"
        raise MgbPvalueNumericalError(msg)
    probability = float(nct.sf(quantile, degrees_freedom, noncentrality))
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        msg = "conditional noncentral-t survival probability is invalid"
        raise MgbPvalueNumericalError(msg)
    return probability


def usgs_mgbt_1_0_7_rank_pvalue(
    n: int,
    rank: int,
    eta: float,
    *,
    absolute_error: float = 1e-8,
    relative_error: float = 1e-8,
) -> tuple[float, float]:
    """Compute one conditional p-value and QUADPACK's absolute error estimate.

    The R function uses 1 - pt() plus Monte Carlo on integration failure.
    This independent Python comparison uses the stable survival tail and
    refuses any integration warning, unrepresentable endpoint or failed budget.
    """
    if n < 5 or not 1 <= rank <= n // 2:
        msg = "reference ranks require n >= 5 and 1 <= rank <= floor(n/2)"
        raise ValueError(msg)
    if not isfinite(eta):
        msg = "reference rank statistic must be finite"
        raise ValueError(msg)
    if not isfinite(absolute_error) or not isfinite(relative_error) or absolute_error <= 0.0 or relative_error <= 0.0:
        msg = "integration absolute/relative errors must be finite positive numbers"
        raise ValueError(msg)

    try:
        with catch_warnings():
            simplefilter("error", IntegrationWarning)
            simplefilter("error", RuntimeWarning)
            probability, error = quad(
                conditional_rank_integrand,
                _USGS_QUAD_LOWER,
                _USGS_QUAD_UPPER,
                args=(n, rank, eta),
                epsabs=absolute_error,
                epsrel=relative_error,
                limit=200,
            )
    except (ArithmeticError, IntegrationWarning, MgbPvalueNumericalError, RuntimeWarning, ValueError) as exc:
        msg = f"conditional MGBT rank {rank} integration failed"
        raise MgbPvalueNumericalError(msg) from exc
    # The source integral omits both endpoint tails. If the integrand
    # saturates at one, QUADPACK can report the interval *width* as a
    # deceptively precise p-value. Likewise very small p-values need a
    # meaningful relative error before being used for classification.
    interval_width = _USGS_QUAD_UPPER - _USGS_QUAD_LOWER
    if not isfinite(probability) or not 0.0 < probability < interval_width - 1e-12:
        msg = f"conditional MGBT rank {rank} probability saturated or invalid"
        raise MgbPvalueNumericalError(msg)
    if (
        not isfinite(error)
        or error < 0
        or error > max(absolute_error, relative_error * abs(probability))
        or (probability < 1e-8 and error > 0.05 * probability)
    ):
        msg = f"conditional MGBT rank {rank} integration error exceeds its tolerance"
        raise MgbPvalueNumericalError(msg)
    return probability, error


def calculate_usgs_mgbt_1_0_7(
    flows: Sequence[float],
    *,
    discharge_unit: str,
    candidate_count: int | None = None,
) -> MgbReferencePvalues:
    """Compute pinned USGS rank p-values and reuse the existing safe three sweeps."""
    convention = MgbStatisticConvention(
        source=f"CRAN MGBT 1.0.7 MGBT17c {_USGS_1_0_7_COMMIT} (external comparator only)",
        discharge_unit=discharge_unit,
        log_floor_discharge=_USGS_LOG_FLOOR,
    )
    statistics = calculate_mgb_rank_statistics(flows, convention=convention, candidate_count=candidate_count)
    computed = tuple(
        usgs_mgbt_1_0_7_rank_pvalue(len(statistics.original_flows), rank, eta)
        for rank, eta in enumerate(statistics.rank_omegas, start=1)
    )
    settings = MgbSweepSettings(
        source_variant="USGS MGBT 1.0.7 MGBT17c (comparison only)",
        alpha_out=0.005,
        alpha_in=0.0,
        alpha_zero_in=0.10,
        candidate_count=statistics.candidate_count,
    )
    decision = select_mgb_candidates(
        statistics.original_flows,
        tuple(value for value, _ in computed),
        settings=settings,
        pvalue_provenance=convention.source,
    )
    verify_mgb_cohort(statistics, decision)
    return MgbReferencePvalues(
        statistics=statistics,
        rank_pvalues=tuple(value for value, _ in computed),
        absolute_error_estimates=tuple(error for _, error in computed),
        decision=decision,
    )
