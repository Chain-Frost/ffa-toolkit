"""Independent generalized Grubbs-Beck rank-statistic evidence, not Flike MGBT inference.

Implements the directly specified order-statistic W(r) only. The one-sided
Cohn/GGBCRITP p-value quadrature, low-outlier acceptance and conversion to a
legacy Flike likelihood are deliberately separate operations.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite, log10, sqrt

from .low_flow import MgbDecision


@dataclass(frozen=True, slots=True, kw_only=True)
class MgbStatisticConvention:
    """An explicit external source convention, *never* a Flike default.

    A log floor changes the test when low/zero flows are present. It applies
    only to the numerical statistic, leaving the original discharge untouched.
    The absolute floor is expressed in the supplied original discharge unit.
    """

    source: str
    discharge_unit: str
    log_floor_discharge: float

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.discharge_unit.strip():
            msg = "the source and physical discharge units must be identified"
            raise ValueError(msg)
        if not isfinite(self.log_floor_discharge) or self.log_floor_discharge <= 0.0:
            msg = "the explicit source log floor must be a finite positive discharge"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class MgbRankStatistics:
    """Traceable one-based ranks and raw source positions for a fixed cohort."""

    convention: MgbStatisticConvention
    original_flows: tuple[float, ...]
    ordered_positions: tuple[int, ...]
    ordered_log10_flows: tuple[float, ...]
    rank_omegas: tuple[float, ...]

    @property
    def candidate_count(self) -> int:
        return len(self.rank_omegas)


def calculate_mgb_rank_statistics(
    flows: Sequence[float], *, convention: MgbStatisticConvention, candidate_count: int | None = None
) -> MgbRankStatistics:
    r"""Compute W_r = [z_r - mean(z_(r+1)..z_n)] / sd_sample(z_(r+1)..z_n).

    Ranks use the ascending *original measured flows*, not arbitrary inputs
    already truncated to a set of candidate values. The upper sample excludes
    the candidate and includes every higher-ranked measurement. Its variance
    denominator is (n-r-1), following the sample-variance definition.

    No p-value or classification follows from W alone: the conditional
    distribution of the rth normal order statistic is essential.
    """
    values = tuple(flows)
    n = len(values)
    if n < 5:
        msg = "at least five measured flows are needed for the rank-statistic comparison"
        raise ValueError(msg)
    if any(not isfinite(flow) or flow < 0.0 for flow in values):
        msg = "the original screening series must contain finite nonnegative discharges"
        raise ValueError(msg)
    maximum = n // 2
    k = maximum if candidate_count is None else candidate_count
    if k < 1 or k > maximum:
        msg = "candidate_count must be in [1, floor(n/2)]"
        raise ValueError(msg)

    ordered = tuple(sorted(range(n), key=lambda position: (values[position], position)))
    transformed = tuple(log10(max(values[position], convention.log_floor_discharge)) for position in ordered)
    omegas: list[float] = []
    for rank in range(1, k + 1):
        upper = transformed[rank:]
        if all(value == upper[0] for value in upper):
            msg = f"rank {rank} has no positive upper-sample log variance (all values equal)"
            raise ValueError(msg)
        size = len(upper)
        mean = fsum(upper) / size
        variance = fsum((value - mean) ** 2 for value in upper) / (size - 1)
        if not isfinite(variance) or variance <= 0.0:
            msg = f"rank {rank} has no positive upper-sample log variance"
            raise ValueError(msg)
        statistic = (transformed[rank - 1] - mean) / sqrt(variance)
        if not isfinite(statistic):
            msg = f"rank {rank} has a non-finite generalized Grubbs-Beck statistic"
            raise ArithmeticError(msg)
        omegas.append(statistic)

    return MgbRankStatistics(
        convention=convention,
        original_flows=values,
        ordered_positions=ordered,
        ordered_log10_flows=transformed,
        rank_omegas=tuple(omegas),
    )


def verify_mgb_cohort(statistics: MgbRankStatistics, decision: MgbDecision) -> None:
    """Fail closed if supplied p-values/decisions use another data cohort.

    P-value provenance is still externally supplied; checking this alignment
    does *not* independently verify the p-value computation or source variant.
    """
    if (
        statistics.original_flows != decision.original_flows
        or statistics.ordered_positions != decision.ordered_positions
        or len(statistics.rank_omegas) != len(decision.rank_pvalues)
    ):
        msg = "rank statistics and p-value decision do not share the same ordered measurement cohort"
        raise ValueError(msg)
