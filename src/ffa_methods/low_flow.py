"""Evidence-preserving MGBT decisions and legacy threshold-coordinate conversions.

This module does not compute MGBT p-values or assert that the USGS settings are
those of TUFLOW/BMT Flike 5.0.306.0. Screening is advisory: original observations,
excluded records and historical count groups are never rewritten into each other.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import exp, isfinite, log, log1p
from typing import Literal

from .contracts import CensorGroup, ExposureKind, ObservationKind, ObservationSeries


@dataclass(frozen=True, slots=True, kw_only=True)
class MgbSweepSettings:
    """Explicit comparison settings, not inferred Flike defaults."""

    source_variant: str
    alpha_out: float
    alpha_in: float
    alpha_zero_in: float
    candidate_count: int | None = None

    def __post_init__(self) -> None:
        if not self.source_variant.strip():
            msg = "the source variant must be identified"
            raise ValueError(msg)
        if not 0.0 < self.alpha_out < 1.0:
            msg = "alpha_out must lie strictly between 0 and 1"
            raise ValueError(msg)
        if not 0.0 <= self.alpha_in < 1.0:
            msg = "alpha_in must lie in [0, 1)"
            raise ValueError(msg)
        if not 0.0 < self.alpha_zero_in < 1.0:
            msg = "alpha_zero_in must lie strictly between 0 and 1"
            raise ValueError(msg)
        if self.candidate_count is not None and self.candidate_count < 1:
            msg = "candidate_count must be positive"
            raise ValueError(msg)

    @classmethod
    def usgs_mgbt17c_1_0_7(cls) -> MgbSweepSettings:
        """Published USGS MGBT17c comparator settings, not a Flike profile."""
        return cls(
            source_variant="USGS MGBT 1.0.7 MGBT17c (comparison only)",
            alpha_out=0.005,
            alpha_in=0.0,
            alpha_zero_in=0.10,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MgbDecision:
    """Potential low-outlier decision with original input positions retained.

    Threshold is the first *retained* order statistic, not necessarily the
    reported Flike censor-group threshold. It is None if no low outliers were
    selected. Never turn this boundary into a grouped count automatically.
    """

    settings: MgbSweepSettings
    pvalue_provenance: str
    original_flows: tuple[float, ...]
    ordered_positions: tuple[int, ...]
    rank_pvalues: tuple[float, ...]
    out_sweep_rank: int
    in_sweep_rank: int
    zero_in_sweep_rank: int
    low_outlier_positions: tuple[int, ...]
    threshold: float | None

    @property
    def low_outlier_count(self) -> int:
        return len(self.low_outlier_positions)


def _validate_mgb_inputs(
    values: tuple[float, ...],
    probabilities: tuple[float, ...],
    settings: MgbSweepSettings,
    provenance: str,
) -> int:
    if len(values) < 5:
        msg = "MGBT comparator requires at least five measured flows"
        raise ValueError(msg)
    if any(not isfinite(value) or value < 0 for value in values):
        msg = "screening flows must be finite, nonnegative measured values"
        raise ValueError(msg)
    if not provenance.strip():
        msg = "p-value calculation provenance is required"
        raise ValueError(msg)
    candidate_count = settings.candidate_count if settings.candidate_count is not None else len(values) // 2
    if candidate_count > len(values) // 2:
        msg = "candidate_count must not exceed floor(n / 2) in this comparator"
        raise ValueError(msg)
    if len(probabilities) != candidate_count:
        msg = "one rank p-value must be supplied for each candidate rank"
        raise ValueError(msg)
    if any(not isfinite(prob) or prob < 0.0 or prob > 1.0 for prob in probabilities):
        msg = "MGBT rank p-values must be finite probabilities in [0, 1]"
        raise ValueError(msg)
    return candidate_count


def _sweep_mgb_rank_probabilities(
    probabilities: tuple[float, ...], settings: MgbSweepSettings, candidate_count: int
) -> tuple[int, int, int]:
    outward = next(
        (rank for rank in range(candidate_count, 0, -1) if probabilities[rank - 1] < settings.alpha_out),
        0,
    )
    # The pinned USGS MGBT 1.0.7 mgbt.sweeper starts both indices at zero.
    # If no p-value meets a stopping condition, a sweep contributes no
    # candidate rank; it must not silently promote the entire lower half.
    inward = 0
    for rank in range(outward + 1, candidate_count + 1):
        if probabilities[rank - 1] >= settings.alpha_in:
            inward = rank - 1
            break
    zero_in = 0
    for rank in range(1, candidate_count + 1):
        if probabilities[rank - 1] >= settings.alpha_zero_in:
            zero_in = rank - 1
            break
    return outward, inward, zero_in


def select_mgb_candidates(
    flows: Sequence[float],
    rank_pvalues: Sequence[float],
    *,
    settings: MgbSweepSettings,
    pvalue_provenance: str,
) -> MgbDecision:
    """Apply a published three-sweep decision to externally calculated p-values.

    rank_pvalues[r - 1] must be the one-sided MGBT p-value for the r-th
    smallest original flow, as defined by the specified source/variant. This
    routine does not calculate or validate the underlying p-value statistic.
    It never applies a logarithmic offset to zero flows or changes observations.
    """
    values = tuple(flows)
    probabilities = tuple(rank_pvalues)
    candidate_count = _validate_mgb_inputs(values, probabilities, settings, pvalue_provenance)
    ordered = tuple(sorted(range(len(values)), key=lambda index: (values[index], index)))
    outward, inward, zero_in = _sweep_mgb_rank_probabilities(probabilities, settings, candidate_count)
    n_low = max(outward, inward, zero_in)
    if n_low and values[ordered[n_low - 1]] == values[ordered[n_low]]:
        msg = "low-outlier cutoff divides tied flows; source tie resolution is not qualified"
        raise ValueError(msg)
    return MgbDecision(
        settings=settings,
        pvalue_provenance=pvalue_provenance,
        original_flows=values,
        ordered_positions=ordered,
        rank_pvalues=probabilities,
        out_sweep_rank=outward,
        in_sweep_rank=inward,
        zero_in_sweep_rank=zero_in,
        low_outlier_positions=ordered[:n_low],
        threshold=values[ordered[n_low]] if n_low else None,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class MgbAnnualAudit:
    """Non-mutating annotations, not a new observation/censor likelihood."""

    original_series: ObservationSeries
    screened_years: tuple[int, ...]
    candidate_low_outlier_years: tuple[int, ...]
    existing_excluded_years: tuple[int, ...]
    decision: MgbDecision


def _check_group_exposure_disjoint(series: ObservationSeries, years: tuple[int, ...]) -> None:
    for group in series.groups:
        if group.start_year is None or group.end_year is None:
            msg = "unbounded censor-group exposure cannot be certified disjoint from screened years"
            raise ValueError(msg)
        if any(group.start_year <= year <= group.end_year for year in years):
            msg = "screening input overlaps a grouped censor exposure period"
            raise ValueError(msg)


def audit_mgb_annual_candidates(
    series: ObservationSeries,
    *,
    screened_years: Sequence[int],
    decision: MgbDecision,
) -> MgbAnnualAudit:
    """Match the decision to explicitly selected systematic exact observations.

    Excluded, missing, historical, already censored, grouped and at/below-zero
    observations are not silently reinterpreted as exact MGBT inputs. The source
    series, original flow values, flags and group exposures remain unchanged.
    """
    by_year = {row.year: row for row in series.annual}
    if len(by_year) != len(series.annual):
        msg = "duplicate annual year prevents an auditable screening mapping"
        raise ValueError(msg)
    years = tuple(screened_years)
    if len(years) != len(set(years)) or not years:
        msg = "screened years must be a nonempty set of distinct annual years"
        raise ValueError(msg)
    measured: list[float] = []
    for year in years:
        row = by_year.get(year)
        if row is None:
            msg = f"screened year {year} is missing from the original annual series"
            raise ValueError(msg)
        if row.exposure != ExposureKind.SYSTEMATIC or row.kind != ObservationKind.EXACT or row.value is None:
            msg = f"year {year} is not an exact systematic measurement"
            raise ValueError(msg)
        measured.append(row.value)
    if tuple(measured) != decision.original_flows:
        msg = "screening flow values/order differ from the original annual observations"
        raise ValueError(msg)

    _check_group_exposure_disjoint(series, years)

    return MgbAnnualAudit(
        original_series=series,
        screened_years=years,
        candidate_low_outlier_years=tuple(years[position] for position in decision.low_outlier_positions),
        existing_excluded_years=tuple(row.year for row in series.annual if row.kind == ObservationKind.EXCLUDED),
        decision=decision,
    )


def probability_at_or_below_threshold(logit_probability: float) -> float:
    """Convert the observed logit coordinate without asserting a zero atom."""
    if not isfinite(logit_probability):
        msg = "the reported logit coordinate must be finite"
        raise ValueError(msg)
    if logit_probability >= 0.0:
        return 1.0 / (1.0 + exp(-logit_probability))
    scaled = exp(logit_probability)
    return scaled / (1.0 + scaled)


def logit_at_or_below_threshold(probability: float) -> float:
    """Inverse for an interior probability; p=0 or 1 needs an infinite logit."""
    if not isfinite(probability) or not 0.0 < probability < 1.0:
        msg = "a finite threshold logit requires probability strictly inside (0, 1)"
        raise ValueError(msg)
    return log(probability) - log1p(-probability)


@dataclass(frozen=True, slots=True, kw_only=True)
class ThresholdProbabilityCoordinate:
    """Reported coordinate only, not a fitted zero-inflation likelihood."""

    family: Literal["lognormal", "log_pearson_iii"]
    threshold: float
    logit_probability: float
    source: str

    def __post_init__(self) -> None:
        if self.family not in ("lognormal", "log_pearson_iii"):
            msg = "only LN and LP3 threshold coordinates are in WP07 scope"
            raise ValueError(msg)
        if not isfinite(self.threshold):
            msg = "threshold must be finite and in the original flow units"
            raise ValueError(msg)
        if not self.source.strip():
            msg = "the coordinate provenance is required"
            raise ValueError(msg)
        probability_at_or_below_threshold(self.logit_probability)

    @property
    def probability(self) -> float:
        return probability_at_or_below_threshold(self.logit_probability)


@dataclass(frozen=True, slots=True, kw_only=True)
class ArrMgbCensoringProposal:
    """Explicit ARR Example 6-style pairing: exclusion and grouped evidence.

    Synthetic group dates are *placeholders*, not historical years. The raw
    original annual series remains available. No native Flike threshold rule
    or source-flag code is inferred from this proposal.
    """

    audit: MgbAnnualAudit
    censor_group: CensorGroup
    threshold_source: str

    @property
    def excluded_years(self) -> tuple[int, ...]:
        return self.audit.candidate_low_outlier_years


def propose_arr_mgb_censoring(
    audit: MgbAnnualAudit,
    *,
    censor_threshold: float,
    threshold_source: str,
    synthetic_start_year: int,
) -> ArrMgbCensoringProposal:
    """Propose the two separate actions documented in ARR Book 3 §2.8.6.4.

    A manually *evidenced* threshold is required because the retained-flow
    cutoff is not the value actually entered into the sample Flike group.
    """
    decision = audit.decision
    if not decision.low_outlier_positions or decision.threshold is None:
        msg = "cannot propose a censor group without selected low-outlier years"
        raise ValueError(msg)
    if not isfinite(censor_threshold) or not threshold_source.strip():
        msg = "a finite source-identified censor threshold is required"
        raise ValueError(msg)
    low_max = max(decision.original_flows[index] for index in decision.low_outlier_positions)
    retained_min = decision.threshold
    if not low_max < censor_threshold < retained_min:
        msg = "the explicit censor threshold must lie strictly between selected and retained flows"
        raise ValueError(msg)
    low_count = decision.low_outlier_count
    synthetic_end_year = synthetic_start_year + low_count - 1
    if synthetic_start_year >= 0 or synthetic_end_year >= 0:
        msg = "dummy year range must remain entirely before year zero"
        raise ValueError(msg)
    for group in audit.original_series.groups:
        if group.start_year is None or group.end_year is None:
            msg = "cannot qualify overlap with an unbounded existing censor group"
            raise ValueError(msg)
        if synthetic_start_year <= group.end_year and group.start_year <= synthetic_end_year:
            msg = "dummy censor-group year range overlaps an existing group"
            raise ValueError(msg)
    group = CensorGroup(
        threshold=censor_threshold,
        exposure=ExposureKind.UNRESOLVED,
        count_at_or_below=low_count,
        count_above=0,
        start_year=synthetic_start_year,
        end_year=synthetic_end_year,
        source_record=f"ARR-style MGB proposal; threshold evidence: {threshold_source}",
    )
    return ArrMgbCensoringProposal(audit=audit, censor_group=group, threshold_source=threshold_source)


def apply_approved_arr_mgb_censoring(proposal: ArrMgbCensoringProposal, *, approve: bool = False) -> ObservationSeries:
    """Materialize a separately approved proposal without deleting source data.

    The resulting excluded annual records retain their original value, year,
    source flag and exposure. Group placeholder years do not represent a second
    measured historical time span; WP05 must qualify downstream likelihood use.
    """
    if not approve:
        msg = "ARR-style MGB censoring requires explicit approval"
        raise ValueError(msg)
    original = proposal.audit.original_series
    excluded = set(proposal.excluded_years)
    annual = tuple(
        replace(row, kind=ObservationKind.EXCLUDED) if row.year in excluded else row for row in original.annual
    )
    return replace(original, annual=annual, groups=(*original.groups, proposal.censor_group))
