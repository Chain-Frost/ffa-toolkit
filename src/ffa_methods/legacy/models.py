"""Evidence-preserving observation and exposure records for legacy Flike inputs.

These types do not define the legacy likelihood. They deliberately do not map
unexplained numeric project flags to exclusion or censoring instructions.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class LegacyInputError(ValueError):
    """Malformed or unsupported legacy record; no implicit fallback is allowed."""


class ObservationKind(StrEnum):
    EXACT = "exact"
    REPORTED_ZERO = "reported_zero_unverified"
    VERIFIED_TRUE_ZERO = "verified_true_zero"
    MISSING = "missing"
    EXCLUDED = "excluded"
    LEFT_CENSORED = "left_censored"
    RIGHT_CENSORED = "right_censored"
    INTERVAL_CENSORED = "interval_censored"
    FLAG_UNRESOLVED = "flag_unresolved"


class ExposureKind(StrEnum):
    SYSTEMATIC = "systematic"
    HISTORICAL = "historical"


@dataclass(frozen=True, slots=True)
class Observation:
    year: int
    kind: ObservationKind
    value: Decimal | None
    source: str
    raw_flag: int | None = None
    lower: Decimal | None = None
    upper: Decimal | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.year <= 9999:
            msg = f"Invalid observation year {self.year}"
            raise LegacyInputError(msg)
        for label, number in (("value", self.value), ("lower", self.lower), ("upper", self.upper)):
            if number is not None and (not number.is_finite() or number < 0):
                msg = f"Invalid {label} for year {self.year}"
                raise LegacyInputError(msg)
        self._validate_kind()

    def _validate_kind(self) -> None:
        if self.kind in {ObservationKind.EXACT, ObservationKind.REPORTED_ZERO, ObservationKind.VERIFIED_TRUE_ZERO}:
            if self.value is None or self.lower is not None or self.upper is not None:
                msg = "Exact/zero observations need a value and no censor bounds"
                raise LegacyInputError(msg)
            if self.kind in {ObservationKind.REPORTED_ZERO, ObservationKind.VERIFIED_TRUE_ZERO} and self.value != 0:
                msg = "A zero classification requires a zero-valued observation"
                raise LegacyInputError(msg)
        elif self.kind == ObservationKind.MISSING:
            if self.value is not None or self.lower is not None or self.upper is not None:
                msg = "Missing observations must not contain observed values"
                raise LegacyInputError(msg)
        elif self.kind == ObservationKind.EXCLUDED:
            if not self.reason:
                msg = "An exclusion needs an explicit reason"
                raise LegacyInputError(msg)
        else:
            self._validate_censoring()

    def _validate_censoring(self) -> None:
        if self.kind == ObservationKind.LEFT_CENSORED:
            if self.value is not None or self.lower is not None or self.upper is None:
                msg = "Left-censored observations need only an upper bound"
                raise LegacyInputError(msg)
        elif self.kind == ObservationKind.RIGHT_CENSORED:
            if self.value is not None or self.lower is None or self.upper is not None:
                msg = "Right-censored observations need only a lower bound"
                raise LegacyInputError(msg)
        elif self.kind == ObservationKind.INTERVAL_CENSORED:
            if self.value is not None or self.lower is None or self.upper is None or self.lower >= self.upper:
                msg = "Interval-censored observations require lower < upper"
                raise LegacyInputError(msg)
        elif self.kind == ObservationKind.FLAG_UNRESOLVED and self.raw_flag is None:
            msg = "Unresolved flags must preserve the original numeric flag"
            raise LegacyInputError(msg)


@dataclass(frozen=True, slots=True)
class ExposurePeriod:
    """An explicit set of years; never infer historical non-exceedances from gaps."""

    identifier: str
    kind: ExposureKind
    years: frozenset[int]
    perception_threshold: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.identifier or not self.years or any(not 1 <= year <= 9999 for year in self.years):
            msg = "Exposure periods need a name and explicit valid years"
            raise LegacyInputError(msg)
        if self.perception_threshold is not None and (
            not self.perception_threshold.is_finite() or self.perception_threshold < 0
        ):
            msg = "Invalid perception threshold"
            raise LegacyInputError(msg)


@dataclass(frozen=True, slots=True)
class CensorGroup:
    """Counts are evidence, not additional observations.

    A group without explicit covered years remains available for auditing but
    cannot be admitted into a likelihood until its exposure is reconciled.
    """

    exposure_id: str
    below_count: int
    above_count: int
    threshold: Decimal
    years: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if self.below_count < 0 or self.above_count < 0 or not self.threshold.is_finite() or self.threshold < 0:
            msg = "Invalid censored count or threshold"
            raise LegacyInputError(msg)
        if self.years is not None and len(self.years) != self.below_count + self.above_count:
            msg = "Group membership size does not equal its total count"
            raise LegacyInputError(msg)


@dataclass(frozen=True, slots=True)
class ExposureAudit:
    covered_years: frozenset[int]
    grouped_years: frozenset[int]
    unresolved_group_ids: tuple[str, ...]

    @property
    def ready_for_likelihood(self) -> bool:
        return not self.unresolved_group_ids


def audit_exposure(
    periods: tuple[ExposurePeriod, ...],
    observations: tuple[Observation, ...],
    groups: tuple[CensorGroup, ...],
) -> ExposureAudit:
    """Reject overlapping evidence and unaccounted group allocations.

    Missing observations are not automatically counted as threshold non-events.
    A bare count is retained but blocks likelihood construction; it does not get
    assigned to the remaining years by guesswork.
    """
    period_by_id, covered = _index_exposure(periods)
    by_year = _observation_years(observations, covered)

    grouped: set[int] = set()
    unresolved: list[str] = []
    for group in groups:
        period = period_by_id.get(group.exposure_id)
        if period is None:
            msg = f"Unknown exposure group: {group.exposure_id}"
            raise LegacyInputError(msg)
        if period.kind == ExposureKind.HISTORICAL and period.perception_threshold is None:
            msg = "Historical group requires a declared perception threshold"
            raise LegacyInputError(msg)
        if group.below_count + group.above_count > len(period.years):
            msg = "Censor count exceeds its exposure length"
            raise LegacyInputError(msg)
        if group.years is None:
            unresolved.append(group.exposure_id)
            continue
        if not group.years.issubset(period.years):
            msg = "Censor group includes years outside its exposure"
            raise LegacyInputError(msg)
        if group.years.intersection(by_year) or group.years.intersection(grouped):
            msg = "Censor groups overlap observations or one another"
            raise LegacyInputError(msg)
        grouped.update(group.years)

    _check_unaccounted_exposure(periods, by_year, grouped, unresolved)

    return ExposureAudit(frozenset(covered), frozenset(grouped), tuple(unresolved))


def _index_exposure(periods: tuple[ExposurePeriod, ...]) -> tuple[dict[str, ExposurePeriod], set[int]]:
    period_by_id: dict[str, ExposurePeriod] = {}
    covered: set[int] = set()
    for period in periods:
        if period.identifier in period_by_id:
            msg = f"Repeated exposure identifier: {period.identifier}"
            raise LegacyInputError(msg)
        overlap = covered.intersection(period.years)
        if overlap:
            msg = f"Overlapping systematic/historical exposure years: {sorted(overlap)}"
            raise LegacyInputError(msg)
        covered.update(period.years)
        period_by_id[period.identifier] = period

    return period_by_id, covered


def _observation_years(observations: tuple[Observation, ...], covered: set[int]) -> set[int]:
    by_year: set[int] = set()
    for observation in observations:
        if observation.year in by_year:
            msg = f"Duplicate observation year: {observation.year}"
            raise LegacyInputError(msg)
        by_year.add(observation.year)
        if observation.year not in covered:
            msg = f"Year {observation.year} is outside the declared exposure"
            raise LegacyInputError(msg)

    return by_year


def _check_unaccounted_exposure(
    periods: tuple[ExposurePeriod, ...],
    by_year: set[int],
    grouped: set[int],
    unresolved: list[str],
) -> None:
    # An unrecorded systematic year is not an observed zero, and an unrecorded
    # historical year is not a verified below-threshold event. Both block
    # likelihood admission rather than acquiring a fabricated observation.
    for period in periods:
        if period.kind == ExposureKind.HISTORICAL and period.perception_threshold is None:
            unresolved.append(f"historical threshold unverified: {period.identifier}")
        unaccounted = period.years - by_year - grouped
        if unaccounted and period.identifier not in unresolved:
            unresolved.append(f"unaccounted exposure years {period.identifier}: {sorted(unaccounted)}")
