"""Conservative WP02 mapping from audited legacy evidence to WP01 annual contracts.

Mapping is not likelihood admission. It does not interpret file3.10 controls,
project flag codes, reported zeroes, historical count semantics or rating options.
Both the original Decimal record and its exact source row remain accessible.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite

from ..contracts import AnnualObservation, ObservationSeries
from ..contracts import ExposureKind as ContractExposureKind
from ..contracts import ObservationKind as ContractObservationKind
from .models import (
    CensorGroup,
    ExposureAudit,
    ExposurePeriod,
    LegacyInputError,
    Observation,
    ObservationKind,
    audit_exposure,
)

_KIND_MAP: dict[ObservationKind, ContractObservationKind] = {
    ObservationKind.EXACT: ContractObservationKind.EXACT,
    ObservationKind.VERIFIED_TRUE_ZERO: ContractObservationKind.EXACT,
    ObservationKind.MISSING: ContractObservationKind.MISSING,
    ObservationKind.EXCLUDED: ContractObservationKind.EXCLUDED,
    ObservationKind.LEFT_CENSORED: ContractObservationKind.LEFT_CENSORED,
    ObservationKind.RIGHT_CENSORED: ContractObservationKind.RIGHT_CENSORED,
    ObservationKind.INTERVAL_CENSORED: ContractObservationKind.INTERVAL_CENSORED,
}

_UNRESOLVED_CONTROLS = (
    "file3.10 controls, prior defaults and rating-error settings are unverified",
    "native likelihood rules and candidate project/report run correspondence are unverified",
)


@dataclass(frozen=True, slots=True)
class MappedAnnualObservation:
    """An annual contract record with its Decimal evidence and original line."""

    annual: AnnualObservation
    legacy: Observation
    raw_source_line: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class MappedAnnualSeries:
    """Exposure-audited records. The remaining native controls block likelihood use."""

    series: ObservationSeries
    source_records: tuple[MappedAnnualObservation, ...]
    exposure_audit: ExposureAudit
    source_periods: tuple[ExposurePeriod, ...]
    unresolved_controls: tuple[str, ...] = _UNRESOLVED_CONTROLS

    @property
    def ready_for_likelihood(self) -> bool:
        """An evidence mapping does not verify unknown Flike controls or priors."""
        return False


def _finite_float(number: Decimal | None, *, field: str) -> float | None:
    if number is None:
        return None
    try:
        converted = float(number)
    except (ValueError, OverflowError) as exc:
        msg = f"Cannot represent {field} as a float without a finite value"
        raise LegacyInputError(msg) from exc
    if not isfinite(converted) or (number != 0 and converted == 0):
        msg = f"Cannot represent {field} as a finite nonzero float"
        raise LegacyInputError(msg)
    return converted


def map_annual_observation(
    observation: Observation,
    *,
    exposure: ContractExposureKind,
    raw_source_line: str,
    source_sha256: str,
) -> MappedAnnualObservation:
    """Map only explicit kinds, retaining unmodified original evidence.

    The input file hash must be supplied by the caller who read the source
    bytes; this function validates its form but does not rehash those bytes.
    Neither a zero printed in an AMS CSV nor a nonzero project flag qualifies.
    """
    if not observation.source.strip() or not raw_source_line.strip():
        msg = "Observation mapping requires source provenance and the original source line"
        raise LegacyInputError(msg)
    if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        msg = "Observation mapping requires a lowercase SHA-256 source digest"
        raise LegacyInputError(msg)
    if exposure not in {ContractExposureKind.SYSTEMATIC, ContractExposureKind.HISTORICAL}:
        msg = "Observation mapping requires explicitly classified exposure"
        raise LegacyInputError(msg)
    if observation.raw_flag not in {None, 0}:
        msg = f"Cannot map unresolved project flag {observation.raw_flag} in {observation.year}"
        raise LegacyInputError(msg)
    mapped_kind = _KIND_MAP.get(observation.kind)
    if mapped_kind is None:
        msg = f"Cannot map unresolved observation kind {observation.kind} in {observation.year}"
        raise LegacyInputError(msg)
    value = _finite_float(observation.value, field="discharge")
    lower = _finite_float(observation.lower, field="lower censor bound")
    upper = _finite_float(observation.upper, field="upper censor bound")
    if (
        observation.kind == ObservationKind.INTERVAL_CENSORED
        and lower is not None
        and upper is not None
        and lower >= upper
    ):
        msg = f"Float conversion collapsed censor interval for year {observation.year}"
        raise LegacyInputError(msg)
    annual = AnnualObservation(
        year=observation.year,
        kind=mapped_kind,
        exposure=exposure,
        value=value,
        lower_bound=lower,
        upper_bound=upper,
        source_flag=str(observation.raw_flag) if observation.raw_flag is not None else None,
    )
    return MappedAnnualObservation(annual, observation, raw_source_line, source_sha256)


def map_annual_series(
    periods: tuple[ExposurePeriod, ...],
    observations: tuple[Observation, ...],
    groups: tuple[CensorGroup, ...],
    *,
    discharge_unit: str,
    source_lines_by_year: dict[int, str],
    source_sha256: str,
    source_file_version: str | None = None,
    year_definition: str | None = None,
) -> MappedAnnualSeries:
    """Produce an evidence-only annual series; fail closed on unknown exposure.

    Grouped counts are not converted into annual rows. The shared contracts
    cannot preserve explicit membership of every grouped year, so a grouping
    requires a separate agreed adapter and native controls before inference.
    """
    audit = audit_exposure(periods, observations, groups)
    if groups:
        msg = "Grouped censor records cannot be losslessly mapped to annual records"
        raise LegacyInputError(msg)
    if not audit.ready_for_likelihood:
        msg = f"Unresolved exposure blocks annual mapping: {audit.unresolved_group_ids}"
        raise LegacyInputError(msg)
    observation_years = {observation.year for observation in observations}
    if set(source_lines_by_year) != observation_years:
        msg = "Source row provenance must identify exactly the mapped observation years"
        raise LegacyInputError(msg)
    kind_by_year = {year: ContractExposureKind(period.kind.value) for period in periods for year in period.years}
    mapped = tuple(
        map_annual_observation(
            observation,
            exposure=kind_by_year[observation.year],
            raw_source_line=source_lines_by_year[observation.year],
            source_sha256=source_sha256,
        )
        for observation in observations
    )
    return MappedAnnualSeries(
        series=ObservationSeries(
            discharge_unit=discharge_unit,
            annual=tuple(item.annual for item in mapped),
            groups=(),
            zero_threshold=None,
            source_file_version=source_file_version,
            year_definition=year_definition,
        ),
        source_records=mapped,
        exposure_audit=audit,
        source_periods=periods,
    )
