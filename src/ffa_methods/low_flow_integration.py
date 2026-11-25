"""Source-safe ARR-style low-outlier censor replacement as a derived view only.

This adapter does not change the WP07 source observations or emit a second
independent synthetic-year count group into a likelihood. It is not a verified
Flike file3.10 flag interpretation, zero model, or enabled fitting mode.
"""

from dataclasses import dataclass, replace
from decimal import Decimal
from math import isfinite

from .contracts import AnnualObservation, ExposureKind, ObservationKind, ObservationSeries
from .legacy import (
    LegacyInputError,
    LegacyProject,
    MappedAnnualObservation,
    MappedAnnualSeries,
    map_annual_series,
    parse_fld,
)
from .legacy import ObservationKind as LegacyObservationKind
from .low_flow import (
    ArrMgbCensoringProposal,
    audit_mgb_annual_candidates,
    propose_arr_mgb_censoring,
    select_mgb_candidates,
)


class PairedCensorEvidenceError(ValueError):
    """The source-to-approval-to-censor linkage cannot be proved."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PairedCensorLink:
    """One source-identified annual measurement represented by one censor term."""

    year: int
    original_value: float
    original_source_flag: str | None
    upper_bound: float


@dataclass(frozen=True, slots=True, kw_only=True)
class PairedCensorLikelihoodView:
    """Audited mathematical view, not an on-disk Flike editor transformation.

    The source and its separate ARR-style synthetic-year group remain intact
    in the proposal; the likelihood view contains only real-year annual rows.
    """

    proposal: ArrMgbCensoringProposal
    likelihood_series: ObservationSeries
    links: tuple[PairedCensorLink, ...]
    approval_source: str
    verified_flag_source: str | None

    @property
    def original_series(self) -> ObservationSeries:
        return self.proposal.audit.original_series


def _check_existing_exposure(series: ObservationSeries) -> None:
    """Reject independent groups overlapping each other or any annual row."""
    actual_years = {row.year for row in series.annual}
    if len(actual_years) != len(series.annual):
        msg = "duplicate original annual years prevent pairing"
        raise PairedCensorEvidenceError(msg)
    intervals: list[tuple[int, int]] = []
    for group in series.groups:
        start, end = group.start_year, group.end_year
        if start is None or end is None or start <= 0 or end < start:
            msg = "pre-existing censor groups require real, bounded calendar years"
            raise PairedCensorEvidenceError(msg)
        if group.exposure == ExposureKind.UNRESOLVED:
            msg = "pre-existing group exposure classification is unresolved"
            raise PairedCensorEvidenceError(msg)
        if group.count_above is None or group.count_at_or_below is None:
            msg = "pre-existing group counts are incomplete"
            raise PairedCensorEvidenceError(msg)
        if group.count_above + group.count_at_or_below != end - start + 1:
            msg = "pre-existing group counts disagree with its real-year exposure"
            raise PairedCensorEvidenceError(msg)
        if any(start <= year <= end for year in actual_years):
            msg = "pre-existing censor group overlaps an original annual record"
            raise PairedCensorEvidenceError(msg)
        if any(start <= previous_end and previous_start <= end for previous_start, previous_end in intervals):
            msg = "pre-existing censor groups overlap each other"
            raise PairedCensorEvidenceError(msg)
        intervals.append((start, end))


def _recheck_proposal(proposal: ArrMgbCensoringProposal) -> None:
    """Re-derive source indices, selection and ARR editor metadata before use."""
    audit = proposal.audit
    old = audit.decision
    try:
        confirmed = select_mgb_candidates(
            old.original_flows,
            old.rank_pvalues,
            settings=old.settings,
            pvalue_provenance=old.pvalue_provenance,
        )
        if confirmed != old:
            msg = "supplied MGBT decision disagrees with recomputed source ranks"
            raise PairedCensorEvidenceError(msg)
        original_audit = audit_mgb_annual_candidates(
            audit.original_series,
            screened_years=audit.screened_years,
            decision=old,
        )
        if original_audit != audit:
            msg = "candidate year mapping differs from original annual evidence"
            raise PairedCensorEvidenceError(msg)
        group = proposal.censor_group
        if group.start_year is None:
            msg = "ARR editor placeholder start year is required"
            raise PairedCensorEvidenceError(msg)
        recreated = propose_arr_mgb_censoring(
            audit,
            censor_threshold=group.threshold,
            threshold_source=proposal.threshold_source,
            synthetic_start_year=group.start_year,
        )
        if recreated != proposal:
            msg = "ARR replacement group metadata do not match source selection"
            raise PairedCensorEvidenceError(msg)
    except ValueError as exc:
        if isinstance(exc, PairedCensorEvidenceError):
            raise
        msg = "invalid or unverified ARR censoring proposal"
        raise PairedCensorEvidenceError(msg) from exc


def paired_censor_likelihood_view(
    proposal: ArrMgbCensoringProposal,
    *,
    approve: bool = False,
    approval_source: str,
    verified_exact_flags: frozenset[str] = frozenset(),
    verified_flag_source: str | None = None,
) -> PairedCensorLikelihoodView:
    """Prepare one genuine annual left-censor per approved PILF year.

    With each selected original year explicitly known and the separate ARR
    group reporting only k at/below events, k independent left-censor terms
    have the same mathematical contribution F(h)**k. The negative placeholder
    group must NOT also be passed to a generic historical-group likelihood.
    """
    if not approve or not approval_source.strip():
        msg = "explicit source-identified approval is required"
        raise PairedCensorEvidenceError(msg)
    _recheck_proposal(proposal)
    source = proposal.audit.original_series
    _check_existing_exposure(source)
    chosen = set(proposal.excluded_years)
    if not chosen or len(chosen) != proposal.censor_group.count_at_or_below:
        msg = "approved group must represent exactly the selected original years"
        raise PairedCensorEvidenceError(msg)
    if proposal.censor_group.count_above != 0:
        msg = "paired replacement supports only zero above-threshold group count"
        raise PairedCensorEvidenceError(msg)

    links: list[PairedCensorLink] = []
    annual: list[AnnualObservation] = []
    for row in source.annual:
        if row.year not in chosen:
            annual.append(row)
            continue
        if (
            row.exposure != ExposureKind.SYSTEMATIC
            or row.kind != ObservationKind.EXACT
            or row.value is None
            or not isfinite(row.value)
            or row.lower_bound is not None
            or row.upper_bound is not None
            or row.value >= proposal.censor_group.threshold
        ):
            msg = "selected year lacks an eligible exact systematic flow below threshold"
            raise PairedCensorEvidenceError(msg)
        if row.source_flag is not None and (
            row.source_flag not in verified_exact_flags or not verified_flag_source or not verified_flag_source.strip()
        ):
            msg = "unverified original project flag prevents likelihood conversion"
            raise PairedCensorEvidenceError(msg)
        annual.append(
            replace(
                row,
                kind=ObservationKind.LEFT_CENSORED,
                upper_bound=proposal.censor_group.threshold,
            )
        )
        links.append(
            PairedCensorLink(
                year=row.year,
                original_value=row.value,
                original_source_flag=row.source_flag,
                upper_bound=proposal.censor_group.threshold,
            )
        )

    if {link.year for link in links} != chosen:
        msg = "replacement linkage is missing selected original years"
        raise PairedCensorEvidenceError(msg)

    derived = replace(source, annual=tuple(annual))
    if derived.groups != source.groups or derived.discharge_unit != source.discharge_unit:
        msg = "derived likelihood view altered unrelated source evidence"
        raise PairedCensorEvidenceError(msg)
    return PairedCensorLikelihoodView(
        proposal=proposal,
        likelihood_series=derived,
        links=tuple(links),
        approval_source=approval_source,
        verified_flag_source=verified_flag_source,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class PairedMappedCensorLikelihoodView:
    """A WP02 Decimal/source-row custody wrapper around the mathematical view.

    This does not override MappedAnnualSeries.ready_for_likelihood, which must
    remain false until independently qualified native Flike controls exist.
    """

    mapped_source: MappedAnnualSeries
    paired: PairedCensorLikelihoodView
    selected_source_records: tuple[MappedAnnualObservation, ...]

    @property
    def likelihood_series(self) -> ObservationSeries:
        return self.paired.likelihood_series


def _same_decimal_repr(original: Decimal | None, mapped: Decimal | None) -> bool:
    """Reject source precision changes even when binary64 values are identical."""
    if original is None or mapped is None:
        return original is None and mapped is None
    return original.as_tuple() == mapped.as_tuple()


def _check_file310_metadata(
    mapping: MappedAnnualSeries,
    records: tuple[MappedAnnualObservation, ...],
    original_project: LegacyProject,
    source_sha256: str,
) -> None:
    """Require exact source hash, units, version and record count."""
    if (
        original_project.sha256 != source_sha256
        or original_project.units != mapping.series.discharge_unit
        or original_project.file_version != mapping.series.source_file_version
    ):
        msg = "original file3.10 source identity, unit or version differs from mapped evidence"
        raise PairedCensorEvidenceError(msg)
    if len(original_project.observations) != len(records) or len(original_project.raw_observation_lines) != len(
        records
    ):
        msg = "original file3.10 source row count differs from mapped annual records"
        raise PairedCensorEvidenceError(msg)


def _validate_original_file310_bytes(
    mapping: MappedAnnualSeries,
    records: tuple[MappedAnnualObservation, ...],
    *,
    source_sha256: str,
    original_file310_bytes: bytes,
) -> None:
    """Authenticate a mapped cohort against separately reparsed original bytes."""
    try:
        original_project = parse_fld(original_file310_bytes)
    except (LegacyInputError, UnicodeError) as exc:
        msg = "original file3.10 source bytes are malformed or unavailable"
        raise PairedCensorEvidenceError(msg) from exc
    _check_file310_metadata(mapping, records, original_project, source_sha256)
    for record, original, source_line in zip(
        records, original_project.observations, original_project.raw_observation_lines, strict=True
    ):
        if record.raw_source_line != source_line or record.legacy != original:
            msg = "original file3.10 full-precision Decimal or source row differs from mapped evidence"
            raise PairedCensorEvidenceError(msg)
        if not all(
            _same_decimal_repr(first, second)
            for first, second in (
                (original.value, record.legacy.value),
                (original.lower, record.legacy.lower),
                (original.upper, record.legacy.upper),
            )
        ):
            msg = "original file3.10 Decimal representation differs from mapped source"
            raise PairedCensorEvidenceError(msg)


def paired_censor_from_mapped_source(
    mapping: MappedAnnualSeries,
    proposal: ArrMgbCensoringProposal,
    *,
    original_file310_bytes: bytes,
    approve: bool = False,
    approval_source: str,
    verified_exact_flags: frozenset[str] = frozenset(),
    verified_flag_source: str | None = None,
) -> PairedMappedCensorLikelihoodView:
    """Bridge real WP02 source custody to WP07's approved mathematical view.

    Independently reparse the caller-supplied original file3.10 bytes before
    replaying WP02's Decimal/line/exposure mapping. Exact source-row identity,
    Decimal representation and independently recomputed SHA-256 must agree.
    This deliberately does not accept an unverified manually assembled mapping
    or a different CSV source format. Native Flike likelihood remains unqualified.
    """
    if mapping.series != proposal.audit.original_series:
        msg = "WP02 mapped annual series does not match the proposed original cohort"
        raise PairedCensorEvidenceError(msg)
    records = mapping.source_records
    if not records or len(records) != len(mapping.series.annual):
        msg = "WP02 source record custody does not cover every original annual row"
        raise PairedCensorEvidenceError(msg)
    sha = {record.source_sha256 for record in records}
    if len(sha) != 1:
        msg = "WP02 annual source rows have inconsistent source SHA-256 identifiers"
        raise PairedCensorEvidenceError(msg)
    original_lines = {record.legacy.year: record.raw_source_line for record in records}
    if len(original_lines) != len(records):
        msg = "WP02 source records contain repeated years"
        raise PairedCensorEvidenceError(msg)

    # Recheck source values independently of the mapper replay. In particular,
    # two different Decimals may convert to the very same binary64 float.
    _validate_original_file310_bytes(
        mapping,
        records,
        source_sha256=next(iter(sha)),
        original_file310_bytes=original_file310_bytes,
    )

    try:
        verified = map_annual_series(
            mapping.source_periods,
            tuple(record.legacy for record in records),
            (),
            discharge_unit=mapping.series.discharge_unit,
            source_lines_by_year=original_lines,
            source_sha256=next(iter(sha)),
            source_file_version=mapping.series.source_file_version,
            year_definition=mapping.series.year_definition,
        )
    except LegacyInputError as exc:
        msg = "WP02 source evidence cannot be re-admitted without unresolved assumptions"
        raise PairedCensorEvidenceError(msg) from exc
    if verified != mapping:
        msg = "WP02 source values, Decimal custody, or exposure mapping have been altered"
        raise PairedCensorEvidenceError(msg)

    selected_years = set(proposal.excluded_years)
    selected_records = tuple(record for record in records if record.annual.year in selected_years)
    if len(selected_records) != len(selected_years) or any(
        record.legacy.kind not in {LegacyObservationKind.EXACT, LegacyObservationKind.VERIFIED_TRUE_ZERO}
        for record in selected_records
    ):
        msg = "selected WP02 source years lack verified exact Decimal records"
        raise PairedCensorEvidenceError(msg)
    paired = paired_censor_likelihood_view(
        proposal,
        approve=approve,
        approval_source=approval_source,
        verified_exact_flags=verified_exact_flags,
        verified_flag_source=verified_flag_source,
    )
    if {record.annual.year for record in selected_records} != {link.year for link in paired.links}:
        msg = "selected WP02 Decimal records do not match approved censor links"
        raise PairedCensorEvidenceError(msg)
    return PairedMappedCensorLikelihoodView(
        mapped_source=mapping,
        paired=paired,
        selected_source_records=selected_records,
    )
