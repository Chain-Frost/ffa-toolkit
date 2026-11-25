"""Contract tests for source-linked ARR-style MGBT censor replacement.

The derived mathematical view is intentionally separate from the on-disk Flike
editor's exclusion plus synthetic-year count group; neither is a native oracle.
"""

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from math import exp, isclose, log1p

import pytest

from ffa_methods.contracts import AnnualObservation, CensorGroup, ExposureKind, ObservationKind, ObservationSeries
from ffa_methods.legacy import (
    ExposureKind as LegacyExposureKind,
)
from ffa_methods.legacy import (
    ExposurePeriod as LegacyExposurePeriod,
)
from ffa_methods.legacy import LegacyInputError, MappedAnnualSeries, map_annual_series, parse_fld
from ffa_methods.legacy import (
    Observation as LegacyObservation,
)
from ffa_methods.legacy import (
    ObservationKind as LegacyObservationKind,
)
from ffa_methods.likelihood import HistoricalPeriod, LikelihoodEvidenceError, log_likelihood
from ffa_methods.low_flow import (
    ArrMgbCensoringProposal,
    MgbSweepSettings,
    audit_mgb_annual_candidates,
    propose_arr_mgb_censoring,
    select_mgb_candidates,
)
from ffa_methods.low_flow_integration import (
    PairedCensorEvidenceError,
    PairedCensorLikelihoodView,
    PairedMappedCensorLikelihoodView,
    paired_censor_from_mapped_source,
    paired_censor_likelihood_view,
)
from tests.analytic_kernels import ExponentialKernel


def _source_series(*, groups: tuple[CensorGroup, ...] | None = None) -> ObservationSeries:
    extra = (
        CensorGroup(
            threshold=4.0,
            exposure=ExposureKind.HISTORICAL,
            count_at_or_below=2,
            count_above=0,
            start_year=1880,
            end_year=1881,
            source_record="source history, distinct from systematic years",
        ),
    )
    observations = tuple(
        AnnualObservation(
            year=2000 + year,
            kind=ObservationKind.EXACT,
            exposure=ExposureKind.SYSTEMATIC,
            value=float(year),
            source_flag="0",
        )
        for year in range(1, 7)
    )
    return ObservationSeries(
        discharge_unit="m3/s",
        source_file_version="3.10",
        year_definition="explicit synthetic systematic calendar years",
        annual=(
            *observations,
            AnnualObservation(
                year=1999,
                kind=ObservationKind.EXCLUDED,
                exposure=ExposureKind.SYSTEMATIC,
                value=999.0,
                source_flag="1",
            ),
            AnnualObservation(year=1998, kind=ObservationKind.MISSING, exposure=ExposureKind.SYSTEMATIC),
        ),
        groups=extra if groups is None else groups,
        zero_threshold=0.0,
    )


def _proposal(series: ObservationSeries | None = None) -> ArrMgbCensoringProposal:
    source = _source_series() if series is None else series
    decision = select_mgb_candidates(
        (1, 2, 3, 4, 5, 6),
        (0.001, 0.001, 0.3),
        settings=MgbSweepSettings.usgs_mgbt17c_1_0_7(),
        pvalue_provenance="synthetic, externally supplied rank probabilities",
    )
    audit = audit_mgb_annual_candidates(
        source,
        screened_years=(2001, 2002, 2003, 2004, 2005, 2006),
        decision=decision,
    )
    return propose_arr_mgb_censoring(
        audit,
        censor_threshold=2.5,
        threshold_source="explicit synthetic threshold 2.5, not a Flike default",
        synthetic_start_year=-1000,
    )


def _approved(proposal: ArrMgbCensoringProposal | None = None) -> PairedCensorLikelihoodView:
    return paired_censor_likelihood_view(
        _proposal() if proposal is None else proposal,
        approve=True,
        approval_source="documented synthetic analyst approval for this test only",
        verified_exact_flags=frozenset({"0"}),
        verified_flag_source="test's explicitly classified synthetic exact records",
    )


def test_derived_likelihood_view_represents_selected_source_years_once() -> None:
    proposal = _proposal()
    view = _approved(proposal)
    original = proposal.audit.original_series
    assert view.original_series is original
    assert view.likelihood_series is not original
    assert view.likelihood_series.groups == original.groups
    assert proposal.censor_group.exposure == ExposureKind.UNRESOLVED
    assert (proposal.censor_group.start_year, proposal.censor_group.end_year) == (-1000, -999)
    assert all(group.start_year is not None and group.start_year > 0 for group in view.likelihood_series.groups)
    assert [row.year for row in view.likelihood_series.annual] == [row.year for row in original.annual]
    assert [row.kind for row in view.likelihood_series.annual[:2]] == [ObservationKind.LEFT_CENSORED] * 2
    assert [row.kind for row in original.annual[:2]] == [ObservationKind.EXACT] * 2
    assert [row.value for row in view.likelihood_series.annual[:2]] == [1, 2]
    assert [row.upper_bound for row in view.likelihood_series.annual[:2]] == [2.5, 2.5]
    assert [row.source_flag for row in view.likelihood_series.annual[:2]] == ["0", "0"]
    assert [row.kind for row in view.likelihood_series.annual[6:]] == [
        ObservationKind.EXCLUDED,
        ObservationKind.MISSING,
    ]
    assert [link.year for link in view.links] == [2001, 2002]
    assert [link.original_value for link in view.links] == [1, 2]
    assert view.likelihood_series.discharge_unit == original.discharge_unit
    assert view.likelihood_series.source_file_version == original.source_file_version


def test_mathematical_count_replacement_is_k_left_censors_without_dummy_years() -> None:
    view = _approved()
    # For an independent unit-rate exponential law, P(Q <= h) = 1-exp(-h).
    # WP05's per-annual left-censored terms each use logcdf(h), hence F(h)^k.
    h = view.proposal.censor_group.threshold
    k = view.proposal.censor_group.count_at_or_below
    assert k == 2
    individual = sum(
        log1p(-exp(-row.upper_bound))
        for row in view.likelihood_series.annual
        if row.kind == ObservationKind.LEFT_CENSORED and row.upper_bound is not None
    )
    expected_group = k * log1p(-exp(-h))
    assert isclose(individual, expected_group, rel_tol=0.0, abs_tol=1e-15)
    assert len(view.likelihood_series.groups) == 1  # Only pre-existing, real-year source groups.


@pytest.mark.parametrize(
    ("approve", "source"),
    [(False, "synthetic explicit approval"), (True, "  ")],
)
def test_requires_explicit_approval(approve: bool, source: str) -> None:
    with pytest.raises(PairedCensorEvidenceError, match="explicit source-identified approval"):
        paired_censor_likelihood_view(_proposal(), approve=approve, approval_source=source)


def test_unknown_project_flags_fail_without_separate_verification() -> None:
    proposal = _proposal()
    with pytest.raises(PairedCensorEvidenceError, match="unverified original project flag"):
        paired_censor_likelihood_view(proposal, approve=True, approval_source="test")
    with pytest.raises(PairedCensorEvidenceError, match="unverified original project flag"):
        paired_censor_likelihood_view(
            proposal,
            approve=True,
            approval_source="test",
            verified_exact_flags=frozenset({"0"}),
        )
    with pytest.raises(PairedCensorEvidenceError, match="unverified original project flag"):
        paired_censor_likelihood_view(
            proposal,
            approve=True,
            approval_source="test",
            verified_exact_flags=frozenset({"1"}),
            verified_flag_source="incorrect classification",
        )


def test_reject_censor_group_count_or_metadata_tampering() -> None:
    proposal = _proposal()
    incorrect = replace(
        proposal,
        censor_group=replace(proposal.censor_group, count_at_or_below=3),
    )
    with pytest.raises(PairedCensorEvidenceError, match="ARR replacement group metadata"):
        _approved(incorrect)
    altered = replace(proposal, censor_group=replace(proposal.censor_group, count_above=1))
    with pytest.raises(PairedCensorEvidenceError, match="ARR replacement group metadata"):
        _approved(altered)
    altered_threshold = replace(proposal, censor_group=replace(proposal.censor_group, threshold=2.0))
    with pytest.raises(PairedCensorEvidenceError, match="invalid or unverified ARR"):
        _approved(altered_threshold)


def test_reject_selected_year_mapping_tampering() -> None:
    proposal = _proposal()
    invalid_audit = replace(proposal.audit, candidate_low_outlier_years=(2002, 2001))
    with pytest.raises(PairedCensorEvidenceError, match="candidate year mapping differs"):
        _approved(replace(proposal, audit=invalid_audit))


def test_reject_other_group_overlaps_and_unbounded_exposure() -> None:
    original = _source_series()
    group = original.groups[0]
    overlapping_annual_group = replace(group, start_year=1999, end_year=2000)
    with pytest.raises(PairedCensorEvidenceError, match="overlaps an original annual"):
        _approved(_proposal(_source_series(groups=(overlapping_annual_group,))))
    overlapping_groups = (
        group,
        replace(group, start_year=1881, end_year=1882),
    )
    with pytest.raises(PairedCensorEvidenceError, match="overlap each other"):
        _approved(_proposal(_source_series(groups=overlapping_groups)))
    incomplete = replace(group, start_year=None)
    with pytest.raises(ValueError, match="unbounded censor-group"):
        _proposal(_source_series(groups=(incomplete,)))
    wrong_count = replace(group, count_at_or_below=1)
    with pytest.raises(PairedCensorEvidenceError, match="disagree with its real-year"):
        _approved(_proposal(_source_series(groups=(wrong_count,))))


def test_selected_exact_source_order_and_flags_are_preserved() -> None:
    proposal = _proposal()
    view = _approved(proposal)
    assert view.original_series.annual == proposal.audit.original_series.annual
    assert view.original_series.groups == proposal.audit.original_series.groups
    assert all(
        original.year == derived.year
        for original, derived in zip(view.original_series.annual, view.likelihood_series.annual, strict=True)
    )
    assert {link.year for link in view.links} == set(proposal.excluded_years)


def test_wp05_real_year_likelihood_without_double_count() -> None:
    view = _approved()
    result = log_likelihood(ExponentialKernel(), view.likelihood_series)
    # Four retained exact source values 3..6, two source-linked left censors
    # at 2.5, and the existing two-year historical group at 4.0.
    expected = -(3 + 4 + 5 + 6) + 2 * log1p(-exp(-2.5)) + 2 * log1p(-exp(-4.0))
    assert isclose(result.log_value, expected, rel_tol=0.0, abs_tol=1e-12)
    terms = {term.source: term for term in result.terms}
    assert "annual:2001:left_censored" in terms
    assert "annual:2002:left_censored" in terms
    assert "group:0" in terms
    assert "group:1" not in terms  # The synthetic -1000/-999 group is audit-only.
    assert not terms["annual:1999:excluded"].included
    assert not terms["annual:1998:missing"].included
    assert len(result.terms) == len(view.original_series.annual) + len(view.original_series.groups)


def test_wp05_historical_disjointness_still_applies_to_paired_view() -> None:
    view = _approved()
    independent = HistoricalPeriod(start_year=1700, end_year=1701, threshold=3.0)
    result = log_likelihood(ExponentialKernel(), view.likelihood_series, historical_periods=(independent,))
    alone = log_likelihood(ExponentialKernel(), view.likelihood_series)
    assert isclose(result.log_value - alone.log_value, 2 * log1p(-exp(-3)), abs_tol=1e-12, rel_tol=0)
    overlapping_annual = HistoricalPeriod(start_year=2001, end_year=2002, threshold=3.0)
    with pytest.raises(LikelihoodEvidenceError, match="overlapping"):
        log_likelihood(ExponentialKernel(), view.likelihood_series, historical_periods=(overlapping_annual,))
    overlapping_group = HistoricalPeriod(start_year=1880, end_year=1881, threshold=3.0)
    with pytest.raises(LikelihoodEvidenceError, match="overlapping"):
        log_likelihood(ExponentialKernel(), view.likelihood_series, historical_periods=(overlapping_group,))


def _wp02_mapping() -> MappedAnnualSeries:
    """Use the actual WP02 mapper on typed Decimal/source-line evidence."""
    observations = (
        *(
            LegacyObservation(
                year=2000 + index,
                kind=LegacyObservationKind.EXACT,
                value=Decimal(f"{index}.000"),
                source="explicit synthetic WP02 record",
                raw_flag=0,
            )
            for index in range(1, 7)
        ),
        LegacyObservation(
            year=1999,
            kind=LegacyObservationKind.EXCLUDED,
            value=Decimal("999.000"),
            source="explicit synthetic prior exclusion",
            reason="source-labelled synthetic exclusion, not inferred from flag",
        ),
        LegacyObservation(
            year=1998,
            kind=LegacyObservationKind.MISSING,
            value=None,
            source="synthetic missing annual record",
        ),
    )
    period = LegacyExposurePeriod(
        identifier="synthetic-systematic",
        kind=LegacyExposureKind.SYSTEMATIC,
        years=frozenset(record.year for record in observations),
    )
    lines = {
        record.year: f"synthetic source row {record.year}; magnitude={record.value}; flag={record.raw_flag}"
        for record in observations
    }
    source_bytes = b"source-identified synthetic WP02 cohort, not legacy Flike data\n"
    return map_annual_series(
        (period,),
        observations,
        (),
        discharge_unit="m3/s",
        source_lines_by_year=lines,
        source_sha256=sha256(source_bytes).hexdigest(),
        source_file_version="3.10",
        year_definition="explicit synthetic calendar years",
    )


def _file310_mapping_with_bytes() -> tuple[MappedAnnualSeries, bytes]:
    """Reparse an actual synthetic file3.10 source instead of a self-attesting row ledger."""
    peak_rows = tuple(f" {year - 2000}.0000000E+00    0 {year}" for year in range(2001, 2007))
    lines = (
        "FLIKE data file  310",
        "Synthetic test station",
        " ",
        "    2    2  0.0000000E+00    1    0    0  0.1000000E+03",
        "Peak flow",
        "m^3/s",
        "    0",
        "    6",
        *peak_rows,
        "    0 10.00   0.50  50.00    1  0.988900E+01",
        "    0",
        " 0.1000000E+01  0.1000000E+01",
    )
    raw = ("\r\n".join(lines) + "\r\n").encode("ascii")
    project = parse_fld(raw)
    assert project.file_version == "3.10"
    assert project.sha256 == sha256(raw).hexdigest()
    assert project.raw_observation_lines == peak_rows
    period = LegacyExposurePeriod(
        identifier="synthetic source file systematic years",
        kind=LegacyExposureKind.SYSTEMATIC,
        years=frozenset(range(2001, 2007)),
    )
    mapping = map_annual_series(
        (period,),
        project.observations,
        (),
        discharge_unit=project.units,
        source_lines_by_year=dict(zip(range(2001, 2007), peak_rows, strict=True)),
        source_sha256=project.sha256,
        source_file_version=project.file_version,
        year_definition="explicit synthetic systematic years",
    )
    return mapping, raw


def test_wp02_mapping_bridge_rejects_changed_custody_and_wrong_approval() -> None:
    mapped, raw = _file310_mapping_with_bytes()
    proposal = _proposal(mapped.series)

    def checked(
        mapping: MappedAnnualSeries,
        *,
        approve: bool = True,
        source: str | None = "synthetic test",
        bytes_override: bytes | None = None,
    ) -> None:
        paired_censor_from_mapped_source(
            mapping,
            proposal,
            original_file310_bytes=raw if bytes_override is None else bytes_override,
            approve=approve,
            approval_source="explicit synthetic source approval",
            verified_exact_flags=frozenset({"0"}),
            verified_flag_source=source,
        )

    with pytest.raises(PairedCensorEvidenceError, match="explicit source-identified approval"):
        checked(mapped, approve=False)
    with pytest.raises(PairedCensorEvidenceError, match="unverified original project flag"):
        checked(mapped, source=None)

    # The critical Codex regression: a different full-precision Decimal can
    # round to the same binary64 float and formerly passed a self-replay.
    changed_decimal = replace(mapped.source_records[0].legacy, value=Decimal("1.00000000000000001"))
    assert changed_decimal.value is not None
    assert float(changed_decimal.value) == mapped.series.annual[0].value
    altered = replace(
        mapped,
        source_records=(replace(mapped.source_records[0], legacy=changed_decimal), *mapped.source_records[1:]),
    )
    with pytest.raises(PairedCensorEvidenceError, match="full-precision Decimal"):
        checked(altered)

    # Even a different precision representation of the same numeric value
    # must not silently replace the actual source token's Decimal exponent.
    altered_zeros = replace(mapped.source_records[0].legacy, value=Decimal("1.0"))
    altered = replace(
        mapped,
        source_records=(replace(mapped.source_records[0], legacy=altered_zeros), *mapped.source_records[1:]),
    )
    with pytest.raises(PairedCensorEvidenceError, match="Decimal representation"):
        checked(altered)

    inconsistent_sha = replace(mapped.source_records[0], source_sha256="a" * 64)
    altered = replace(mapped, source_records=(inconsistent_sha, *mapped.source_records[1:]))
    with pytest.raises(PairedCensorEvidenceError, match="inconsistent source SHA-256"):
        checked(altered)
    missing_line = replace(mapped.source_records[0], raw_source_line=" ")
    altered = replace(mapped, source_records=(missing_line, *mapped.source_records[1:]))
    with pytest.raises(PairedCensorEvidenceError, match="source row differs"):
        checked(altered)

    with pytest.raises(PairedCensorEvidenceError, match="source identity"):
        checked(mapped, bytes_override=raw.replace(b"1.0000000E+00", b"1.0000001E+00"))
    with pytest.raises(PairedCensorEvidenceError, match="source bytes are malformed"):
        checked(mapped, bytes_override=b"not a Flike file3.10 data source")


def test_wp02_actual_mapper_rejects_raw_flags_and_unverified_zeroes() -> None:
    mapped = _wp02_mapping()
    source = tuple(item.legacy for item in mapped.source_records)
    lines = {item.legacy.year: item.raw_source_line for item in mapped.source_records}
    sha = mapped.source_records[0].source_sha256
    observations = (
        replace(source[0], kind=LegacyObservationKind.FLAG_UNRESOLVED, raw_flag=3),
        *source[1:],
    )
    with pytest.raises(LegacyInputError, match="unresolved project flag"):
        map_annual_series(
            mapped.source_periods,
            observations,
            (),
            discharge_unit="m3/s",
            source_lines_by_year=lines,
            source_sha256=sha,
        )
    zero_without_verification = replace(
        source[0], kind=LegacyObservationKind.REPORTED_ZERO, value=Decimal(0), raw_flag=None
    )
    with pytest.raises(LegacyInputError, match="unresolved observation kind"):
        map_annual_series(
            mapped.source_periods,
            (zero_without_verification, *source[1:]),
            (),
            discharge_unit="m3/s",
            source_lines_by_year=lines,
            source_sha256=sha,
        )


def test_flike_file310_parse_to_approved_math_likelihood_preserves_byte_provenance() -> None:
    """Use independently reparsed file3.10 bytes, not merely a WP02 mapper replay."""
    mapping, raw = _file310_mapping_with_bytes()
    assert all(item.raw_source_line in raw.decode("ascii") for item in mapping.source_records)
    proposal = _proposal(mapping.series)
    view = paired_censor_from_mapped_source(
        mapping,
        proposal,
        original_file310_bytes=raw,
        approve=True,
        approval_source="synthetic analyst approval, not verified Flike defaults",
        verified_exact_flags=frozenset({"0"}),
        verified_flag_source="synthetic source fixture explicitly treats raw 0 as exact",
    )
    assert isinstance(view, PairedMappedCensorLikelihoodView)
    assert tuple(row.legacy.value for row in view.selected_source_records) == (
        Decimal("1.0000000"),
        Decimal("2.0000000"),
    )
    assert {row.source_sha256 for row in view.selected_source_records} == {sha256(raw).hexdigest()}
    assert not view.mapped_source.ready_for_likelihood
    assert not view.likelihood_series.groups  # Editor dummy-year group is not extra exposure.
    assert len(view.likelihood_series.annual) == 6
    result = log_likelihood(ExponentialKernel(), view.likelihood_series)
    assert isclose(
        result.log_value,
        -(3 + 4 + 5 + 6) + 2 * log1p(-exp(-2.5)),
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert len(result.terms) == len(mapping.series.annual)
