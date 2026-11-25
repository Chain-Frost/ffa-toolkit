"""WP02: synthetic boundary checks plus original candidate corpus admission checks."""

from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest

from ffa_methods.legacy import (
    CaseStatus,
    CensorGroup,
    ExposureKind,
    ExposurePeriod,
    LegacyInputError,
    Observation,
    ObservationKind,
    audit_exposure,
    parse_ams_csv,
    parse_fld,
    parse_report,
    review_corpus,
)
from ffa_methods.legacy.provenance import review_candidate, verify_manifest_files

PROJECT = (
    b"\r\n".join(
        (
            b"FLIKE data file  310",
            b"Synthetic station",
            b" ",
            b"    2    2  0.0000000E+00    1    0    0  0.1000000E+03",
            b"Peak flow",
            b"m^3/s",
            b"    0",
            b"    2",
            b" 0.4200000E+02    0 2001",
            b" 0.0000000E+00    0 2002",
            b"    0 10.00   0.50  50.00    1  0.988900E+01",
            b"    0",
            b" 0.1000000E+01  0.1000000E+01",
        )
    )
    + b"\r\n"
)

REPORT = (
    b"\r\n".join(
        (
            b" Report created on 1/ 1/2026 at 10:18",
            b" FLIKE program version 5.0.306.0",
            b" FLIKE file version 3.10",
            b" Data file: [REDACTED_SOURCE_PATH]",
            b" Title: Synthetic station",
            b" Input Data for Flood Frequency Analysis for Model: Log Pearson III",
            b" Gauged Annual Maximum Discharge Data",
            b" Obs   Discharge Year AEP plot AEP",
            b"                      position 1 in Y yrs",
            b" ----------------------------------------",
            b"   1      42.00 2001  0.90000      10.00",
            b"   2       0.00 2002  0.10000       1.11",
            b"",
            b" Flood model: Log Pearson III",
            b" Zero flow threshold:    0.000",
            b" Number of gauged flows at or below flow threshold =    1",
            b" Parameter 4 = loge(P/(1-P)) where P is probability of flow <= flow threshold",
            b" Summary of Posterior Moments from Importance Sampling",
            b" No Parameter                           Mean       Std dev  Correlation",
            b" ------------------------------------------------------------------------------",
            b"  1 Mean (loge flow)               5.4000   0.4000  1.000",
            b"  2 loge [Std dev (loge flow)]     0.5000   0.2000 -0.500 1.000",
            b"  3 Skew (loge flow)              -0.7000   0.5000  0.020 -0.450 1.000",
            b"  4 \x00\x00\x00 2060.52          -2.02714  0.50015 -0.007 0.009 0.004 1.000",
            b" ------------------------------------------------------------------------------",
            b" Upper bound = 20386.9",
            b" AEP 1 in Y    Exp parameter     Monte Carlo 90% quantile Mean(log10(q)) Stdev(log10(q))",
            b"                quantile             probability limits",
            b" ---------------------------------------------------------------------------------------",
            b"      1.250           5.21           0.00           17.6         0.6060         0.5685",
            b"      2.000          83.25          38.32          159.4         1.9033         0.1897",
            b"",
            b"      Flood        Expected    <------------AEP------------>",
            b"      magnitude    probability    1 in Y       90% limits",
            b" -----------------------------------------------------------",
            b"           0.00        0.12580      1.14      1.05       1.28",
            b"         100.00        0.49357      1.97      1.51       2.97",
        )
    )
    + b"\r\n"
)


def test_file310_exposes_records_and_opaque_controls() -> None:
    project = parse_fld(PROJECT)
    assert project.title == "Synthetic station"
    assert project.file_version == "3.10"
    assert project.units == "m^3/s"
    assert project.sha256 == sha256(PROJECT).hexdigest()
    assert project.control_fields == ("2", "2", "0.0000000E+00", "1", "0", "0", "0.1000000E+03")
    assert project.unknown_tail_lines[0].strip().endswith("0.988900E+01")
    assert project.observations[0].value == Decimal(42)
    assert project.observations[1].kind == ObservationKind.REPORTED_ZERO
    assert project.flags_are_interpreted is False


def test_nonzero_project_flag_is_never_treated_as_exclusion() -> None:
    project = parse_fld(PROJECT.replace(b"42", b"42").replace(b"E+02    0 2001", b"E+02    3 2001"))
    assert project.observations[0].kind == ObservationKind.FLAG_UNRESOLVED
    assert project.observations[0].raw_flag == 3


@pytest.mark.parametrize(
    "data",
    [
        PROJECT.replace(b"file  310", b"file  309"),
        PROJECT.replace(b"    2\r\n 0.420", b"    3\r\n 0.420"),
        PROJECT.replace(b"E+02    0 2001", b"E+02    0 2002"),
        PROJECT.replace(b"0.4200000E+02", b"NaN"),
        PROJECT.replace(b"0.4200000E+02", b"-0.4200000E+02"),
        PROJECT.replace(b"0.4200000E+02    0 2001", b"0.4200000E+02 0 2001 X"),
        PROJECT + b"\x00",
    ],
)
def test_bad_project_fails_explicitly(data: bytes) -> None:
    with pytest.raises(LegacyInputError):
        parse_fld(data)


def test_csv_missing_and_zero_are_different() -> None:
    parsed = parse_ams_csv(b"Year,AMS\r\n2001,0\r\n2002,\r\n2003,2.1\r\n")
    assert [record.kind for record in parsed] == [
        ObservationKind.REPORTED_ZERO,
        ObservationKind.MISSING,
        ObservationKind.EXACT,
    ]
    assert parsed[1].value is None


@pytest.mark.parametrize(
    "data",
    [
        b"Year,AMS\n2001,1\n2001,2\n",
        b"Year,AMS\n2001,-1\n",
        b"Year,AMS\n2001,1,excess\n",
        b"Year,AMS\n2001,NaN\n",
        b"year,AMS\n2001,1\n",
        b"Year,AMS\n2001,1\n2002,1\x00\n",
    ],
)
def test_bad_csv_fails_explicitly(data: bytes) -> None:
    with pytest.raises(LegacyInputError):
        parse_ams_csv(data)


def test_report_keeps_corrupt_label_and_separate_output_definitions() -> None:
    report = parse_report(REPORT)
    assert report.program_version == "5.0.306.0"
    assert report.reported_at_or_below_threshold == 1
    assert report.reported_upper_bound == Decimal("20386.9")
    assert report.reported_most_probable_threshold_probability is None
    assert report.observations[1].year == 2002
    assert report.posterior_parameters[1].raw_label == "loge [Std dev (loge flow)]"
    assert report.posterior_parameters[3].label_contains_nul
    assert report.posterior_parameters[3].mean == Decimal("-2.02714")
    assert report.posterior_parameters[3].standard_deviation == Decimal("0.50015")
    assert report.parameter_4_definition is not None
    assert len(report.quantiles) == 2
    assert report.quantiles[0].mean_log10_quantile == Decimal("0.6060")
    assert report.probabilities[0].raw_expected_probability == Decimal("0.12580")
    assert report.unknown_sections == ()


def test_report_version_and_table_fail_closed() -> None:
    with pytest.raises(LegacyInputError):
        parse_report(REPORT.replace(b"file version 3.10", b"file version 3.00"))
    with pytest.raises(LegacyInputError):
        parse_report(REPORT.replace(b"0.6060         0.5685", b"X              0.5685"))


def test_explicit_exposure_never_double_counts() -> None:
    historical = ExposurePeriod("old", ExposureKind.HISTORICAL, frozenset({1995, 1996}), Decimal(100))
    systematic = ExposurePeriod("gauge", ExposureKind.SYSTEMATIC, frozenset({2000, 2001}))
    observed = (
        Observation(2000, ObservationKind.EXACT, Decimal(5), "synthetic"),
        Observation(2001, ObservationKind.MISSING, None, "synthetic"),
    )
    grouped = (CensorGroup("old", 2, 0, Decimal(100), frozenset({1995, 1996})),)
    audited = audit_exposure((historical, systematic), observed, grouped)
    assert audited.ready_for_likelihood
    assert audited.grouped_years == frozenset({1995, 1996})
    with pytest.raises(LegacyInputError, match="overlap"):
        audit_exposure(
            (historical, systematic), observed, (CensorGroup("gauge", 1, 0, Decimal(100), frozenset({2000})),)
        )
    with pytest.raises(LegacyInputError, match="Overlapping"):
        audit_exposure((historical, ExposurePeriod("bad", ExposureKind.SYSTEMATIC, frozenset({1996, 1997}))), (), ())


def test_unallocated_group_retained_but_not_likelihood_ready() -> None:
    period = ExposurePeriod("historic", ExposureKind.HISTORICAL, frozenset({1990, 1991}), Decimal(8))
    audit = audit_exposure((period,), (), (CensorGroup("historic", 2, 0, Decimal(8)),))
    assert not audit.ready_for_likelihood
    assert audit.unresolved_group_ids == ("historic",)
    with pytest.raises(LegacyInputError, match="exceeds"):
        audit_exposure((period,), (), (CensorGroup("historic", 3, 0, Decimal(8)),))


def test_source_hash_and_transformation_metadata_are_enforced(tmp_path: Path) -> None:
    raw = b"year,discharge\n2001,1\n"
    (tmp_path / "record.txt").write_bytes(raw)
    sha = sha256(raw).hexdigest()
    item: dict[str, object] = {
        "staged_path": "record.txt",
        "source_member": "original/record.txt",
        "source_archive_sha256": "a" * 64,
        "original_sha256": sha,
        "staged_sha256": sha,
        "original_size_bytes": len(raw),
        "staged_size_bytes": len(raw),
        "transformations": [],
    }
    assert verify_manifest_files(tmp_path, {"schema_version": 1, "files": [item]})[0] == 1
    with pytest.raises(LegacyInputError, match="hash mismatch"):
        verify_manifest_files(tmp_path, {"schema_version": 1, "files": [{**item, "staged_sha256": "0" * 64}]})
    with pytest.raises(LegacyInputError, match="Unsafe"):
        verify_manifest_files(tmp_path, {"schema_version": 1, "files": [{**item, "staged_path": "../record.txt"}]})
    with pytest.raises(LegacyInputError, match="Undocumented"):
        verify_manifest_files(tmp_path, {"schema_version": 1, "files": [{**item, "original_sha256": "b" * 64}]})


def test_pinned_candidate_corpus_only_if_available() -> None:
    root = Path(__file__).resolve().parents[1] / "reference" / "legacy-candidates"
    if not root.is_dir():
        pytest.skip("Original six-file reference corpus not available in this local test workspace")
    review = review_corpus(root)
    assert review.checked_files == 6
    assert {case.case_id for case in review.cases} == {"case-a", "case-b"}
    assert [case.status for case in review.cases] == [CaseStatus.CANDIDATE] * 2
    assert [case.project_observations for case in review.cases] == [19, 40]
    assert [case.report_observations for case in review.cases] == [19, 40]
    assert all(not case.discrepancies for case in review.cases)
    assert all(case.unresolved for case in review.cases)
    assert len(review.unresolved_originals) == 6


def test_candidate_review_checks_declared_settings_without_promoting_status(tmp_path: Path) -> None:
    (tmp_path / "synthetic.fld").write_bytes(PROJECT)
    (tmp_path / "synthetic.csv").write_bytes(b"Year,AMS\n2001,42\n2002,0\n")
    (tmp_path / "synthetic.txt").write_bytes(REPORT)
    files: list[dict[str, object]] = [
        {"case_id": "synthetic", "staged_path": name} for name in ("synthetic.fld", "synthetic.csv", "synthetic.txt")
    ]
    case: dict[str, object] = {
        "case_id": "synthetic",
        "pairing_status": "candidate_settings_unverified",
        "distribution": "Log Pearson III",
        "zero_threshold": 0.0,
        "project_observation_flags_all_zero": True,
        "observations": 2,
        "quantile_rows": 2,
        "reported_at_or_below_threshold": 1,
    }
    reviewed = review_candidate(tmp_path, case, files)
    assert reviewed.status == CaseStatus.CANDIDATE
    assert not reviewed.discrepancies
    assert reviewed.unresolved
    assert review_candidate(tmp_path, {**case, "pairing_status": "unavailable"}, files).status == CaseStatus.UNAVAILABLE
    assert review_candidate(tmp_path, {**case, "distribution": "GEV"}, files).status == CaseStatus.MISMATCHED
    assert review_candidate(tmp_path, {**case, "zero_threshold": 1.0}, files).status == CaseStatus.MISMATCHED
    assert (
        review_candidate(tmp_path, {**case, "project_observation_flags_all_zero": False}, files).status
        == CaseStatus.MISMATCHED
    )
    with pytest.raises(LegacyInputError, match="promotion"):
        review_candidate(tmp_path, {**case, "pairing_status": "verified_pair"}, files)


def test_unrecorded_exposure_is_not_inferred_to_be_a_non_event() -> None:
    period = ExposurePeriod("gauge", ExposureKind.SYSTEMATIC, frozenset({2001, 2002}))
    observed = (Observation(2001, ObservationKind.EXACT, Decimal(5), "synthetic"),)
    audit = audit_exposure((period,), observed, ())
    assert not audit.ready_for_likelihood
    assert "2002" in audit.unresolved_group_ids[0]

    historical = ExposurePeriod("history", ExposureKind.HISTORICAL, frozenset({1990}))
    audit_history = audit_exposure((historical,), (), ())
    assert not audit_history.ready_for_likelihood
    assert any("threshold" in reason for reason in audit_history.unresolved_group_ids)


def test_truncated_numeric_tables_are_not_accepted_as_complete() -> None:
    bad_probabilities = REPORT.replace(
        b"100.00        0.49357      1.97      1.51       2.97",
        b"100.00        0.49357      1.97      1.51",
    )
    bad_quantiles = REPORT.replace(
        b"83.25          38.32          159.4         1.9033         0.1897",
        b"83.25          38.32          159.4         1.9033",
    )
    assert bad_probabilities != REPORT
    assert bad_quantiles != REPORT
    with pytest.raises(LegacyInputError, match="Malformed probability table row"):
        parse_report(bad_probabilities)
    with pytest.raises(LegacyInputError, match="Malformed quantile table row"):
        parse_report(bad_quantiles)


def test_malformed_later_gauged_row_is_not_a_table_terminator() -> None:
    bad_observation = REPORT.replace(
        b"2       0.00 2002  0.10000       1.11",
        b"2       0.00 2002  0.10000",
    )
    assert bad_observation != REPORT
    with pytest.raises(LegacyInputError, match="Malformed report observation"):
        parse_report(bad_observation)
    bad_rank = REPORT.replace(
        b"2       0.00 2002  0.10000       1.11",
        b"X       0.00 2002  0.10000       1.11",
    )
    with pytest.raises(LegacyInputError, match="Malformed report observation"):
        parse_report(bad_rank)
