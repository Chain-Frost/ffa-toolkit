"""Independent checks of archived output identity classification and custody."""

import json
from math import log

from scripts.audit_pyflood2022_results import (
    block_audit,
    columns,
    interval_order,
    keyed_table,
    mean_vs_interval,
    metadata_excerpt,
    parse_cell,
    surprise_identity,
    water_year_audit,
)


def test_equation4_central_and_rounding_compatible() -> None:
    # Exact zero is an identity, not evidence of a successful fit.
    exact = surprise_identity(parse_cell("12.34"), parse_cell("12.34"), parse_cell("0"))
    assert exact.status == "central_match"
    rounded = surprise_identity(parse_cell("100.00"), parse_cell("80.00"), parse_cell("0.22314"))
    assert rounded.status == "compatible_display_rounding"
    assert rounded.residual is not None
    assert abs(rounded.residual - abs(log(100 / 80) - 0.22314)) < 1e-12


def test_quantile_rounding_not_only_surprise_rounding() -> None:
    # 10.04 and 9.96 both round to 10.0; their log-ratio rounds to 0.008.
    outcome = surprise_identity(parse_cell("10.0"), parse_cell("10.0"), parse_cell("0.008"))
    assert outcome.status == "compatible_display_rounding"


def test_equation4_inconsistent_missing_invalid_and_nonpositive() -> None:
    a, j = parse_cell("100.00"), parse_cell("80.00")
    assert surprise_identity(a, j, parse_cell("0.25")).status == "inconsistent_under_rounding"
    assert surprise_identity(a, j, parse_cell("")).status == "missing"
    assert surprise_identity(a, j, parse_cell("NaN")).status == "invalid_token"
    assert surprise_identity(parse_cell("-1.0"), j, parse_cell("0.2")).status == "nonpositive_quantile"
    assert parse_cell("1e-500").state == "unrepresentable"


def test_interval_order_and_mean_outside_interval() -> None:
    lo, hi = parse_cell("10.0"), parse_cell("20.0")
    assert interval_order(lo, hi) == "ordered"
    assert interval_order(hi, lo) == "reversed"
    assert mean_vs_interval(parse_cell("30.0"), lo, hi) == "outside"
    assert mean_vs_interval(parse_cell("15.0"), lo, hi) == "inside"
    assert mean_vs_interval(parse_cell("15.0"), hi, lo) == "not_evaluable"


def test_archive_block_counts_and_site_repeated_fields() -> None:
    header = ("SITEID", "WATERYEAR", "FLOW_PEAK[m3/sec]", *columns("FLOW_PEAK", "m3/sec", "GEV"))
    data = (
        ",".join(header)
        + "\n001,2000,10,100.00,80.00,0.22314,20,200\n"
        + "001,2001,20,100.00,100.00,0,20,200\n"
        + "001,2002,30,120.00,100.00,0.5,200,20\n"
        + "002,2000,40,,,,,\n"
    ).encode()
    h, rows = keyed_table(data)
    report = block_audit(h, rows, "FLOW_PEAK", "m3/sec", "GEV")
    assert report["rows"] == 4
    assert report["interval_order"] == {"ordered": 2, "reversed": 1, "missing": 1}
    # Inspect the structured receipt without depending on raw reference observations.
    receipt = json.loads(json.dumps(report))
    assert receipt["equation4"]["central_match"]["count"] == 1
    assert receipt["equation4"]["compatible_display_rounding"]["count"] == 1
    assert receipt["equation4"]["inconsistent_under_rounding"]["count"] == 1
    assert "example_keys" not in receipt["equation4"]["inconsistent_under_rounding"]
    assert receipt["equation4"]["missing"]["count"] == 1
    assert receipt["site_repeated_fields"]["ALL"]["sites_with_conflicting_valid_values"] == 1
    assert receipt["site_repeated_fields"]["ALL"]["sites_without_valid_value"] == 1


def test_water_year_hypotheses_do_not_relabel_source() -> None:
    data = b"SITEID,WATERYEAR,FLOW_TIME_OF_PEAK\n001,2001,2000-07-01\n001,2002,2002-01-01\n"
    header, rows = keyed_table(data)
    result = water_year_audit(header, rows)
    assert result["counts"] == {
        "pre1950_water_year_labels": 0,
        "parsed_timestamp": 2,
        "pre1950_peak_calendar_years": 0,
    }
    assert ("001", 2001) in rows
    receipt = json.loads(json.dumps(result))
    candidates = receipt["candidate_year_label_agreement_counts"]
    assert candidates["start_month_07_end"] == 2
    assert candidates["start_month_01_start"] == 1


def test_source_header_custody_does_not_emit_identifying_paths_or_email() -> None:
    raw = (
        b"# version : v5\n"
        b"# source file : C:\\private\\record.csv\n"
        b"# source file : /home/synthetic/record.csv\n"
        b"# date : author@example.invalid\n"
        b"SITEID,WATERYEAR\n001,2000\n"
    )
    assert metadata_excerpt(raw) == [
        "# version : v5",
        "[identifying source-header value omitted]",
        "[identifying source-header value omitted]",
        "[identifying source-header value omitted]",
    ]
