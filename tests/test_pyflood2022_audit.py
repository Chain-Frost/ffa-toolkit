"""Synthetic tests for the external archive audit; no downloads in ordinary CI."""

import io
from hashlib import sha1
from math import isclose
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.audit_pyflood2022 import (
    PEAK,
    archive_csv,
    describe,
    flood_table,
    independent_l_moments,
    pair_audit,
    selected_cases,
    verify_asset,
)


def _zip(name: str, data: bytes) -> bytes:
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr(name, data)
    return stream.getvalue()


def _csv(rows: str) -> bytes:
    return f"# synthetic data\nSITEID,WATERYEAR,{PEAK}\n{rows}".encode()


def test_verify_exact_git_object(tmp_path: Path) -> None:
    data = b"immutable synthetic asset"
    path = tmp_path / "asset"
    path.write_bytes(data)
    blob = sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()
    assert verify_asset(path, size=len(data), blob=blob) == data
    with pytest.raises(ValueError, match="size mismatch"):
        verify_asset(path, size=len(data) + 1, blob=blob)
    path.write_bytes(b"X" + data[1:])
    with pytest.raises(ValueError, match="blob mismatch"):
        verify_asset(path, size=len(data), blob=blob)


def test_archive_member_is_read_not_extracted() -> None:
    data = _csv("001,2001,3\n")
    assert archive_csv(_zip("events.csv", data)) == ("events.csv", data)
    with pytest.raises(ValueError, match="unexpected"):
        archive_csv(_zip("../events.csv", data))


def test_duplicate_and_missing_keys_not_silently_promoted() -> None:
    a = flood_table(_csv("001,2001,3\n001,2002,\n001,2002,5\n002,2001,0\n"))
    b = flood_table(_csv("001,2001,3.0\n001,2002,4\n003,2001,5\n"))
    assert describe(a)["duplicate_keys"] == 1
    assert describe(a)["rows"] == 4
    assert describe(a)["unique_key_peak_classifications"] == {"positive": 1, "missing": 1, "zero_or_negative": 1}
    assert pair_audit(a, b) == {
        "left_only_keys": 1,
        "right_only_keys": 1,
        "different_common_raw_peaks": 1,
        "exact_unique_raw_pair": False,
    }


def test_preserve_leading_zero_site_and_source_year() -> None:
    a = flood_table(_csv("001,2001,3\n001,2003,4\n1,2001,5\n"))
    assert describe(a)["sites"] == 2
    assert describe(a)["sites_with_water_year_gaps"] == 1
    assert pair_audit(a, a)["exact_unique_raw_pair"] is True


def test_require_declared_peak_not_specific_discharge_or_runoff() -> None:
    with pytest.raises(ValueError, match="instantaneous discharge schema"):
        flood_table(b"SITEID,WATERYEAR,SPECIFICFLOW_PEAK[m3/sec/km2]\n001,2001,1\n")


def test_independent_pwm_known_example() -> None:
    # For 1,2,3,4 the sample L1 is 2.5, L2 is 5/6, L3 is zero by symmetry.
    first, second, third = independent_l_moments((4.0, 1.0, 3.0, 2.0))
    assert isclose(first, 2.5, abs_tol=1e-14)
    assert isclose(second, 5 / 6, abs_tol=1e-14)
    assert isclose(third, 0.0, abs_tol=1e-14)


def test_selection_deterministic_and_no_dropping_bad_rows() -> None:
    data = "".join(
        f"{site},{2000 + i},{i + 1}\n"
        for site, count in (("001", 30), ("002", 31), ("003", 32), ("004", 33))
        for i in range(count)
    )
    data += "004,2040,0\n"
    cases = selected_cases(flood_table(_csv(data)), {"001", "002", "003", "004"})
    assert [case[0] for case in cases] == ["001", "002", "003"]
    assert [len(case[1]) for case in cases] == [30, 31, 32]
