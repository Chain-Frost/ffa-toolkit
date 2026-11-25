"""Read-only independent pyflood2022 v5 evidence audit; no native qualification."""

import csv
import io
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha1, sha256
from math import comb, fsum, isclose, isfinite
from pathlib import Path
from zipfile import ZipFile

from scipy.stats import genextreme

from ffa_methods.estimators.gev_lh import fit_gev_lh

REVISION = "d603936ae642feef94228b39c33b7307dc236b8e"
ASSETS = {
    "flood_data_v5.zip": (1388679, "d739e149d92c9569b68250881098b83b226d1b47"),
    "flood_data_censored_v5.zip": (1384695, "d7bc53bb04975e2a7524fb588e9b2071cbba52de"),
    "sites_info.csv": (73895, "75cdd56a08c6d075ac8974b746b28d478c3a6ec3"),
}
PEAK = "FLOW_PEAK[m3/sec]"
MAX_CSV_BYTES = 64 * 1024 * 1024
PROBABILITIES = (0.5, 0.9, 0.99)


def verify_asset(path: Path, *, size: int, blob: str) -> bytes:
    """Verify exact pinned Git object identity, not just the filename."""
    if path.stat().st_size != size:
        msg = f"pinned size mismatch: {path.name}"
        raise ValueError(msg)
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    if sha1(header + data, usedforsecurity=False).hexdigest() != blob:
        msg = f"pinned Git blob mismatch: {path.name}"
        raise ValueError(msg)
    return data


def archive_csv(data: bytes) -> tuple[str, bytes]:
    """Read one bounded CSV member in memory, without extracting or executing anything."""
    with ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) != 1:
            msg = "expected one CSV member"
            raise ValueError(msg)
        member = members[0]
        if (
            member.is_dir()
            or "/" in member.filename
            or "\\" in member.filename
            or not member.filename.lower().endswith(".csv")
            or not 0 < member.file_size <= MAX_CSV_BYTES
        ):
            msg = "unexpected or oversized ZIP member"
            raise ValueError(msg)
        return member.filename, archive.read(member)


def parse_table(data: bytes) -> tuple[tuple[str, ...], list[list[str]], int]:
    """Preserve column names and skip only the initial source comment block."""
    stream = io.StringIO(data.decode("utf-8-sig"), newline="")
    comments = 0
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line.startswith("#"):
            stream.seek(position)
            break
        comments += 1
    reader = csv.reader(stream, strict=True)
    header_row = next(reader, None)
    header = tuple(header_row) if header_row is not None else ()
    if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
        msg = "empty or duplicated CSV column names"
        raise ValueError(msg)
    rows = list(reader)
    if not rows or any(len(row) != len(header) for row in rows):
        msg = "empty or malformed CSV records"
        raise ValueError(msg)
    return header, rows, comments


def _peak_value(token: str) -> Decimal | None:
    if not token.strip():
        return None
    try:
        return Decimal(token)
    except InvalidOperation as exc:
        msg = "unexpected nonnumeric peak token"
        raise ValueError(msg) from exc


@dataclass(frozen=True, slots=True)
class FloodTable:
    header: tuple[str, ...]
    rows: dict[tuple[str, int], Decimal | None]
    duplicates: int
    duplicate_sites: frozenset[str]
    comments: int


def flood_table(data: bytes) -> FloodTable:
    header, records, comments = parse_table(data)
    if not {"SITEID", "WATERYEAR", PEAK}.issubset(header):
        msg = f"unverified instantaneous discharge schema: {header!r}"
        raise ValueError(msg)
    si, yi, qi = (header.index(name) for name in ("SITEID", "WATERYEAR", PEAK))
    rows: dict[tuple[str, int], Decimal | None] = {}
    duplicates = 0
    duplicate_sites: set[str] = set()
    for row in records:
        site, year = row[si], row[yi]
        if not site.strip() or not year.isascii() or not year.isdecimal() or not 1 <= int(year) <= 9999:
            msg = "invalid source site or water-year key"
            raise ValueError(msg)
        key = (site, int(year))
        value = _peak_value(row[qi])
        if key in rows:
            duplicates += 1
            duplicate_sites.add(site)
        else:
            rows[key] = value
    return FloodTable(header, rows, duplicates, frozenset(duplicate_sites), comments)


def _classification(value: Decimal | None) -> str:
    if value is None:
        return "missing"
    if not value.is_finite():
        return "nonfinite"
    if value <= 0:
        return "zero_or_negative"
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0:
        return "unrepresentable"
    return "positive"


def group_series(table: FloodTable) -> dict[str, list[tuple[int, Decimal | None]]]:
    grouped: dict[str, list[tuple[int, Decimal | None]]] = {}
    for (site, year), value in table.rows.items():
        grouped.setdefault(site, []).append((year, value))
    return grouped


def describe(table: FloodTable) -> dict[str, object]:
    grouped = group_series(table)
    counts: dict[str, int] = {}
    for value in table.rows.values():
        label = _classification(value)
        counts[label] = counts.get(label, 0) + 1
    lengths = sorted(len(rows) for rows in grouped.values())
    return {
        "header": table.header,
        "source_comment_lines": table.comments,
        "rows": len(table.rows) + table.duplicates,
        "unique_keys": len(table.rows),
        "duplicate_keys": table.duplicates,
        "sites": len(grouped),
        "first_water_year": min(year for _, year in table.rows),
        "last_water_year": max(year for _, year in table.rows),
        "minimum_records": lengths[0],
        "median_record_count": lengths[len(lengths) // 2],
        "maximum_records": lengths[-1],
        "sites_with_water_year_gaps": sum(
            len(rows) != max(year for year, _ in rows) - min(year for year, _ in rows) + 1 for rows in grouped.values()
        ),
        "unique_key_peak_classifications": counts,
    }


def pair_audit(first: FloodTable, second: FloodTable) -> dict[str, int | bool]:
    """Only compare raw annual observations, not posterior fields."""
    left, right = set(first.rows), set(second.rows)
    changed = 0
    for key in left & right:
        a, b = first.rows[key], second.rows[key]
        if a is not None and b is not None and (not a.is_finite() or not b.is_finite()):
            changed += int(str(a) != str(b))
        else:
            changed += int(a != b)
    return {
        "left_only_keys": len(left - right),
        "right_only_keys": len(right - left),
        "different_common_raw_peaks": changed,
        "exact_unique_raw_pair": not (left ^ right or changed or first.duplicates or second.duplicates),
    }


def independent_l_moments(values: tuple[float, ...]) -> tuple[float, float, float]:
    """Direct ascending-rank unbiased PWM oracle, independent of LH implementation."""
    ordered = sorted(values)
    n = len(ordered)
    if n < 3:
        msg = "three or more observations required"
        raise ValueError(msg)
    b = tuple(
        fsum(comb(i, rank) / comb(n - 1, rank) * value for i, value in enumerate(ordered) if i >= rank) / n
        for rank in range(3)
    )
    return b[0], 2 * b[1] - b[0], fsum((6 * b[2], -6 * b[1], b[0]))


def _eligible(rows: list[tuple[int, Decimal | None]]) -> bool:
    return len(rows) >= 30 and all(_classification(value) == "positive" for _, value in rows)


def selected_cases(table: FloodTable, sites: set[str]) -> list[tuple[str, tuple[float, ...], int]]:
    """Choose shortest, median and longest eligible record by (length, site ID)."""
    candidates: list[tuple[str, tuple[float, ...], int]] = []
    for site, rows in group_series(table).items():
        if site not in sites or site in table.duplicate_sites or not _eligible(rows):
            continue
        values = tuple(float(value) for _, value in sorted(rows) if value is not None)
        if min(values) != max(values):
            gaps = max(year for year, _ in rows) - min(year for year, _ in rows) + 1 - len(rows)
            candidates.append((site, values, gaps))
    candidates.sort(key=lambda case: (len(case[1]), case[0]))
    if len(candidates) < 3:
        msg = "fewer than three eligible positive nonconstant annual records"
        raise ValueError(msg)
    return [candidates[0], candidates[len(candidates) // 2], candidates[-1]]


def check_case(site: str, values: tuple[float, ...], gaps: int) -> dict[str, object]:
    """Numerical consistency checks, NOT the published MCMC estimates."""
    try:
        fitted = fit_gev_lh(values, shift=0)
    except (ValueError, ArithmeticError) as exc:
        return {"site": site, "records": len(values), "gaps": gaps, "status": "fit_rejected", "reason": str(exc)}
    reference = independent_l_moments(values)
    scale = max(abs(value) for value in values)
    moment_ok = all(
        isclose(expected, actual, rel_tol=1e-8, abs_tol=scale * 1e-10)
        for expected, actual in zip(reference, fitted.fitted_moments[:3], strict=True)
    )
    quantile_ok = all(
        isclose(
            fitted.quantile(p),
            float(genextreme.ppf(p, fitted.shape, loc=fitted.location, scale=fitted.scale)),
            rel_tol=1e-10,
            abs_tol=scale * 1e-12,
        )
        for p in PROBABILITIES
    )
    return {
        "site": site,
        "records": len(values),
        "gaps": gaps,
        "status": "pass" if moment_ok and quantile_ok else "numerical_mismatch",
        "independent_pwm_moments": moment_ok,
        "scipy_quantile_consistency": quantile_ok,
    }


def audit(directory: Path) -> bool:
    assets: dict[str, bytes] = {}
    evidence: dict[str, object] = {}
    for name, (size, blob) in ASSETS.items():
        data = verify_asset(directory / name, size=size, blob=blob)
        assets[name] = data
        evidence[name] = {"git_blob": blob, "sha256": sha256(data).hexdigest(), "bytes": len(data)}
    header, records, _ = parse_table(assets["sites_info.csv"])
    if "STATIONID" not in header:
        msg = "station metadata missing STATIONID"
        raise ValueError(msg)
    site_index = header.index("STATIONID")
    sites = {row[site_index] for row in records}
    if "" in sites or len(sites) != len(records):
        msg = "missing or duplicate metadata station IDs"
        raise ValueError(msg)
    first_name, first_bytes = archive_csv(assets["flood_data_v5.zip"])
    second_name, second_bytes = archive_csv(assets["flood_data_censored_v5.zip"])
    first, second = flood_table(first_bytes), flood_table(second_bytes)
    cases = [check_case(site, values, gaps) for site, values, gaps in selected_cases(first, sites)]
    result = {
        "scope": "read-only independent mathematical audit; not matched MCMC or native Flike",
        "revision": REVISION,
        "assets": evidence,
        "site_metadata_count": len(sites),
        "unmatched_uncensored_sites": len(set(group_series(first)) - sites),
        "unmatched_censored_sites": len(set(group_series(second)) - sites),
        "uncensored_member": first_name,
        "uncensored_csv_sha256": sha256(first_bytes).hexdigest(),
        "uncensored": describe(first),
        "censored_member": second_name,
        "censored_csv_sha256": sha256(second_bytes).hexdigest(),
        "censored": describe(second),
        "paired_raw_peaks": pair_audit(first, second),
        "fixed_h_zero_checks": cases,
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    return all(case["status"] == "pass" for case in cases)


def main() -> int:
    if len(sys.argv) != 2:
        msg = "usage: python scripts/audit_pyflood2022.py PINNED_ASSET_DIRECTORY"
        raise SystemExit(msg)
    return 0 if audit(Path(sys.argv[1])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
