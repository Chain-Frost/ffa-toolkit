"""Audit published pyflood2022 result identities without refitting or reproducing source rows."""

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, DecimalException, InvalidOperation, localcontext
from hashlib import sha256
from math import inf, isfinite, log, nextafter, ulp
from pathlib import Path
from typing import TypeAlias

from scripts.audit_pyflood2022 import ASSETS, REVISION, archive_csv, flood_table, pair_audit, parse_table, verify_asset

FIELDS = ("ALL", "JACKKNIFE", "SURPRISE", "CI5", "CI95")
VARIABLES = (("FLOW_PEAK", "m3/sec"), ("RUNOFF_240H", "mm"))
FAMILIES = ("GEV", "LOGPEARSON3")
Key: TypeAlias = tuple[str, int]
Rows: TypeAlias = dict[Key, list[str]]


@dataclass(frozen=True, slots=True)
class Cell:
    """A source decimal token and its validation status; preserve displayed precision."""

    state: str
    value: Decimal | None = None


def parse_cell(token: str) -> Cell:
    if not token.strip():
        return Cell("missing")
    try:
        value = Decimal(token)
    except InvalidOperation:
        return Cell("nonnumeric")
    if not value.is_finite():
        return Cell("nonfinite")
    numeric = float(value)
    if not isfinite(numeric) or (value != 0 and abs(numeric) < sys.float_info.min):
        return Cell("unrepresentable")
    return Cell("valid", value)


def rounding_interval(value: Decimal) -> tuple[float, float] | None:
    """Conditional nearest rounding at the last displayed decimal place, not proven export precision."""
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):
        return None
    try:
        with localcontext() as ctx:
            ctx.prec = max(50, len(value.as_tuple().digits) + 8)
            half = Decimal("0.5").scaleb(exponent)
            lower = float(value - half)
            upper = float(value + half)
    except (DecimalException, OverflowError):
        return None
    if not isfinite(lower) or not isfinite(upper):
        return None
    # Expand the float conversion outwards; log/subtraction use a separate allowance.
    lower, upper = nextafter(lower, -inf), nextafter(upper, inf)
    return (lower, upper) if isfinite(lower) and isfinite(upper) else None


@dataclass(frozen=True, slots=True)
class Identity:
    status: str
    residual: float | None = None


def _unavailable(cells: tuple[Cell, ...]) -> str | None:
    if any(cell.state not in ("valid", "missing") for cell in cells):
        return "invalid_token"
    if any(cell.value is None for cell in cells):
        return "missing"
    return None


def surprise_identity(all_fit: Cell, jackknife: Cell, surprise: Cell) -> Identity:
    """Equation (4): ln(Q100_ALL) - ln(Q100_JACKKNIFE), compared to archived S."""
    unavailable = _unavailable((all_fit, jackknife, surprise))
    if unavailable is not None:
        return Identity(unavailable)
    a, j, s = all_fit.value, jackknife.value, surprise.value
    if a is None or j is None or s is None:
        msg = "validated surprise fields must have decimal values"
        raise ValueError(msg)
    if a <= 0 or j <= 0:
        return Identity("nonpositive_quantile")
    la, lj, observed = log(float(a)), log(float(j)), float(s)
    residual = abs(la - lj - observed)
    numerical = 32 * ulp(1.0) * (1 + abs(la) + abs(lj) + abs(observed))
    if residual <= numerical:
        return Identity("central_match", residual)
    ia, ij, si = rounding_interval(a), rounding_interval(j), rounding_interval(s)
    if ia is None or ij is None or si is None or ia[0] <= 0 or ij[0] <= 0:
        return Identity("rounding_unresolved", residual)
    predicted_lo = log(ia[0]) - log(ij[1])
    predicted_hi = log(ia[1]) - log(ij[0])
    if predicted_lo <= si[1] + numerical and predicted_hi >= si[0] - numerical:
        return Identity("compatible_display_rounding", residual)
    return Identity("inconsistent_under_rounding", residual)


def interval_order(lower: Cell, upper: Cell) -> str:
    unavailable = _unavailable((lower, upper))
    if unavailable is not None:
        return unavailable
    if lower.value is None or upper.value is None:
        msg = "validated interval fields must have decimal values"
        raise ValueError(msg)
    return "ordered" if lower.value <= upper.value else "reversed"


def mean_vs_interval(mean: Cell, lower: Cell, upper: Cell) -> str:
    """A posterior mean need not be in its central credible interval."""
    if _unavailable((mean, lower, upper)) is not None or interval_order(lower, upper) != "ordered":
        return "not_evaluable"
    m, lo, hi = mean.value, lower.value, upper.value
    if m is None or lo is None or hi is None:
        msg = "validated mean and interval fields must have decimal values"
        raise ValueError(msg)
    return "inside" if lo <= m <= hi else "outside"


def columns(variable: str, unit: str, family: str) -> tuple[str, ...]:
    prefix = f"{variable}_{family}-Q100"
    return (
        f"{prefix}-ALL[{unit}]",
        f"{prefix}-JACKKNIFE[{unit}]",
        f"{prefix}-SURPRISE[adim]",
        f"{prefix}CI5-ALL[{unit}]",
        f"{prefix}CI95-ALL[{unit}]",
    )


def keyed_table(data: bytes) -> tuple[tuple[str, ...], Rows]:
    header, records, _ = parse_table(data)
    si, yi = header.index("SITEID"), header.index("WATERYEAR")
    rows: Rows = {}
    for row in records:
        key = (row[si], int(row[yi]))
        if key in rows:
            msg = "duplicate source station/water-year key"
            raise ValueError(msg)
        rows[key] = row
    return header, rows


def key_receipt(keys: list[Key]) -> dict[str, object]:
    """Record membership reproducibly without emitting source data rows."""
    ordered = sorted(keys)
    encoded = json.dumps(ordered, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return {"count": len(ordered), "sha256_sorted_keys": sha256(encoded).hexdigest()}


def repeated_field_summary(rows: Rows, index: int) -> dict[str, object]:
    values: dict[str, set[Decimal]] = {}
    unavailable: set[str] = set()
    for (site, _), row in rows.items():
        cell = parse_cell(row[index])
        site_values = values.setdefault(site, set[Decimal]())
        if cell.value is None:
            unavailable.add(site)
        else:
            site_values.add(cell.value)
    conflicts = sorted(site for site, v in values.items() if len(v) > 1)
    return {
        "sites": len(values),
        "sites_without_valid_value": sum(not v for v in values.values()),
        "sites_with_unavailable_rows": len(unavailable),
        "sites_with_conflicting_valid_values": len(conflicts),
    }


def _record_keys(ledger: dict[str, list[Key]], status: str, key: Key) -> None:
    ledger.setdefault(status, list[Key]()).append(key)


def block_audit(header: tuple[str, ...], rows: Rows, variable: str, unit: str, family: str) -> dict[str, object]:
    source_index = header.index(f"{variable}[{unit}]")
    indices = tuple(header.index(column) for column in columns(variable, unit, family))
    identity_keys: dict[str, list[Key]] = {}
    field_counts = {name: Counter[str]() for name in ("source", *FIELDS)}
    ci = Counter[str]()
    mean = Counter[str]()
    residual_max = 0.0
    missing_source_with_results = 0
    for key, row in rows.items():
        source = parse_cell(row[source_index])
        cells = tuple(parse_cell(row[index]) for index in indices)
        all_fit, jackknife, surprise, lower, upper = cells
        for name, cell in zip(("source", *FIELDS), (source, *cells), strict=True):
            field_counts[name][cell.state] += 1
            if cell.value is not None and cell.value <= 0 and name != "SURPRISE":
                field_counts[name]["nonpositive_valid"] += 1
        result = surprise_identity(all_fit, jackknife, surprise)
        _record_keys(identity_keys, result.status, key)
        if result.residual is not None:
            residual_max = max(residual_max, result.residual)
        ci[interval_order(lower, upper)] += 1
        mean[mean_vs_interval(all_fit, lower, upper)] += 1
        if source.value is None and any(cell.value is not None for cell in cells):
            missing_source_with_results += 1
    return {
        "variable": variable,
        "unit": unit,
        "family": family,
        "rows": len(rows),
        "field_counts": field_counts,
        "equation4": {status: key_receipt(keys) for status, keys in sorted(identity_keys.items())},
        "max_abs_central_residual": residual_max,
        "interval_order": ci,
        "posterior_mean_vs_interval": mean,
        "unavailable_source_with_valid_result_rows": missing_source_with_results,
        "site_repeated_fields": {
            field: repeated_field_summary(rows, indices[i]) for i, field in ((0, "ALL"), (3, "CI5"), (4, "CI95"))
        },
    }


def water_year_audit(header: tuple[str, ...], rows: Rows) -> dict[str, object]:
    """Enumerate possible water-year conventions; agreement is not source definition."""
    ti = header.index("FLOW_TIME_OF_PEAK")
    counts = Counter[str]()
    candidates = {f"start_month_{month:02}_{label}": 0 for month in range(1, 13) for label in ("start", "end")}
    earliest, latest = 9999, 0
    for (_, year), row in rows.items():
        counts["pre1950_water_year_labels"] += int(year < 1950)
        try:
            stamp = datetime.fromisoformat(row[ti])
        except ValueError:
            counts["missing_or_unparsed_timestamp"] += 1
            continue
        counts["parsed_timestamp"] += 1
        earliest, latest = min(earliest, stamp.year), max(latest, stamp.year)
        counts["pre1950_peak_calendar_years"] += int(stamp.year < 1950)
        for month in range(1, 13):
            starting = stamp.year - int(stamp.month < month)
            ending = starting + int(month != 1)
            candidates[f"start_month_{month:02}_start"] += int(year == starting)
            candidates[f"start_month_{month:02}_end"] += int(year == ending)
    return {
        "counts": counts,
        "earliest_peak_calendar_year": earliest if latest else None,
        "latest_peak_calendar_year": latest or None,
        "candidate_year_label_agreement_counts": candidates,
    }


def metadata_excerpt(data: bytes) -> list[str]:
    """Bounded relevant source header metadata only, never data rows."""
    terms = ("version", "source file", "date :", "wateryear", "water year", "surprise", "censor", "precision")
    lines: list[str] = []
    for line in data.decode("utf-8-sig").splitlines():
        if not line.startswith("#"):
            break
        if any(term in line.lower() for term in terms):
            if re.search(r"[A-Za-z]:[\\/]|\\\\|[\w.+-]+@[\w.-]+", line):
                lines.append("[identifying source-header value omitted]")
            else:
                lines.append(line[:240])
    return lines[:18]


def run(directory: Path) -> None:
    assets = {name: verify_asset(directory / name, size=size, blob=blob) for name, (size, blob) in ASSETS.items()}
    first = archive_csv(assets["flood_data_v5.zip"])[1]
    second = archive_csv(assets["flood_data_censored_v5.zip"])[1]
    if not pair_audit(flood_table(first), flood_table(second))["exact_unique_raw_pair"]:
        msg = "pinned archives lack identical unique raw peak keys"
        raise ValueError(msg)
    header, rows = keyed_table(first)
    other_header, other_rows = keyed_table(second)
    paired_fields = ("FLOW_TIME_OF_PEAK", "SPECIFICFLOW_PEAK[m3/sec/km2]", "RUNOFF_240H[mm]")
    differences = {
        field: sum(row[header.index(field)] != other_rows[key][other_header.index(field)] for key, row in rows.items())
        for field in paired_fields
    }
    print(
        json.dumps(
            {
                "scope": "archived output identity only; no refitting or native equivalence",
                "revision": REVISION,
                "paired_raw_keys": len(rows),
                "paired_additional_field_token_differences": differences,
                "water_year_evidence": water_year_audit(header, rows),
                "archive_header_metadata": {
                    "uncensored": metadata_excerpt(first),
                    "censored": metadata_excerpt(second),
                },
            },
            allow_nan=False,
        ),
        flush=True,
    )
    for label, table_header, table_rows in (("uncensored", header, rows), ("censored", other_header, other_rows)):
        for variable, unit in VARIABLES:
            for family in FAMILIES:
                result = block_audit(table_header, table_rows, variable, unit, family)
                print(json.dumps({"archive": label, **result}, allow_nan=False), flush=True)
    print("Archive identity audit completed; classification counts are evidence, not fit qualification.", flush=True)


def main() -> int:
    if len(sys.argv) != 2:
        msg = "usage: python -m scripts.audit_pyflood2022_results PINNED_ASSET_DIRECTORY"
        raise SystemExit(msg)
    run(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
