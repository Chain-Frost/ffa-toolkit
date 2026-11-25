"""Read-only, strict adapters for evidenced Flike file3.10 and Year/AMS text.

The first and trailing project control blocks are intentionally opaque. Known
records are extracted without assigning meanings to undocumented positional
settings, coefficients, prior blocks or flag values.
"""

import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from io import StringIO

from .models import LegacyInputError, Observation, ObservationKind

_FILE_HEADER = re.compile(r"FLIKE data file\s+310\s*\Z")
_NUMERIC = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?\Z")
_INTEGER = re.compile(r"[+-]?\d+\Z")


def parse_number(token: str, *, context: str) -> Decimal:
    """Parse an explicit finite decimal/Fortran exponent without guessing formats."""
    if not _NUMERIC.fullmatch(token):
        msg = f"Unsupported numeric record in {context}: {token!r}"
        raise LegacyInputError(msg)
    try:
        value = Decimal(token.replace("D", "E").replace("d", "E"))
    except InvalidOperation as exc:
        msg = f"Invalid number in {context}: {token!r}"
        raise LegacyInputError(msg) from exc
    if not value.is_finite():
        msg = f"Non-finite value in {context}"
        raise LegacyInputError(msg)
    return value


def parse_integer(token: str, *, context: str) -> int:
    if not _INTEGER.fullmatch(token):
        msg = f"Invalid integer in {context}: {token!r}"
        raise LegacyInputError(msg)
    return int(token)


@dataclass(frozen=True, slots=True)
class LegacyProject:
    file_version: str
    title: str
    subtitle: str
    control_fields: tuple[str, ...]
    variable_name: str
    units: str
    pre_observation_control: str
    observations: tuple[Observation, ...]
    unknown_tail_lines: tuple[str, ...]
    sha256: str
    raw_observation_lines: tuple[str, ...]

    @property
    def flags_are_interpreted(self) -> bool:
        # Actual file3.10 flag semantics must be evidenced separately.
        return False


def parse_fld(data: bytes) -> LegacyProject:
    """Read the admitted 3-column observation layout of Flike file3.10.

    A file with a different header/count layout is rejected, not treated as a
    different version. Unknown trailing settings are retained, never executed.
    """
    if b"\x00" in data:
        msg = "Project contains a NUL byte"
        raise LegacyInputError(msg)
    lines = data.decode("latin-1").splitlines()
    if len(lines) < 9 or _FILE_HEADER.fullmatch(lines[0]) is None:
        msg = "Only the evidenced FLIKE data file 310 layout is supported"
        raise LegacyInputError(msg)
    controls = tuple(lines[3].split())
    if len(controls) != 7 or not all(_NUMERIC.fullmatch(token) for token in controls):
        msg = "Unsupported project primary control record"
        raise LegacyInputError(msg)
    for token in controls:
        parse_number(token, context="project primary controls")
    if not lines[1].strip() or not lines[4].strip() or not lines[5].strip():
        msg = "Missing project title, variable or units"
        raise LegacyInputError(msg)
    parse_integer(lines[6].strip(), context="pre-observation control")
    count = parse_integer(lines[7].strip(), context="observation count")
    if count <= 0 or len(lines) < 8 + count + 1:
        msg = "Invalid or truncated project observation block"
        raise LegacyInputError(msg)
    observations = _parse_project_observations(lines[8 : 8 + count])
    tail = tuple(lines[8 + count :])
    if not tail or not tail[0].strip():
        msg = "Truncated project trailing control block"
        raise LegacyInputError(msg)
    return LegacyProject(
        file_version="3.10",
        title=lines[1].strip(),
        subtitle=lines[2],
        control_fields=controls,
        variable_name=lines[4].strip(),
        units=lines[5].strip(),
        pre_observation_control=lines[6],
        observations=tuple(observations),
        unknown_tail_lines=tail,
        sha256=sha256(data).hexdigest(),
        raw_observation_lines=tuple(lines[8 : 8 + count]),
    )


def parse_ams_csv(data: bytes) -> tuple[Observation, ...]:
    """Read a year/AMS series, preserving empty cells as explicitly missing."""
    if b"\x00" in data:
        msg = "CSV contains NUL bytes"
        raise LegacyInputError(msg)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        msg = "CSV must be UTF-8 or UTF-8 with BOM"
        raise LegacyInputError(msg) from exc
    reader = csv.reader(StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except StopIteration as exc:
        msg = "Empty AMS CSV"
        raise LegacyInputError(msg) from exc
    if header != ["Year", "AMS"]:
        msg = "Expected exact Year,AMS CSV header"
        raise LegacyInputError(msg)
    observations = _parse_csv_rows(reader)
    if not observations:
        msg = "No AMS observations"
        raise LegacyInputError(msg)
    return tuple(observations)


def _parse_project_observations(lines: list[str]) -> list[Observation]:
    observations: list[Observation] = []
    seen: set[int] = set()
    for index, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 3:
            msg = f"Malformed project observation record {index}"
            raise LegacyInputError(msg)
        flow = parse_number(parts[0], context=f"observation {index}")
        if flow < 0:
            msg = f"Negative flow in observation {index}"
            raise LegacyInputError(msg)
        flag = parse_integer(parts[1], context=f"observation flag {index}")
        year = parse_integer(parts[2], context=f"observation year {index}")
        if year in seen:
            msg = f"Duplicate project year {year}"
            raise LegacyInputError(msg)
        seen.add(year)
        kind = (
            ObservationKind.FLAG_UNRESOLVED
            if flag != 0
            else ObservationKind.REPORTED_ZERO
            if flow == 0
            else ObservationKind.EXACT
        )
        observations.append(Observation(year, kind, flow, "flike_file3.10", raw_flag=flag))
    return observations


def _parse_csv_rows(reader: Iterable[list[str]]) -> list[Observation]:
    observations: list[Observation] = []
    years: set[int] = set()
    try:
        for line_number, columns in enumerate(reader, start=2):
            if len(columns) != 2:
                msg = f"CSV row {line_number}: expected two columns"
                raise LegacyInputError(msg)
            year = parse_integer(columns[0].strip(), context=f"CSV row {line_number}")
            if year in years:
                msg = f"Duplicate CSV year {year}"
                raise LegacyInputError(msg)
            years.add(year)
            token = columns[1].strip()
            if not token:
                observations.append(Observation(year, ObservationKind.MISSING, None, "year_ams_csv"))
                continue
            flow = parse_number(token, context=f"CSV row {line_number}")
            if flow < 0:
                msg = f"Negative AMS in CSV row {line_number}"
                raise LegacyInputError(msg)
            kind = ObservationKind.REPORTED_ZERO if flow == 0 else ObservationKind.EXACT
            observations.append(Observation(year, kind, flow, "year_ams_csv"))
    except csv.Error as exc:
        msg = f"Malformed CSV: {exc}"
        raise LegacyInputError(msg) from exc
    return observations
