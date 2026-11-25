"""Read-only extractor for evidenced versioned Flike numerical report tables.

Reported values are Decimals and retain source tokens, including anomalous
parameter labels. No report quantity is treated as a validated estimator oracle.
"""

import re
from dataclasses import dataclass
from decimal import Decimal

from .models import LegacyInputError
from .project import parse_integer, parse_number

_REPORT_ROW = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\d{4})\s+(\S+)\s+(\S+)\s*$")
_VERSION = re.compile(r"^\s*FLIKE (program|file) version\s+(\S+)\s*$")
_FIELD = re.compile(r"^\s*([A-Za-z ]+):\s*(.*?)\s*$")


@dataclass(frozen=True, slots=True)
class ReportObservation:
    rank: int
    year: int
    discharge: Decimal
    plot_position_raw: Decimal
    return_period_raw: Decimal
    printed_discharge: str


@dataclass(frozen=True, slots=True)
class PosteriorParameter:
    source_index: int
    raw_label: str
    mean: Decimal
    standard_deviation: Decimal
    correlations: tuple[Decimal, ...]
    label_contains_nul: bool
    raw_line: str


@dataclass(frozen=True, slots=True)
class QuantileRow:
    return_period: Decimal
    expected_parameter_quantile: Decimal
    lower_90: Decimal
    upper_90: Decimal
    mean_log10_quantile: Decimal
    sd_log10_quantile: Decimal
    raw_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProbabilityRow:
    flood_magnitude: Decimal
    raw_expected_probability: Decimal
    raw_return_period: Decimal
    raw_lower_90_return_period: Decimal
    raw_upper_90_return_period: Decimal
    raw_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LegacyReport:
    program_version: str
    file_version: str
    report_created: str
    title: str
    source_path_header: str
    distribution: str
    zero_threshold: Decimal
    reported_at_or_below_threshold: int
    reported_upper_bound: Decimal
    reported_most_probable_threshold_probability: Decimal | None
    parameter_4_definition: str | None
    observations: tuple[ReportObservation, ...]
    posterior_parameters: tuple[PosteriorParameter, ...]
    quantiles: tuple[QuantileRow, ...]
    probabilities: tuple[ProbabilityRow, ...]
    unknown_sections: tuple[str, ...]
    raw_lines: tuple[str, ...]


def _single_field(lines: tuple[str, ...], label: str) -> str:
    prefix = label + ":"
    values = [line.strip()[len(prefix) :].strip() for line in lines if line.strip().startswith(prefix)]
    if len(values) != 1:
        msg = f"Expected exactly one report field {label!r}"
        raise LegacyInputError(msg)
    return values[0]


def _find_heading(lines: tuple[str, ...], fragment: str) -> int:
    positions = [index for index, line in enumerate(lines) if fragment in line]
    if len(positions) != 1:
        msg = f"Expected exactly one report section {fragment!r}"
        raise LegacyInputError(msg)
    return positions[0]


def _read_number_table(lines: tuple[str, ...], start: int, width: int, context: str) -> tuple[tuple[str, ...], ...]:
    """Read rows only after the first table-rule line following a header."""
    rule = next((index for index in range(start + 1, len(lines)) if lines[index].strip().startswith("---")), None)
    if rule is None or rule > start + 6:
        msg = f"Missing table separator in {context}"
        raise LegacyInputError(msg)
    rows: list[tuple[str, ...]] = []
    for line in lines[rule + 1 :]:
        parts = tuple(line.split())
        if len(parts) != width:
            if not parts and not rows:
                continue
            if rows and (not parts or not re.match(r"^[+-]?(?:\d|\.\d)", parts[0])):
                break
            msg = f"Malformed {context} row: {line!r}"
            raise LegacyInputError(msg)
        for token in parts:
            parse_number(token, context=context)
        rows.append(parts)
    if not rows:
        msg = f"No data rows in {context}"
        raise LegacyInputError(msg)
    return tuple(rows)


def _parse_observations(lines: tuple[str, ...]) -> tuple[ReportObservation, ...]:
    heading = _find_heading(lines, "Gauged Annual Maximum Discharge Data")
    rule = next(
        (index for index in range(heading + 1, min(len(lines), heading + 8)) if lines[index].strip().startswith("---")),
        None,
    )
    if rule is None:
        msg = "Missing gauged-observation header rule"
        raise LegacyInputError(msg)
    rows: list[ReportObservation] = []
    for line in lines[rule + 1 :]:
        matched = _REPORT_ROW.fullmatch(line)
        if matched is None:
            if not line.strip():
                if rows:
                    break
                continue
            # This strict, evidenced layout permits only explicit section
            # terminators. A malformed later row must not truncate evidence.
            if rows and (line.lstrip().startswith("Flood model:") or line.lstrip().startswith("---")):
                break
            msg = f"Malformed report observation record: {line!r}"
            raise LegacyInputError(msg)
        rank, flow, year, plot, period = matched.groups()
        index = parse_integer(rank, context="report observation index")
        if index != len(rows) + 1:
            msg = "Report observation indices are not consecutive"
            raise LegacyInputError(msg)
        rows.append(
            ReportObservation(
                rank=index,
                year=parse_integer(year, context="report observation year"),
                discharge=parse_number(flow, context="report observation discharge"),
                plot_position_raw=parse_number(plot, context="report plotting position"),
                return_period_raw=parse_number(period, context="report observation return period"),
                printed_discharge=flow,
            )
        )
    if not rows or len({row.year for row in rows}) != len(rows):
        msg = "No report observations or duplicate gauged years"
        raise LegacyInputError(msg)
    return tuple(rows)


def _parse_parameters(lines: tuple[str, ...]) -> tuple[PosteriorParameter, ...]:
    heading = _find_heading(lines, "Summary of Posterior Moments from Importance Sampling")
    rule = next(
        (index for index in range(heading + 1, min(len(lines), heading + 8)) if lines[index].strip().startswith("---")),
        None,
    )
    if rule is None:
        msg = "Missing posterior-moments header rule"
        raise LegacyInputError(msg)
    records: list[PosteriorParameter] = []
    for line in lines[rule + 1 :]:
        if line.lstrip().startswith("---") and records:
            break
        if not line.strip():
            continue
        parts = line.split()
        if not parts or not parts[0].isdigit():
            msg = f"Unexpected posterior parameter record {line!r}"
            raise LegacyInputError(msg)
        records.append(_parse_parameter(parts, line, len(records) + 1))
    if not records:
        msg = "No posterior parameters"
        raise LegacyInputError(msg)
    return tuple(records)


def parse_report(data: bytes) -> LegacyReport:
    """Extract four evidenced tables; flag every unimplemented censor section.

    Latin-1 is one-to-one with legacy single-byte data; embedded NUL bytes in a
    *label* are retained. Unknown sections are preserved and mark the source as
    unsuitable for automatically qualified comparisons.
    """
    lines = tuple(data.decode("latin-1").splitlines())
    versions = _parse_versions(lines)
    created = next((line.strip() for line in lines if line.lstrip().startswith("Report created on ")), None)
    if created is None:
        msg = "Missing report creation provenance"
        raise LegacyInputError(msg)
    distribution = _single_field(lines, "Flood model")
    if not distribution:
        msg = "Missing reported flood model"
        raise LegacyInputError(msg)
    observations = _parse_observations(lines)
    parameters = _parse_parameters(lines)
    threshold = parse_number(_single_field(lines, "Zero flow threshold"), context="zero flow threshold")
    if threshold < 0:
        msg = "Negative zero flow threshold"
        raise LegacyInputError(msg)
    count_lines = [line for line in lines if "Number of gauged flows at or below flow threshold" in line]
    if len(count_lines) != 1 or "=" not in count_lines[0]:
        msg = "Missing or duplicate threshold count"
        raise LegacyInputError(msg)
    count = parse_integer(count_lines[0].split("=", 1)[1].strip(), context="threshold count")
    if not 0 <= count <= len(observations):
        msg = "Invalid threshold count"
        raise LegacyInputError(msg)
    qstart = _find_heading(lines, "Exp parameter")
    # The heading spans two rows: the AEP-return-period table is keyed by this
    # unique first-header phrase rather than a fixed number of data rows.
    quantiles = tuple(
        QuantileRow(
            return_period=parse_number(row[0], context="quantile return period"),
            expected_parameter_quantile=parse_number(row[1], context="quantile at expected parameters"),
            lower_90=parse_number(row[2], context="quantile 90% lower limit"),
            upper_90=parse_number(row[3], context="quantile 90% upper limit"),
            mean_log10_quantile=parse_number(row[4], context="quantile log10 mean"),
            sd_log10_quantile=parse_number(row[5], context="quantile log10 standard deviation"),
            raw_tokens=row,
        )
        for row in _read_number_table(lines, qstart, 6, "quantile table")
    )
    pstart = _find_heading(lines, "Flood        Expected")
    probabilities = tuple(
        ProbabilityRow(
            flood_magnitude=parse_number(row[0], context="probability flood magnitude"),
            raw_expected_probability=parse_number(row[1], context="expected probability"),
            raw_return_period=parse_number(row[2], context="probability return period"),
            raw_lower_90_return_period=parse_number(row[3], context="probability return-period lower limit"),
            raw_upper_90_return_period=parse_number(row[4], context="probability return-period upper limit"),
            raw_tokens=row,
        )
        for row in _read_number_table(lines, pstart, 5, "probability table")
    )
    if any(row.return_period <= 1 or row.lower_90 > row.upper_90 for row in quantiles):
        msg = "Invalid reported quantile return period or limits"
        raise LegacyInputError(msg)
    if any(row.raw_expected_probability < 0 or row.raw_expected_probability > 1 for row in probabilities):
        msg = "Invalid expected-probability column"
        raise LegacyInputError(msg)
    reported_upper_bound, reported_p_mode, param4 = _parse_output_controls(lines)
    unknown = tuple(
        line.strip()
        for line in lines
        if "Censored Data" in line or "Censored Annual Maximum" in line or "Historical Data" in line
    )
    return LegacyReport(
        program_version=versions["program"],
        file_version=versions["file"],
        report_created=created,
        title=_single_field(lines, "Title"),
        source_path_header=_single_field(lines, "Data file"),
        distribution=distribution,
        zero_threshold=threshold,
        reported_at_or_below_threshold=count,
        reported_upper_bound=reported_upper_bound,
        reported_most_probable_threshold_probability=reported_p_mode,
        parameter_4_definition=param4[0] if param4 else None,
        observations=observations,
        posterior_parameters=parameters,
        quantiles=quantiles,
        probabilities=probabilities,
        unknown_sections=unknown,
        raw_lines=lines,
    )


def _parse_parameter(parts: list[str], line: str, expected_index: int) -> PosteriorParameter:
    index = int(parts[0])
    if index != expected_index:
        msg = "Posterior parameter order/index is invalid"
        raise LegacyInputError(msg)
    expected_numbers = index + 2  # mean, sd, and triangular correlation row
    if len(parts) < expected_numbers + 2:
        msg = f"Incomplete posterior parameter {index}"
        raise LegacyInputError(msg)
    suffix = parts[-expected_numbers:]
    numbers = tuple(parse_number(token, context=f"posterior parameter {index}") for token in suffix)
    label = " ".join(parts[1:-expected_numbers])
    if not label:
        msg = f"Missing posterior parameter label {index}"
        raise LegacyInputError(msg)
    if numbers[1] < 0 or any(abs(correlation) > 1 for correlation in numbers[2:]):
        msg = f"Invalid moments/correlations in parameter {index}"
        raise LegacyInputError(msg)
    return PosteriorParameter(index, label, numbers[0], numbers[1], numbers[2:], "\x00" in label, line)


def _parse_versions(lines: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in lines:
        match = _VERSION.fullmatch(line)
        if match:
            if match[1] in versions:
                msg = f"Repeated {match[1]} version"
                raise LegacyInputError(msg)
            versions[match[1]] = match[2]
    if set(versions) != {"program", "file"} or versions["file"] != "3.10":
        msg = "Unsupported or missing Flike report version"
        raise LegacyInputError(msg)
    return versions


def _parse_output_controls(lines: tuple[str, ...]) -> tuple[Decimal, Decimal | None, list[str]]:
    upper = [line.strip() for line in lines if line.lstrip().startswith("Upper bound =")]
    if len(upper) != 1:
        msg = "Missing or repeated reported upper bound"
        raise LegacyInputError(msg)
    reported_upper_bound = parse_number(upper[0].split("=", 1)[1].strip(), context="upper bound")
    if reported_upper_bound <= 0:
        msg = "Invalid reported upper bound"
        raise LegacyInputError(msg)
    p_mode = [line.strip() for line in lines if "The most probable value of P is" in line]
    if len(p_mode) > 1:
        msg = "Repeated threshold-probability mode"
        raise LegacyInputError(msg)
    reported_p_mode = (
        parse_number(p_mode[0].split("is", 1)[1].strip(), context="threshold probability mode") if p_mode else None
    )
    if reported_p_mode is not None and not 0 <= reported_p_mode <= 1:
        msg = "Invalid most-probable threshold probability"
        raise LegacyInputError(msg)
    param4 = [line.strip() for line in lines if line.lstrip().startswith("Parameter 4 =")]
    if len(param4) > 1:
        msg = "Repeated fourth-parameter statement"
        raise LegacyInputError(msg)
    return reported_upper_bound, reported_p_mode, param4
