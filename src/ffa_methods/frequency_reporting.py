"""Static frequency presentation of already computed, admitted deterministic fits."""

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from io import StringIO
from math import ceil, isfinite, log10
from typing import Self

from ffa_methods.contracts import AnalysisResult, ProbabilityKind, ResultMeaning
from ffa_methods.legacy.models import ObservationKind
from ffa_methods.legacy.project import parse_ams_csv

# Deliberately version-gated here: future fitter schemas do not automatically gain plotting admission.
_ADMITTED_SCHEMAS = ("ffa-gev-fixed-h/1", "ffa-ordinary-lmoments/1")
_EXACT_ASSERTION = "caller_asserted_complete_exact_positive_annual_maxima"
_SOURCE = "ARR Book 3 v4.2, Chapter 2, section 2.6.2, eq. (3.2.36), printed pp. 35–36"
_LIMITS = (
    "Observed ranks are plotting positions, not fitted probabilities. Fitted markers are the requested "
    "deterministic quantiles; connecting segments are visual guides, not additional calculations. "
    "No uncertainty intervals are calculated. Native compatibility and engineering adoption are not established."
)
_LEFT, _RIGHT, _TOP, _BOTTOM = 110, 960, 55, 445


@dataclass(frozen=True, slots=True)
class PlotPoint:
    series: str
    discharge: float
    aep: float
    non_exceedance: float
    year: int | None = None
    rank: int | None = None
    source_index: int | None = None


def _observed_points(raw_ams: bytes) -> tuple[PlotPoint, ...]:
    """Use all admitted source rows; do not infer exposure in unrepresented years."""
    rows: list[tuple[int, int, float]] = []
    for index, record in enumerate(parse_ams_csv(raw_ams), start=1):
        if record.kind != ObservationKind.EXACT or record.value is None:
            msg = "frequency report requires exact positive systematic annual maxima"
            raise ValueError(msg)
        discharge = float(record.value)
        if not isfinite(discharge) or discharge <= 0:
            msg = "frequency report requires finite positive systematic annual maxima"
            raise ValueError(msg)
        rows.append((index, record.year, discharge))
    n = len(rows)
    years = [row[1] for row in rows]
    if max(years) - min(years) + 1 != n:
        msg = "frequency report requires consecutive systematic years; missing exposure is not inferred"
        raise ValueError(msg)
    ranked = sorted(rows, key=lambda row: (-row[2], row[1]))
    points: list[PlotPoint] = []
    for rank, (index, year, discharge) in enumerate(ranked, start=1):
        aep = (rank - 0.4) / (n + 0.2)
        points.append(PlotPoint("observed", discharge, aep, 1.0 - aep, year, rank, index))
    return tuple(points)


def _fitted_points(result: AnalysisResult, unit: str) -> tuple[PlotPoint, ...]:
    """Preserve every requested deterministic result exactly; reject other estimands."""
    points: list[PlotPoint] = []
    previous = 0.0
    for point in result.values:
        f = point.input_probability
        if (
            point.meaning != ResultMeaning.DETERMINISTIC_FITTED_QUANTILE
            or point.input_probability_kind != ProbabilityKind.NON_EXCEEDANCE
            or point.unit != unit
            or f is None
            or not previous < f < 1.0
            or not isfinite(point.value)
        ):
            msg = "frequency report supports only ordered, finite deterministic fitted quantiles in the source unit"
            raise ValueError(msg)
        points.append(PlotPoint("fitted", point.value, 1.0 - f, f))
        previous = f
    if not points:
        msg = "frequency report requires fitted quantiles"
        raise ValueError(msg)
    return tuple(points)


@dataclass(frozen=True, slots=True)
class PlotScale:
    decades: int
    magnitude: float
    low: float
    high: float

    @classmethod
    def from_points(cls, points: tuple[PlotPoint, ...]) -> Self:
        decades = max(1, ceil(-log10(min(point.aep for point in points))))
        magnitude = max(abs(point.discharge) for point in points)
        low = min(0.0, min(point.discharge / magnitude for point in points))
        high = max(point.discharge / magnitude for point in points)
        return cls(decades, magnitude, low, high)

    def xy(self, point: PlotPoint) -> tuple[float, float]:
        x = _LEFT + (_RIGHT - _LEFT) * (-log10(point.aep) / self.decades)
        y = _BOTTOM - (_BOTTOM - _TOP) * ((point.discharge / self.magnitude - self.low) / (self.high - self.low))
        return x, y


def _axis_svg(scale: PlotScale, unit: str) -> list[str]:
    lines: list[str] = []
    step = 1 if scale.decades <= 8 else 2
    for decade in sorted({*range(0, scale.decades + 1, step), scale.decades}):
        x = _LEFT + (_RIGHT - _LEFT) * decade / scale.decades
        lines.append(f'<path d="M{x} {_TOP} V{_BOTTOM}" stroke="#d5dbe2"/>')
        lines.append(f'<text x="{x}" y="469" text-anchor="middle">{10.0 ** (2 - decade):g}%</text>')
    # For short log ranges, label the engineering-relevant intermediate AEPs.
    # Avoid crowding labels when the requested tail spans many decades.
    if scale.decades <= 3:
        for decade in range(scale.decades):
            for multiplier in (5, 2):
                aep = multiplier * 10.0 ** (-decade - 1)
                x = _LEFT + (_RIGHT - _LEFT) * (-log10(aep) / scale.decades)
                lines.append(f'<path d="M{x} {_TOP} V{_BOTTOM}" stroke="#edf0f3"/>')
                lines.append(f'<text x="{x}" y="469" text-anchor="middle">{100 * aep:g}%</text>')
    for index in range(6):
        fraction = index / 5
        y = _BOTTOM - (_BOTTOM - _TOP) * fraction
        value = ((1.0 - fraction) * scale.low + fraction * scale.high) * scale.magnitude
        lines.append(f'<path d="M{_LEFT} {y} H{_RIGHT}" stroke="#d5dbe2"/>')
        lines.append(f'<text x="100" y="{y + 4}" text-anchor="end">{value:.5g}</text>')
    lines.extend(
        (
            f'<path d="M{_LEFT} {_TOP} V{_BOTTOM} H{_RIGHT}" fill="none" stroke="#202c39"/>',
            '<text x="535" y="500" text-anchor="middle">Annual exceedance probability (AEP, log scale)</text>',
            f'<text x="20" y="250" text-anchor="middle" transform="rotate(-90 20 250)">Discharge ({escape(unit)})</text>',
        )
    )
    return lines


def _support_notice(result: AnalysisResult) -> str:
    support = result.fitted_gev_support
    details = "See analysis report for support and other warnings."
    if support is not None:
        details = f"GEV support: {len(support.outside)} outside; {len(support.at_endpoint)} at endpoint. {details}"
    if any(point.value < 0 for point in result.values):
        details = f"Negative mathematical quantiles are nonphysical. {details}"
    return details


def _svg(points: tuple[PlotPoint, ...], scale: PlotScale, unit: str, notice: str) -> str:
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 580" role="img" aria-labelledby="title desc">',
        '<title id="title">Annual maximum flood frequency — independent, not native-qualified</title>',
        f'<desc id="desc">{escape(_LIMITS)} {escape(notice)}</desc>',
        '<rect width="1000" height="580" fill="white"/>',
        '<g font-family="sans-serif" font-size="13" fill="#202c39">',
        '<text x="110" y="28" font-size="18" font-weight="bold">Annual maximum flood frequency</text>',
        *_axis_svg(scale, unit),
    ]
    fitted = [scale.xy(point) for point in points if point.series == "fitted"]
    if len(fitted) > 1:
        coords = " ".join(f"{x},{y}" for x, y in fitted)
        lines.append(f'<polyline points="{coords}" fill="none" stroke="#1767a0" stroke-width="2"/>')
    for point in points:
        x, y = scale.xy(point)
        # Open fitted markers leave coincident observed points visible.
        style = (
            'r="4" fill="#202c39"'
            if point.series == "observed"
            else 'r="5.5" fill="none" stroke="#1767a0" stroke-width="1.7"'
        )
        label = f"{point.series}: year={point.year}, AEP={point.aep!r}, Q={point.discharge!r} {unit}"
        lines.append(
            f'<circle data-series="{point.series}" cx="{x}" cy="{y}" {style}>'
            f"<title>{escape(label)}</title></circle>"
        )
    lines.extend(
        (
            '<circle cx="120" cy="522" r="4" fill="#202c39"/>',
            '<text x="132" y="526">Observed Cunnane positions</text>',
            '<circle cx="370" cy="522" r="5.5" fill="none" stroke="#1767a0" stroke-width="1.7"/>',
            '<text x="382" y="526">Requested fitted quantiles (segments are guides)</text>',
            '<text x="110" y="545">No intervals; independent mathematical fit, not native-qualified.</text>',
            f'<text x="110" y="563" font-size="11">{escape(notice)}</text>',
            "</g></svg>",
        )
    )
    return "\n".join(lines) + "\n"


def _coordinate_csv(points: tuple[PlotPoint, ...], scale: PlotScale, unit: str) -> str:
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        ("series", "source_index", "year", "descending_rank", "F", "AEP", "discharge", "unit", "svg_x", "svg_y")
    )
    for point in points:
        writer.writerow(
            (
                point.series,
                point.source_index,
                point.year,
                point.rank,
                point.non_exceedance,
                point.aep,
                point.discharge,
                unit,
                *scale.xy(point),
            )
        )
    return stream.getvalue()


def _html_report(svg: str, points: tuple[PlotPoint, ...], report_text: str, unit: str, year_definition: str) -> str:
    observed = tuple(point for point in points if point.series == "observed")
    rows = "".join(
        f"<tr><td>{point.source_index}</td><td>{point.year}</td><td>{point.rank}</td>"
        f"<td>{point.discharge!r}</td><td>{point.non_exceedance!r}</td><td>{point.aep!r}</td></tr>"
        for point in sorted(observed, key=lambda point: point.source_index or 0)
    )
    fitted_rows = "".join(
        f"<tr><td>{point.non_exceedance!r}</td><td>{point.aep!r}</td><td>{point.discharge!r}</td></tr>"
        for point in points
        if point.series == "fitted"
    )
    return (
        '<!doctype html>\n<html lang="en"><meta charset="utf-8"><title>FFA frequency report</title>'
        '<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem}'
        'svg{width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{padding:.4rem;border:1px solid #aaa}'
        'td{text-align:right}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.85rem}'
        '@media print{body{margin:0}tr{break-inside:avoid}svg{break-inside:avoid}}</style>'
        '<main><h1>Flood frequency analysis</h1><p><strong>Not an adopted engineering estimate.</strong> '
        f'{escape(_LIMITS)}</p><h2>Frequency plot</h2>{svg}'
        f'<p>Plotting basis: {escape(_SOURCE)}. Cunnane AEP = (rank − 0.4)/(n + 0.2); F = 1 − AEP. '
        'Equal discharges receive consecutive ranks ordered by ascending year (presentation decision). '
        'The logarithmic AEP axis is independent of fitted family; discharge is linear and signed.</p>'
        f'<h2>Systematic record and exposure</h2><p>{len(observed)} consecutive supplied years; '
        f'year definition: {escape(year_definition)}. The source classification and units are caller assertions. '
        'No historical, censored, missing, excluded or zero observations are admitted to this report.</p>'
        f'<table><thead><tr><th>Source record</th><th>Year</th><th>Rank</th><th>Q ({escape(unit)})</th>'
        f'<th>F</th><th>AEP</th></tr></thead><tbody>{rows}</tbody></table>'
        '<h2>Requested deterministic design quantiles</h2>'
        f'<table><thead><tr><th>F</th><th>AEP</th><th>Q ({escape(unit)})</th></tr></thead>'
        f'<tbody>{fitted_rows}</tbody></table>'
        '<p>No uncertainty intervals are calculated. These are not adopted estimates.</p>'
        '<h2>Calculated results, assumptions, diagnostics and warnings</h2>'
        f'<pre>{escape(report_text)}</pre>'
        '<h2>Plot reference and replay</h2><p>Cunnane, C. (1978), Unbiased plotting positions — A review, '
        'Journal of Hydrology 37, 205–222, doi:10.1016/0022-1694(78)90017-3. '
        'ARR: Commonwealth of Australia (Geoscience Australia), 2019. '
        'Use the byte-exact source snapshots and the same explicit fit settings to replay; '
        'frequency-coordinates.csv preserves the plotted coordinates. '
        'See also the offline command ffa explain frequency-plots --format markdown.</p></main></html>\n'
    )


def render_frequency_files(
    result: AnalysisResult, *, settings: Mapping[str, object], raw_ams: bytes, report_text: str
) -> dict[str, str]:
    """Produce additive static files; no fitting, uncertainty or source defaults."""
    unit, year_definition = settings.get("discharge_unit"), settings.get("year_definition")
    if (
        settings.get("schema_version") not in _ADMITTED_SCHEMAS
        or settings.get("source_classification") != _EXACT_ASSERTION
        or not isinstance(unit, str)
        or not isinstance(year_definition, str)
    ):
        msg = "frequency report requires an admitted deterministic systematic-AMS settings schema"
        raise ValueError(msg)
    if any(
        not (0x20 <= ord(char) <= 0xD7FF or 0xE000 <= ord(char) <= 0xFFFD or 0x10000 <= ord(char) <= 0x10FFFF)
        for char in unit
    ):
        msg = "discharge unit contains a character forbidden in SVG XML"
        raise ValueError(msg)
    observed = _observed_points(raw_ams)
    points = (*observed, *_fitted_points(result, unit))
    scale = PlotScale.from_points(points)
    svg = _svg(points, scale, unit, _support_notice(result))
    return {
        "frequency.svg": svg,
        "frequency-coordinates.csv": _coordinate_csv(points, scale, unit),
        "report.html": _html_report(svg, points, report_text, unit, year_definition),
    }
