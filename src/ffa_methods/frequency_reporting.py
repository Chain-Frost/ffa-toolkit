"""Static frequency presentation of already computed, admitted deterministic fits."""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from html import escape
from io import StringIO
from math import ceil, isfinite, log10
from typing import Self

from ffa_methods.contracts import AnalysisResult, ProbabilityKind, ResultMeaning
from ffa_methods.frequency_quantiles import (
    BAYESIAN_PLOT_SCHEMA,
    BAYESIAN_SERIES,
    BOOTSTRAP_LIMIT,
    DETERMINISTIC_SCHEMAS,
    POSTERIOR_LIMIT,
    SELECTED_SCHEMA,
    QuantilePlotRow,
    bayesian_quantile_rows,
    deterministic_quantile_rows,
)
from ffa_methods.legacy.models import ObservationKind
from ffa_methods.legacy.project import parse_ams_csv

# Deliberately version-gated here: future fitter schemas do not automatically gain plotting admission.
_EXACT_ASSERTION = "caller_asserted_complete_exact_positive_annual_maxima"
_SOURCE = "ARR Book 3 v4.2, Chapter 2, section 2.6.2, eq. (3.2.36), printed pp. 35-36"
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
    rows: list[tuple[int, int, Decimal]] = []
    for index, record in enumerate(parse_ams_csv(raw_ams), start=1):
        if record.kind != ObservationKind.EXACT or record.value is None:
            msg = "frequency report requires exact positive systematic annual maxima"
            raise ValueError(msg)
        discharge = float(record.value)
        if not isfinite(discharge) or discharge <= 0:
            msg = "frequency report requires finite positive systematic annual maxima"
            raise ValueError(msg)
        rows.append((index, record.year, record.value))
    n = len(rows)
    years = [row[1] for row in rows]
    if max(years) - min(years) + 1 != n:
        msg = "frequency report requires consecutive systematic years; missing exposure is not inferred"
        raise ValueError(msg)
    ranked = sorted(rows, key=lambda row: (row[2].copy_negate(), row[1]))
    points: list[PlotPoint] = []
    for rank, (index, year, discharge) in enumerate(ranked, start=1):
        aep = (rank - 0.4) / (n + 0.2)
        points.append(PlotPoint("observed", float(discharge), aep, 1.0 - aep, year, rank, index))
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
        minimum = min(point.discharge for point in points)
        low = min(0.0, minimum / magnitude)
        high = max(point.discharge / magnitude for point in points)
        return cls(decades, magnitude, low, high)

    def xy(self, point: PlotPoint) -> tuple[float, float]:
        x = _LEFT + (_RIGHT - _LEFT) * (-log10(point.aep) / self.decades)
        y = _BOTTOM - (_BOTTOM - _TOP) * ((point.discharge / self.magnitude - self.low) / (self.high - self.low))
        return x, y


def _axis_svg(scale: PlotScale, unit: str) -> list[str]:
    lines: list[str] = []
    step = 1 if scale.decades <= 8 else 2
    # Leave room for the final label when a two-decade step ends one decade early.
    decades = [decade for decade in range(0, scale.decades, step) if scale.decades - decade >= step]
    for decade in (*decades, scale.decades):
        x = _LEFT + (_RIGHT - _LEFT) * decade / scale.decades
        lines.append(f'<path d="M{x} {_TOP} V{_BOTTOM}" stroke="#d5dbe2"/>')
        exponent = 2 - decade
        label = f"{10.0**exponent:g}" if exponent >= -3 else f"1e{exponent}"
        lines.append(f'<text x="{x}" y="469" text-anchor="middle">{label}%</text>')
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
        normalized = (1.0 - fraction) * scale.low + fraction * scale.high
        # Tick labels are real-valued interpolations. Avoid underflow between
        # subnormal inputs and retain readable scientific notation at large scales.
        value = Decimal(normalized) * Decimal(scale.magnitude)
        label = f"{value:.5g}" if value else "0"
        if value and (abs(value) < Decimal("0.001") or abs(value) >= Decimal("1e6")):
            label = f"{value:.4e}"
        lines.append(f'<path d="M{_LEFT} {y} H{_RIGHT}" stroke="#d5dbe2"/>')
        lines.append(f'<text x="100" y="{y + 4}" text-anchor="end">{label}</text>')
    lines.extend(
        (
            f'<path d="M{_LEFT} {_TOP} V{_BOTTOM} H{_RIGHT}" fill="none" stroke="#202c39"/>',
            '<text x="535" y="500" text-anchor="middle">Annual exceedance probability (AEP, log scale)</text>',
            f'<text x="12" y="250" text-anchor="middle" transform="rotate(-90 12 250)">Discharge ({escape(unit)})</text>',
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
        identity = (
            f"Observed year {point.year}, rank {point.rank}, source record {point.source_index}"
            if point.series == "observed"
            else "Requested fitted quantile"
        )
        label = f"{identity}, F={point.non_exceedance!r}, AEP={point.aep!r}, Q={point.discharge!r} {unit}"
        lines.append(
            f'<circle data-series="{point.series}" cx="{x}" cy="{y}" {style}><title>{escape(label)}</title></circle>'
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
        "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem}"
        "svg{width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{padding:.4rem;border:1px solid #aaa}"
        "td{text-align:right}td,th{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.85rem}"
        "h2{break-after:avoid-page}code{overflow-wrap:anywhere}"
        "@media screen and (max-width:650px){table{table-layout:fixed}td,th{padding:.2rem;font-size:.83rem}}"
        "@media print{body{margin:0}tr,svg{break-inside:avoid}thead{display:table-header-group}}</style>"
        "<main><h1>Flood frequency analysis</h1><p><strong>Not an adopted engineering estimate.</strong> "
        f"{escape(_LIMITS)}</p><h2>Frequency plot</h2>{svg}"
        f"<p>Plotting basis: {escape(_SOURCE)}. Cunnane AEP = (rank - 0.4)/(n + 0.2); F = 1 - AEP. "
        "Equal discharges receive consecutive ranks ordered by ascending year (presentation decision). "
        "The horizontal axis shows AEP in percent. Tables show F and AEP as fractions. Discharge is linear and signed.</p>"
        "<h2>Requested deterministic design quantiles</h2>"
        f"<table><thead><tr><th>F</th><th>AEP</th><th>Q ({escape(unit)})</th></tr></thead>"
        f"<tbody>{fitted_rows}</tbody></table>"
        f"<h2>Systematic record and exposure</h2><p>{len(observed)} consecutive supplied years; "
        f"year definition: {escape(year_definition)}. The source classification and units are caller assertions. "
        "No historical, censored, missing, excluded or zero observations are admitted to this report.</p>"
        f"<table><thead><tr><th>Source record</th><th>Year</th><th>Rank</th><th>Q ({escape(unit)})</th>"
        f"<th>F</th><th>AEP</th></tr></thead><tbody>{rows}</tbody></table>"
        "<h2>Calculated results, assumptions, diagnostics and warnings</h2>"
        f"<pre>{escape(report_text)}</pre>"
        "<h2>Plot reference and replay</h2><p>Cunnane, C. (1978), Unbiased plotting positions — A review, "
        "Journal of Hydrology 37, 205-222, doi:10.1016/0022-1694(78)90017-3. "
        "ARR: Commonwealth of Australia (Geoscience Australia), 2019. "
        "Replay from this directory using the byte-exact snapshots and the same package and dependency versions.</p>"
        "<p><code>ffa fit --ams source-ams.csv --settings source-settings.json --output replay "
        "--frequency-report</code></p><p>"
        "frequency-coordinates.csv preserves the plotted coordinates. "
        "See also the offline command ffa explain frequency-plots --format markdown.</p></main></html>\n"
    )


# The deterministic point-only renderer and its coordinate CSV remain unchanged.
_QUANTILE_STYLES: dict[ResultMeaning, tuple[str, str, str, float]] = {
    ResultMeaning.DETERMINISTIC_FITTED_QUANTILE: ("Deterministic fitted quantile", "#1767a0", "", 5.5),
    ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS: ("At expected parameters", "#1767a0", "", 5.5),
    ResultMeaning.POSTERIOR_MEAN_QUANTILE: ("Mean of posterior quantiles", "#a34f05", "7 4", 7.5),
    ResultMeaning.POSTERIOR_MEDIAN_QUANTILE: ("Median of posterior quantiles", "#7552a0", "2 3", 9.0),
    ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE: ("Posterior predictive quantile", "#16725f", "9 3 2 3", 11.0),
}
_BAYESIAN_LIMITS = (
    "Posterior quantile probability intervals are pointwise in F and describe parameter-induced uncertainty. "
    "They are not predictive intervals, simultaneous bands or Monte Carlo error bounds. "
    "The numerical precision state belongs to each reported estimate; unaudited outputs remain unassessed."
)


def _model_plot_point(row: QuantilePlotRow) -> PlotPoint:
    return PlotPoint(row.point.meaning.value, row.point.value, row.aep, row.f)


def _quantile_marker(row: QuantilePlotRow, scale: PlotScale) -> str:
    x, y = scale.xy(_model_plot_point(row))
    if row.point.meaning in (POSTERIOR_LIMIT, BOOTSTRAP_LIMIT):
        color, radius = "#64748b", 3.0
        interval = "Posterior quantile" if row.point.meaning == POSTERIOR_LIMIT else "Parametric bootstrap sampling"
        label = f"{interval} interval {row.point.probability_limit_side}, mass={row.point.confidence_level!r}"
    else:
        label, color, _, radius = _QUANTILE_STYLES[row.point.meaning]
    title = f"{label}, F={row.f!r}, AEP={row.aep!r}, Q={row.point.value!r} {row.point.unit}"
    if row.precision_state:
        title += f"; numerical precision={row.precision_state}"
    return (
        f'<circle data-result-row="{row.index}" data-series="{row.point.meaning.value}" '
        f'cx="{x}" cy="{y}" r="{radius}" fill="none" stroke="{color}" stroke-width="1.7">'
        f"<title>{escape(title)}</title></circle>"
    )


def _quantile_segments(rows: tuple[QuantilePlotRow, ...], scale: PlotScale) -> list[str]:
    """Only join adjacent requested probabilities; never bridge an omitted mean."""
    lines: list[str] = []
    probabilities = sorted({row.f for row in rows})
    for meaning in _QUANTILE_STYLES:
        by_f = {row.f: row for row in rows if row.point.meaning == meaning}
        _, color, dash, _ = _QUANTILE_STYLES[meaning]
        for left, right in zip(probabilities[:-1], probabilities[1:], strict=True):
            if left in by_f and right in by_f:
                x1, y1 = scale.xy(_model_plot_point(by_f[left]))
                x2, y2 = scale.xy(_model_plot_point(by_f[right]))
                lines.append(
                    f'<path data-series="{meaning.value}" d="M{x1} {y1} L{x2} {y2}" '
                    f'fill="none" stroke="{color}" stroke-width="1.7" stroke-dasharray="{dash}"/>'
                )
    return lines


def _interval_bars(rows: tuple[QuantilePlotRow, ...], scale: PlotScale) -> list[str]:
    lowers = {row.f: row for row in rows if row.point.probability_limit_side == "lower"}
    uppers = {row.f: row for row in rows if row.point.probability_limit_side == "upper"}
    lines: list[str] = []
    for f, lower in lowers.items():
        x, low_y = scale.xy(_model_plot_point(lower))
        _, high_y = scale.xy(_model_plot_point(uppers[f]))
        kind = "posterior_quantile" if lower.point.meaning == POSTERIOR_LIMIT else "parametric_bootstrap"
        lines.append(
            f'<path data-interval="{kind}" d="M{x - 5} {low_y} H{x + 5} '
            f'M{x} {low_y} V{high_y} M{x - 5} {high_y} H{x + 5}" '
            'fill="none" stroke="#64748b" stroke-width="1.5"/>'
        )
    return lines


def _bayesian_legend(rows: tuple[QuantilePlotRow, ...]) -> list[str]:
    mass = next(row.point.confidence_level for row in rows if row.point.meaning == POSTERIOR_LIMIT)
    lines = [
        '<circle cx="120" cy="522" r="4" fill="#202c39"/>',
        '<text x="138" y="526">Observed Cunnane positions</text>',
        '<path d="M115 536 V552 M110 536 H120 M110 552 H120" stroke="#64748b" fill="none"/>',
        f'<text x="138" y="548">Pointwise posterior quantile interval (mass={mass})</text>',
    ]
    for index, meaning in enumerate(BAYESIAN_SERIES):
        label, color, dash, radius = _QUANTILE_STYLES[meaning]
        y = 522 + 22 * index
        if any(row.point.meaning == meaning for row in rows):
            lines.extend(
                (
                    f'<path d="M505 {y} H555" stroke="{color}" stroke-dasharray="{dash}" fill="none"/>',
                    f'<circle cx="530" cy="{y}" r="{radius}" stroke="{color}" fill="none"/>',
                    f'<text x="570" y="{y + 4}">{label}</text>',
                )
            )
    if any(row.point.meaning == ResultMeaning.POSTERIOR_MEAN_QUANTILE for row in rows):
        note = "An absent posterior mean is undefined where omitted by the calculation; no substitute is plotted."
    else:
        note = "Posterior mean quantiles are undefined at the requested probabilities; no substitute is plotted."
    lines.extend(
        (
            '<text x="110" y="620">Only requested quantiles; segments are visual guides. Not native-qualified.</text>',
            '<text x="110" y="642">Intervals are pointwise; not predictive intervals, simultaneous bands or MC error.</text>',
            f'<text x="110" y="664" font-size="12">{escape(note)}</text>',
            '<text x="110" y="686" font-size="12">Numerical precision states and omitted quantities: see the full report.</text>',
        )
    )
    return lines


@dataclass(frozen=True, slots=True)
class QuantilePlotDescription:
    heading: str
    summary: str
    explanation: str
    legend: tuple[str, ...]


def _deterministic_description(
    result: AnalysisResult, settings: Mapping[str, object], rows: tuple[QuantilePlotRow, ...]
) -> QuantilePlotDescription:
    selected = settings.get("schema_version") == SELECTED_SCHEMA
    fit_label = f"Selected H={settings['lh_shift']} (provisional calibration)" if selected else "Deterministic point fit"
    mass = next((row.point.confidence_level for row in rows if row.point.meaning == BOOTSTRAP_LIMIT), None)
    legend = [
        '<circle cx="120" cy="522" r="4" fill="#202c39"/>',
        '<text x="138" y="526">Observed Cunnane positions</text>',
        '<circle cx="505" cy="522" r="5.5" fill="none" stroke="#1767a0" stroke-width="1.7"/>',
        '<text x="520" y="526">Requested fitted quantile</text>',
        f'<text x="110" y="620">{escape(fit_label)}. Not native-qualified; segments are visual guides.</text>',
    ]
    summary = "No uncertainty intervals were requested. Only the selected point fit is displayed."
    if mass is not None:
        summary = (
            "Pointwise parametric-bootstrap percentile sampling intervals, conditional on successful refits. "
            "They are not posterior or predictive intervals, simultaneous bands or endpoint Monte Carlo brackets."
        )
        legend.extend(
            (
                '<path d="M115 536 V552 M110 536 H120 M110 552 H120" stroke="#64748b" fill="none"/>',
                f'<text x="138" y="548">Pointwise bootstrap sampling interval (mass={mass})</text>',
                '<text x="110" y="642">Conditional on successful refits. Not posterior uncertainty or MC error.</text>',
                '<text x="110" y="664">Endpoint Monte Carlo diagnostics and refit attempt ledger: see report.</text>',
            )
        )
    legend.append(f'<text x="110" y="686" font-size="12">{escape(_support_notice(result))}</text>')
    explanation = (
        f"{fit_label}. {summary} The numerical report preserves all candidate diagnostics and, "
        "when requested, the complete bootstrap attempt ledger and endpoint Monte Carlo brackets. "
        "Blank per-result precision state means no Bayesian per-row audit field is supplied."
    )
    return QuantilePlotDescription("Annual maximum flood frequency", summary, explanation, tuple(legend))


def _bayesian_description(rows: tuple[QuantilePlotRow, ...]) -> QuantilePlotDescription:
    return QuantilePlotDescription(
        "Bayesian annual maximum flood frequency",
        _BAYESIAN_LIMITS,
        (
            "The four central estimands are different quantities. The posterior predictive quantile inverts "
            "the averaged CDF; it is not the mean or median of component quantiles. Probability-at-flow and "
            "return-period rows remain in the numerical report, not the quantile plot. "
            "Missing or undefined outputs are not reconstructed."
        ),
        tuple(_bayesian_legend(rows)),
    )


def _quantile_svg(
    observed: tuple[PlotPoint, ...],
    rows: tuple[QuantilePlotRow, ...],
    scale: PlotScale,
    unit: str,
    description: QuantilePlotDescription,
) -> str:
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 705" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(description.heading)} — independent, not native-qualified</title>',
        f'<desc id="desc">{escape(description.summary)}</desc>',
        '<rect width="1000" height="705" fill="white"/>',
        '<g font-family="sans-serif" font-size="13" fill="#202c39">',
        f'<text x="110" y="28" font-size="18" font-weight="bold">{escape(description.heading)}</text>',
        *_axis_svg(scale, unit),
        *_interval_bars(rows, scale),
        *_quantile_segments(rows, scale),
    ]
    for point in observed:
        x, y = scale.xy(point)
        title = f"Observed year {point.year}, rank {point.rank}, F={point.non_exceedance!r}, Q={point.discharge!r} {unit}"
        lines.append(
            f'<circle data-series="observed" cx="{x}" cy="{y}" r="4" fill="#202c39">'
            f"<title>{escape(title)}</title></circle>"
        )
    lines.extend(_quantile_marker(row, scale) for row in rows)
    lines.extend(description.legend)
    lines.append("</g></svg>")
    return "\n".join(lines) + "\n"


def _quantile_csv(
    observed: tuple[PlotPoint, ...], rows: tuple[QuantilePlotRow, ...], scale: PlotScale, unit: str
) -> str:
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "series", "source_index", "year", "descending_rank", "F", "AEP", "discharge", "unit", "svg_x", "svg_y",
            "result_row", "meaning", "reported_probability_level", "probability_limit_side",
            "probability_limit_construction", "precision_state",
        )
    )
    for point in observed:
        writer.writerow(
            (
                point.series, point.source_index, point.year, point.rank, point.non_exceedance,
                point.aep, point.discharge, unit, *scale.xy(point), "", "", "", "", "", "",
            )
        )
    for row in rows:
        point = row.point
        writer.writerow(
            (
                point.meaning.value, "", "", "", row.f, row.aep, point.value, unit,
                *scale.xy(_model_plot_point(row)), row.index, point.meaning.value, point.confidence_level,
                point.probability_limit_side, point.probability_limit_construction, row.precision_state,
            )
        )
    return stream.getvalue()


def _quantile_html(
    svg: str,
    observed: tuple[PlotPoint, ...],
    rows: tuple[QuantilePlotRow, ...],
    unit: str,
    year_definition: str,
    report_text: str,
    description: QuantilePlotDescription,
) -> str:
    def cell(value: object) -> str:
        return f"<td>{escape('' if value is None else str(value))}</td>"

    source_rows = "".join(
        "<tr>" + "".join(cell(v) for v in (p.source_index, p.year, p.rank, p.discharge, p.non_exceedance, p.aep)) + "</tr>"
        for p in sorted(observed, key=lambda p: p.source_index or 0)
    )
    model_rows = "".join(
        "<tr>" + "".join(
            cell(v) for v in (
                row.index, row.point.meaning.value, row.f, row.aep, row.point.value, row.point.confidence_level,
                row.point.probability_limit_side, row.point.probability_limit_construction, row.precision_state,
            )
        ) + "</tr>"
        for row in rows
    )
    return (
        '<!doctype html>\n<html lang="en"><meta charset="utf-8">'
        f"<title>{escape(description.heading)}</title>"
        '<style>body{font:15px/1.5 system-ui,sans-serif;max-width:1150px;margin:2rem auto;padding:0 1rem}'
        'svg{width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{border:1px solid #aaa;padding:.3rem}'
        'td,th{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.85rem}'
        'h2{break-after:avoid-page}@media print{body{margin:0}tr,svg{break-inside:avoid}thead{display:table-header-group}}</style>'
        f"<main><h1>{escape(description.heading)}</h1><p><strong>Not an adopted engineering estimate.</strong> "
        f"{escape(description.summary)}</p><h2>Frequency plot</h2>{svg}"
        f"<p>{escape(description.explanation)}</p>"
        '<h2>Requested quantiles and pointwise limits</h2><p>Result row is the one-based data row of results.csv '
        '(excluding its header), or the corresponding results.json entry. F and AEP are fractions.</p>'
        '<table><thead><tr><th>Result row</th><th>Meaning</th><th>F</th><th>AEP</th>'
        f'<th>Q ({escape(unit)})</th><th>Interval mass</th><th>Side</th><th>Construction</th>'
        f'<th>Numerical precision</th></tr></thead><tbody>{model_rows}</tbody></table>'
        f'<h2>Systematic record and exposure</h2><p>{len(observed)} consecutive exact positive supplied annual maxima; '
        f'year definition: {escape(year_definition)}. {escape(_SOURCE)}. '
        'Cunnane AEP=(rank-0.4)/(n+0.2); rank at original decimal precision, ascending year for ties. '
        'Units and classification are caller assertions. No censored or historical exposure is inferred.</p>'
        f'<table><thead><tr><th>Source record</th><th>Year</th><th>Rank</th><th>Q ({escape(unit)})</th>'
        f'<th>F</th><th>AEP</th></tr></thead><tbody>{source_rows}</tbody></table>'
        '<h2>Calculated results, diagnostics and warnings</h2>'
        f'<pre>{escape(report_text)}</pre><h2>Replay and references</h2>'
        '<p>Use the same package and dependency versions, settings, seed and source snapshots. '
        'Cunnane (1978), DOI 10.1016/0022-1694(78)90017-3; ARR 2019, Commonwealth of Australia (Geoscience Australia). '
        'See ffa explain quantile-estimands and ffa explain frequency-plots.</p>'
        '<p><code>ffa fit --ams source-ams.csv --settings source-settings.json --output replay --frequency-report</code></p>'
        '</main></html>\n'
    )


def _render_extended_frequency(
    result: AnalysisResult,
    *,
    settings: Mapping[str, object],
    raw_ams: bytes,
    report_text: str,
    unit: str,
    year_definition: str,
    numerical_rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    if settings.get("schema_version") == BAYESIAN_PLOT_SCHEMA:
        rows = bayesian_quantile_rows(result, settings=settings, unit=unit, numerical_rows=numerical_rows)
        description = _bayesian_description(rows)
    else:
        rows = deterministic_quantile_rows(result, settings=settings, unit=unit, numerical_rows=numerical_rows)
        description = _deterministic_description(result, settings, rows)
    observed = _observed_points(raw_ams)
    scale = PlotScale.from_points((*observed, *(_model_plot_point(row) for row in rows)))
    svg = _quantile_svg(observed, rows, scale, unit, description)
    return {
        "frequency.svg": svg,
        "frequency-coordinates.csv": _quantile_csv(observed, rows, scale, unit),
        "report.html": _quantile_html(svg, observed, rows, unit, year_definition, report_text, description),
    }


def render_frequency_files(
    result: AnalysisResult,
    *,
    settings: Mapping[str, object],
    raw_ams: bytes,
    report_text: str,
    numerical_rows: Sequence[Mapping[str, object]] = (),
) -> dict[str, str]:
    """Produce additive static files; no fitting, interval construction or source defaults."""
    unit, year_definition = settings.get("discharge_unit"), settings.get("year_definition")
    if not isinstance(unit, str) or not isinstance(year_definition, str):
        msg = "frequency report requires explicit discharge units and year definition"
        raise ValueError(msg)
    if any(
        not (0x20 <= ord(char) <= 0xD7FF or 0xE000 <= ord(char) <= 0xFFFD or 0x10000 <= ord(char) <= 0x10FFFF)
        for char in unit
    ):
        msg = "discharge unit contains a character forbidden in SVG XML"
        raise ValueError(msg)
    if (
        settings.get("schema_version") in (BAYESIAN_PLOT_SCHEMA, SELECTED_SCHEMA)
        or settings.get("bootstrap_status") == "completed"
    ):
        return _render_extended_frequency(
            result,
            settings=settings,
            raw_ams=raw_ams,
            report_text=report_text,
            unit=unit,
            year_definition=year_definition,
            numerical_rows=numerical_rows,
        )
    if (
        settings.get("schema_version") not in DETERMINISTIC_SCHEMAS
        or settings.get("source_classification") != _EXACT_ASSERTION
    ):
        msg = "frequency report requires an admitted deterministic systematic-AMS settings schema"
        raise ValueError(msg)
    if "bootstrap_status" in settings:
        msg = "frequency report requires a point fit without a bootstrap request"
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
