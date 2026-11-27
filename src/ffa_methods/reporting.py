"""Internal deterministic, non-qualifying numerical report export for future CLI analysis.

This module formats a completed AnalysisResult; it performs no estimation and must
not be exposed as an independent unverified-input-to-analysis CLI command.
"""

import csv
import json
import shutil
import tempfile
from collections.abc import Mapping
from hashlib import sha256
from io import StringIO
from math import isfinite
from pathlib import Path
from typing import TypeAlias, cast

from ffa_methods import __version__
from ffa_methods.compatibility import LEGACY_FLIKE_PROFILE
from ffa_methods.contracts import (
    AnalysisResult,
    ParameterSummary,
    ProbabilityKind,
    ResultMeaning,
    ResultPoint,
    RunDiagnostic,
)
from ffa_methods.frequency_reporting import render_frequency_files
from ffa_methods.gev_support import GEVFittedSupport
from ffa_methods.guidance import get_method, references

ReportScalar: TypeAlias = str | int | float | bool | None

SCHEMA_VERSION = "ffa-analysis-report/1"
OUTPUT_FILENAMES = ("results.json", "results.csv", "report.txt")

MODELLED_DISCHARGE_MEANINGS: frozenset[ResultMeaning] = frozenset(
    {
        ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS,
        ResultMeaning.DETERMINISTIC_FITTED_QUANTILE,
        ResultMeaning.POSTERIOR_MEAN_QUANTILE,
        ResultMeaning.POSTERIOR_MEDIAN_QUANTILE,
        ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE,
        ResultMeaning.QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT,
    }
)

RESULT_MEANINGS: dict[ResultMeaning, str] = {
    ResultMeaning.DETERMINISTIC_FITTED_QUANTILE: (
        "Quantile of explicitly configured fixed-H GEV L/LH point estimate; "
        "not Bayesian, posterior predictive, bootstrapped or native Flike-qualified."
    ),
    ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS: (
        "Quantile at expected source-coordinate parameters; not the expected quantile."
    ),
    ResultMeaning.POSTERIOR_MEAN_QUANTILE: (
        "Posterior mean of per-parameter quantiles; not the quantile at expected parameters."
    ),
    ResultMeaning.POSTERIOR_MEDIAN_QUANTILE: (
        "Posterior median of per-parameter quantiles; not the mean or predictive quantile."
    ),
    ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE: (
        "Generalized inverse of the posterior-averaged CDF; not the mean of parameter-wise quantiles."
    ),
    ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY: (
        "Expected non-exceedance probability F at a given discharge; AEP is 1-F, not F."
    ),
    ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY: (
        "Expected annual exceedance probability E[S(q|theta)] at a fixed discharge."
    ),
    ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP: (
        "Return period 1/E[S(q|theta)]; not posterior mean of per-parameter return periods."
    ),
    ResultMeaning.RETURN_PERIOD_MONTE_CARLO_PROBABILITY_LIMIT: (
        "Pointwise posterior empirical limit of component return periods 1/S(q|theta); "
        "not a confidence interval on the reciprocal of expected AEP."
    ),
    ResultMeaning.QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT: (
        "Quantile Monte Carlo probability limit; side, level and construction must be stated; "
        "not automatically a bootstrap interval or simultaneous band."
    ),
    ResultMeaning.LOG10_QUANTILE_POSTERIOR_MOMENT: (
        "Specified posterior mean, standard deviation or variance of log10(discharge); "
        "distinct from natural-log distribution parameters."
    ),
}


def _text(value: str, *, label: str) -> str:
    if not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
        msg = f"{label} must be nonempty and contain no control characters"
        raise ValueError(msg)
    return value


def _number(value: float, *, label: str) -> float:
    if not isfinite(value):
        msg = f"{label} must be finite; undefined/infinite numerical results need a typed state"
        raise ValueError(msg)
    return value


def _hash(value: str | None, *, label: str) -> str | None:
    if value is not None and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
        msg = f"{label} must be a lowercase SHA-256 hex digest"
        raise ValueError(msg)
    return value


def _record_probability_context(point: ResultPoint, record: dict[str, object]) -> None:
    """Carry both F and AEP; neither is silently relabelled as the other."""
    if (point.input_probability is None) != (point.input_probability_kind is None):
        msg = "input probability and its F/AEP kind must be provided together"
        raise ValueError(msg)
    if point.input_probability is None:
        return
    probability = _number(point.input_probability, label="input probability")
    if not 0.0 <= probability <= 1.0:
        msg = "input probability must be within [0, 1]"
        raise ValueError(msg)
    kind = point.input_probability_kind
    if kind == ProbabilityKind.NON_EXCEEDANCE:
        f, aep = probability, 1.0 - probability
    elif kind == ProbabilityKind.ANNUAL_EXCEEDANCE:
        f, aep = 1.0 - probability, probability
    else:
        msg = "unrecognised input probability kind"
        raise ValueError(msg)
    record.update(
        input_probability=probability,
        input_probability_kind=kind.value,
        input_non_exceedance_probability=f,
        input_annual_exceedance_probability=aep,
    )


def _record_discharge_and_level(point: ResultPoint, record: dict[str, object]) -> None:
    """Retain input-discharge and probability-limit metadata only when supplied."""
    if point.input_discharge is not None:
        discharge = _number(point.input_discharge, label="input discharge")
        if discharge < 0:
            msg = "input discharge must be nonnegative"
            raise ValueError(msg)
        record["input_discharge"] = discharge
    if point.input_discharge_unit is not None:
        record["input_discharge_unit"] = _text(point.input_discharge_unit, label="input discharge unit")
    if point.confidence_level is not None:
        level = _number(point.confidence_level, label="reported probability level")
        if not 0.0 < level < 1.0:
            msg = "reported probability level must be strictly between 0 and 1"
            raise ValueError(msg)
        record["reported_probability_level"] = level


def _validate_quantile_context(point: ResultPoint) -> None:
    if point.input_probability is None:
        msg = f"{point.meaning.value} requires an input probability with F/AEP kind"
        raise ValueError(msg)
    if point.input_discharge is not None or point.input_discharge_unit is not None:
        msg = "quantile outputs cannot also declare an input discharge"
        raise ValueError(msg)
    if point.unit == "probability":
        msg = "quantile output unit must be a discharge or a specified log10 discharge unit"
        raise ValueError(msg)


def _validate_discharge_context(point: ResultPoint) -> None:
    if point.input_discharge is None or point.input_discharge_unit is None:
        msg = "probability/return-period at discharge requires input discharge and discharge unit"
        raise ValueError(msg)
    if point.input_probability is not None:
        msg = "probability/return-period at discharge must not also specify an input probability"
        raise ValueError(msg)
    if point.meaning in (
        ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY,
        ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY,
    ):
        if point.unit != "probability":
            msg = "expected F/AEP must be labelled with unit 'probability'"
            raise ValueError(msg)
    elif point.unit != "years":
        msg = "return-period output requires unit 'years'"
        raise ValueError(msg)


def _validate_point_context(point: ResultPoint) -> None:
    if point.meaning in (
        ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY,
        ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY,
        ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP,
        ResultMeaning.RETURN_PERIOD_MONTE_CARLO_PROBABILITY_LIMIT,
    ):
        _validate_discharge_context(point)
    elif point.meaning in (
        ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS,
        ResultMeaning.DETERMINISTIC_FITTED_QUANTILE,
        ResultMeaning.POSTERIOR_MEAN_QUANTILE,
        ResultMeaning.POSTERIOR_MEDIAN_QUANTILE,
        ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE,
        ResultMeaning.QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT,
        ResultMeaning.LOG10_QUANTILE_POSTERIOR_MOMENT,
    ):
        _validate_quantile_context(point)
    else:
        msg = f"unsupported result meaning: {point.meaning}"
        raise ValueError(msg)


def _record_probability_limit(point: ResultPoint, record: dict[str, object]) -> None:
    if point.meaning in (
        ResultMeaning.QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT,
        ResultMeaning.RETURN_PERIOD_MONTE_CARLO_PROBABILITY_LIMIT,
    ):
        if (
            point.confidence_level is None
            or point.probability_limit_side not in ("lower", "upper")
            or point.probability_limit_construction is None
        ):
            msg = "probability limit requires level, lower/upper side and construction"
            raise ValueError(msg)
        record["probability_limit_side"] = point.probability_limit_side
        record["probability_limit_construction"] = _text(
            point.probability_limit_construction, label="probability limit construction"
        )
    elif (
        point.confidence_level is not None
        or point.probability_limit_side is not None
        or point.probability_limit_construction is not None
    ):
        msg = "probability limit metadata is invalid for a non-limit result"
        raise ValueError(msg)


def _record_log10_moment(point: ResultPoint, record: dict[str, object], value: float) -> None:
    if point.meaning == ResultMeaning.LOG10_QUANTILE_POSTERIOR_MOMENT:
        if point.log10_moment_kind not in ("mean", "standard_deviation", "variance"):
            msg = "log10 quantile posterior moment requires its mean/standard_deviation/variance kind"
            raise ValueError(msg)
        if not point.unit.startswith("log10("):
            msg = "log10 posterior moment requires a labelled log10 discharge unit"
            raise ValueError(msg)
        if point.log10_moment_kind != "mean" and value < 0:
            msg = "log10 posterior standard deviation or variance must be nonnegative"
            raise ValueError(msg)
        record["log10_moment_kind"] = point.log10_moment_kind
    elif point.log10_moment_kind is not None:
        msg = "log10 moment kind is invalid for a non-moment result"
        raise ValueError(msg)


def _result_interpretation(meaning: ResultMeaning, method_id: str) -> str:
    if meaning == ResultMeaning.DETERMINISTIC_FITTED_QUANTILE and method_id == "l-moments":
        return (
            "Quantile of explicitly configured ordinary H=0 L-moment point estimate; "
            "not Bayesian, posterior predictive, bootstrapped or native Flike-qualified."
        )
    return RESULT_MEANINGS[meaning]


def _point(point: ResultPoint, *, method_id: str) -> dict[str, object]:
    """Reject incomplete quantity labels before any JSON, CSV or prose export."""
    value = _number(point.value, label="result value")
    if (
        point.meaning
        in (
            ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY,
            ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY,
        )
        and not 0.0 <= value <= 1.0
    ):
        msg = "expected non-exceedance/exceedance probability must be within [0, 1]"
        raise ValueError(msg)
    if (
        point.meaning
        in (
            ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP,
            ResultMeaning.RETURN_PERIOD_MONTE_CARLO_PROBABILITY_LIMIT,
        )
        and value < 1.0
    ):
        msg = "annual return period must be at least one year"
        raise ValueError(msg)
    record: dict[str, object] = {
        "meaning": point.meaning.value,
        "interpretation": _result_interpretation(point.meaning, method_id),
        "value": value,
        "unit": _text(point.unit, label="result unit"),
        "input_probability": None,
        "input_probability_kind": None,
        "input_non_exceedance_probability": None,
        "input_annual_exceedance_probability": None,
        "input_discharge": None,
        "input_discharge_unit": None,
        "reported_probability_level": None,
        "probability_limit_side": None,
        "probability_limit_construction": None,
        "log10_moment_kind": None,
    }
    _record_probability_context(point, record)
    _record_discharge_and_level(point, record)
    _validate_point_context(point)
    _record_probability_limit(point, record)
    _record_log10_moment(point, record, value)
    return record


def _report_settings(settings: Mapping[str, object]) -> dict[str, ReportScalar]:
    """Check and order explicit run settings without inventing target defaults."""
    settings_data: dict[str, ReportScalar] = {}
    for key, value in sorted(settings.items()):
        _text(key, label="setting name")
        if not (value is None or isinstance(value, str | int | float | bool)):
            msg = f"setting {key!r} is not a JSON scalar"
            raise TypeError(msg)
        if isinstance(value, float):
            _number(value, label=f"setting {key!r}")
        if isinstance(value, str):
            _text(value, label=f"setting {key!r}")
        settings_data[key] = value
    return settings_data


def _parameter(parameter: ParameterSummary) -> dict[str, object]:
    """Preserve deterministic point estimates separately from posterior means."""
    if parameter.summary_kind not in ("posterior_mean", "point_estimate"):
        msg = "unrecognised parameter summary kind"
        raise ValueError(msg)
    if parameter.standard_deviation is not None and parameter.standard_deviation < 0:
        msg = "parameter standard deviation cannot be negative"
        raise ValueError(msg)
    if parameter.summary_kind == "point_estimate" and parameter.standard_deviation is not None:
        msg = "point estimate cannot be labelled with a posterior standard deviation"
        raise ValueError(msg)
    value = _number(parameter.mean, label="parameter value")
    return {
        "name": _text(parameter.name, label="parameter name"),
        "coordinate": _text(str(parameter.coordinate), label="parameter coordinate"),
        "summary_kind": parameter.summary_kind,
        "mean": value if parameter.summary_kind == "posterior_mean" else None,
        "point_estimate": value if parameter.summary_kind == "point_estimate" else None,
        "standard_deviation": (
            _number(parameter.standard_deviation, label="parameter standard deviation")
            if parameter.standard_deviation is not None
            else None
        ),
    }


def _diagnostic(diagnostic: RunDiagnostic) -> dict[str, object]:
    """Keep computed diagnostics apart from user-controlled run settings."""
    return {
        "name": _text(diagnostic.name, label="diagnostic name"),
        "value": _number(diagnostic.value, label="diagnostic value"),
        "unit": _text(diagnostic.unit, label="diagnostic unit"),
    }


def _citations(method: dict[str, object]) -> dict[str, str]:
    """Embed original citation register entries for a standalone offline report."""
    keys = method.get("source_keys")
    if not isinstance(keys, list):
        msg = "method does not supply a valid source reference key list"
        raise TypeError(msg)
    raw_keys = cast(list[object], keys)
    if any(not isinstance(key, str) for key in raw_keys):
        msg = "method does not supply a valid source reference key list"
        raise TypeError(msg)
    return {key: references(key) for key in cast(list[str], keys)}


def _nonphysical_quantile_warning(point: ResultPoint) -> str:
    """A signed mathematical quantile is not a feasible physical discharge."""
    return (
        f"Modelled {point.meaning.value} is negative ({point.value:.17g} {point.unit}): "
        "this is a mathematical distribution output, not a physically possible discharge; "
        "review model support and engineering suitability."
    )


def _fitted_support_fields(support: GEVFittedSupport | None) -> dict[str, object]:
    """Serialize the optional fixed-GEV diagnostic without changing other report schemas."""
    if support is None:
        return {}
    return {
        "gev_fitted_support": {
            "kind": support.kind,
            "endpoint_exact": support.endpoint_exact,
            "endpoint_approximate": support.endpoint_approximate,
            "unit": support.unit,
            "observation_count": support.observation_count,
            "outside_count": len(support.outside),
            "at_endpoint_count": len(support.at_endpoint),
            "outside": [{"year": row.year, "discharge": row.discharge} for row in support.outside],
            "at_endpoint": [{"year": row.year, "discharge": row.discharge} for row in support.at_endpoint],
            "comparison": "exact rational endpoint of fitted binary64 coordinates and admitted binary64 flows",
            "interpretation": (
                "The algebraic L/LH fit is retained. Strictly outside observations have zero exact continuous "
                "density; endpoint equality is distinct and does not certify likelihood admissibility. "
                "This is not the Wang goodness-of-fit statistic or an H-selection rule."
            ),
        }
    }


def _fitted_support_text(support: GEVFittedSupport | None) -> list[str]:
    if support is None:
        return []
    lines = [
        "",
        "FITTED GEV SUPPORT (not an H-selection or likelihood test)",
        f"  Kind: {support.kind}",
        f"  Exact endpoint (rational): {support.endpoint_exact} {support.unit}",
        f"  Approximate endpoint (binary64; None if unavailable): {support.endpoint_approximate} {support.unit}",
        (
            f"  Observations: {support.observation_count}; outside: {len(support.outside)}; "
            f"at endpoint: {len(support.at_endpoint)}"
        ),
    ]
    for status, rows in (("outside", support.outside), ("at_endpoint", support.at_endpoint)):
        lines.extend(f"  {status}: year={row.year}, discharge={row.discharge!r} {support.unit}" for row in rows)
    return lines


def report_document(result: AnalysisResult, *, settings: Mapping[str, object]) -> dict[str, object]:
    """Prepare replayable, finite JSON data from an already computed result."""
    provenance = result.provenance
    if provenance.compatibility_profile != LEGACY_FLIKE_PROFILE.identifier:
        msg = "result uses a different or unspecified legacy compatibility profile"
        raise ValueError(msg)
    method_id = _text(provenance.method_id, label="method ID")
    method = get_method(method_id)
    for label, value in (
        ("method version", provenance.method_version),
        ("package version", provenance.package_version),
    ):
        _text(value, label=label)
    if provenance.package_version != __version__:
        msg = "provenance package version does not match the installed package"
        raise ValueError(msg)
    settings_data = _report_settings(settings)
    if not result.values:
        msg = "cannot export an analysis with no numerical result points"
        raise ValueError(msg)
    parameters = [_parameter(parameter) for parameter in result.parameters]
    diagnostics = [_diagnostic(diagnostic) for diagnostic in result.diagnostics]
    points = [_point(point, method_id=method_id) for point in result.values]
    warnings = [_text(warning, label="warning") for warning in result.warnings]
    warnings.extend(
        _nonphysical_quantile_warning(point)
        for point in result.values
        if point.meaning in MODELLED_DISCHARGE_MEANINGS and point.value < 0
    )
    if provenance.input_sha256 is None:
        warnings.append("Input SHA-256 is absent: this file does not identify immutable input bytes.")
    if provenance.settings_sha256 is None:
        warnings.append("Settings SHA-256 is absent: the exact analysis settings file cannot be replayed.")
    if not settings_data:
        warnings.append("No run settings supplied: this report cannot reproduce the analysis configuration.")
    warnings.append(
        "This formatter does not verify numerical correctness, legacy compatibility or engineering suitability."
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "formatting_status": "completed",
        "method_qualification": "not_established_by_formatter",
        "legacy_comparison_status": "not_established_by_formatter",
        "compatibility_target": {
            "profile": LEGACY_FLIKE_PROFILE.identifier,
            "legacy_program_version": LEGACY_FLIKE_PROFILE.legacy_program_version,
            "legacy_file_version": LEGACY_FLIKE_PROFILE.legacy_file_version,
        },
        "provenance": {
            "method_id": method_id,
            "method_version": provenance.method_version,
            "package_version": provenance.package_version,
            "reference_evidence_status": provenance.evidence_status.value,
            "input_sha256": _hash(provenance.input_sha256, label="input hash"),
            "settings_sha256": _hash(provenance.settings_sha256, label="settings hash"),
        },
        "method_guidance": {
            "register_implementation_status": method["implementation_status"],
            "register_qualification_status": method["qualification_status"],
            "source_keys": method.get("source_keys", []),
            "source_location": method.get("source_location"),
            "specification": method.get("documentation"),
            "reference_register": "resources/references.md",
            "reference_entries": _citations(method),
        },
        "settings": settings_data,
        "parameters": parameters,
        "diagnostics": diagnostics,
        "results": points,
        "warnings": warnings,
        **_fitted_support_fields(result.fitted_gev_support),
    }


def _display(value: object) -> str:
    """Use compact engineering display precision in prose; JSON/CSV retain full floats."""
    return format(value, ".12g") if isinstance(value, float) else str(value)


def _parameter_text_lines(params: list[dict[str, object]]) -> list[str]:
    """Render deterministic point estimates without an invented posterior mean."""
    if not params:
        return ["  None supplied"]
    lines: list[str] = []
    for parameter in params:
        if parameter["summary_kind"] == "point_estimate":
            lines.append(
                f"  {parameter['name']} [{parameter['coordinate']}] point_estimate={parameter['point_estimate']}"
            )
        else:
            lines.append(
                f"  {parameter['name']} [{parameter['coordinate']}] "
                f"mean={parameter['mean']}, sd={parameter['standard_deviation']}"
            )
    return lines


def render_report_files(result: AnalysisResult, *, settings: Mapping[str, object]) -> dict[str, str]:
    """Return a deterministic JSON, tabular CSV and text presentation of identical values."""
    doc = report_document(result, settings=settings)
    typed_points = cast(list[dict[str, object]], doc["results"])
    keys = (
        "meaning",
        "value",
        "unit",
        "input_probability",
        "input_probability_kind",
        "input_non_exceedance_probability",
        "input_annual_exceedance_probability",
        "input_discharge",
        "input_discharge_unit",
        "reported_probability_level",
        "probability_limit_side",
        "probability_limit_construction",
        "log10_moment_kind",
    )
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    support = doc.get("gev_fitted_support")
    support_header = ("gev_fitted_support_json",) if support is not None else ()
    support_cell = (json.dumps(support, ensure_ascii=False, allow_nan=False),) if support is not None else ()
    writer.writerow(
        (
            "method_id",
            "method_version",
            "method_qualification",
            "legacy_comparison_status",
            "reference_evidence_status",
            *keys,
            *support_header,
        )
    )
    provenance = cast(dict[str, object], doc["provenance"])
    for point in typed_points:
        writer.writerow(
            (
                provenance["method_id"],
                provenance["method_version"],
                doc["method_qualification"],
                doc["legacy_comparison_status"],
                provenance["reference_evidence_status"],
                *("" if point[key] is None else str(point[key]) for key in keys),
                *support_cell,
            )
        )
    method_guidance = cast(dict[str, object], doc["method_guidance"])
    lines = [
        "FFA ANALYSIS RESULT — UNQUALIFIED FORMATTER OUTPUT",
        "This is not a verified legacy Flike comparison or an adopted engineering estimate.",
        f"Compatibility target: {LEGACY_FLIKE_PROFILE.identifier}",
        f"Method: {provenance['method_id']} (algorithm {provenance['method_version']})",
        f"Package version: {provenance['package_version']}",
        f"Reference case evidence status: {provenance['reference_evidence_status']} (not a solver qualification)",
        f"Input SHA-256: {provenance['input_sha256'] or 'not supplied'}",
        f"Settings SHA-256: {provenance['settings_sha256'] or 'not supplied'}",
        f"Method source keys: {method_guidance['source_keys']}",
        f"Method source location: {method_guidance['source_location']}",
        "Full citations: packaged resources/references.md; method guidance: ffa explain METHOD_ID",
        "",
        "RUN SETTINGS",
    ]
    for key, value in cast(dict[str, ReportScalar], doc["settings"]).items():
        lines.append(f"  {key}: {value!r}")
    lines.extend(("", "PARAMETERS (reported source coordinates)"))
    params = cast(list[dict[str, object]], doc["parameters"])
    lines.extend(_parameter_text_lines(params))
    lines.extend(("", "COMPUTED DIAGNOSTICS (not run settings)"))
    diagnostics = cast(list[dict[str, object]], doc["diagnostics"])
    if not diagnostics:
        lines.append("  None supplied")
    lines.extend(f"  {item['name']}: {item['value']} {item['unit']}" for item in diagnostics)
    lines.extend(_fitted_support_text(result.fitted_gev_support))
    lines.extend(("", "RESULT POINTS"))
    for point in typed_points:
        components = [
            f"{point['meaning']} = {point['value']} {point['unit']}",
            f"meaning: {point['interpretation']}",
        ]
        if point["input_probability"] is not None:
            components.append(
                f"input {point['input_probability_kind']}={point['input_probability']}; "
                f"F={_display(point['input_non_exceedance_probability'])}; "
                f"AEP={_display(point['input_annual_exceedance_probability'])}"
            )
        if point["input_discharge"] is not None:
            components.append(f"input discharge={point['input_discharge']} {point['input_discharge_unit']}")
        if point["reported_probability_level"] is not None:
            components.append(
                f"probability limit: {point['probability_limit_side']} "
                f"at level={point['reported_probability_level']}; "
                f"construction={point['probability_limit_construction']}"
            )
        if point["log10_moment_kind"] is not None:
            components.append(f"log10 moment={point['log10_moment_kind']}")
        lines.append("  " + " | ".join(components))
    lines.extend(("", "WARNINGS"))
    warnings = cast(list[str], doc["warnings"])
    lines.extend(f"  - {warning}" for warning in warnings)
    lines.extend(("", "SOURCE REFERENCES (bundled source-register entries)"))
    citations = cast(dict[str, str], method_guidance["reference_entries"])
    lines.extend(citation.rstrip() for citation in citations.values())
    return {
        "results.json": json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        "results.csv": stream.getvalue(),
        "report.txt": "\n".join(lines) + "\n",
    }


def _write_source_inputs(staging: Path, result: AnalysisResult, source_inputs: tuple[bytes, bytes]) -> None:
    ams, settings = source_inputs
    if (
        sha256(ams).hexdigest() != result.provenance.input_sha256
        or sha256(settings).hexdigest() != result.provenance.settings_sha256
    ):
        msg = "source snapshot bytes do not match analysis provenance"
        raise ValueError(msg)
    (staging / "source-ams.csv").write_bytes(ams)
    (staging / "source-settings.json").write_bytes(settings)


def write_report_bundle(
    result: AnalysisResult,
    destination: Path,
    *,
    settings: Mapping[str, object],
    source_inputs: tuple[bytes, bytes] | None = None,
    frequency_report: bool = False,
) -> tuple[Path, Path, Path]:
    """Publish a staged bundle without silently overwriting an existing directory.

    A new directory is required. Files are staged beside the final destination, then
    the directory is renamed so consumers do not observe partially generated files.
    """
    files = render_report_files(result, settings=settings)
    if frequency_report:
        if source_inputs is None:
            msg = "frequency report requires immutable source inputs"
            raise ValueError(msg)
        files.update(
            render_frequency_files(result, settings=settings, raw_ams=source_inputs[0], report_text=files["report.txt"])
        )
    target = destination.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        msg = f"output directory already exists: {destination}"
        raise FileExistsError(msg)
    if not target.parent.is_dir():
        msg = f"output parent directory does not exist: {target.parent}"
        raise FileNotFoundError(msg)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for filename, content in files.items():
            (staging / filename).write_text(content, encoding="utf-8", newline="\n")
        if source_inputs is not None:
            _write_source_inputs(staging, result, source_inputs)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return (target / OUTPUT_FILENAMES[0], target / OUTPUT_FILENAMES[1], target / OUTPUT_FILENAMES[2])
