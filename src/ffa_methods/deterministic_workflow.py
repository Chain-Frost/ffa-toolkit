"""Explicit file-to-fixed-GEV L/LH mathematical fit; never infer Flike defaults.

The Year,AMS CSV is parsed by the existing WP02 evidence reader. Only complete,
positive, exact, equally weighted annual maxima are admitted. The separate
settings document identifies every supported choice and is SHA-256 pinned.
This is a deterministic independently specified calculation, not legacy replay.
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import cast

from ffa_methods import __version__
from ffa_methods.compatibility import LEGACY_FLIKE_PROFILE
from ffa_methods.contracts import (
    AnalysisResult,
    EvidenceStatus,
    ParameterSummary,
    ProbabilityKind,
    ResultMeaning,
    ResultPoint,
    RunDiagnostic,
    RunProvenance,
)
from ffa_methods.estimators.gev_lh import GEVLHFit, fit_gev_lh
from ffa_methods.gev_support import GEVFittedSupport, GEVSupportObservation, diagnose_gev_support
from ffa_methods.legacy.models import ObservationKind
from ffa_methods.legacy.project import parse_ams_csv

SETTINGS_SCHEMA = "ffa-gev-fixed-h/1"
EXACT_ASSERTION = "caller_asserted_complete_exact_positive_annual_maxima"
ALGORITHM_VERSION = "arr-wang-gev-fixed-h/1"
_FIELDS = frozenset(
    {
        "schema_version",
        "estimator",
        "lh_shift",
        "discharge_unit",
        "year_definition",
        "source_classification",
        "non_exceedance_probabilities",
    }
)


@dataclass(frozen=True, slots=True)
class FixedGEVSettings:
    """Required settings: no implicit estimator, year, unit or probability defaults."""

    lh_shift: int
    discharge_unit: str
    year_definition: str
    probabilities: tuple[float, ...]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys, including nested settings keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate settings JSON key: {key}"
            raise ValueError(msg)
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    msg = f"nonstandard numeric constant in settings: {value}"
    raise ValueError(msg)


def _source_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        msg = f"{field} must be a nonempty string without control characters"
        raise ValueError(msg)
    return value


def _ordered_probabilities(raw: object) -> tuple[float, ...]:
    """Reject unsupported, nonfinite, repeated and unordered probability levels."""
    if not isinstance(raw, list):
        msg = "non_exceedance_probabilities must contain 1 to 200 explicit probabilities"
        raise TypeError(msg)
    items = cast(list[object], raw)
    if not 1 <= len(items) <= 200:
        msg = "non_exceedance_probabilities must contain 1 to 200 explicit probabilities"
        raise ValueError(msg)
    probabilities: list[float] = []
    for value in items:
        if type(value) not in (int, float):
            msg = "probabilities must be numeric (not boolean)"
            raise ValueError(msg)
        probability = float(cast(int | float, value))
        if not isfinite(probability) or not 0.0 < probability < 1.0:
            msg = "non-exceedance probabilities must be finite and strictly between zero and one"
            raise ValueError(msg)
        if probabilities and not probability > probabilities[-1]:
            msg = "non-exceedance probabilities must be strictly increasing without duplicates"
            raise ValueError(msg)
        probabilities.append(probability)
    return tuple(probabilities)


def load_fixed_gev_settings(data: bytes) -> FixedGEVSettings:
    """Read JSON without accepting unknown switches, aliases or vendor defaults."""
    try:
        raw: object = json.loads(
            data.decode("utf-8-sig"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"settings must be a valid UTF-8 JSON document: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = "settings JSON root must be an object"
        raise TypeError(msg)
    values = cast(dict[str, object], raw)
    if set(values) != set(_FIELDS):
        msg = f"settings keys must match the fixed-H schema exactly; missing={sorted(_FIELDS - set(values))}, unknown={sorted(set(values) - _FIELDS)}"
        raise ValueError(msg)
    if values["schema_version"] != SETTINGS_SCHEMA or values["estimator"] != "gev_lh_fixed_h":
        msg = "settings require explicit schema_version='ffa-gev-fixed-h/1' and estimator='gev_lh_fixed_h'"
        raise ValueError(msg)
    if values["source_classification"] != EXACT_ASSERTION:
        msg = "caller must explicitly assert complete exact positive annual maxima; no missing/zero/censored admission"
        raise ValueError(msg)
    shift = values["lh_shift"]
    if type(shift) is not int or shift not in range(5):
        msg = "lh_shift must be an integer H=0,1,2,3,4; optimized-H is unavailable"
        raise ValueError(msg)
    unit = _source_text(values["discharge_unit"], field="discharge_unit")
    if unit in {"years", "probability"}:
        msg = "discharge_unit must name a physical flow unit"
        raise ValueError(msg)
    definition = _source_text(values["year_definition"], field="year_definition")
    return FixedGEVSettings(
        lh_shift=shift,
        discharge_unit=unit,
        year_definition=definition,
        probabilities=_ordered_probabilities(values["non_exceedance_probabilities"]),
    )


def _positive_exact_ams_records(data: bytes) -> tuple[GEVSupportObservation, ...]:
    """Refuse non-exact, zero, missing or unrepresentable records; never drop rows."""
    observations = parse_ams_csv(data)
    values: list[GEVSupportObservation] = []
    for record in observations:
        if record.kind != ObservationKind.EXACT or record.value is None or record.value <= 0:
            msg = f"year {record.year} is {record.kind}; only complete exact strictly positive maxima are supported"
            raise ValueError(msg)
        try:
            flow = float(record.value)
        except (OverflowError, ValueError) as exc:
            msg = f"year {record.year} has a discharge not representable as binary64"
            raise ValueError(msg) from exc
        if not isfinite(flow) or flow <= 0:
            msg = f"year {record.year} has a nonfinite/underflowed flow"
            raise ValueError(msg)
        values.append(GEVSupportObservation(record.year, flow))
    return tuple(values)


def _fit_diagnostics(fit: GEVLHFit, n: int, unit: str) -> tuple[RunDiagnostic, ...]:
    """Report actual fixed-H fit diagnostics; fourth ratio is NOT standardized z."""
    points = [RunDiagnostic(name="exact_annual_maxima_count", value=float(n), unit="count")]
    points.extend(
        RunDiagnostic(name=f"sample_lh_moment_{index}", value=value, unit=unit)
        for index, value in enumerate(fit.sample_moments, start=1)
    )
    if fit.lh_kurtosis_residual is not None:
        points.append(
            RunDiagnostic(
                name="raw_lh_fourth_to_second_ratio_residual_unstandardized",
                value=fit.lh_kurtosis_residual,
                unit="dimensionless",
            )
        )
    return tuple(points)


def _support_warnings(support: GEVFittedSupport) -> tuple[str, ...]:
    warnings: list[str] = []
    if support.outside:
        warnings.append(
            f"{len(support.outside)} exact annual maxima outside fitted GEV support: "
            "the algebraic fit is retained, but the fitted distribution cannot describe the full sample. "
            "Outside observations have zero exact continuous likelihood density."
        )
    if support.at_endpoint:
        warnings.append(
            f"{len(support.at_endpoint)} exact annual maxima at the fitted GEV endpoint: "
            "not strictly outside; boundary density/admissibility is separate from the LH fit."
        )
    return tuple(warnings)


def calculate_fixed_gev_from_files(
    ams_path: Path, settings_path: Path
) -> tuple[AnalysisResult, dict[str, str | int | float | bool | None]]:
    """Fit the admitted independent mathematical model and return typed report inputs."""
    raw_ams = ams_path.read_bytes()
    raw_settings = settings_path.read_bytes()
    config = load_fixed_gev_settings(raw_settings)
    records = _positive_exact_ams_records(raw_ams)
    values = tuple(record.discharge for record in records)
    fitted = fit_gev_lh(values, shift=config.lh_shift)
    support = diagnose_gev_support(
        records, location=fitted.location, scale=fitted.scale, shape=fitted.shape, unit=config.discharge_unit
    )
    points = tuple(
        ResultPoint(
            meaning=ResultMeaning.DETERMINISTIC_FITTED_QUANTILE,
            value=fitted.quantile(probability),
            unit=config.discharge_unit,
            input_probability=probability,
            input_probability_kind=ProbabilityKind.NON_EXCEEDANCE,
        )
        for probability in config.probabilities
    )
    parameters = tuple(
        ParameterSummary(name=name, coordinate=coordinate, mean=value, summary_kind="point_estimate")
        for name, coordinate, value in (
            ("GEV location", "arr.gev.location", fitted.location),
            ("GEV scale", "arr.gev.scale", fitted.scale),
            ("GEV shape k", "arr.gev.shape_k", fitted.shape),
        )
    )
    result = AnalysisResult(
        provenance=RunProvenance(
            compatibility_profile=LEGACY_FLIKE_PROFILE.identifier,
            method_id="lh-moments-gev",
            method_version=ALGORITHM_VERSION,
            package_version=__version__,
            evidence_status=EvidenceStatus.OBSERVED_ARTIFACT,
            input_sha256=sha256(raw_ams).hexdigest(),
            settings_sha256=sha256(raw_settings).hexdigest(),
        ),
        parameters=parameters,
        values=points,
        diagnostics=_fit_diagnostics(fitted, len(values), config.discharge_unit),
        fitted_gev_support=support,
        warnings=(
            "Mathematical fixed-H GEV L/LH fit only: NOT Flike-native, Bayesian, bootstrap or optimized-H.",
            "Source classification, year definition and discharge units are caller assertions, not independently verified.",
            "The fourth LH ratio residual is unstandardized and is NOT the Wang optimized-H goodness-of-fit statistic.",
            *_support_warnings(support),
        ),
    )
    settings: dict[str, str | int | float | bool | None] = {
        "schema_version": SETTINGS_SCHEMA,
        "estimator": "gev_lh_fixed_h",
        "lh_shift": config.lh_shift,
        "year_definition": config.year_definition,
        "discharge_unit": config.discharge_unit,
        "source_classification": EXACT_ASSERTION,
        "probability_count": len(config.probabilities),
        "non_exceedance_probabilities": ",".join(str(p) for p in config.probabilities),
    }
    return result, settings
