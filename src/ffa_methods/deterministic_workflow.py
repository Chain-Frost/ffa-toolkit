"""Explicit file-to-L/LH mathematical fits; never infer Flike defaults.

The Year,AMS CSV is parsed by the existing WP02 evidence reader. Only complete,
positive, exact, equally weighted annual maxima are admitted. The separate
settings document identifies every supported choice and is SHA-256 pinned.
This is a deterministic independently specified calculation, not legacy replay.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from math import exp, isfinite, sqrt
from pathlib import Path
from typing import Literal, cast

from ffa_methods import __version__
from ffa_methods.bayesian_workflow import BAYESIAN_SCHEMA, calculate_bayesian
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
from ffa_methods.distributions import GeneralizedPareto, Gumbel, Lognormal, LogPearsonIII
from ffa_methods.estimators.gev_lh import GEVLHFit, fit_gev_lh
from ffa_methods.estimators.gev_lh_calibration import evaluate_candidate_h
from ffa_methods.estimators.ordinary_lmoments import (
    OrdinaryLMomentFit,
    fit_gp_lmoments,
    fit_gumbel_lmoments,
    fit_lognormal_lmoments,
    fit_lp3_lmoments,
)
from ffa_methods.estimators.parametric_bootstrap import (
    BootstrapEndpoint,
    BootstrapSettings,
    Quantile,
    parametric_bootstrap,
)
from ffa_methods.gev_support import GEVFittedSupport, GEVSupportObservation, diagnose_gev_support
from ffa_methods.legacy.models import ObservationKind
from ffa_methods.legacy.project import parse_ams_csv

SETTINGS_SCHEMA = "ffa-gev-fixed-h/1"
SELECTED_SCHEMA = "ffa-gev-selected-h/1"
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


def _settings_object(data: bytes) -> dict[str, object]:
    """Read one settings object without duplicate keys or nonstandard constants."""
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
    return cast(dict[str, object], raw)


def load_fixed_gev_settings(data: bytes) -> FixedGEVSettings:
    """Read JSON without accepting unknown switches, aliases or vendor defaults."""
    return _fixed_gev_settings(_settings_object(data))


def _fixed_gev_settings(values: dict[str, object]) -> FixedGEVSettings:
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
        msg = "lh_shift must be an integer H=0,1,2,3,4 for the fixed-H schema"
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


def _positive_exact_ams(data: bytes) -> tuple[float, ...]:
    """Reuse the source-preserving admission boundary for ordinary-family values."""
    return tuple(record.discharge for record in _positive_exact_ams_records(data))


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
    values = _settings_object(raw_settings)
    bootstrap = _bootstrap_settings(values.pop("bootstrap")) if "bootstrap" in values else None
    return _calculate_fixed_gev(raw_ams, raw_settings, _fixed_gev_settings(values), bootstrap)


def _calculate_fixed_gev(
    raw_ams: bytes, raw_settings: bytes, config: FixedGEVSettings, bootstrap: BootstrapSettings | None = None
) -> tuple[AnalysisResult, dict[str, str | int | float | bool | None]]:
    records = _positive_exact_ams_records(raw_ams)
    values = tuple(record.discharge for record in records)
    fitted = fit_gev_lh(values, shift=config.lh_shift)
    report, settings = _report_fixed_gev(raw_ams, raw_settings, config, records, fitted)
    if bootstrap is not None:
        report = _with_bootstrap(
            report,
            settings,
            bootstrap,
            config.probabilities,
            len(records),
            fitted.quantile,
            lambda record: fit_gev_lh(record, shift=config.lh_shift).quantile,
        )
    return report, settings


def _report_fixed_gev(
    raw_ams: bytes,
    raw_settings: bytes,
    config: FixedGEVSettings,
    records: tuple[GEVSupportObservation, ...],
    fitted: GEVLHFit,
) -> tuple[AnalysisResult, dict[str, str | int | float | bool | None]]:
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
        diagnostics=_fit_diagnostics(fitted, len(records), config.discharge_unit),
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


def _calculate_selected_gev(
    raw_ams: bytes, raw_settings: bytes, settings: dict[str, object], bootstrap: BootstrapSettings | None
) -> tuple[AnalysisResult, dict[str, str | int | float | bool | None]]:
    if (
        frozenset(settings) != (_FIELDS - {"lh_shift"}) | {"critical_z", "calibration_seed"}
        or settings.get("estimator") != "gev_lh_selected_h"
    ):
        msg = "selected-H settings require gev_lh_selected_h, critical_z and calibration_seed in place of lh_shift"
        raise ValueError(msg)
    seed = settings["calibration_seed"]
    if type(seed) is not int or seed < 0:
        msg = "calibration_seed must be a nonnegative integer"
        raise ValueError(msg)
    critical_z = _finite_setting(settings["critical_z"], field="critical_z")
    fixed_settings = {key: val for key, val in settings.items() if key in _FIELDS}
    fixed_settings.update(schema_version=SETTINGS_SCHEMA, estimator="gev_lh_fixed_h", lh_shift=0)
    config = _fixed_gev_settings(fixed_settings)
    records = _positive_exact_ams_records(raw_ams)
    values = tuple(record.discharge for record in records)
    candidates, selected = evaluate_candidate_h(values, critical_z=critical_z, seed=seed)
    candidate_fit = candidates[0 if selected is None else selected].fit
    report, report_settings = _report_fixed_gev(
        raw_ams,
        raw_settings,
        replace(config, lh_shift=candidate_fit.shift, probabilities=() if selected is None else config.probabilities),
        records,
        candidate_fit,
    )
    diagnostics = [RunDiagnostic(name="exact_annual_maxima_count", value=float(len(values)), unit="count")]
    support: list[GEVFittedSupport] = []
    for candidate in candidates:
        fit = candidate.fit
        calibration = candidate.calibration
        support.append(
            diagnose_gev_support(
                records, location=fit.location, scale=fit.scale, shape=fit.shape, unit=config.discharge_unit
            )
        )
        diagnostics.extend(
            RunDiagnostic(name=f"h{fit.shift}_{name}", value=value, unit=unit)
            for name, value, unit in (
                ("location", fit.location, config.discharge_unit),
                ("scale", fit.scale, config.discharge_unit),
                ("shape_k", fit.shape, "dimensionless"),
                ("sample_skew", fit.sample_moments[2] / fit.sample_moments[1], "dimensionless"),
                ("fitted_skew", fit.fitted_moments[2] / fit.fitted_moments[1], "dimensionless"),
                ("sample_kurtosis", fit.sample_moments[3] / fit.sample_moments[1], "dimensionless"),
                ("fitted_kurtosis", fit.fitted_moments[3] / fit.fitted_moments[1], "dimensionless"),
                ("skew_variance", calibration.covariance.skew_variance, "dimensionless"),
                ("kurtosis_variance", calibration.covariance.kurtosis_variance, "dimensionless"),
                ("skew_kurtosis_covariance", calibration.covariance.covariance, "dimensionless"),
                ("conditional_variance", calibration.covariance.conditional_variance, "dimensionless"),
                ("conditional_sd", sqrt(calibration.covariance.conditional_variance), "dimensionless"),
                ("variance_lower", calibration.variance_lower, "dimensionless"),
                ("variance_upper", calibration.variance_upper, "dimensionless"),
                ("calibration_count", float(calibration.covariance.count), "count"),
                ("z", candidate.z, "dimensionless"),
                ("passes_cutoff", float(abs(candidate.z) <= critical_z), "indicator"),
            )
        )
    report_settings.update(
        schema_version=SELECTED_SCHEMA,
        estimator="gev_lh_selected_h",
        lh_shift=selected,
        critical_z=critical_z,
        calibration_seed=str(seed),
        calibration_rng="NumPy PCG64 (base seed + H)",
        non_exceedance_probabilities=",".join(str(p) for p in config.probabilities),
        probability_count=len(config.probabilities),
        selection="lowest_passing_h" if selected is not None else "no_pass",
    )
    selected_support = None if selected is None else support[selected]
    report = replace(
        report,
        provenance=replace(report.provenance, method_version="wang-direct-lh-selected-h/1"),
        parameters=() if selected is None else report.parameters,
        values=() if selected is None else report.values,
        diagnostics=tuple(diagnostics),
        fitted_gev_support=selected_support,
        candidate_gev_support=tuple(support),
        warnings=(
            "Provisional independent approximate Wang diagnostic with direct simulation calibration.",
            "Source classification, year definition and discharge units are caller assertions.",
            "No candidate passed the declared cutoff."
            if selected_support is None
            else f"Lowest passing shift H={selected}.",
            *(_support_warnings(selected_support) if selected_support is not None else ()),
        ),
    )
    if bootstrap is not None:
        if selected is None:
            report_settings["bootstrap_status"] = "unavailable_without_selected_fit"
            report = replace(
                report,
                warnings=(*report.warnings, "Bootstrap unavailable because the original record has no selected fit."),
            )
        else:

            def refit(record: tuple[float, ...]) -> Quantile:
                replicate, selected_h = evaluate_candidate_h(record, critical_z=critical_z, seed=seed)
                if selected_h is None:
                    msg = "no bootstrap candidate passed the declared cutoff"
                    raise ValueError(msg)
                return replicate[selected_h].fit.quantile

            report = _with_bootstrap(
                report, report_settings, bootstrap, config.probabilities, len(records), candidate_fit.quantile, refit
            )
    return report, report_settings


ORDINARY_SCHEMA = "ffa-ordinary-lmoments/1"
_ORDINARY_FIELDS = _FIELDS - {"lh_shift"}
_ORDINARY_ESTIMATORS = frozenset({"gumbel_lmoments", "lognormal_lmoments", "lp3_lmoments", "gp_lmoments"})


@dataclass(frozen=True, slots=True)
class OrdinarySettings:
    """Explicit H=0 family configuration; no implicit reference or GP location."""

    estimator: str
    discharge_unit: str
    year_definition: str
    probabilities: tuple[float, ...]
    reference_discharge: float | None = None
    gp_location_mode: str | None = None
    gp_location: float | None = None


def _finite_setting(value: object, *, field: str) -> float:
    if type(value) not in (int, float):
        msg = f"{field} must be a finite number (not boolean)"
        raise ValueError(msg)
    try:
        result = float(cast(int | float, value))
    except OverflowError as exc:
        msg = f"{field} must be representable as a finite binary64 number"
        raise ValueError(msg) from exc
    if not isfinite(result):
        msg = f"{field} must be finite"
        raise ValueError(msg)
    return result


def _ordinary_extra_fields(estimator: str, values: dict[str, object]) -> frozenset[str]:
    if estimator in ("lognormal_lmoments", "lp3_lmoments"):
        return frozenset({"reference_discharge"})
    if estimator == "gp_lmoments":
        mode = values.get("gp_location_mode")
        if mode == "fixed":
            return frozenset({"gp_location_mode", "gp_location"})
        if mode == "free":
            return frozenset({"gp_location_mode"})
        msg = "gp_location_mode must explicitly be 'fixed' or 'free'"
        raise ValueError(msg)
    return frozenset()


def _ordinary_settings(values: dict[str, object]) -> OrdinarySettings:
    estimator = values.get("estimator")
    if not isinstance(estimator, str) or estimator not in _ORDINARY_ESTIMATORS:
        msg = "ordinary L-moment schema requires an explicit supported estimator"
        raise ValueError(msg)
    expected = _ORDINARY_FIELDS | _ordinary_extra_fields(estimator, values)
    if frozenset(values) != expected:
        msg = f"ordinary settings keys must match exactly; missing={sorted(expected - set(values))}, unknown={sorted(set(values) - expected)}"
        raise ValueError(msg)
    if values["source_classification"] != EXACT_ASSERTION:
        msg = "caller must explicitly assert complete exact positive annual maxima; no missing/zero/censored admission"
        raise ValueError(msg)
    unit = _source_text(values["discharge_unit"], field="discharge_unit")
    if unit in {"years", "probability"}:
        msg = "discharge_unit must name a physical flow unit"
        raise ValueError(msg)
    reference = (
        _finite_setting(values["reference_discharge"], field="reference_discharge")
        if "reference_discharge" in values
        else None
    )
    if reference is not None and reference <= 0:
        msg = "reference_discharge must be strictly positive"
        raise ValueError(msg)
    return OrdinarySettings(
        estimator=estimator,
        discharge_unit=unit,
        year_definition=_source_text(values["year_definition"], field="year_definition"),
        probabilities=_ordered_probabilities(values["non_exceedance_probabilities"]),
        reference_discharge=reference,
        gp_location_mode=cast(str | None, values.get("gp_location_mode")),
        gp_location=_finite_setting(values["gp_location"], field="gp_location") if "gp_location" in values else None,
    )


def _ordinary_fit(
    values: tuple[float, ...], config: OrdinarySettings
) -> (
    OrdinaryLMomentFit[Gumbel]
    | OrdinaryLMomentFit[Lognormal]
    | OrdinaryLMomentFit[LogPearsonIII]
    | OrdinaryLMomentFit[GeneralizedPareto]
):
    if config.estimator == "gumbel_lmoments":
        return fit_gumbel_lmoments(values)
    if config.estimator == "gp_lmoments":
        return fit_gp_lmoments(values, location=config.gp_location)
    reference = config.reference_discharge
    if reference is None:
        msg = "explicit reference_discharge required for lognormal or LP3"
        raise ValueError(msg)
    if config.estimator == "lognormal_lmoments":
        return fit_lognormal_lmoments(values, reference_discharge=reference)
    if config.estimator == "lp3_lmoments":
        return fit_lp3_lmoments(values, reference_discharge=reference)
    msg = "unsupported ordinary estimator"
    raise ValueError(msg)


def _ordinary_parameters(
    kernel: Gumbel | Lognormal | LogPearsonIII | GeneralizedPareto, unit: str
) -> tuple[ParameterSummary, ...]:
    entries: list[tuple[str, str, float]]
    if isinstance(kernel, Gumbel | GeneralizedPareto):
        family = "gumbel" if isinstance(kernel, Gumbel) else "gp"
        entries = [
            (f"Location ({unit})", f"{family}.location", kernel.location),
            (f"Scale ({unit})", f"{family}.scale", exp(kernel.log_scale)),
        ]
        if isinstance(kernel, GeneralizedPareto):
            entries.append(("EVT GP shape xi (ARR kappa = -xi)", "gp.shape_xi", kernel.xi))
    else:
        family = "lp3" if isinstance(kernel, LogPearsonIII) else "lognormal"
        entries = [
            ("Mean of ln(Q/reference_discharge)", f"{family}.mean_loge_flow", kernel.log.mean),
            ("ln(SD of ln(Q/reference_discharge))", f"{family}.loge_sd_loge_flow", kernel.log.log_sd),
        ]
        if isinstance(kernel, LogPearsonIII):
            entries.append(("Skew of ln(Q/reference_discharge)", "lp3.skew_loge_flow", kernel.skew))
    return tuple(
        ParameterSummary(name=name, coordinate=coordinate, mean=value, summary_kind="point_estimate")
        for name, coordinate, value in entries
    )


def _ordinary_report_settings(config: OrdinarySettings, moments: str) -> dict[str, str | int | float | bool | None]:
    settings: dict[str, str | int | float | bool | None] = {
        "schema_version": ORDINARY_SCHEMA,
        "estimator": config.estimator,
        "lh_shift": 0,
        "year_definition": config.year_definition,
        "discharge_unit": config.discharge_unit,
        "source_classification": EXACT_ASSERTION,
        "source_format": "Year,AMS CSV (year_ams_csv)",
        "moment_coordinates": moments,
        "probability_count": len(config.probabilities),
        "non_exceedance_probabilities": ",".join(str(p) for p in config.probabilities),
        "source_ams_file": "source-ams.csv",
        "source_settings_file": "source-settings.json",
    }
    if config.reference_discharge is not None:
        settings["reference_discharge"] = config.reference_discharge
    if config.gp_location_mode is not None:
        settings["gp_location_mode"] = config.gp_location_mode
    if config.gp_location is not None:
        settings["gp_location"] = config.gp_location
    return settings


def _calculate_ordinary(
    raw_ams: bytes, raw_settings: bytes, config: OrdinarySettings, bootstrap: BootstrapSettings | None = None
) -> tuple[AnalysisResult, dict[str, str | int | float | bool | None]]:
    values = _positive_exact_ams(raw_ams)
    fit = _ordinary_fit(values, config)
    points = tuple(
        ResultPoint(
            meaning=ResultMeaning.DETERMINISTIC_FITTED_QUANTILE,
            value=fit.quantile(probability),
            unit=config.discharge_unit,
            input_probability=probability,
            input_probability_kind=ProbabilityKind.NON_EXCEEDANCE,
        )
        for probability in config.probabilities
    )
    moment_unit = config.discharge_unit if fit.moment_coordinates == "discharge" else "dimensionless"
    diagnostics = (
        RunDiagnostic(name="exact_annual_maxima_count", value=float(len(values)), unit="count"),
        *(
            RunDiagnostic(name=f"sample_l_moment_{index}", value=value, unit=moment_unit)
            for index, value in enumerate(fit.sample_moments, start=1)
        ),
    )
    warnings = [
        "Independent H=0 ordinary L-moment fit only: NOT Flike-native, Bayesian, bootstrap or optimized-H.",
        "Source classification, year definition and discharge units are caller assertions, not independently verified.",
    ]
    if config.estimator == "gp_lmoments":
        warnings.append("GP fitted to annual maxima only: this is not a peaks-over-threshold event/exposure model.")
    result = AnalysisResult(
        provenance=RunProvenance(
            compatibility_profile=LEGACY_FLIKE_PROFILE.identifier,
            method_id="l-moments",
            method_version="arr-hosking-ordinary-lmoments/1",
            package_version=__version__,
            evidence_status=EvidenceStatus.OBSERVED_ARTIFACT,
            input_sha256=sha256(raw_ams).hexdigest(),
            settings_sha256=sha256(raw_settings).hexdigest(),
        ),
        parameters=_ordinary_parameters(fit.distribution, config.discharge_unit),
        values=points,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
    )
    settings = _ordinary_report_settings(config, fit.moment_coordinates)
    if bootstrap is not None:
        result = _with_bootstrap(
            result,
            settings,
            bootstrap,
            config.probabilities,
            len(values),
            fit.quantile,
            lambda record: _ordinary_fit(record, config).quantile,
        )
    return result, settings


def _bootstrap_settings(raw: object) -> BootstrapSettings:
    expected = {"replications", "seed", "confidence", "maximum_failure_fraction", "endpoint_tolerance"}
    if not isinstance(raw, dict) or set(cast(dict[object, object], raw)) != expected:
        msg = f"bootstrap settings require exactly {sorted(expected)}"
        raise ValueError(msg)
    values = cast(dict[str, object], raw)
    replications, seed = values["replications"], values["seed"]
    confidence = _finite_setting(values["confidence"], field="bootstrap confidence")
    failures = _finite_setting(values["maximum_failure_fraction"], field="maximum_failure_fraction")
    tolerance = _finite_setting(values["endpoint_tolerance"], field="endpoint_tolerance")
    if (
        type(replications) is not int
        or not 1 <= replications <= 100_000
        or type(seed) is not int
        or seed < 0
        or not 0 < confidence < 1
        or not 0 <= failures < 1
        or tolerance <= 0
    ):
        msg = "invalid bootstrap count, seed, confidence, failure fraction or endpoint tolerance"
        raise ValueError(msg)
    return BootstrapSettings(replications, seed, confidence, failures, tolerance)


def _with_bootstrap(
    report: AnalysisResult,
    report_settings: dict[str, str | int | float | bool | None],
    settings: BootstrapSettings,
    probabilities: tuple[float, ...],
    count: int,
    generate: Quantile,
    refit: Callable[[tuple[float, ...]], Quantile],
) -> AnalysisResult:
    result = parametric_bootstrap(generate, refit, count=count, probabilities=probabilities, settings=settings)
    failed = sum(count for _, count in result.failures)
    report_settings.update(
        bootstrap_status="completed",
        bootstrap_replications=settings.replications,
        bootstrap_seed=str(settings.seed),
        bootstrap_confidence=settings.confidence,
        bootstrap_maximum_failure_fraction=settings.maximum_failure_fraction,
        bootstrap_endpoint_tolerance=settings.endpoint_tolerance,
        bootstrap_successful=settings.replications - failed,
        bootstrap_failed=failed,
        bootstrap_failure_reasons=json.dumps(dict(result.failures)),
    )
    points = list(report.values)
    diagnostics = list(report.diagnostics)
    unit = report.values[0].unit
    for f, (low, high) in zip(probabilities, result.endpoints, strict=True):
        endpoints: tuple[tuple[Literal["lower", "upper"], BootstrapEndpoint], ...] = (("lower", low), ("upper", high))
        for side, endpoint in endpoints:
            points.append(
                ResultPoint(
                    meaning=ResultMeaning.QUANTILE_BOOTSTRAP_LIMIT,
                    value=endpoint.estimate,
                    unit=unit,
                    input_probability=f,
                    input_probability_kind=ProbabilityKind.NON_EXCEEDANCE,
                    confidence_level=settings.confidence,
                    probability_limit_side=side,
                    probability_limit_construction="parametric_percentile_type_7_successful_refits",
                )
            )
            diagnostics.extend(
                RunDiagnostic(name=f"bootstrap_f={f}_{side}_mc_{bound}", value=value, unit=unit)
                for bound, value in (("lower", endpoint.mc_lower), ("upper", endpoint.mc_upper))
            )
    return replace(
        report,
        values=tuple(points),
        diagnostics=tuple(diagnostics),
        warnings=(
            *(warning for warning in report.warnings if "fit only" not in warning),
            "Bootstrap sampling intervals are conditional on successful refits. Endpoint Monte Carlo brackets describe simulation error.",
        ),
    )


def calculate_from_bytes(
    raw_ams: bytes, raw_settings: bytes
) -> tuple[AnalysisResult, dict[str, str | int | float | bool | None]]:
    """Single explicit schema dispatcher; consumed bytes are also the hashed bytes."""
    values = _settings_object(raw_settings)
    if values.get("schema_version") == BAYESIAN_SCHEMA:
        return calculate_bayesian(raw_ams, raw_settings, values)
    bootstrap = _bootstrap_settings(values.pop("bootstrap")) if "bootstrap" in values else None
    if values.get("schema_version") == ORDINARY_SCHEMA:
        return _calculate_ordinary(raw_ams, raw_settings, _ordinary_settings(values), bootstrap)
    if values.get("schema_version") == SELECTED_SCHEMA:
        return _calculate_selected_gev(raw_ams, raw_settings, values, bootstrap)
    return _calculate_fixed_gev(raw_ams, raw_settings, _fixed_gev_settings(values), bootstrap)
