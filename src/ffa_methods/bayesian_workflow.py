"""Explicit lognormal Bayesian file-to-report orchestration (independent, not native)."""

import csv
import json
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from hashlib import sha256
from io import StringIO
from itertools import pairwise
from math import isfinite
from re import fullmatch
from typing import cast

from ffa_methods import __version__
from ffa_methods.bayesian_outputs import probability_outputs, quantile_outputs
from ffa_methods.compatibility import LEGACY_FLIKE_PROFILE
from ffa_methods.contracts import (
    AnalysisResult,
    AnnualObservation,
    EvidenceStatus,
    ExposureKind,
    ObservationKind,
    ObservationSeries,
    ResultMeaning,
    RunDiagnostic,
    RunProvenance,
)
from ffa_methods.family_posterior import PosteriorFamily, exact_family_series
from ffa_methods.importance_precision import PrecisionPlan, PrecisionRequest, audit_lognormal_precision
from ffa_methods.importance_proposal import CurvatureControls, build_curvature_proposal
from ffa_methods.importance_sampling import ExplicitSamplingControls, independent_importance_sample
from ffa_methods.lognormal_posterior import (
    LOG_SD_COORDINATE,
    MEAN_COORDINATE,
    SOURCE_CLASSIFICATION,
    ClassifiedLognormalPosterior,
    bind_exact_lognormal_csv,
)
from ffa_methods.priors import JointGaussianPrior, LegacyPriorMode
from ffa_methods.shape_posterior import BoundedShapePrior, ClassifiedShapePosterior, execute_shape_posterior
from ffa_methods.wp06_report_bridge import wp06_output_analysis_result

BAYESIAN_SCHEMA = "ffa-lognormal-bayesian/1"
SHAPE_SCHEMA = "ffa-bounded-shape-bayesian/1"
BAYESIAN_SCHEMAS = (BAYESIAN_SCHEMA, SHAPE_SCHEMA)
METHOD_VERSION = "explicit-lognormal-bayesian-cli/1"
_FIELDS = frozenset(
    {
        "schema_version",
        "estimator",
        "source_format",
        "source",
        "source_classification",
        "discharge_unit",
        "year_definition",
        "reference_discharge",
        "fixed_log_sd",
        "prior",
        "proposal",
        "sampling",
        "outputs",
        "precision",
    }
)


def _object(value: object, fields: frozenset[str], *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"{name} must be a JSON object"
        raise TypeError(msg)
    result = cast(dict[str, object], value)
    if frozenset(result) != fields:
        msg = f"{name} keys must match exactly; missing={sorted(fields - set(result))}, unknown={sorted(set(result) - fields)}"
        raise ValueError(msg)
    return result


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        msg = f"{name} must be a nonempty string without control characters"
        raise ValueError(msg)
    return value


def _number(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        msg = f"{name} must be a finite number (not boolean)"
        raise ValueError(msg)
    result = float(cast(int | float, value))
    if not isfinite(result):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return result


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        msg = f"{name} must be an integer (not boolean)"
        raise ValueError(msg)
    return value


def _items(value: object, *, name: str, maximum: int = 200) -> list[object]:
    if not isinstance(value, list) or len(cast(list[object], value)) > maximum:
        msg = f"{name} must be an array of at most {maximum} entries"
        raise ValueError(msg)
    return cast(list[object], value)


def _vector(value: object, *, name: str, maximum: int = 2) -> tuple[float, ...]:
    return tuple(_number(item, name=name) for item in _items(value, name=name, maximum=maximum))


def _optional_number(value: object, *, name: str) -> float | None:
    return None if value is None else _number(value, name=name)


def _level(value: object, *, name: str) -> float:
    result = _number(value, name=name)
    if not 0 < result < 1:
        msg = f"{name} must be strictly between zero and one"
        raise ValueError(msg)
    return result


def _ordered(value: object, *, name: str, probability: bool) -> tuple[float, ...]:
    numbers = _vector(value, name=name, maximum=200)
    if any(n <= 0 or (probability and n >= 1) for n in numbers) or any(b <= a for a, b in pairwise(numbers)):
        msg = f"{name} must be strictly increasing, positive and within the requested domain"
        raise ValueError(msg)
    return numbers


@dataclass(frozen=True, slots=True)
class OutputSettings:
    probabilities: tuple[float, ...]
    discharges: tuple[float, ...]
    interval_mass: float


def _outputs(raw: object) -> OutputSettings:
    values = _object(raw, frozenset({"non_exceedance_probabilities", "discharges", "interval_mass"}), name="outputs")
    probs = _ordered(values["non_exceedance_probabilities"], name="non_exceedance_probabilities", probability=True)
    flows = _ordered(values["discharges"], name="discharges", probability=False)
    mass = _level(values["interval_mass"], name="interval_mass")
    if not 0 < len(probs) + len(flows) <= 200 or not 0 < (1 - mass) / 2 < 1 - (1 - mass) / 2 < 1:
        msg = "one to 200 output levels/flows and representable central interval endpoints are required"
        raise ValueError(msg)
    return OutputSettings(probs, flows, mass)


def _prior(raw: object, coordinates: tuple[str, ...]) -> JointGaussianPrior:
    values = _object(raw, frozenset({"mode", "source", "coordinates", "means", "covariance"}), name="prior")
    named = tuple(
        _text(v, name="prior coordinate") for v in _items(values["coordinates"], name="coordinates", maximum=2)
    )
    if values["mode"] != "supplied" or named != coordinates:
        msg = "prior must be supplied Gaussian in the exact ordered free lognormal coordinates"
        raise ValueError(msg)
    covariance = tuple(
        _vector(row, name="prior covariance row")
        for row in _items(values["covariance"], name="prior covariance", maximum=2)
    )
    return JointGaussianPrior(
        coordinates=named,
        means=_vector(values["means"], name="prior means"),
        covariance=covariance,
        source=_text(values["source"], name="prior source"),
        mode=LegacyPriorMode.SUPPLIED,
    )


@dataclass(frozen=True, slots=True)
class ProposalSettings:
    start: tuple[tuple[str, float], ...]
    controls: CurvatureControls
    gamma: float


def _proposal(raw: object, coordinates: tuple[str, ...]) -> ProposalSettings:
    keys = frozenset(
        {
            "start",
            "gamma",
            "scales",
            "initial_step",
            "max_refinements",
            "max_iterations",
            "gradient_tolerance",
            "mode_tolerance",
            "curvature_tolerance",
            "max_condition",
        }
    )
    v = _object(raw, keys, name="proposal")
    return ProposalSettings(
        tuple(zip(coordinates, _vector(v["start"], name="proposal start"), strict=True)),
        CurvatureControls(
            scales=_vector(v["scales"], name="proposal scales"),
            initial_step=_number(v["initial_step"], name="initial_step"),
            max_refinements=_integer(v["max_refinements"], name="max_refinements"),
            max_iterations=_integer(v["max_iterations"], name="max_iterations"),
            gradient_tolerance=_number(v["gradient_tolerance"], name="gradient_tolerance"),
            mode_tolerance=_number(v["mode_tolerance"], name="mode_tolerance"),
            curvature_tolerance=_number(v["curvature_tolerance"], name="curvature_tolerance"),
            max_condition=_number(v["max_condition"], name="max_condition"),
        ),
        _number(v["gamma"], name="gamma"),
    )


def _sampling(raw: object) -> ExplicitSamplingControls:
    v = _object(
        raw,
        frozenset({"rng", "draw_count", "random_seed", "minimum_effective_size", "maximum_single_weight"}),
        name="sampling",
    )
    if v["rng"] != "python_random_gauss":
        msg = "sampling requires the explicitly supported python_random_gauss stream"
        raise ValueError(msg)
    return ExplicitSamplingControls(
        draw_count=_integer(v["draw_count"], name="draw_count"),
        random_seed=_integer(v["random_seed"], name="random_seed"),
        minimum_effective_size=_number(v["minimum_effective_size"], name="minimum_effective_size"),
        maximum_single_weight=_number(v["maximum_single_weight"], name="maximum_single_weight"),
    )


def _precision_request(raw: object, outputs: OutputSettings, *, fixed: bool) -> PrecisionRequest:
    v = _object(
        raw,
        frozenset(
            {"meaning", "absolute_tolerance", "non_exceedance", "discharge", "interval_mass", "side", "log_moment"}
        ),
        name="precision request",
    )
    meaning = _text(v["meaning"], name="precision meaning")
    side, moment = v["side"], v["log_moment"]
    if side not in (None, "lower", "upper") or moment not in (None, "mean", "standard_deviation"):
        msg = "unsupported precision side or logarithmic moment"
        raise ValueError(msg)
    request = PrecisionRequest(
        meaning="parameter_mean" if meaning == "parameter_mean" else ResultMeaning(meaning),
        absolute_tolerance=_number(v["absolute_tolerance"], name="absolute_tolerance"),
        non_exceedance=_optional_number(v["non_exceedance"], name="non_exceedance"),
        discharge=_optional_number(v["discharge"], name="discharge"),
        interval_mass=_optional_number(v["interval_mass"], name="interval_mass"),
        side=side,
        log_moment=moment,
    )
    if (
        (request.non_exceedance is not None and request.non_exceedance not in outputs.probabilities)
        or (request.discharge is not None and request.discharge not in outputs.discharges)
        or (request.interval_mass is not None and request.interval_mass != outputs.interval_mass)
        or (request.meaning == "parameter_mean" and not fixed)
    ):
        msg = "precision request does not identify a configured output (parameter mean audit requires known scale)"
        raise ValueError(msg)
    return request


@dataclass(frozen=True, slots=True)
class AuditSettings:
    plan: PrecisionPlan
    requests: tuple[PrecisionRequest, ...]
    report_scalar_reservation: int


def _precision(
    raw: object, sampling: ExplicitSamplingControls, outputs: OutputSettings, *, fixed: bool
) -> AuditSettings | None:
    if raw == {"mode": "not_assessed"}:
        return None
    keys = frozenset(
        {
            "mode",
            "audit_seeds",
            "maximum_total_draws",
            "maximum_scalar_evaluations",
            "alpha",
            "minimum_side_count",
            "minimum_side_ess",
            "maximum_log_ratio",
            "criterion_source",
            "requests",
        }
    )
    v = _object(raw, keys, name="precision")
    if v["mode"] != "fixed_budget":
        msg = "precision mode must explicitly be not_assessed or fixed_budget"
        raise ValueError(msg)
    requests = tuple(
        _precision_request(item, outputs, fixed=fixed) for item in _items(v["requests"], name="precision requests")
    )
    if len({replace(request, absolute_tolerance=1.0) for request in requests}) != len(requests):
        msg = "precision output identities must be unique, regardless of tolerance"
        raise ValueError(msg)
    # The report sampler runs once; the existing audit replays that primary
    # stream and independently samples its audit streams. Reserve all draws.
    report_reservation = sampling.draw_count * (300 * len(outputs.probabilities) + 20 * len(outputs.discharges) + 10)
    plan = PrecisionPlan(
        primary=sampling,
        audit_seeds=tuple(_integer(seed, name="audit seed") for seed in _items(v["audit_seeds"], name="audit_seeds")),
        maximum_total_draws=_integer(v["maximum_total_draws"], name="maximum_total_draws") - sampling.draw_count,
        maximum_scalar_evaluations=(
            _integer(v["maximum_scalar_evaluations"], name="maximum_scalar_evaluations") - report_reservation
        ),
        alpha=_level(v["alpha"], name="alpha"),
        minimum_side_count=_integer(v["minimum_side_count"], name="minimum_side_count"),
        minimum_side_ess=_number(v["minimum_side_ess"], name="minimum_side_ess"),
        maximum_log_ratio=_number(v["maximum_log_ratio"], name="maximum_log_ratio"),
        criterion_source=_text(v["criterion_source"], name="criterion_source"),
    )
    plan.preflight(requests)
    return AuditSettings(plan, requests, report_reservation)


def _csv_flow(token: str, *, retained_excluded: bool = False) -> float | None:
    if token == "":
        return None
    if fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", token) is None:
        msg = "classified CSV flow must be a decimal number or an explicitly empty cell"
        raise ValueError(msg)
    value = float(token)
    if retained_excluded and value == 0 and not Decimal(token).is_zero():
        msg = "nonzero retained excluded flow underflows binary64"
        raise ValueError(msg)
    if not isfinite(value) or (value <= 0 and not retained_excluded):
        msg = "classified CSV nonempty flow and bound cells must be strictly positive finite binary64 values"
        raise ValueError(msg)
    return value


def _classified_rows(data: bytes, *, retain_excluded_value: bool = False) -> tuple[AnnualObservation, ...]:
    """Strict explicit per-year classifications; never interpret native flags."""
    if b"\x00" in data:
        msg = "classified CSV contains NUL bytes"
        raise ValueError(msg)
    reader = csv.reader(StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
    rows: list[AnnualObservation] = []
    try:
        if next(reader, None) != ["Year", "Kind", "Value", "Lower", "Upper"]:
            msg = "expected exact classified CSV header Year,Kind,Value,Lower,Upper"
            raise ValueError(msg)
        for columns in reader:
            if len(columns) != 5 or fullmatch(r"[1-9][0-9]*", columns[0]) is None:
                msg = "classified CSV row needs five columns and a positive integer year"
                raise ValueError(msg)
            year, kind, value, lower, upper = columns
            rows.append(
                AnnualObservation(
                    year=int(year),
                    kind=ObservationKind(kind),
                    exposure=ExposureKind.SYSTEMATIC,
                    value=_csv_flow(
                        value, retained_excluded=retain_excluded_value and kind == ObservationKind.EXCLUDED
                    ),
                    lower_bound=_csv_flow(lower),
                    upper_bound=_csv_flow(upper),
                )
            )
    except csv.Error as exc:
        msg = f"malformed classified CSV: {exc}"
        raise ValueError(msg) from exc
    return tuple(rows)


def _model(data: bytes, v: dict[str, object]) -> ClassifiedLognormalPosterior:
    if v["source_classification"] != SOURCE_CLASSIFICATION:
        msg = "source classification must explicitly assert classified systematic annual maxima"
        raise ValueError(msg)
    source = _text(v["source"], name="source")
    unit = _text(v["discharge_unit"], name="discharge_unit")
    year = _text(v["year_definition"], name="year_definition")
    reference = _number(v["reference_discharge"], name="reference_discharge")
    fixed = _optional_number(v["fixed_log_sd"], name="fixed_log_sd")
    coordinates = (MEAN_COORDINATE,) if fixed is not None else (MEAN_COORDINATE, LOG_SD_COORDINATE)
    prior = _prior(v["prior"], coordinates)
    if unit in {"years", "probability"}:
        msg = "discharge_unit must name a physical flow unit"
        raise ValueError(msg)
    if v["source_format"] == "year_ams_csv":
        return bind_exact_lognormal_csv(
            data,
            source=source,
            discharge_unit=unit,
            year_definition=year,
            source_classification=SOURCE_CLASSIFICATION,
            reference_discharge=reference,
            fixed_log_sd=fixed,
            prior=prior,
        )
    if v["source_format"] != "classified_systematic_csv":
        msg = "unsupported Bayesian source format"
        raise ValueError(msg)
    return ClassifiedLognormalPosterior(
        series=ObservationSeries(discharge_unit=unit, annual=_classified_rows(data), year_definition=year),
        source=source,
        source_sha256=sha256(data).hexdigest(),
        source_classification=SOURCE_CLASSIFICATION,
        reference_discharge=reference,
        fixed_log_sd=fixed,
        prior=prior,
    )


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def calculate_bayesian(
    raw_ams: bytes, raw_settings: bytes, values: dict[str, object]
) -> tuple[AnalysisResult, dict[str, str | int | float | bool | None]]:
    """Compose the admitted target, scaled proposal, actual sampler and typed report."""
    if values.get("schema_version") == SHAPE_SCHEMA:
        return _calculate_shape(raw_ams, raw_settings, values)
    v = _object(values, _FIELDS, name="Bayesian settings")
    if v["schema_version"] != BAYESIAN_SCHEMA or v["estimator"] != "lognormal_bayesian":
        msg = "explicit Bayesian lognormal schema and estimator are required"
        raise ValueError(msg)
    model = _model(raw_ams, v)
    proposal_settings = _proposal(v["proposal"], model.coordinates)
    sampling = _sampling(v["sampling"])
    outputs = _outputs(v["outputs"])
    audit_settings = _precision(v["precision"], sampling, outputs, fixed=model.fixed_log_sd is not None)
    provenance = RunProvenance(
        compatibility_profile=LEGACY_FLIKE_PROFILE.identifier,
        method_id="quantile-estimands",
        method_version=METHOD_VERSION,
        package_version=__version__,
        evidence_status=EvidenceStatus.OBSERVED_ARTIFACT,
        input_sha256=model.source_sha256,
        settings_sha256=sha256(raw_settings).hexdigest(),
    )
    target = model.target()
    construction = build_curvature_proposal(
        target=target,
        starting_point=proposal_settings.start,
        controls=proposal_settings.controls,
        gamma=proposal_settings.gamma,
    )
    run = independent_importance_sample(
        proposal=construction.proposal,
        target=target,
        controls=sampling,
        provenance=provenance,
    )
    warnings = [
        "Independent explicitly configured lognormal Bayesian fit; not native Flike-qualified.",
        "Source classification, unit, year definition and prior provenance are caller assertions.",
        "Local curvature and global weight gates do not establish coverage or output-specific numerical precision.",
    ]
    result = wp06_output_analysis_result(
        provenance=provenance,
        parameters=run.moments.parameters,
        discharge_unit=model.series.discharge_unit,
        quantiles=tuple(
            quantile_outputs(
                run.particles,
                p,
                model.kernel,
                interval_mass=outputs.interval_mass,
                mean_undefined_reason=(
                    "upper design-flood mean diverges for the full Gaussian estimated-log-SD model"
                    if model.fixed_log_sd is None and p > 0.5
                    else None
                ),
            )
            for p in outputs.probabilities
        ),
        probabilities=tuple(
            probability_outputs(run.particles, q, model.kernel, interval_mass=outputs.interval_mass)
            for q in outputs.discharges
        ),
        warnings=warnings,
    )
    audit = None
    if audit_settings is not None:
        audit = audit_lognormal_precision(
            model=model,
            proposal=construction.proposal,
            provenance=provenance,
            plan=audit_settings.plan,
            requests=audit_settings.requests,
        )
    precision_status = "not_assessed" if audit is None else f"selected_audit_estimates_{audit.state.value}"
    result = replace(
        result,
        diagnostics=(
            RunDiagnostic(name="importance_draw_count", value=float(sampling.draw_count), unit="count"),
            RunDiagnostic(
                name="importance_effective_sample_size", value=run.moments.effective_sample_size, unit="count"
            ),
            RunDiagnostic(name="importance_largest_weight", value=run.moments.largest_weight, unit="fraction"),
            RunDiagnostic(
                name="importance_zero_weight_count", value=float(run.moments.zero_weight_count), unit="count"
            ),
            RunDiagnostic(
                name="local_scaled_gradient_norm", value=construction.scaled_gradient_norm, unit="dimensionless"
            ),
            RunDiagnostic(name="local_scaled_condition", value=construction.scaled_condition, unit="dimensionless"),
            *(
                RunDiagnostic(
                    name=f"approximate_parameter_mean_mcse_{coordinate}", value=mcse, unit="source coordinate"
                )
                for coordinate, mcse in zip(model.coordinates, run.approximate_mean_mcse, strict=True)
            ),
        ),
        precision_audit=audit,
        warnings=(
            *result.warnings,
            (
                f"Monte Carlo precision: {precision_status}; unaudited outputs remain unassessed. "
                "Conditional audit acceptance is not posterior uncertainty or native qualification."
            ),
        ),
    )
    settings: dict[str, str | int | float | bool | None] = {
        "schema_version": BAYESIAN_SCHEMA,
        "estimator": "lognormal_bayesian",
        "source": model.source,
        "source_format": _text(v["source_format"], name="source_format"),
        "source_classification": SOURCE_CLASSIFICATION,
        "year_definition": model.series.year_definition,
        "discharge_unit": model.series.discharge_unit,
        "reference_discharge": model.reference_discharge,
        "fixed_log_sd": model.fixed_log_sd,
        "prior_source": model.prior.source,
        "prior_coordinates": _json(model.coordinates),
        "prior_means": _json(model.prior.means),
        "prior_covariance": _json(model.prior.covariance),
        "proposal_gamma": construction.gamma,
        "proposal_means": _json(construction.proposal.means),
        "proposal_covariance": _json(construction.proposal.covariance),
        "proposal_controls": _json(asdict(proposal_settings.controls)),
        "proposal_start": _json(proposal_settings.start),
        "random_stream": run.particles.random_stream_description,
        "sampling_draw_count": sampling.draw_count,
        "sampling_seed": sampling.random_seed,
        "minimum_effective_size": sampling.minimum_effective_size,
        "maximum_single_weight": sampling.maximum_single_weight,
        "interval_mass": outputs.interval_mass,
        "non_exceedance_probabilities": _json(outputs.probabilities),
        "discharges": _json(outputs.discharges),
        "precision_status": precision_status,
        "precision_audit_attempted_primary_replay": audit is not None and audit.attempts[0].attempted,
        "source_ams_file": "source-ams.csv",
        "source_settings_file": "source-settings.json",
    }
    if audit is not None and audit_settings is not None:
        settings["precision_scheduled_draws_including_report"] = audit.plan.total_draws + sampling.draw_count
        settings["precision_report_scalar_reservation"] = audit_settings.report_scalar_reservation
    return result, settings


def _shape_prior(raw: object) -> BoundedShapePrior:
    v = _object(
        raw, frozenset({"law", "coordinates", "means", "lower_factor", "shape_bounds", "source"}), name="shape prior"
    )
    if v["law"] != "gaussian_scaled_logit":
        msg = "bounded shape requires a supplied gaussian_scaled_logit law"
        raise ValueError(msg)
    bounds = _vector(v["shape_bounds"], name="shape bounds")
    if len(bounds) != 2:
        msg = "two shape bounds are required"
        raise ValueError(msg)
    return BoundedShapePrior(
        coordinates=tuple(
            _text(c, name="prior coordinate") for c in _items(v["coordinates"], name="coordinates", maximum=3)
        ),
        means=_vector(v["means"], name="latent means", maximum=3),
        lower_factor=tuple(
            _vector(row, name="factor row", maximum=3) for row in _items(v["lower_factor"], name="factor", maximum=3)
        ),
        bounds=(bounds[0], bounds[1]),
        source=_text(v["source"], name="prior source"),
    )


def _shape_model(data: bytes, v: dict[str, object]) -> ClassifiedShapePosterior:
    estimator = _text(v["estimator"], name="estimator")
    families = {"gev_bayesian": PosteriorFamily.GEV, "lp3_bayesian": PosteriorFamily.LP3}
    if estimator not in families:
        msg = "expected gev_bayesian or lp3_bayesian for bounded-shape inference"
        raise ValueError(msg)
    unit = _text(v["discharge_unit"], name="discharge unit")
    year = _text(v["year_definition"], name="year definition")
    if v["source_format"] == "classified_systematic_csv":
        series = ObservationSeries(
            discharge_unit=unit, year_definition=year, annual=_classified_rows(data, retain_excluded_value=True)
        )
    elif v["source_format"] == "year_ams_csv":
        series = exact_family_series(data, unit, year)
    else:
        msg = "unsupported Bayesian source format"
        raise ValueError(msg)
    return ClassifiedShapePosterior(
        family=families[estimator],
        series=series,
        source=_text(v["source"], name="source"),
        source_classification=_text(v["source_classification"], name="source_classification"),
        source_sha256=sha256(data).hexdigest(),
        reference_discharge=_optional_number(v["reference_discharge"], name="reference discharge"),
        prior=_shape_prior(v["prior"]),
    )


def _calculate_shape(
    data: bytes, raw_settings: bytes, values: dict[str, object]
) -> tuple[AnalysisResult, dict[str, str | int | float | bool | None]]:
    v = _object(values, _FIELDS - {"fixed_log_sd"}, name="bounded shape settings")
    model = _shape_model(data, v)
    prop = _object(v["proposal"], frozenset({"law", "gamma"}), name="shape proposal")
    if (
        prop["law"] != "scaled_prior_innovations"
        or _object(v["precision"], frozenset({"mode"}), name="shape precision")["mode"] != "not_assessed"
    ):
        msg = "shape inference requires scaled_prior_innovations and not_assessed output precision"
        raise ValueError(msg)
    gamma = _number(prop["gamma"], name="proposal gamma")
    sampling = _sampling(v["sampling"])
    outputs = _outputs(v["outputs"])
    reasons = tuple(model.raw_mean_undefined_reason(p) for p in outputs.probabilities)
    run = execute_shape_posterior(
        model, gamma=gamma, sampling=sampling, settings_sha256=sha256(raw_settings).hexdigest()
    )
    warnings = ["Output-specific integration precision is unassessed. Posterior SDs describe parameter uncertainty."]
    if model.family == PosteriorFamily.LP3:
        warnings.append("All positive raw predictive discharge moments diverge under this unbounded log-scale prior.")
    else:
        warnings.append(
            f"Predictive absolute moments of order h>0 exist exactly when h times upper EVT shape bound {model.prior.bounds[1]} is at most one."
        )
    probabilities = tuple(
        probability_outputs(run.particles, q, model.kernel, interval_mass=outputs.interval_mass)
        for q in outputs.discharges
    )
    if any(p.expected_aep == 0.0 or p.expected_non_exceedance == 0.0 for p in probabilities):
        msg = "sampled mixture cannot resolve a mathematically positive expected tail"
        raise ValueError(msg)
    result = wp06_output_analysis_result(
        provenance=run.particles.provenance,
        parameters=run.moments.parameters,
        discharge_unit=model.series.discharge_unit,
        quantiles=tuple(
            quantile_outputs(
                run.particles,
                p,
                model.kernel,
                interval_mass=outputs.interval_mass,
                mean_undefined_reason=reason,
                direct_upper_tail=True,
                log_undefined_reason="the population assigns positive mass to nonpositive GEV design quantiles"
                if model.family == PosteriorFamily.GEV
                else None,
            )
            for p, reason in zip(outputs.probabilities, reasons, strict=True)
        ),
        probabilities=probabilities,
        warnings=warnings,
    )
    diag = [
        RunDiagnostic(
            name="log_evidence_estimate", value=run.log_evidence, unit="log likelihood in declared flow units"
        ),
        RunDiagnostic(name="importance_effective_sample_size", value=run.moments.effective_sample_size, unit="count"),
        RunDiagnostic(name="importance_largest_weight", value=run.moments.largest_weight, unit="fraction"),
    ]
    diag.extend(
        RunDiagnostic(
            name=f"covariance_{name}_{model.coordinates[j]}",
            value=run.moments.covariance[i][j],
            unit="product of named coordinate units",
        )
        for i, name in enumerate(model.coordinates)
        for j in range(i, 3)
    )
    settings: dict[str, str | int | float | bool | None] = {
        "schema_version": SHAPE_SCHEMA,
        "estimator": _text(v["estimator"], name="estimator"),
        "source": model.source,
        "source_format": _text(v["source_format"], name="source_format"),
        "source_classification": model.source_classification,
        "year_definition": model.series.year_definition,
        "discharge_unit": model.series.discharge_unit,
        "reference_discharge": model.reference_discharge,
        "prior": _json(v["prior"]),
        "proposal": _json(prop),
        "sampling": _json(v["sampling"]),
        "outputs": _json(v["outputs"]),
        "random_stream": run.particles.random_stream_description,
        "precision_status": "not_assessed",
        "source_ams_file": "source-ams.csv",
        "source_settings_file": "source-settings.json",
    }
    return replace(result, diagnostics=tuple(diag)), settings
