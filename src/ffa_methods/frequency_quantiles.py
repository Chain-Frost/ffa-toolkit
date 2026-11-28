"""Validate report-backed quantile series and intervals without performing inference."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import cast

from ffa_methods.contracts import AnalysisResult, ProbabilityKind, ResultMeaning, ResultPoint

BAYESIAN_PLOT_SCHEMA = "ffa-lognormal-bayesian/1"
BAYESIAN_PLOT_SOURCE = "caller_asserted_classified_systematic_annual_maxima"
POSTERIOR_LIMIT = ResultMeaning.QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT
POSTERIOR_CONSTRUCTION = "WP06 left-continuous weighted empirical inverse CDF; Flike interpolation unverified"
BAYESIAN_SERIES = (
    ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS,
    ResultMeaning.POSTERIOR_MEAN_QUANTILE,
    ResultMeaning.POSTERIOR_MEDIAN_QUANTILE,
    ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE,
)
_TABLE_ONLY = frozenset(
    {
        ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY,
        ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY,
        ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP,
        ResultMeaning.RETURN_PERIOD_MONTE_CARLO_PROBABILITY_LIMIT,
        ResultMeaning.LOG10_QUANTILE_POSTERIOR_MOMENT,
    }
)


@dataclass(frozen=True, slots=True)
class QuantilePlotRow:
    """One actual numerical result row, not a new model evaluation."""

    index: int
    point: ResultPoint
    f: float
    precision_state: str

    @property
    def aep(self) -> float:
        return 1.0 - self.f


def _requested_probabilities(settings: Mapping[str, object]) -> tuple[float, ...]:
    raw = settings.get("non_exceedance_probabilities")
    if not isinstance(raw, str):
        msg = "Bayesian frequency reporting requires recorded requested probabilities"
        raise ValueError(msg)
    data: object = json.loads(raw)
    if not isinstance(data, list):
        msg = "Bayesian frequency reporting requires recorded requested probabilities"
        raise ValueError(msg)
    values: list[float] = []
    for value in cast(list[object], data):
        if isinstance(value, bool) or not isinstance(value, int | float):
            msg = "Bayesian plot probabilities must be numbers"
            raise ValueError(msg)
        f = float(value)
        if not 0.0 < f < 1.0 or (values and f <= values[-1]):
            msg = "Bayesian plot probabilities must be ordered and strictly inside (0, 1)"
            raise ValueError(msg)
        values.append(f)
    if not values:
        msg = "Bayesian frequency reporting requires requested quantiles; probability-at-flow tables are not curves"
        raise ValueError(msg)
    return tuple(values)


def _plot_row(
    point: ResultPoint,
    record: Mapping[str, object],
    index: int,
    unit: str,
    requested: tuple[float, ...],
    *,
    require_precision: bool = True,
) -> QuantilePlotRow:
    f = point.input_probability
    if (
        f is None
        or f not in requested
        or point.input_probability_kind != ProbabilityKind.NON_EXCEEDANCE
        or point.unit != unit
        or not isfinite(point.value)
    ):
        msg = "Plot result has an invalid probability, unit or discharge"
        raise ValueError(msg)
    expected: dict[str, object] = {
        "meaning": point.meaning.value,
        "value": point.value,
        "unit": point.unit,
        "input_probability": f,
        "input_probability_kind": ProbabilityKind.NON_EXCEEDANCE.value,
        "input_non_exceedance_probability": f,
        "input_annual_exceedance_probability": 1.0 - f,
        "reported_probability_level": point.confidence_level,
        "probability_limit_side": point.probability_limit_side,
        "probability_limit_construction": point.probability_limit_construction,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        msg = "Plot result does not match its canonical numerical report row"
        raise ValueError(msg)
    if not require_precision:
        if "precision_state" in record:
            msg = "deterministic plot cannot claim a Bayesian per-result precision state"
            raise ValueError(msg)
        return QuantilePlotRow(index, point, f, "")
    precision = record.get("precision_state")
    if not isinstance(precision, str) or not precision:
        msg = "Bayesian plot requires the canonical per-result numerical precision state"
        raise ValueError(msg)
    return QuantilePlotRow(index, point, f, precision)


def _check_group(
    group: tuple[QuantilePlotRow, ...], *, f: float, mass: float, mean_defined: bool
) -> None:
    core = {row.point.meaning for row in group if row.point.meaning in BAYESIAN_SERIES}
    expected = set(BAYESIAN_SERIES)
    if not mean_defined:
        expected.remove(ResultMeaning.POSTERIOR_MEAN_QUANTILE)
    if core != expected or len(group) != len(expected) + 2:
        msg = f"Bayesian plot has incomplete or duplicate quantile estimands at F={f!r}"
        raise ValueError(msg)
    limits = [row.point for row in group if row.point.meaning == POSTERIOR_LIMIT]
    by_side = {point.probability_limit_side: point for point in limits}
    if len(limits) != 2 or set(by_side) != {"lower", "upper"}:
        msg = f"Bayesian plot requires a lower/upper posterior quantile interval pair at F={f!r}"
        raise ValueError(msg)
    if any(
        point.confidence_level != mass or point.probability_limit_construction != POSTERIOR_CONSTRUCTION
        for point in limits
    ) or by_side["lower"].value > by_side["upper"].value:
        msg = f"Bayesian plot interval level, construction or endpoint ordering is inconsistent at F={f!r}"
        raise ValueError(msg)
    if any(
        row.point.confidence_level is not None
        or row.point.probability_limit_side is not None
        or row.point.probability_limit_construction is not None
        for row in group
        if row.point.meaning in BAYESIAN_SERIES
    ):
        msg = "Bayesian central estimands cannot carry interval endpoint metadata"
        raise ValueError(msg)


def bayesian_quantile_rows(
    result: AnalysisResult,
    *,
    settings: Mapping[str, object],
    unit: str,
    numerical_rows: Sequence[Mapping[str, object]],
) -> tuple[QuantilePlotRow, ...]:
    """Select only design quantiles and their actual pointwise posterior limits."""
    if (
        settings.get("schema_version") != BAYESIAN_PLOT_SCHEMA
        or settings.get("estimator") != "lognormal_bayesian"
        or settings.get("source_format") != "year_ams_csv"
        or settings.get("source_classification") != BAYESIAN_PLOT_SOURCE
        or result.provenance.method_id != "quantile-estimands"
        or result.provenance.method_version != "explicit-lognormal-bayesian-cli/1"
    ):
        msg = "Bayesian frequency plotting requires the admitted exact systematic Year,AMS workflow"
        raise ValueError(msg)
    requested = _requested_probabilities(settings)
    mass = settings.get("interval_mass")
    if isinstance(mass, bool) or not isinstance(mass, int | float) or not 0.0 < mass < 1.0:
        msg = "Bayesian frequency plotting requires the recorded central posterior interval mass"
        raise ValueError(msg)
    if "fixed_log_sd" not in settings or len(numerical_rows) != len(result.values):
        msg = "Bayesian frequency plotting requires complete canonical numerical rows and scale configuration"
        raise ValueError(msg)
    groups: dict[float, list[QuantilePlotRow]] = {f: [] for f in requested}
    for index, (point, record) in enumerate(zip(result.values, numerical_rows, strict=True), start=1):
        if point.meaning in _TABLE_ONLY:
            continue
        if point.meaning not in (*BAYESIAN_SERIES, POSTERIOR_LIMIT):
            msg = "unsupported Bayesian plot estimand"
            raise ValueError(msg)
        row = _plot_row(point, record, index, unit, requested)
        groups[row.f].append(row)
    fixed = settings["fixed_log_sd"] is not None
    for f, rows in groups.items():
        # The admitted estimated-log-SD workflow explicitly omits the divergent upper-tail mean.
        _check_group(tuple(rows), f=f, mass=mass, mean_defined=fixed or f <= 0.5)
    return tuple(row for rows in groups.values() for row in rows)


SELECTED_SCHEMA = "ffa-gev-selected-h/1"
DETERMINISTIC_SCHEMAS = ("ffa-gev-fixed-h/1", "ffa-ordinary-lmoments/1", SELECTED_SCHEMA)
DETERMINISTIC_SOURCE = "caller_asserted_complete_exact_positive_annual_maxima"
BOOTSTRAP_LIMIT = ResultMeaning.QUANTILE_BOOTSTRAP_LIMIT
BOOTSTRAP_CONSTRUCTION = "parametric_percentile_type_7_successful_refits"


def _deterministic_probabilities(settings: Mapping[str, object]) -> tuple[float, ...]:
    raw = settings.get("non_exceedance_probabilities")
    if not isinstance(raw, str) or not raw:
        msg = "frequency report requires recorded requested deterministic probabilities"
        raise ValueError(msg)
    values = tuple(float(token) for token in raw.split(","))
    if any(not 0.0 < f < 1.0 for f in values) or any(
        left >= right for left, right in zip(values[:-1], values[1:], strict=True)
    ):
        msg = "deterministic plot probabilities must be ordered and strictly inside (0, 1)"
        raise ValueError(msg)
    return values


def _bootstrap_mass(settings: Mapping[str, object]) -> float | None:
    if "bootstrap_status" not in settings:
        return None
    if settings["bootstrap_status"] != "completed":
        msg = "frequency report requires a point fit without a bootstrap request or a completed bootstrap"
        raise ValueError(msg)
    mass = settings.get("bootstrap_confidence")
    if (
        isinstance(mass, bool)
        or not isinstance(mass, int | float)
        or not 0.0 < mass < 1.0
        or settings.get("bootstrap_interval_target") != "conditional_on_successful_refits"
    ):
        msg = "bootstrap frequency plot requires recorded interval mass and successful-refit target"
        raise ValueError(msg)
    return float(mass)


def _check_deterministic_group(group: list[QuantilePlotRow], *, mass: float | None) -> None:
    fits = [row.point for row in group if row.point.meaning == ResultMeaning.DETERMINISTIC_FITTED_QUANTILE]
    if len(fits) != 1 or any(
        point.confidence_level is not None
        or point.probability_limit_side is not None
        or point.probability_limit_construction is not None
        for point in fits
    ):
        msg = "frequency report requires one deterministic point fit per requested probability"
        raise ValueError(msg)
    limits = [row.point for row in group if row.point.meaning == BOOTSTRAP_LIMIT]
    if mass is None:
        if limits:
            msg = "bootstrap endpoints require a completed bootstrap"
            raise ValueError(msg)
        return
    by_side = {point.probability_limit_side: point for point in limits}
    if len(limits) != 2 or set(by_side) != {"lower", "upper"}:
        msg = "bootstrap frequency plot requires paired lower and upper endpoints"
        raise ValueError(msg)
    if any(
        point.confidence_level != mass or point.probability_limit_construction != BOOTSTRAP_CONSTRUCTION
        for point in limits
    ) or by_side["lower"].value > by_side["upper"].value:
        msg = "bootstrap interval level, construction or endpoint ordering is inconsistent"
        raise ValueError(msg)


def deterministic_quantile_rows(
    result: AnalysisResult,
    *,
    settings: Mapping[str, object],
    unit: str,
    numerical_rows: Sequence[Mapping[str, object]],
) -> tuple[QuantilePlotRow, ...]:
    """Admit selected-H point fits and completed, typed percentile bootstrap endpoints."""
    schema = settings.get("schema_version")
    if schema not in DETERMINISTIC_SCHEMAS or settings.get("source_classification") != DETERMINISTIC_SOURCE:
        msg = "frequency report requires an admitted deterministic systematic-AMS settings schema"
        raise ValueError(msg)
    if schema == SELECTED_SCHEMA:
        shift = settings.get("lh_shift")
        if (
            settings.get("selection") != "lowest_passing_h"
            or type(shift) is not int
            or shift not in range(5)
            or result.provenance.method_version != "wang-direct-lh-selected-h/1"
        ):
            msg = "selected-H frequency report requires a successfully selected candidate"
            raise ValueError(msg)
    requested = _deterministic_probabilities(settings)
    mass = _bootstrap_mass(settings)
    if len(numerical_rows) != len(result.values):
        msg = "frequency report requires complete canonical numerical rows"
        raise ValueError(msg)
    groups: dict[float, list[QuantilePlotRow]] = {f: [] for f in requested}
    for index, (point, record) in enumerate(zip(result.values, numerical_rows, strict=True), start=1):
        if point.meaning not in (ResultMeaning.DETERMINISTIC_FITTED_QUANTILE, BOOTSTRAP_LIMIT):
            msg = "unsupported deterministic plot estimand"
            raise ValueError(msg)
        row = _plot_row(point, record, index, unit, requested, require_precision=False)
        groups[row.f].append(row)
    for rows in groups.values():
        _check_deterministic_group(rows, mass=mass)
    return tuple(row for rows in groups.values() for row in rows)
