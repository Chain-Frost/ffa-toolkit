"""Internal presentation adapter for explicitly supplied WP06 weighted mathematical outputs.

The WP06 objects are already calculated. No proposal, prior, likelihood, source
file, Flike default or fitted model is constructed here. Native qualification
is not implied by the report's compatibility target or supplied evidence label.
"""

from collections.abc import Sequence
from math import inf, isclose, isfinite
from typing import Literal

from ffa_methods.bayesian_outputs import ProbabilityOutputs, QuantileOutputs
from ffa_methods.compatibility import LEGACY_FLIKE_PROFILE
from ffa_methods.contracts import (
    AnalysisResult,
    ParameterSummary,
    ProbabilityKind,
    ResultMeaning,
    ResultPoint,
    RunProvenance,
)

# Explicit WP06 implementation choice; never presented as verified FLIKE interpolation.
_EMPIRICAL_LIMIT = "WP06 left-continuous weighted empirical inverse CDF; Flike interpolation unverified"


def _unit(unit: str) -> str:
    if not unit.strip() or unit in ("probability", "years") or any(ord(c) < 32 or ord(c) == 127 for c in unit):
        msg = "a physical discharge unit must be explicitly supplied"
        raise ValueError(msg)
    return unit


def _discharge(value: float, *, name: str) -> float:
    """Check physical input discharge; never use for unbounded model quantiles."""
    if not isfinite(value) or value < 0:
        msg = f"{name} must be finite and nonnegative"
        raise ValueError(msg)
    return value


def _modelled_quantile(value: float, *, name: str) -> float:
    """A continuous GEV/Gumbel prediction can be negative without being nonfinite."""
    if not isfinite(value):
        msg = f"{name} must be finite; negative modelled quantiles remain reportable"
        raise ValueError(msg)
    return value


def _mass(mass: float) -> None:
    if not isfinite(mass) or not 0.0 < mass < 1.0:
        msg = "WP06 central interval mass must be strictly between zero and one"
        raise ValueError(msg)


def _quantile_points(output: QuantileOutputs, unit: str) -> tuple[list[ResultPoint], list[str]]:
    """Represent every defined scalar WP06 quantile estimand without relabelling."""
    _mass(output.interval_mass)
    f = output.non_exceedance
    if not isfinite(f) or not 0.0 < f < 1.0:
        msg = "WP06 quantile input must be a non-exceedance probability in (0, 1)"
        raise ValueError(msg)
    values = [
        (ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS, output.at_expected_parameters),
        (ResultMeaning.POSTERIOR_MEDIAN_QUANTILE, output.median_of_quantiles),
        (ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE, output.predictive_quantile),
    ]
    if output.mean_of_quantiles is not None:
        if output.mean_undefined_reason is not None:
            msg = "a defined posterior mean quantile cannot carry an undefined reason"
            raise ValueError(msg)
        values.insert(1, (ResultMeaning.POSTERIOR_MEAN_QUANTILE, output.mean_of_quantiles))
    elif not output.mean_undefined_reason or not output.mean_undefined_reason.strip():
        msg = "omitted posterior mean quantile requires an explicit undefined reason"
        raise ValueError(msg)
    points = [
        ResultPoint(
            meaning=meaning,
            value=_modelled_quantile(value, name=meaning.value),
            unit=unit,
            input_probability=f,
            input_probability_kind=ProbabilityKind.NON_EXCEEDANCE,
        )
        for meaning, value in values
    ]
    lower, upper = output.probability_limits
    _modelled_quantile(lower, name="lower quantile probability limit")
    _modelled_quantile(upper, name="upper quantile probability limit")
    if lower > upper:
        msg = "lower quantile probability limit exceeds the upper limit"
        raise ValueError(msg)
    limits: tuple[tuple[Literal["lower", "upper"], float], ...] = (("lower", lower), ("upper", upper))
    for side, value in limits:
        points.append(
            ResultPoint(
                meaning=ResultMeaning.QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT,
                value=value,
                unit=unit,
                input_probability=f,
                input_probability_kind=ProbabilityKind.NON_EXCEEDANCE,
                confidence_level=output.interval_mass,
                probability_limit_side=side,
                probability_limit_construction=_EMPIRICAL_LIMIT,
            )
        )

    if not isfinite(output.nonpositive_quantile_weight) or not 0 <= output.nonpositive_quantile_weight <= 1:
        msg = "nonpositive-quantile posterior weight must lie within [0, 1]"
        raise ValueError(msg)
    warnings = [
        (
            f"WP06 quantile F={f:.17g}: weighted tail resolution ESS "
            f"lower={output.lower_limit_effective_size:.17g}, upper={output.upper_limit_effective_size:.17g}; "
            "these are not Monte Carlo standard errors."
        )
    ]
    if output.mean_undefined_reason is not None:
        warnings.append(f"WP06 quantile F={f:.17g}: UNDEFINED posterior mean quantile; {output.mean_undefined_reason}.")
    moment_points, moment_warnings = _log_quantile_moments(output, unit)
    return points + moment_points, warnings + moment_warnings


def _log_quantile_moments(output: QuantileOutputs, unit: str) -> tuple[list[ResultPoint], list[str]]:
    f = output.non_exceedance
    points: list[ResultPoint] = []
    warnings: list[str] = []
    log_mean = output.mean_log10_quantile
    log_sd = output.standard_deviation_log10_quantile
    if output.nonpositive_quantile_weight:
        if log_mean is not None or log_sd is not None:
            msg = "base-10 quantile moments must be undefined when a positive-weight quantile is nonpositive"
            raise ValueError(msg)
        warnings.append(
            f"WP06 quantile F={f:.17g}: log10 moments undefined because "
            f"nonpositive quantiles have posterior weight {output.nonpositive_quantile_weight:.17g}."
        )
    elif log_mean is None or log_sd is None:
        msg = "WP06 log10 moments missing despite zero nonpositive-quantile weight"
        raise ValueError(msg)
    else:
        if not isfinite(log_mean) or not isfinite(log_sd) or log_sd < 0:
            msg = "WP06 base-10 quantile moments must be finite, with nonnegative standard deviation"
            raise ValueError(msg)
        moments: tuple[tuple[Literal["mean", "standard_deviation"], float], ...] = (
            ("mean", log_mean),
            ("standard_deviation", log_sd),
        )
        for kind, value in moments:
            points.append(
                ResultPoint(
                    meaning=ResultMeaning.LOG10_QUANTILE_POSTERIOR_MOMENT,
                    value=value,
                    unit=f"log10({unit})",
                    input_probability=f,
                    input_probability_kind=ProbabilityKind.NON_EXCEEDANCE,
                    log10_moment_kind=kind,
                )
            )
    return points, warnings


def _return_period_points(output: ProbabilityOutputs, unit: str) -> tuple[list[ResultPoint], list[str]]:
    """Preserve finite periods and explicitly flag unrepresentable infinity."""
    q = output.discharge
    values: tuple[tuple[ResultMeaning, Literal["lower", "upper"] | None, float], ...] = (
        (ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP, None, output.return_period),
        (ResultMeaning.RETURN_PERIOD_MONTE_CARLO_PROBABILITY_LIMIT, "lower", output.return_period_limits[0]),
        (ResultMeaning.RETURN_PERIOD_MONTE_CARLO_PROBABILITY_LIMIT, "upper", output.return_period_limits[1]),
    )
    points: list[ResultPoint] = []
    warnings: list[str] = []
    lower_limit, upper_limit = output.return_period_limits
    if lower_limit > upper_limit:
        msg = "lower return-period probability limit exceeds the upper limit"
        raise ValueError(msg)
    if output.expected_aep == 0.0 and output.return_period != inf:
        msg = "zero expected AEP requires an infinite return period"
        raise ValueError(msg)
    for meaning, side, value in values:
        if value == inf:
            warnings.append(
                f"WP06 discharge={q:.17g} {unit}: {meaning.value}"
                f"{' ' + side if side is not None else ''} is +infinity; no finite surrogate exported."
            )
            continue
        if not isfinite(value) or value < 1:
            msg = f"{meaning.value} must be at least one year or positive infinity"
            raise ValueError(msg)
        points.append(
            ResultPoint(
                meaning=meaning,
                value=value,
                unit="years",
                input_discharge=q,
                input_discharge_unit=unit,
                confidence_level=output.interval_mass if side is not None else None,
                probability_limit_side=side,
                probability_limit_construction=_EMPIRICAL_LIMIT if side is not None else None,
            )
        )
    if (
        output.expected_aep > 0
        and isfinite(output.return_period)
        and not isclose(output.return_period * output.expected_aep, 1.0, rel_tol=1e-12)
    ):
        msg = "expected return period is not reciprocal expected AEP"
        raise ValueError(msg)
    return points, warnings


def _probability_points(output: ProbabilityOutputs, unit: str) -> tuple[list[ResultPoint], list[str]]:
    _mass(output.interval_mass)
    q = _discharge(output.discharge, name="input discharge")
    f, aep = output.expected_non_exceedance, output.expected_aep
    if not isfinite(f) or not isfinite(aep) or not 0 <= f <= 1 or not 0 <= aep <= 1:
        msg = "WP06 expected F and AEP must be finite probabilities"
        raise ValueError(msg)
    if not isclose(f + aep, 1.0, rel_tol=0.0, abs_tol=2e-14):
        msg = "WP06 expected non-exceedance and exceedance probabilities are inconsistent"
        raise ValueError(msg)
    if not isfinite(output.zero_survival_weight) or not 0 <= output.zero_survival_weight <= 1:
        msg = "WP06 zero-survival posterior weight must lie within [0, 1]"
        raise ValueError(msg)
    points = [
        ResultPoint(
            meaning=meaning,
            value=value,
            unit="probability",
            input_discharge=q,
            input_discharge_unit=unit,
        )
        for meaning, value in (
            (ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY, f),
            (ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY, aep),
        )
    ]
    periods, warnings = _return_period_points(output, unit)
    points.extend(periods)
    if output.zero_survival_weight:
        warnings.append(
            f"WP06 discharge={q:.17g} {unit}: zero-survival posterior weight "
            f"{output.zero_survival_weight:.17g}; infinite component periods are not replaced by sentinels."
        )
    return points, warnings


def wp06_output_analysis_result(
    *,
    provenance: RunProvenance,
    parameters: tuple[ParameterSummary, ...],
    discharge_unit: str,
    quantiles: Sequence[QuantileOutputs] = (),
    probabilities: Sequence[ProbabilityOutputs] = (),
    warnings: Sequence[str] = (),
) -> AnalysisResult:
    """Bridge supplied WP06 outputs to the WP08 typed formatter, without fitting.

    Use method_id='quantile-estimands' and provide the true upstream version and
    source-coordinate parameter summaries. The formatter later checks provenance,
    rejects invalid values and ALWAYS marks legacy conformance unestablished.
    """
    unit = _unit(discharge_unit)
    if provenance.method_id != "quantile-estimands":
        msg = "WP06 output reporting requires provenance method_id='quantile-estimands'"
        raise ValueError(msg)
    if provenance.compatibility_profile != LEGACY_FLIKE_PROFILE.identifier:
        msg = "WP06 output reporting requires the explicitly declared legacy target profile"
        raise ValueError(msg)
    if not parameters or not (quantiles or probabilities):
        msg = "WP06 output reporting needs source-coordinate parameters and at least one table"
        raise ValueError(msg)
    points: list[ResultPoint] = []
    notes = list(warnings)
    notes.append("WP06 outputs are mathematical components; legacy Flike fitting and interval rules are unqualified.")
    for output in quantiles:
        extra, labels = _quantile_points(output, unit)
        points.extend(extra)
        notes.extend(labels)
    for output in probabilities:
        extra, labels = _probability_points(output, unit)
        points.extend(extra)
        notes.extend(labels)
    return AnalysisResult(
        provenance=provenance,
        parameters=parameters,
        values=tuple(points),
        warnings=tuple(notes),
    )
