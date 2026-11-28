"""WP08 presentation of actual WP06 mathematical outputs, never native FLIKE goldens."""

import csv
import io
import json
from dataclasses import replace
from math import erf, exp, inf, log, sqrt
from statistics import NormalDist
from typing import cast

import pytest

from ffa_methods import __version__
from ffa_methods.bayesian_outputs import (
    ImportanceParticles,
    ProbabilityOutputs,
    QuantileOutputs,
    posterior_moments,
    probability_outputs,
    quantile_outputs,
)
from ffa_methods.compatibility import LEGACY_FLIKE_PROFILE
from ffa_methods.contracts import (
    AnalysisResult,
    EvidenceStatus,
    ParameterSummary,
    ResultMeaning,
    ResultPoint,
    RunProvenance,
)
from ffa_methods.distributions import Gumbel, LogCoordinates, Lognormal
from ffa_methods.reporting import render_report_files, report_document
from ffa_methods.wp06_report_bridge import wp06_output_analysis_result


def _near(expected: float) -> object:
    """Isolate pytest's incomplete typed approximation stub."""
    return pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


def _provenance(*, evidence: EvidenceStatus = EvidenceStatus.CANDIDATE_PAIR) -> RunProvenance:
    return RunProvenance(
        compatibility_profile=LEGACY_FLIKE_PROFILE.identifier,
        method_id="quantile-estimands",
        method_version="test-only-source-coordinate-model/1",
        package_version=__version__,
        evidence_status=evidence,
        input_sha256=None,
        settings_sha256=None,
    )


class _TestContinuousKernel:
    """Adapt keyword-only protocol names for the WP03 mathematical kernels in tests.

    The production kernel's ppf/isf parameter names differ from the WP01
    protocol; this is a typed TEST shim, not a new public kernel or fitting API.
    """

    def __init__(self, kernel: Lognormal | Gumbel) -> None:
        self.kernel = kernel

    def logpdf(self, discharge: float) -> float:
        return self.kernel.logpdf(discharge)

    def logcdf(self, discharge: float) -> float:
        return self.kernel.logcdf(discharge)

    def logsf(self, discharge: float) -> float:
        return self.kernel.logsf(discharge)

    def cdf(self, discharge: float) -> float:
        return self.kernel.cdf(discharge)

    def sf(self, discharge: float) -> float:
        return self.kernel.sf(discharge)

    def ppf(self, non_exceedance_probability: float) -> float:
        return self.kernel.ppf(non_exceedance_probability)

    def isf(self, survival_probability: float) -> float:
        return self.kernel.isf(survival_probability)


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        msg = "expected typed finite numeric output"
        raise TypeError(msg)
    return float(value)


def _actual_wp06_outputs() -> tuple[tuple[ParameterSummary, ...], QuantileOutputs, ProbabilityOutputs]:
    """Compute two-component posterior using actual WP03/WP06 mathematical code."""
    p = ImportanceParticles(
        coordinates=("lp3.mean_loge_flow",),
        draws=((0.0,), (log(4.0),)),
        log_likelihood=(log(3.0), 0.0),
        log_prior=(0.0, 0.0),
        log_proposal=(0.0, 0.0),
        provenance=_provenance(),
        proposal_description="synthetic known two-point proposal; not Flike",
        random_stream_description="no random numbers: exact discrete weights",
    )

    def kernel(coords: tuple[float, ...]) -> _TestContinuousKernel:
        return _TestContinuousKernel(Lognormal(LogCoordinates(mean=coords[0], log_sd=0.0)))

    q = quantile_outputs(p, 0.8, kernel, interval_mass=0.9)
    prob = probability_outputs(p, 2.0, kernel, interval_mass=0.9)
    return posterior_moments(p).parameters, q, prob


def test_wp06_producer_to_wp08_json_csv_text_preserves_distinct_estimands() -> None:
    parameters, q, p = _actual_wp06_outputs()
    result = wp06_output_analysis_result(
        provenance=_provenance(evidence=EvidenceStatus.QUALIFIED_REFERENCE),
        parameters=parameters,
        discharge_unit="m3/s",
        quantiles=(q,),
        probabilities=(p,),
        warnings=("Synthetic quadrature, not a Flike run.",),
    )
    files = render_report_files(result, settings={"synthetic_test": True, "interval_mass": 0.9})
    doc = json.loads(files["results.json"])
    rows = cast(list[dict[str, object]], doc["results"])

    def find(meaning: ResultMeaning) -> list[dict[str, object]]:
        return [row for row in rows if row["meaning"] == meaning.value]

    # Analytic lognormal mixture with weights 3/4 and 1/4 in natural-log coordinates.
    z = NormalDist().inv_cdf(0.8)
    q_base = exp(z)
    assert _number(find(ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS)[0]["value"]) == _near(sqrt(2.0) * q_base)
    assert _number(find(ResultMeaning.POSTERIOR_MEAN_QUANTILE)[0]["value"]) == _near(1.75 * q_base)
    assert _number(find(ResultMeaning.POSTERIOR_MEDIAN_QUANTILE)[0]["value"]) == _near(q_base)
    predictive = _number(find(ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE)[0]["value"])
    assert predictive != _number(find(ResultMeaning.POSTERIOR_MEAN_QUANTILE)[0]["value"])
    quantile_limits = find(ResultMeaning.QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT)
    assert [(r["probability_limit_side"], r["reported_probability_level"]) for r in quantile_limits] == [
        ("lower", 0.9),
        ("upper", 0.9),
    ]
    assert "Flike interpolation unverified" in str(quantile_limits[0]["probability_limit_construction"])
    assert quantile_limits[0]["input_non_exceedance_probability"] == 0.8
    assert _number(quantile_limits[0]["input_annual_exceedance_probability"]) == _near(0.2)

    cdf0 = (1.0 + erf(log(2.0) / sqrt(2.0))) / 2.0
    cdf1 = (1.0 + erf(log(0.5) / sqrt(2.0))) / 2.0
    expected_f = 0.75 * cdf0 + 0.25 * cdf1
    expected_aep = 1.0 - expected_f
    assert _number(find(ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY)[0]["value"]) == _near(expected_f)
    assert _number(find(ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY)[0]["value"]) == _near(expected_aep)
    period = _number(find(ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP)[0]["value"])
    assert period == _near(1.0 / expected_aep)
    assert len(find(ResultMeaning.RETURN_PERIOD_MONTE_CARLO_PROBABILITY_LIMIT)) == 2
    assert find(ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP)[0]["input_discharge"] == 2.0
    assert find(ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP)[0]["input_discharge_unit"] == "m3/s"
    log_moments = find(ResultMeaning.LOG10_QUANTILE_POSTERIOR_MOMENT)
    assert [row["log10_moment_kind"] for row in log_moments] == ["mean", "standard_deviation"]

    assert doc["method_qualification"] == "not_established_by_formatter"
    assert doc["legacy_comparison_status"] == "not_established_by_formatter"
    assert doc["provenance"]["reference_evidence_status"] == "qualified_reference"
    assert any("not Monte Carlo standard errors" in warning for warning in doc["warnings"])
    assert "not a Flike run" in files["report.txt"]
    assert "posterior_predictive_quantile" in files["report.txt"]
    csv_rows = list(csv.DictReader(io.StringIO(files["results.csv"])))
    assert len(csv_rows) == len(rows)
    assert csv_rows[0]["meaning"] == "quantile_at_expected_parameters"
    assert any(row["meaning"] == "expected_annual_exceedance_probability" for row in csv_rows)


def test_undefined_log10_and_infinite_periods_are_explicit_omissions() -> None:
    parameters, q, p = _actual_wp06_outputs()
    q = replace(
        q,
        nonpositive_quantile_weight=0.25,
        mean_log10_quantile=None,
        standard_deviation_log10_quantile=None,
        probability_limits=(0.0, q.probability_limits[1]),
    )
    p = replace(
        p,
        expected_non_exceedance=1.0,
        expected_aep=0.0,
        return_period=inf,
        return_period_limits=(inf, inf),
        zero_survival_weight=1.0,
    )
    result = wp06_output_analysis_result(
        provenance=_provenance(),
        parameters=parameters,
        discharge_unit="m3/s",
        quantiles=(q,),
        probabilities=(p,),
    )
    doc = report_document(result, settings={"synthetic_test": True})
    kinds = [str(row["meaning"]) for row in cast(list[dict[str, object]], doc["results"])]
    assert "log10_quantile_posterior_moment" not in kinds
    assert "return_period_from_expected_aep" not in kinds
    assert "return_period_monte_carlo_probability_limit" not in kinds
    assert kinds.count("quantile_monte_carlo_probability_limit") == 2
    assert kinds.count("expected_annual_exceedance_probability") == 1
    warnings = " ".join(cast(list[str], doc["warnings"]))
    assert "log10 moments undefined" in warnings
    assert "+infinity; no finite surrogate" in warnings
    assert "infinite component periods" in warnings
    assert doc["legacy_comparison_status"] == "not_established_by_formatter"
    assert "Infinity" not in render_report_files(result, settings={"synthetic_test": True})["results.json"]


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("incorrect_method", "provenance method_id"),
        ("cross_profile", "target profile"),
        ("empty", "at least one table"),
        ("missing_parameters", "source-coordinate parameters"),
        ("unit", "physical discharge unit"),
        ("invalid_q_probability", "non-exceedance probability"),
        ("reversed_limits", "lower quantile probability limit exceeds"),
        ("bad_mass", "central interval mass"),
        ("undefined_moments_inconsistent", "must be undefined"),
        ("missing_moments", "log10 moments missing"),
        ("f_aep_mismatch", "inconsistent"),
        ("wrong_period", "reciprocal expected AEP"),
        ("negative_flow", "finite and nonnegative"),
        ("nan_period", "at least one year"),
        ("zero_aep_finite_period", "zero expected AEP"),
        ("reversed_period_limits", "lower return-period probability limit exceeds"),
    ],
)
def test_bridge_rejects_mislabelled_and_inconsistent_wp06_outputs(kind: str, match: str) -> None:
    parameters, q, p = _actual_wp06_outputs()
    provenance = _provenance()
    unit = "m3/s"
    qs: tuple[QuantileOutputs, ...] = (q,)
    ps: tuple[ProbabilityOutputs, ...] = (p,)
    q_variants = {
        "invalid_q_probability": replace(q, non_exceedance=2.0),
        "reversed_limits": replace(q, probability_limits=(1000.0, 1.0)),
        "bad_mass": replace(q, interval_mass=1.0),
        "undefined_moments_inconsistent": replace(q, nonpositive_quantile_weight=0.2),
        "missing_moments": replace(q, mean_log10_quantile=None),
    }
    p_variants = {
        "f_aep_mismatch": replace(p, expected_aep=0.5),
        "wrong_period": replace(p, return_period=4.0),
        "negative_flow": replace(p, discharge=-5.0),
        "nan_period": replace(p, return_period=float("nan")),
        "zero_aep_finite_period": replace(p, expected_non_exceedance=1.0, expected_aep=0.0),
        "reversed_period_limits": replace(p, return_period_limits=(100.0, 1.0)),
    }
    if kind == "incorrect_method":
        provenance = replace(provenance, method_id="importance-sampling-reference")
    elif kind == "cross_profile":
        provenance = replace(provenance, compatibility_profile="other")
    elif kind == "empty":
        qs, ps = (), ()
    elif kind == "missing_parameters":
        parameters = ()
    elif kind == "unit":
        unit = "probability"
    elif kind in q_variants:
        qs = (q_variants[kind],)
    else:
        qs, ps = (), (p_variants[kind],)
    with pytest.raises(ValueError, match=match):
        wp06_output_analysis_result(
            provenance=provenance,
            parameters=parameters,
            discharge_unit=unit,
            quantiles=qs,
            probabilities=ps,
        )


def test_new_result_meanings_reject_missing_context_and_wrong_units() -> None:
    parameters, _, _ = _actual_wp06_outputs()
    provenance = _provenance()
    invalid = (
        (ResultPoint(meaning=ResultMeaning.POSTERIOR_MEAN_QUANTILE, value=5.0, unit="m3/s"), "input probability"),
        (
            ResultPoint(
                meaning=ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY,
                value=2.0,
                unit="probability",
                input_discharge=1.0,
                input_discharge_unit="m3/s",
            ),
            "within",
        ),
        (
            ResultPoint(
                meaning=ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP,
                value=0.5,
                unit="years",
                input_discharge=1.0,
                input_discharge_unit="m3/s",
            ),
            "one year",
        ),
        (
            ResultPoint(
                meaning=ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP,
                value=10.0,
                unit="probability",
                input_discharge=1.0,
                input_discharge_unit="m3/s",
            ),
            "years",
        ),
        (
            ResultPoint(
                meaning=ResultMeaning.RETURN_PERIOD_MONTE_CARLO_PROBABILITY_LIMIT,
                value=10.0,
                unit="years",
                input_discharge=1.0,
                input_discharge_unit="m3/s",
            ),
            "requires level",
        ),
    )
    for point, message in invalid:
        result = AnalysisResult(provenance=provenance, parameters=parameters, values=(point,))
        with pytest.raises(ValueError, match=message):
            report_document(result, settings={"synthetic_test": True})


def test_no_fabricated_fit_or_qualification_claims_in_bridge() -> None:
    parameters, q, p = _actual_wp06_outputs()
    result = wp06_output_analysis_result(
        provenance=_provenance(),
        parameters=parameters,
        discharge_unit="m3/s",
        quantiles=(q,),
        probabilities=(p,),
    )
    report = report_document(result, settings={"test_only": True})
    provenance = cast(dict[str, object], report["provenance"])
    warnings = cast(list[str], report["warnings"])
    assert provenance["input_sha256"] is None
    assert any("Input SHA-256 is absent" in x for x in warnings)
    assert "native" not in str(report["method_qualification"])
    assert report["method_qualification"] == "not_established_by_formatter"


def test_negative_gumbel_limit_is_preserved_with_explicit_physical_warning() -> None:
    """P2 regression: a valid WP06 negative lower limit must not discard the table."""
    p = ImportanceParticles(
        coordinates=("gumbel.location",),
        draws=((-2.0,), (2.0,)),
        log_likelihood=(log(0.1), log(0.9)),
        log_prior=(0.0, 0.0),
        log_proposal=(0.0, 0.0),
        provenance=_provenance(),
        proposal_description="synthetic two-component Gumbel mixture; not native Flike",
        random_stream_description="deterministic fixed weighted components",
    )

    def kernel(coords: tuple[float, ...]) -> _TestContinuousKernel:
        return _TestContinuousKernel(Gumbel(location=coords[0], log_scale=0.0))

    q = quantile_outputs(p, 0.8, kernel, interval_mass=0.9)
    assert q.probability_limits[0] < 0 < q.probability_limits[1]
    assert q.at_expected_parameters > 0
    assert q.mean_of_quantiles is not None
    assert q.mean_of_quantiles > 0
    assert q.median_of_quantiles > 0
    assert q.predictive_quantile > 0
    assert q.nonpositive_quantile_weight == _near(0.1)
    assert q.mean_log10_quantile is None

    result = wp06_output_analysis_result(
        provenance=_provenance(),
        parameters=posterior_moments(p).parameters,
        discharge_unit="m3/s",
        quantiles=(q,),
    )
    files = render_report_files(result, settings={"synthetic_test": "gumbel_unbounded"})
    doc = json.loads(files["results.json"])
    points = cast(list[dict[str, object]], doc["results"])
    limits = [point for point in points if point["meaning"] == "quantile_monte_carlo_probability_limit"]
    assert len(limits) == 2
    assert limits[0]["probability_limit_side"] == "lower"
    assert _number(limits[0]["value"]) == _near(q.probability_limits[0])
    assert _number(limits[0]["value"]) < 0
    assert _number(limits[1]["value"]) == _near(q.probability_limits[1])
    assert all(point["meaning"] != "log10_quantile_posterior_moment" for point in points)
    warnings = cast(list[str], doc["warnings"])
    assert any("Modelled quantile_monte_carlo_probability_limit is negative" in x for x in warnings)
    assert any("log10 moments undefined" in x for x in warnings)
    assert "mathematical distribution output, not a physically possible discharge" in files["report.txt"]
    csv_rows = list(csv.DictReader(io.StringIO(files["results.csv"])))
    assert any(
        row["meaning"] == "quantile_monte_carlo_probability_limit"
        and row["probability_limit_side"] == "lower"
        and row["value"] is not None
        and float(row["value"]) < 0
        for row in csv_rows
    )
