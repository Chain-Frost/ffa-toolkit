"""Independent mathematical and admission checks for the real-record harness."""

from dataclasses import replace
from math import exp, isclose, log, sqrt

import pytest

from ffa_methods.distributions import GEV, GeneralizedPareto, LogCoordinates, LogPearsonIII
from scripts import validate_pyflood2022_fits as matrix
from scripts.audit_pyflood2022 import PEAK, flood_table
from scripts.validate_pyflood2022_fits import (
    UNIT_FACTOR,
    CheckEvidence,
    Fitted,
    check,
    cohort,
    fit,
    gev_population,
    independent_lh,
    ordinary_population,
    quantile_checks,
)


def test_normalized_pwm_order_statistics_on_five_ordered_values() -> None:
    # Uniform spacing: maxima of subsets of size 2,3,4,5 are 4,4.5,4.8,5.
    actual = independent_lh((5.0, 1.0, 4.0, 2.0, 3.0), 1)
    for value, expected in zip(actual, (4.0, 0.75, 0.0, 0.0), strict=True):
        assert isclose(value, expected, rel_tol=0, abs_tol=1e-13)


def test_population_reference_known_exponential_and_gumbel() -> None:
    # A shifted exponential with mean 2, scale 1 has L2=1/2 and L3=1/6.
    gp = GeneralizedPareto(1.0, 0.0, 0.0)
    lp = LogPearsonIII(LogCoordinates(2.0, 0.0), 2.0)
    for kernel in (gp, lp):
        for value, expected in zip(ordinary_population(kernel), (2.0, 0.5, 1 / 6), strict=True):
            assert isclose(value, expected, rel_tol=0, abs_tol=1e-13)
    # Max of two standard Gumbels has mean gamma + log(2), unchanged scale.
    moments = gev_population(GEV(0.0, 0.0, 0.0), 1)
    assert isclose(moments[0], 0.5772156649015329 + log(2), abs_tol=1e-14)
    assert isclose(moments[1], 1.5 * log(1.5), abs_tol=1e-14)


def test_cohort_retains_gaps_and_does_not_drop_invalid_or_short_rows() -> None:
    rows = "".join(f"{site},{1900 + i * 2},{i + 1}\n" for site in ("001", "002", "003") for i in range(30))
    rows += "002,2001,0\n003,1900,7\n004,2001,1\n"
    table = flood_table(f"SITEID,WATERYEAR,{PEAK}\n{rows}".encode())
    cases, exclusions = cohort(table, {"001", "002", "003", "004"})
    assert cases == [("001", tuple(float(i) for i in range(1, 31)), 29)]
    assert exclusions == {
        "invalid_peak": ["002"],
        "duplicate_water_year": ["003"],
        "fewer_than_30_records": ["004"],
    }


def test_actual_fixed_h_fit_and_corrupted_quantile_detection() -> None:
    # Smooth positive synthetic series, not a published FFA estimate.
    values = tuple(10 + exp(i / 15) for i in range(30))
    result = check("gev-h2", ("synthetic", values, 0))
    assert result.status == "pass", result.reason
    model = fit("gev-h2", values, reference=1.0)
    scaled = fit("gev-h2", tuple(q * UNIT_FACTOR for q in values), reference=UNIT_FACTOR)

    def wrong_quantile(p: float) -> float:
        return model.quantile(p) + 1.0

    wrong = replace(model, quantile=wrong_quantile)
    evidence = CheckEvidence()
    quantile_checks(values, wrong, scaled, evidence)
    assert not evidence.exceptions
    checks = evidence.ratios
    assert checks["reference_quantile"] > 1
    assert checks["cdf_and_survival"] > 1
    assert checks["unit_scaling"] > 1


def test_lognormal_raw_flow_moment_reference() -> None:
    values = tuple(exp(i / sqrt(20)) for i in range(1, 31))
    result = check("lognormal", ("synthetic", values, 0))
    assert result.status == "pass", result.reason


def test_retains_base_diagnostics_when_scaled_refit_and_later_quantile_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    values = tuple(float(i) for i in range(1, 31))
    # Deliberately inconsistent fit: finite upper endpoint of 15 and incorrect sample moments.
    kernel = GEV(10.0, log(5.0), -1.0)

    def partial_quantile(p: float) -> float:
        if p == 0.99:
            msg = "injected later quantile failure"
            raise ValueError(msg)
        return kernel.ppf(p)

    admitted = Fitted(kernel, partial_quantile, (1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0), 0.0)

    def partial_fit(_method: str, _values: tuple[float, ...], *, reference: float) -> Fitted:
        if reference == UNIT_FACTOR:
            msg = "injected scaled refit failure"
            raise ValueError(msg)
        return admitted

    monkeypatch.setattr(matrix, "fit", partial_fit)
    result = check("gev-h0", ("synthetic", values, 0))
    assert result.status == "check_error"
    assert result.outside_support is True
    assert result.errors["sample_moments"] > 1
    assert result.errors["reference_quantile"] <= 1
    assert result.errors["cdf_and_survival"] <= 1
    assert "unit_scaling" not in result.errors
    assert result.check_exceptions == {
        "scaled_refit": "injected scaled refit failure",
        "base_quantile:0.99": "injected later quantile failure",
    }
    assert "sample_moments" in result.reason
    report = matrix.report_method("gev-h0", [("synthetic", values, 0)])
    assert report["outside_fitted_support_sites"] == ["synthetic"]
    assert report["support_unresolved_sites"] == []
    assert report["outcomes"] == {"check_error": 1}


def test_quantile_reference_failure_does_not_stop_probability_or_unit_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    values = tuple(10 + exp(i / 15) for i in range(30))
    model = fit("gev-h2", values, reference=1.0)
    scaled = fit("gev-h2", tuple(q * UNIT_FACTOR for q in values), reference=UNIT_FACTOR)
    original = matrix.independent_quantile

    def partial_reference(kernel: matrix.Kernel, p: float) -> float:
        if p == 0.9:
            msg = "injected reference failure"
            raise ValueError(msg)
        return original(kernel, p)

    monkeypatch.setattr(matrix, "independent_quantile", partial_reference)
    evidence = CheckEvidence()
    quantile_checks(values, model, scaled, evidence)
    assert evidence.exceptions == {"reference_quantile:0.9": "injected reference failure"}
    assert set(evidence.ratios) == {"reference_quantile", "cdf_and_survival", "unit_scaling"}
    assert all(ratio <= 1 for ratio in evidence.ratios.values())
