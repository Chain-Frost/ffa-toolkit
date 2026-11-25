"""Source-pinned report-internal LP3 consistency; NOT a verified native Flike rerun.

The two reports are only candidate pairs: this test does not infer which saved
project, controls, priors, seeds or rating settings produced either report.
All 40 reported expected-parameter quantiles are checked at *printed* precision,
with uncertainty propagated from the rounded posterior-mean coordinates.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pytest

from ffa_methods.distributions import LogCoordinates, LogPearsonIII


@dataclass(frozen=True, slots=True)
class _ReportCase:
    name: str
    filename: str
    staged_sha256: str
    mean: float
    log_sd: float
    skew: float
    upper_bound: float
    expected_rows: int
    mean_zero_logit: float | None = None


_CASES = (
    _ReportCase(
        name="case-a",
        filename="case-a/case-a_Bayes_Out.txt",
        staged_sha256="7372edd66bf5d9b919c344841a091c15d6ac9cacb0d637760c9214018b5c796c",
        mean=5.45975,
        log_sd=0.51810,
        skew=-0.75235,
        upper_bound=20386.9,
        expected_rows=21,
    ),
    _ReportCase(
        name="case-b",
        filename="case-b/case-b_Bayes_Out.txt",
        staged_sha256="af73a8a30b0de9308804001b2507b13c9f688fa0d86f287cc05135d65f653cc1",
        mean=4.51784,
        log_sd=0.73566,
        skew=-0.71925,
        upper_bound=30354.4,
        expected_rows=19,
        mean_zero_logit=-2.02714,
    ),
)

# The printed posterior parameter means have five digits after the decimal,
# while expected-parameter quantiles have two and upper bounds have one.
_PARAMETER_HALF_UNIT = 0.000005
_QUANTILE_HALF_UNIT = 0.005
_BOUND_HALF_UNIT = 0.05
_REPO = Path(__file__).resolve().parents[1]


def _load_source_report(case: _ReportCase) -> str:
    path = _REPO / "reference" / "legacy-candidates" / case.filename
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == case.staged_sha256
    body = raw.decode("latin-1")
    assert body.count("FLIKE program version 5.0.306.0") == 1
    assert body.count("FLIKE file version 3.10") == 1
    assert f"1 Mean (loge flow)                 {case.mean:.5f}" in body
    assert f"2 loge [Std dev (loge flow)]       {case.log_sd:.5f}" in body
    assert f"3 Skew (loge flow)                {case.skew:.5f}" in body
    if case.mean_zero_logit is not None:
        assert f"{case.mean_zero_logit:.5f}" in body
        assert "The most probable value of P is    0.1250" in body
    return body


def _reported_expected_parameter_rows(body: str, expected_count: int) -> tuple[tuple[float, float], ...]:
    # Test-only extraction of one fixed six-column report table. The WP02
    # parser owns general report parsing and preservation of raw fields.
    heading = "  AEP 1 in Y    Exp parameter"
    assert body.count(heading) == 1
    tail = body.split(heading, 1)[1]
    assert "Flood        Expected" in tail
    table = tail.split("Flood        Expected", 1)[0]
    rows: list[tuple[float, float]] = []
    for line in table.splitlines():
        fields = line.split()
        if len(fields) == 6 and re.fullmatch(r"\d+\.\d{3}", fields[0]):
            rows.append((float(fields[0]), float(fields[1])))
    assert len(rows) == expected_count
    assert len({years for years, _ in rows}) == expected_count
    assert all(years > 1.0 and discharge > 0.0 for years, discharge in rows)
    return tuple(rows)


def _zero_threshold_probability(mean_logit: float | None) -> float:
    # These report vectors use the expected logit parameter, not the reported
    # separate most-probable threshold probability or a posterior mean of P.
    return 0.0 if mean_logit is None else 1.0 / (1.0 + math.exp(-mean_logit))


def _replayed_quantile(
    return_period: float, mean: float, log_sd: float, skew: float, mean_zero_logit: float | None
) -> float:
    p_zero = _zero_threshold_probability(mean_zero_logit)
    # Above the declared zero threshold:
    # Pr(Q > q) = (1-p_zero) Pr(Q_positive > q).
    conditional_survival = (1.0 / return_period) / (1.0 - p_zero)
    return LogPearsonIII(LogCoordinates(mean, log_sd), skew).isf(conditional_survival)


def _rounded_parameter_corners(case: _ReportCase) -> tuple[tuple[float, float, float, float | None], ...]:
    count = 4 if case.mean_zero_logit is not None else 3
    corners: list[tuple[float, float, float, float | None]] = []
    for changes in product((-_PARAMETER_HALF_UNIT, _PARAMETER_HALF_UNIT), repeat=count):
        logit = case.mean_zero_logit + changes[3] if case.mean_zero_logit is not None else None
        corners.append((case.mean + changes[0], case.log_sd + changes[1], case.skew + changes[2], logit))
    return tuple(corners)


@pytest.mark.parametrize("case", _CASES, ids=("case-a", "case-b"))
def test_expected_parameter_quantiles_match_report_with_declared_print_rounding(case: _ReportCase) -> None:
    report = _load_source_report(case)
    rows = _reported_expected_parameter_rows(report, case.expected_rows)
    corners = _rounded_parameter_corners(case)
    for years, printed in rows:
        predicted = [_replayed_quantile(years, *parameters) for parameters in corners]
        assert all(math.isfinite(q) and q > 0.0 for q in predicted)
        # Compare the allowable source-parameter interval to the reported
        # output rounding interval, not to a broad relative flood tolerance.
        assert min(predicted) <= printed + _QUANTILE_HALF_UNIT + 1e-10, (case.name, years, printed)
        assert max(predicted) >= printed - _QUANTILE_HALF_UNIT - 1e-10, (case.name, years, printed)


@pytest.mark.parametrize("case", _CASES, ids=("case-a", "case-b"))
def test_reported_upper_bound_is_consistent_with_negative_skew_lp3(case: _ReportCase) -> None:
    report = _load_source_report(case)
    match = re.search(r"Upper bound\s*=\s*([0-9.]+)", report)
    assert match is not None
    assert float(match.group(1)) == case.upper_bound
    endpoints = [
        LogPearsonIII(LogCoordinates(mean, log_sd), skew).support()[1]
        for mean, log_sd, skew, _ in _rounded_parameter_corners(case)
    ]
    assert all(math.isfinite(endpoint) for endpoint in endpoints)
    assert min(endpoints) <= case.upper_bound + _BOUND_HALF_UNIT
    assert max(endpoints) >= case.upper_bound - _BOUND_HALF_UNIT


def test_case_b_parameter_four_is_expected_logit_not_most_probable_probability() -> None:
    case = _CASES[1]
    assert case.mean_zero_logit is not None
    rows = dict(_reported_expected_parameter_rows(_load_source_report(case), case.expected_rows))
    reported_two_year = rows[2.0]
    replayed = _replayed_quantile(2.0, case.mean, case.log_sd, case.skew, case.mean_zero_logit)
    most_probable_as_logit = math.log(0.1250 / (1.0 - 0.1250))
    incorrect = _replayed_quantile(2.0, case.mean, case.log_sd, case.skew, most_probable_as_logit)
    assert math.isclose(replayed, reported_two_year, rel_tol=0.0, abs_tol=0.005)
    # The incorrect choice differs by orders of magnitude more than the
    # display rounding interval and cannot be used as a substitution.
    assert abs(incorrect - reported_two_year) > 2.0


def test_shipped_wp03_resources_exclude_candidate_specific_values() -> None:
    """Regression for review P1: source-only data must not enter wheel/sdist resources.

    The CI distribution check separately verifies that reference/ and tests/
    are not included in the built artifacts. This checks their *copied
    contents* do not leak into the two shipped WP03 text resources.
    """
    source_only_terms = ["case_b", "case_b", "case_a", "scenario-a", "scenario-b", "CASE_A", "CASE_B"]
    for case in _CASES:
        source_only_terms.extend(
            (
                case.staged_sha256,
                f"{case.mean:.5f}",
                f"{case.log_sd:.5f}",
                f"{case.skew:.5f}",
                f"{case.upper_bound:.1f}",
            )
        )
        if case.mean_zero_logit is not None:
            source_only_terms.append(f"{case.mean_zero_logit:.5f}")
    for filename in ("docs/distribution-kernels.md", "docs/method-registry.json"):
        shipped_resource = (_REPO / filename).read_text(encoding="utf-8").lower()
        for forbidden in source_only_terms:
            assert forbidden.lower() not in shipped_resource, (filename, forbidden)
