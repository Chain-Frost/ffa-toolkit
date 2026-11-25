"""Source-pinned and independent quadrature tests for the USGS 1.0.7 comparator.

The 08066300 rank probabilities are printed R examples, not native Flike results.
"""

import json
from math import isclose, sqrt
from sys import float_info

import pytest
from numpy.polynomial.legendre import leggauss
from scipy.integrate import IntegrationWarning

from ffa_methods import mgbt_pvalues
from ffa_methods.guidance import explain
from ffa_methods.mgbt_pvalues import (
    MgbPvalueNumericalError,
    calculate_usgs_mgbt_1_0_7,
    usgs_mgbt_1_0_7_rank_pvalue,
)
from tests.test_wp07_low_flow import USGS_08066300, USGS_RANK_PVALUES


def test_all_25_printed_source_rank_probabilities_and_selection() -> None:
    result = calculate_usgs_mgbt_1_0_7(USGS_08066300, discharge_unit="cfs")
    assert result.source_revision == "aa4356208f05d6036f3f425ec37c656da345bba0"
    assert result.statistics.original_flows == USGS_08066300
    assert result.statistics.convention.discharge_unit == "cfs"
    assert result.statistics.convention.log_floor_discharge == sqrt(float_info.epsilon)
    assert len(result.rank_pvalues) == len(result.absolute_error_estimates) == 25
    # Printed eight digits are not eight-digit numeric accuracy: R 1.0.7
    # integrates with a looser default, and has a Monte Carlo fallback.
    # The independent deterministic calculation's maximum observed difference
    # is about 1.57e-7; this source-comparator assertion allows 2e-6 absolute.
    assert (
        max(
            abs(computed - published)
            for computed, published in zip(result.rank_pvalues, USGS_RANK_PVALUES, strict=True)
        )
        < 2e-6
    )
    assert all(0.0 <= error <= 1e-8 for error in result.absolute_error_estimates)
    assert result.decision.low_outlier_count == 1
    assert result.decision.threshold == 284
    assert result.decision.original_flows[result.decision.low_outlier_positions[0]] == 55
    assert (result.decision.out_sweep_rank, result.decision.in_sweep_rank, result.decision.zero_in_sweep_rank) == (
        0,
        0,
        1,
    )


def test_gauss_legendre_independent_quadrature_agrees_at_first_rank() -> None:
    result = calculate_usgs_mgbt_1_0_7((1, 20, 40, 75, 120, 170, 190, 250, 300, 500), discharge_unit="m3/s")
    eta = result.statistics.rank_omegas[0]
    a = sqrt(float_info.epsilon)
    centre = 0.5
    half_width = (1.0 - 2.0 * a) * 0.5
    nodes, weights = leggauss(1024)
    # Different quadrature family (fixed-order Gaussian vs adaptive
    # Gauss-Kronrod); 1,024 nodes resolve this endpoint-sensitive integral
    # to about 2.4e-8, verified separately from QUADPACK's 1e-9 budget.
    independent = half_width * sum(
        float(weight) * mgbt_pvalues.conditional_rank_integrand(centre + half_width * float(node), 10, 1, eta)
        for node, weight in zip(nodes, weights, strict=True)
    )
    p, err = usgs_mgbt_1_0_7_rank_pvalue(10, 1, eta, absolute_error=1e-9, relative_error=1e-9)
    assert err <= 1e-9
    assert isclose(p, independent, abs_tol=5e-8, rel_tol=0)


def test_rank_probability_increases_with_less_negative_statistic() -> None:
    values = [usgs_mgbt_1_0_7_rank_pvalue(12, 2, eta)[0] for eta in (-3.0, -2.0, -1.0)]
    assert 0.0 < values[0] < values[1] < values[2] < 1.0


def test_original_source_permutation_does_not_reassign_rank_pvalues_to_years() -> None:
    flows = (1.0, 12.0, 25.0, 30.0, 40.0, 45.0, 75.0, 140.0, 200.0, 300.0)
    first = calculate_usgs_mgbt_1_0_7(flows, discharge_unit="m3/s")
    reversed_flows = tuple(reversed(flows))
    second = calculate_usgs_mgbt_1_0_7(reversed_flows, discharge_unit="m3/s")
    assert first.statistics.ordered_positions != second.statistics.ordered_positions
    assert first.rank_pvalues == second.rank_pvalues
    assert first.decision.threshold == second.decision.threshold
    assert second.statistics.original_flows == reversed_flows


def test_explicit_zeros_and_calculation_floor_do_not_change_original_flows() -> None:
    flows = (0, 0, 4, 12, 30, 45, 50, 70, 95, 120)
    result = calculate_usgs_mgbt_1_0_7(flows, discharge_unit="cfs")
    assert result.statistics.original_flows == flows
    assert result.statistics.ordered_log10_flows[0] == result.statistics.ordered_log10_flows[1]
    assert all(0 < p < 1 for p in result.rank_pvalues)


@pytest.mark.parametrize(
    ("n", "rank", "eta"),
    [(4, 1, -1.0), (10, 0, -1.0), (10, 6, -1.0), (10, 2, float("nan"))],
)
def test_invalid_statistic_or_rank_rejected(n: int, rank: int, eta: float) -> None:
    with pytest.raises(ValueError, match=r"reference ranks|reference rank statistic"):
        usgs_mgbt_1_0_7_rank_pvalue(n, rank, eta)


@pytest.mark.parametrize("eta", [-1e6, 1e6])
def test_numerical_endpoint_saturation_fails_closed(eta: float) -> None:
    with pytest.raises(MgbPvalueNumericalError):
        usgs_mgbt_1_0_7_rank_pvalue(12, 2, eta)


def test_invalid_integration_tolerances_rejected() -> None:
    for absolute_error, relative_error in ((-1.0, 1e-8), (1e-8, 0.0), (float("nan"), 1e-8)):
        with pytest.raises(ValueError, match="errors must be"):
            usgs_mgbt_1_0_7_rank_pvalue(10, 1, -2.0, absolute_error=absolute_error, relative_error=relative_error)


def test_integration_warning_and_error_budget_are_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_integration(*args: object, **kwargs: object) -> tuple[float, float]:
        del args, kwargs
        msg = "source quadrature failure"
        raise IntegrationWarning(msg)

    monkeypatch.setattr(mgbt_pvalues, "quad", failed_integration)
    with pytest.raises(MgbPvalueNumericalError, match="integration failed"):
        usgs_mgbt_1_0_7_rank_pvalue(10, 1, -2.0)

    def over_budget(*args: object, **kwargs: object) -> tuple[float, float]:
        del args, kwargs
        return 0.1, 1e-3

    monkeypatch.setattr(mgbt_pvalues, "quad", over_budget)
    with pytest.raises(MgbPvalueNumericalError, match="exceeds its tolerance"):
        usgs_mgbt_1_0_7_rank_pvalue(10, 1, -2.0)


def test_invalid_source_cohort_is_rejected_by_existing_statistic_adapter() -> None:
    with pytest.raises(ValueError, match="finite nonnegative"):
        calculate_usgs_mgbt_1_0_7((1, 2, 3, -1, 5), discharge_unit="cfs")
    with pytest.raises(ValueError, match="positive upper-sample"):
        calculate_usgs_mgbt_1_0_7((1, 3, 3, 3, 3, 3), discharge_unit="cfs")
    with pytest.raises(ValueError, match="physical discharge units"):
        calculate_usgs_mgbt_1_0_7((1, 2, 3, 4, 5), discharge_unit=" ")


def test_source_pinned_comparator_is_visible_in_packaged_guidance() -> None:
    """Keep research implementation details distinct from the unqualified legacy method."""
    markdown = explain("mgb-low-flow-screen", output_format="markdown")
    summary = explain("mgb-low-flow-screen", output_format="text")
    data = json.loads(explain("mgb-low-flow-screen", output_format="json"))
    method = data["methods"][0]

    assert method["method_id"] == "mgb-low-flow-screen"
    assert method["implementation_status"] == "not_implemented"
    assert method["qualification_status"] == "not_started"
    assert any("conditional rank p-value quadrature" in item for item in method["evidence_gaps"])
    assert "mgbt_pvalues.calculate_usgs_mgbt_1_0_7" in markdown
    assert "estimated absolute integration error" in markdown
    assert "This comparison is **not** an original Flike run" in markdown
    assert "External USGS MGBT 1.0.7" in summary
    assert "not_implemented" in summary
