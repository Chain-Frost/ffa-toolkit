"""MGBT rank-statistic verification against published and analytic comparators."""

from math import isclose, log10, sqrt
from sys import float_info

import pytest

from ffa_methods.low_flow import MgbSweepSettings, select_mgb_candidates
from ffa_methods.mgbt_statistic import MgbStatisticConvention, calculate_mgb_rank_statistics, verify_mgb_cohort
from tests.test_wp07_low_flow import USGS_08066300, USGS_RANK_PVALUES

# Printed omegas for station 08066300 in the MGBT 1.0.7 public reference manual.
# https://mirrors.sau.edu.cn/CRAN/web/packages/MGBT/refman/MGBT.html
# Six printed decimal places are a formatting precision, not a native Flike tolerance.
PUBLISHED_08066300_OMEGAS = (
    -3.781980,
    -2.268554,
    -2.393569,
    -2.341027,
    -2.309990,
    -2.237571,
    -2.028614,
    -1.928391,
    -1.720404,
    -1.673523,
    -1.727138,
    -1.671534,
    -1.661346,
    -1.391819,
    -1.293324,
    -1.246974,
    -1.276485,
    -1.272878,
    -1.280917,
    -1.310286,
    -1.372402,
    -1.434898,
    -1.226588,
    -1.237743,
    -1.276794,
)


def _source(floor: float = sqrt(float_info.epsilon), unit: str = "cfs") -> MgbStatisticConvention:
    source = (
        "CRAN MGBT 1.0.7 MGBT17c (sqrt machine epsilon)"
        if floor == sqrt(float_info.epsilon)
        else "explicit numerical log-floor comparison; not MGBT17c defaults"
    )
    return MgbStatisticConvention(source=source, discharge_unit=unit, log_floor_discharge=floor)


def test_all_25_published_usgs_rank_statistics_not_just_outlier_count() -> None:
    result = calculate_mgb_rank_statistics(USGS_08066300, convention=_source())
    assert result.candidate_count == 25
    assert len(result.ordered_log10_flows) == 51
    assert [result.original_flows[i] for i in result.ordered_positions[:4]] == [55, 284, 292, 351]
    for computed, printed in zip(result.rank_omegas, PUBLISHED_08066300_OMEGAS, strict=True):
        assert isclose(computed, printed, rel_tol=0, abs_tol=5.5e-7)

    decision = select_mgb_candidates(
        USGS_08066300,
        USGS_RANK_PVALUES,
        settings=MgbSweepSettings.usgs_mgbt17c_1_0_7(),
        pvalue_provenance="USGS 1.0.7 published numerical example",
    )
    verify_mgb_cohort(result, decision)
    assert result.ordered_positions[0] == decision.low_outlier_positions[0]
    assert result.original_flows == USGS_08066300


def test_simple_hand_calculated_upper_sample_variance_and_floor() -> None:
    flows = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
    result = calculate_mgb_rank_statistics(flows, convention=_source())
    assert result.ordered_log10_flows == (0, 1, 2, 3, 4, 5)
    assert isclose(result.rank_omegas[0], -3 / sqrt(2.5), rel_tol=1e-13)
    assert isclose(result.rank_omegas[1], -2.5 / sqrt(5 / 3), rel_tol=1e-13)
    assert result.candidate_count == 3


def test_positive_flow_statistic_is_scale_and_permutation_invariant_without_floor_activation() -> None:
    base = (2, 500, 40, 4, 200, 80, 7, 12)
    reference = calculate_mgb_rank_statistics(base, convention=_source())
    rescaled = calculate_mgb_rank_statistics(tuple(100 * value for value in base), convention=_source(unit="scaled"))
    assert all(
        isclose(a, b, rel_tol=0, abs_tol=1e-13)
        for a, b in zip(reference.rank_omegas, rescaled.rank_omegas, strict=True)
    )
    permuted = (base[3], base[7], base[0], base[5], base[2], base[1], base[4], base[6])
    alternate = calculate_mgb_rank_statistics(permuted, convention=_source())
    assert all(
        isclose(a, b, rel_tol=0, abs_tol=1e-13)
        for a, b in zip(reference.rank_omegas, alternate.rank_omegas, strict=True)
    )
    assert reference.original_flows != alternate.original_flows
    assert reference.ordered_positions != alternate.ordered_positions


def test_zero_log_floor_is_source_choice_not_changed_measurement() -> None:
    observed = (0.0, 0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
    r = calculate_mgb_rank_statistics(observed, convention=_source(1e-8))
    peakfq = calculate_mgb_rank_statistics(observed, convention=_source(1e-88))
    assert r.original_flows == peakfq.original_flows == observed
    assert r.ordered_log10_flows[:2] == (-8, -8)
    assert peakfq.ordered_log10_flows[:2] == (-88, -88)
    assert r.rank_omegas != peakfq.rank_omegas
    assert r.ordered_positions[:2] == (0, 1)
    assert isclose(r.ordered_log10_flows[2], log10(1), abs_tol=0.0)


def test_pvalue_and_rank_evidence_fail_closed_for_different_cohorts() -> None:
    statistics = calculate_mgb_rank_statistics((1, 2, 3, 4, 5, 6), convention=_source())
    settings = MgbSweepSettings.usgs_mgbt17c_1_0_7()
    same = select_mgb_candidates((1, 2, 3, 4, 5, 6), (0.4, 0.4, 0.4), settings=settings, pvalue_provenance="unit test")
    verify_mgb_cohort(statistics, same)
    mismatched = select_mgb_candidates(
        (2, 1, 3, 4, 5, 6), (0.4, 0.4, 0.4), settings=settings, pvalue_provenance="unit test"
    )
    with pytest.raises(ValueError, match="same ordered measurement cohort"):
        verify_mgb_cohort(statistics, mismatched)
    fewer_ranks = calculate_mgb_rank_statistics((1, 2, 3, 4, 5, 6), convention=_source(), candidate_count=2)
    with pytest.raises(ValueError, match="same ordered measurement cohort"):
        verify_mgb_cohort(fewer_ranks, same)


def test_reject_unidentified_units_invalid_floor_unusable_variance_and_counts() -> None:
    with pytest.raises(ValueError, match="physical discharge units"):
        MgbStatisticConvention(source="manual", discharge_unit="", log_floor_discharge=1e-8)
    for floor in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite positive"):
            MgbStatisticConvention(source="manual", discharge_unit="cfs", log_floor_discharge=floor)
    with pytest.raises(ValueError, match="five"):
        calculate_mgb_rank_statistics((1, 2, 3, 4), convention=_source())
    with pytest.raises(ValueError, match="nonnegative"):
        calculate_mgb_rank_statistics((1, 2, -3, 4, 5, 6), convention=_source())
    with pytest.raises(ValueError, match="candidate_count"):
        calculate_mgb_rank_statistics((1, 2, 3, 4, 5, 6), convention=_source(), candidate_count=4)
    with pytest.raises(ValueError, match="variance"):
        calculate_mgb_rank_statistics((2, 2, 2, 2, 2, 2), convention=_source())
    with pytest.raises(ValueError, match="variance"):
        calculate_mgb_rank_statistics((1, 2, 3, 4, 4, 4), convention=_source())


def test_constant_upper_sample_never_creates_spurious_roundoff_variance() -> None:
    # Codex independent review: with Python 3.14 and n=10, the sample mean
    # can be one ULP from identical log10(7) values. Even if a computed
    # squared residual is positive, W(r) is undefined for a constant upper set.
    flows = (1, 2, 3, 4, 5, 7, 7, 7, 7, 7)
    with pytest.raises(ValueError, match=r"rank 5.*all values equal"):
        calculate_mgb_rank_statistics(flows, convention=_source())

    # A genuinely nonconstant upper sample remains admissible, without a
    # guessed absolute or relative near-constant variance cutoff.
    distinct = (1, 2, 3, 4, 5, 7, 7, 7, 7, 7.0001)
    result = calculate_mgb_rank_statistics(distinct, convention=_source())
    assert result.candidate_count == 5


def test_equality_at_sweep_significance_is_not_strictly_significant() -> None:
    settings = MgbSweepSettings.usgs_mgbt17c_1_0_7()
    flows = (1, 2, 3, 4, 5, 6, 7, 8)
    equality = select_mgb_candidates(
        flows, (0.1, 0.005, 0.1, 0.1), settings=settings, pvalue_provenance="synthetic equality"
    )
    assert (equality.out_sweep_rank, equality.in_sweep_rank, equality.zero_in_sweep_rank) == (0, 0, 0)
    lower = select_mgb_candidates(
        flows, (0.099, 0.0049, 0.1, 0.1), settings=settings, pvalue_provenance="synthetic boundary"
    )
    assert (lower.out_sweep_rank, lower.in_sweep_rank, lower.zero_in_sweep_rank) == (2, 2, 2)
    custom = MgbSweepSettings(
        source_variant="synthetic significance comparison", alpha_out=0.005, alpha_in=0.05, alpha_zero_in=0.1
    )
    inward = select_mgb_candidates(
        flows, (0.01, 0.02, 0.05, 0.2), settings=custom, pvalue_provenance="synthetic inward equality"
    )
    assert inward.in_sweep_rank == 2
    assert inward.zero_in_sweep_rank == 3
