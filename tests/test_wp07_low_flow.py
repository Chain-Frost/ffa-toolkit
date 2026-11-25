"""Independent MGBT17c decision fixtures and invariant-based WP07 tests."""

from math import isclose, isnan, log

import pytest

from ffa_methods.contracts import AnnualObservation, CensorGroup, ExposureKind, ObservationKind, ObservationSeries
from ffa_methods.low_flow import (
    ArrMgbCensoringProposal,
    MgbSweepSettings,
    ThresholdProbabilityCoordinate,
    apply_approved_arr_mgb_censoring,
    audit_mgb_annual_candidates,
    logit_at_or_below_threshold,
    probability_at_or_below_threshold,
    propose_arr_mgb_censoring,
    select_mgb_candidates,
)

USGS_08066300 = (
    3530,
    284,
    1810,
    9660,
    489,
    292,
    1000,
    2640,
    2910,
    1900,
    1120,
    1020,
    632,
    7160,
    1750,
    2730,
    1630,
    8210,
    4270,
    1730,
    13200,
    2550,
    915,
    11000,
    2370,
    2230,
    4650,
    2750,
    1860,
    13700,
    2290,
    3390,
    5160,
    13200,
    410,
    1890,
    4120,
    3930,
    4290,
    1890,
    1480,
    10300,
    1190,
    2320,
    2480,
    55,
    7480,
    351,
    738,
    2430,
    6700,
)
# Published MGBT 1.0.7 documentation, 51 values, 25 lower-rank p-values.
# These are *external p-values*, not p-values calculated by this package.
USGS_RANK_PVALUES = (
    0.01192184,
    0.30337879,
    0.08198836,
    0.04903091,
    0.02949836,
    0.02700114,
    0.07802324,
    0.11185553,
    0.31531749,
    0.34257170,
    0.21560086,
    0.25950150,
    0.24113157,
    0.72747052,
    0.86190920,
    0.89914152,
    0.84072131,
    0.82381908,
    0.78750571,
    0.70840262,
    0.55379730,
    0.40255392,
    0.79430336,
    0.75515103,
    0.66031442,
)


def _usgs_settings() -> MgbSweepSettings:
    return MgbSweepSettings.usgs_mgbt17c_1_0_7()


def test_published_usgs_51_year_three_sweep_decision() -> None:
    result = select_mgb_candidates(
        USGS_08066300, USGS_RANK_PVALUES, settings=_usgs_settings(), pvalue_provenance="MGBT 1.0.7 example"
    )
    assert result.low_outlier_count == 1
    assert result.threshold == 284
    assert tuple(result.original_flows[i] for i in result.low_outlier_positions) == (55,)
    assert (result.out_sweep_rank, result.in_sweep_rank, result.zero_in_sweep_rank) == (0, 0, 1)
    assert result.original_flows == USGS_08066300
    assert result.settings.source_variant.startswith("USGS")


def test_outward_inward_and_zero_sweeps_are_distinct() -> None:
    settings = _usgs_settings()
    flows = (1, 2, 3, 4, 5, 6, 7, 8)
    outward = select_mgb_candidates(flows, (0.02, 0.02, 0.004, 0.5), settings=settings, pvalue_provenance="synthetic")
    assert (outward.out_sweep_rank, outward.in_sweep_rank, outward.zero_in_sweep_rank) == (3, 3, 3)
    assert outward.threshold == 4
    shallow = select_mgb_candidates(flows, (0.2, 0.001, 0.5, 0.5), settings=settings, pvalue_provenance="synthetic")
    assert (shallow.out_sweep_rank, shallow.in_sweep_rank, shallow.zero_in_sweep_rank) == (2, 2, 0)
    assert shallow.threshold == 3
    no_outlier = select_mgb_candidates(flows, (0.2, 0.2, 0.2, 0.2), settings=settings, pvalue_provenance="synthetic")
    assert no_outlier.threshold is None
    assert no_outlier.low_outlier_count == 0
    sweep_in = select_mgb_candidates(
        flows,
        (0.2, 0.001, 0.005, 0.9),
        settings=MgbSweepSettings(
            source_variant="explicit comparison", alpha_out=0.005, alpha_in=0.01, alpha_zero_in=0.1
        ),
        pvalue_provenance="synthetic",
    )
    assert (sweep_in.out_sweep_rank, sweep_in.in_sweep_rank, sweep_in.zero_in_sweep_rank) == (2, 3, 0)


def test_explicit_zeros_are_preserved_and_ties_across_cutoff_rejected() -> None:
    result = select_mgb_candidates(
        (0, 5, 0, 10, 20, 30),
        (0.0001, 0.0001, 0.3),
        settings=_usgs_settings(),
        pvalue_provenance="synthetic zero test",
    )
    assert result.low_outlier_count == 2
    assert result.threshold == 5
    assert result.low_outlier_positions == (0, 2)
    assert result.original_flows[0] == result.original_flows[2] == 0
    with pytest.raises(ValueError, match="tied flows"):
        select_mgb_candidates(
            (0, 0, 4, 5, 6, 7), (0.001, 0.3, 0.3), settings=_usgs_settings(), pvalue_provenance="synthetic"
        )


@pytest.mark.parametrize("pvalues", [(0.5,), (0.2, float("nan"), 0.3), (-0.01, 0.2, 0.3), (1.1, 0.2, 0.3)])
def test_reject_incomplete_or_invalid_rank_pvalues(pvalues: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="p-value"):
        select_mgb_candidates((1, 2, 3, 4, 5, 6), pvalues, settings=_usgs_settings(), pvalue_provenance="test")


def test_reject_malformed_flow_settings_and_missing_provenance() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        select_mgb_candidates((1, -1, 3, 4, 5, 6), (0.3, 0.3, 0.3), settings=_usgs_settings(), pvalue_provenance="x")
    with pytest.raises(ValueError, match="at least five"):
        select_mgb_candidates((1, 2, 3), (0.2,), settings=_usgs_settings(), pvalue_provenance="x")
    with pytest.raises(ValueError, match="provenance"):
        select_mgb_candidates((1, 2, 3, 4, 5, 6), (0.2, 0.2, 0.2), settings=_usgs_settings(), pvalue_provenance="")
    with pytest.raises(ValueError, match="candidate_count"):
        select_mgb_candidates(
            (1, 2, 3, 4, 5, 6),
            (0.2, 0.2, 0.2, 0.2),
            settings=MgbSweepSettings(
                source_variant="test", alpha_out=0.005, alpha_in=0, alpha_zero_in=0.1, candidate_count=4
            ),
            pvalue_provenance="test",
        )
    with pytest.raises(ValueError, match="source variant"):
        MgbSweepSettings(source_variant="  ", alpha_out=0.005, alpha_in=0.0, alpha_zero_in=0.1)


def _series(*, grouped: bool = True) -> ObservationSeries:
    years = (2001, 2002, 2003, 2004, 2005, 2006)
    flows = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    annual = (
        *(
            AnnualObservation(
                year=y, kind=ObservationKind.EXACT, exposure=ExposureKind.SYSTEMATIC, value=q, source_flag="0"
            )
            for y, q in zip(years, flows, strict=True)
        ),
        AnnualObservation(
            year=1999, kind=ObservationKind.EXCLUDED, exposure=ExposureKind.SYSTEMATIC, value=999.0, source_flag="1"
        ),
        AnnualObservation(year=1998, kind=ObservationKind.MISSING, exposure=ExposureKind.SYSTEMATIC),
    )
    groups = (
        (
            CensorGroup(
                threshold=26.54,
                exposure=ExposureKind.HISTORICAL,
                count_at_or_below=12,
                count_above=0,
                start_year=1880,
                end_year=1891,
                source_record="reported count record",
            ),
        )
        if grouped
        else ()
    )
    return ObservationSeries(discharge_unit="m3/s", annual=annual, groups=groups, zero_threshold=0.0)


def test_preserves_original_rows_flags_exclusions_and_grouped_exposure() -> None:
    series = _series()
    result = select_mgb_candidates(
        (1, 2, 3, 4, 5, 6), (0.001, 0.3, 0.3), settings=_usgs_settings(), pvalue_provenance="synthetic"
    )
    audited = audit_mgb_annual_candidates(series, screened_years=(2001, 2002, 2003, 2004, 2005, 2006), decision=result)
    assert audited.original_series is series
    assert audited.candidate_low_outlier_years == (2001,)
    assert audited.existing_excluded_years == (1999,)
    assert audited.original_series.groups[0].count_at_or_below == 12
    assert audited.original_series.groups[0].source_record == "reported count record"
    assert audited.original_series.annual[6].value == 999.0
    assert audited.original_series.annual[6].source_flag == "1"


def test_screening_audit_rejects_unqualified_inputs_and_overlaps() -> None:
    series = _series()
    decision = select_mgb_candidates(
        (1, 2, 3, 4, 5, 6), (0.2, 0.2, 0.2), settings=_usgs_settings(), pvalue_provenance="synthetic"
    )
    with pytest.raises(ValueError, match="not an exact systematic"):
        audit_mgb_annual_candidates(series, screened_years=(1999, 2002, 2003, 2004, 2005, 2006), decision=decision)
    with pytest.raises(ValueError, match="flow values/order"):
        audit_mgb_annual_candidates(series, screened_years=(2002, 2001, 2003, 2004, 2005, 2006), decision=decision)
    with pytest.raises(ValueError, match="distinct annual"):
        audit_mgb_annual_candidates(series, screened_years=(2001, 2001, 2003, 2004, 2005, 2006), decision=decision)
    overlapping = ObservationSeries(
        discharge_unit=series.discharge_unit,
        annual=series.annual,
        groups=(
            CensorGroup(
                threshold=3.0, exposure=ExposureKind.SYSTEMATIC, count_at_or_below=1, start_year=2001, end_year=2002
            ),
        ),
    )
    with pytest.raises(ValueError, match="overlaps a grouped"):
        audit_mgb_annual_candidates(overlapping, screened_years=(2001, 2002, 2003, 2004, 2005, 2006), decision=decision)
    unbounded = ObservationSeries(
        discharge_unit=series.discharge_unit,
        annual=series.annual,
        groups=(CensorGroup(threshold=3.0, exposure=ExposureKind.UNRESOLVED, count_at_or_below=1),),
    )
    with pytest.raises(ValueError, match="unbounded censor-group"):
        audit_mgb_annual_candidates(unbounded, screened_years=(2001, 2002, 2003, 2004, 2005, 2006), decision=decision)


def test_zero_threshold_logit_is_coordinate_only_not_empirical_p0() -> None:
    coord = ThresholdProbabilityCoordinate(
        family="log_pearson_iii", threshold=0.0, logit_probability=log(4), source="report parameter column"
    )
    assert isclose(coord.probability, 0.8, rel_tol=1e-12)
    assert probability_at_or_below_threshold(0.0) == 0.5
    assert probability_at_or_below_threshold(1000.0) == 1.0
    assert probability_at_or_below_threshold(-1000.0) == 0.0
    for p in (0.000001, 0.2, 0.8, 0.999999):
        assert isclose(probability_at_or_below_threshold(logit_at_or_below_threshold(p)), p, rel_tol=1e-12)
    for p in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="strictly inside"):
            logit_at_or_below_threshold(p)
    with pytest.raises(ValueError, match="finite"):
        probability_at_or_below_threshold(float("inf"))
    with pytest.raises(ValueError, match="provenance"):
        ThresholdProbabilityCoordinate(family="lognormal", threshold=0, logit_probability=1.0, source="")
    assert not isnan(coord.probability)


def test_arr_approved_censoring_preserves_both_record_and_group_representations() -> None:
    series = _series()
    decision = select_mgb_candidates(
        (1, 2, 3, 4, 5, 6),
        (0.001, 0.001, 0.3),
        settings=_usgs_settings(),
        pvalue_provenance="synthetic",
    )
    audit = audit_mgb_annual_candidates(series, screened_years=(2001, 2002, 2003, 2004, 2005, 2006), decision=decision)
    proposal = propose_arr_mgb_censoring(
        audit,
        censor_threshold=2.5,
        threshold_source="ARR example style explicit numerical value",
        synthetic_start_year=-1000,
    )
    assert isinstance(proposal, ArrMgbCensoringProposal)
    assert proposal.excluded_years == (2001, 2002)
    assert proposal.censor_group.threshold == 2.5
    assert proposal.censor_group.count_at_or_below == 2
    assert proposal.censor_group.count_above == 0
    assert (proposal.censor_group.start_year, proposal.censor_group.end_year) == (-1000, -999)
    assert proposal.censor_group.exposure == ExposureKind.UNRESOLVED
    assert proposal.audit.original_series is series
    with pytest.raises(ValueError, match="explicit approval"):
        apply_approved_arr_mgb_censoring(proposal)
    transformed = apply_approved_arr_mgb_censoring(proposal, approve=True)
    assert [row.kind for row in transformed.annual[:2]] == [ObservationKind.EXCLUDED] * 2
    assert [row.kind for row in series.annual[:2]] == [ObservationKind.EXACT] * 2
    assert transformed.annual[0].value == 1.0
    assert transformed.annual[0].source_flag == "0"
    assert transformed.annual[0].year == 2001
    assert transformed.annual[6].kind == ObservationKind.EXCLUDED
    assert transformed.annual[6].value == 999.0
    assert transformed.groups[0] is series.groups[0]
    assert transformed.groups[1] is proposal.censor_group
    assert len(transformed.annual) == len(series.annual)


def test_arr_mgb_censoring_rejects_endpoint_guess_and_placeholder_overlap() -> None:
    series = _series()
    decision = select_mgb_candidates(
        (1, 2, 3, 4, 5, 6),
        (0.001, 0.2, 0.2),
        settings=_usgs_settings(),
        pvalue_provenance="synthetic",
    )
    audit = audit_mgb_annual_candidates(series, screened_years=(2001, 2002, 2003, 2004, 2005, 2006), decision=decision)
    for threshold in (1.0, 2.0, 3.0):
        with pytest.raises(ValueError, match="strictly between"):
            propose_arr_mgb_censoring(
                audit, censor_threshold=threshold, threshold_source="test", synthetic_start_year=-1000
            )
    with pytest.raises(ValueError, match="before year zero"):
        propose_arr_mgb_censoring(audit, censor_threshold=1.5, threshold_source="test", synthetic_start_year=0)
    with pytest.raises(ValueError, match="source-identified"):
        propose_arr_mgb_censoring(audit, censor_threshold=1.5, threshold_source="", synthetic_start_year=-1000)
    overlapping = ObservationSeries(
        discharge_unit=series.discharge_unit,
        annual=series.annual,
        groups=(
            *series.groups,
            CensorGroup(
                threshold=4.0,
                exposure=ExposureKind.UNRESOLVED,
                count_at_or_below=2,
                start_year=-1000,
                end_year=-999,
            ),
        ),
    )
    overlapping_audit = audit_mgb_annual_candidates(
        overlapping, screened_years=(2001, 2002, 2003, 2004, 2005, 2006), decision=decision
    )
    with pytest.raises(ValueError, match="overlaps an existing group"):
        propose_arr_mgb_censoring(
            overlapping_audit, censor_threshold=1.5, threshold_source="test", synthetic_start_year=-1000
        )


def test_no_stopping_probability_does_not_promote_half_sample() -> None:
    # USGS 1.0.7 R/MGBT.R, mgbt.sweeper: j2 and j3 begin at zero and
    # remain zero if their respective >=-alpha stopping tests never succeed.
    flows = (1, 2, 3, 4, 5, 6, 7, 8)
    no_stop = select_mgb_candidates(
        flows, (0.01, 0.02, 0.03, 0.04), settings=_usgs_settings(), pvalue_provenance="synthetic"
    )
    assert (no_stop.out_sweep_rank, no_stop.in_sweep_rank, no_stop.zero_in_sweep_rank) == (0, 0, 0)
    assert no_stop.low_outlier_count == 0
    assert no_stop.threshold is None

    # A significant outermost rank still counts, but a missing inward or
    # zero-in stop does not independently select all lower-half ranks.
    all_below = select_mgb_candidates(
        flows, (0.001, 0.002, 0.003, 0.004), settings=_usgs_settings(), pvalue_provenance="synthetic"
    )
    assert (all_below.out_sweep_rank, all_below.in_sweep_rank, all_below.zero_in_sweep_rank) == (4, 0, 0)
    assert all_below.low_outlier_count == 4
    assert all_below.threshold == 5

    custom_in = select_mgb_candidates(
        flows,
        (0.2, 0.001, 0.01, 0.02),
        settings=MgbSweepSettings(
            source_variant="USGS 1.0.7 custom alpha comparison", alpha_out=0.005, alpha_in=0.05, alpha_zero_in=0.1
        ),
        pvalue_provenance="synthetic",
    )
    assert (custom_in.out_sweep_rank, custom_in.in_sweep_rank, custom_in.zero_in_sweep_rank) == (2, 0, 0)
    assert custom_in.low_outlier_count == 2


def test_strict_alpha_equalities_in_each_sweep() -> None:
    flows = (1, 2, 3, 4, 5, 6, 7, 8)
    # The outward sweep uses p < alpha_out: equality does not count.
    equal_out = select_mgb_candidates(
        flows, (0.2, 0.005, 0.2, 0.2), settings=_usgs_settings(), pvalue_provenance="synthetic"
    )
    just_below_out = select_mgb_candidates(
        flows, (0.2, 0.005 - 1e-12, 0.2, 0.2), settings=_usgs_settings(), pvalue_provenance="synthetic"
    )
    assert equal_out.out_sweep_rank == 0
    assert just_below_out.out_sweep_rank == 2

    # The zero-in sweep stops at p >= alpha_zero_in: equality retains rank 1.
    equal_zero = select_mgb_candidates(
        flows, (0.1, 0.2, 0.2, 0.2), settings=_usgs_settings(), pvalue_provenance="synthetic"
    )
    below_zero = select_mgb_candidates(
        flows, (0.1 - 1e-12, 0.1, 0.2, 0.2), settings=_usgs_settings(), pvalue_provenance="synthetic"
    )
    assert (equal_zero.zero_in_sweep_rank, below_zero.zero_in_sweep_rank) == (0, 1)

    # With alpha_in > 0, equality stops the inward sweep at the preceding rank.
    settings = MgbSweepSettings(
        source_variant="USGS 1.0.7 custom alpha comparison", alpha_out=0.005, alpha_in=0.01, alpha_zero_in=0.1
    )
    equal_in = select_mgb_candidates(flows, (0.2, 0.001, 0.01, 0.2), settings=settings, pvalue_provenance="synthetic")
    below_in = select_mgb_candidates(
        flows, (0.2, 0.001, 0.01 - 1e-12, 0.2), settings=settings, pvalue_provenance="synthetic"
    )
    assert (equal_in.in_sweep_rank, below_in.in_sweep_rank) == (2, 3)
    assert (equal_in.zero_in_sweep_rank, below_in.zero_in_sweep_rank) == (0, 0)


def test_published_ranks_are_invariant_to_original_record_order() -> None:
    # The published one-sided probabilities are for order-statistic ranks,
    # not the 51 annual records' original positions. No p-values are recomputed.
    original = select_mgb_candidates(
        USGS_08066300, USGS_RANK_PVALUES, settings=_usgs_settings(), pvalue_provenance="USGS 1.0.7 printed"
    )
    reversed_values = tuple(reversed(USGS_08066300))
    shuffled = select_mgb_candidates(
        reversed_values, USGS_RANK_PVALUES, settings=_usgs_settings(), pvalue_provenance="USGS 1.0.7 printed"
    )
    assert original.threshold == shuffled.threshold == 284
    assert (original.low_outlier_count, shuffled.low_outlier_count) == (1, 1)
    assert shuffled.low_outlier_positions == (reversed_values.index(55),)
    assert shuffled.original_flows == reversed_values
    assert tuple(shuffled.original_flows[index] for index in shuffled.low_outlier_positions) == (55,)


def test_small_sample_boundary_is_only_a_supplied_probability_comparator() -> None:
    # n=5 is admitted by this conservative decision adapter; it is not
    # Numerics' n>=10 p-value generator or BestFit's n>=10 UI entry point.
    decision = select_mgb_candidates(
        (1, 2, 3, 4, 5), (0.2, 0.3), settings=_usgs_settings(), pvalue_provenance="synthetic"
    )
    assert decision.threshold is None
    assert decision.ordered_positions == (0, 1, 2, 3, 4)
