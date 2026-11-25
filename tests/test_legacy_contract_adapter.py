"""WP02 source-preserving annual contract mapping and fail-closed boundaries."""

from decimal import Decimal

import pytest

from ffa_methods.contracts import ExposureKind as ContractExposureKind
from ffa_methods.contracts import ObservationKind as ContractObservationKind
from ffa_methods.legacy import CensorGroup, ExposureKind, ExposurePeriod, LegacyInputError, Observation, ObservationKind
from ffa_methods.legacy.adapter import (
    MappedAnnualObservation,
    MappedAnnualSeries,
    map_annual_observation,
    map_annual_series,
)

SOURCE_DIGEST = "a" * 64


def _map(observation: Observation, *, line: str = "2001,12.50") -> MappedAnnualObservation:
    return map_annual_observation(
        observation,
        exposure=ContractExposureKind.SYSTEMATIC,
        raw_source_line=line,
        source_sha256=SOURCE_DIGEST,
    )


def test_exact_maps_without_losing_decimal_precision_or_source_evidence() -> None:
    original = Observation(2001, ObservationKind.EXACT, Decimal("12.500"), "year_ams_csv")
    result = _map(original)
    assert result.annual.kind == ContractObservationKind.EXACT
    assert result.annual.value == 12.5
    assert result.legacy.value == Decimal("12.500")
    assert result.raw_source_line == "2001,12.50"
    assert result.source_sha256 == SOURCE_DIGEST
    assert result.annual.source_flag is None


def test_reported_zero_and_unclassified_project_flags_remain_unresolved() -> None:
    reported = Observation(2001, ObservationKind.REPORTED_ZERO, Decimal(0), "year_ams_csv")
    flagged = Observation(2001, ObservationKind.FLAG_UNRESOLVED, Decimal(12), "flike_file3.10", raw_flag=4)
    with pytest.raises(LegacyInputError, match="unresolved observation"):
        _map(reported)
    with pytest.raises(LegacyInputError, match="unresolved project flag"):
        _map(flagged)
    manually_reclassified = Observation(2001, ObservationKind.EXCLUDED, Decimal(12), "flike_file3.10", 4, reason="test")
    with pytest.raises(LegacyInputError, match="unresolved project flag"):
        _map(manually_reclassified)


def test_verified_true_zero_is_explicit_not_inferred_from_parser() -> None:
    proven = Observation(2001, ObservationKind.VERIFIED_TRUE_ZERO, Decimal(0), "external_checked_record")
    mapped = _map(proven)
    assert mapped.annual.kind == ContractObservationKind.EXACT
    assert mapped.annual.value == 0
    assert mapped.legacy.kind == ObservationKind.VERIFIED_TRUE_ZERO


def test_missing_excluded_and_censor_constraints_preserved() -> None:
    missing = _map(Observation(2001, ObservationKind.MISSING, None, "year_ams_csv"))
    excluded = _map(
        Observation(2001, ObservationKind.EXCLUDED, Decimal("42.125"), "checked", reason="manual exclusion")
    )
    censored = _map(
        Observation(
            2001, ObservationKind.INTERVAL_CENSORED, None, "checked", lower=Decimal("1.5"), upper=Decimal("2.5")
        )
    )
    assert missing.annual.kind == ContractObservationKind.MISSING
    assert missing.annual.value is None
    assert excluded.annual.kind == ContractObservationKind.EXCLUDED
    assert excluded.annual.value == 42.125
    assert excluded.legacy.reason == "manual exclusion"
    assert censored.annual.kind == ContractObservationKind.INTERVAL_CENSORED
    assert (censored.annual.lower_bound, censored.annual.upper_bound) == (1.5, 2.5)


@pytest.mark.parametrize("flow", ["1e9999", "1e-9999", "1e309", "1e-400"])
def test_decimal_conversion_rejects_overflow_and_underflow(flow: str) -> None:
    with pytest.raises(LegacyInputError, match="represent"):
        _map(Observation(2001, ObservationKind.EXACT, Decimal(flow), "synthetic"))


def test_float_maximum_and_minimum_subnormal_are_representable() -> None:
    assert _map(Observation(2001, ObservationKind.EXACT, Decimal("1.7976931348623157e308"), "test")).annual.value
    assert _map(Observation(2001, ObservationKind.EXACT, Decimal("5e-324"), "test")).annual.value == 5e-324


def test_mapping_requires_full_provenance_and_classified_exposure() -> None:
    observation = Observation(2001, ObservationKind.EXACT, Decimal(12), "synthetic")
    with pytest.raises(LegacyInputError, match="source provenance"):
        _map(observation, line="")
    with pytest.raises(LegacyInputError, match="SHA-256"):
        map_annual_observation(
            observation, exposure=ContractExposureKind.SYSTEMATIC, raw_source_line="2001,12", source_sha256="abc"
        )
    with pytest.raises(LegacyInputError, match="classified exposure"):
        map_annual_observation(
            observation,
            exposure=ContractExposureKind.UNRESOLVED,
            raw_source_line="2001,12",
            source_sha256=SOURCE_DIGEST,
        )


def test_series_preserves_missing_and_excluded_and_stays_evidence_only() -> None:
    periods = (ExposurePeriod("gauge", ExposureKind.SYSTEMATIC, frozenset({2001, 2002, 2003})),)
    observations = (
        Observation(2001, ObservationKind.EXACT, Decimal(12), "year_ams_csv"),
        Observation(2002, ObservationKind.MISSING, None, "year_ams_csv"),
        Observation(2003, ObservationKind.EXCLUDED, Decimal(19), "checked", reason="reason recorded"),
    )
    result = map_annual_series(
        periods,
        observations,
        (),
        discharge_unit="m^3/s",
        source_lines_by_year={2001: "2001,12", 2002: "2002,", 2003: "2003,19"},
        source_sha256=SOURCE_DIGEST,
        source_file_version="3.10",
    )
    assert [record.kind for record in result.series.annual] == [
        ContractObservationKind.EXACT,
        ContractObservationKind.MISSING,
        ContractObservationKind.EXCLUDED,
    ]
    assert result.exposure_audit.ready_for_likelihood
    assert not result.ready_for_likelihood
    assert result.series.zero_threshold is None
    assert result.series.groups == ()
    assert result.series.source_file_version == "3.10"


def test_series_rejects_unaccounted_exposure_and_group_overlap() -> None:
    periods = (ExposurePeriod("gauge", ExposureKind.SYSTEMATIC, frozenset({2001, 2002})),)
    observations = (Observation(2001, ObservationKind.EXACT, Decimal(12), "synthetic"),)
    with pytest.raises(LegacyInputError, match="Unresolved exposure"):
        map_annual_series(
            periods,
            observations,
            (),
            discharge_unit="m^3/s",
            source_lines_by_year={2001: "2001,12"},
            source_sha256=SOURCE_DIGEST,
        )
    overlap_group = (CensorGroup("gauge", 1, 0, Decimal(20), frozenset({2001})),)
    with pytest.raises(LegacyInputError, match="overlap"):
        map_annual_series(
            periods,
            observations,
            overlap_group,
            discharge_unit="m^3/s",
            source_lines_by_year={2001: "2001,12"},
            source_sha256=SOURCE_DIGEST,
        )


def test_series_does_not_erase_group_membership_or_infer_unlabelled_rows() -> None:
    period = ExposurePeriod("history", ExposureKind.HISTORICAL, frozenset({1937}), Decimal(12))
    group = (CensorGroup("history", 1, 0, Decimal(12), frozenset({1937})),)
    with pytest.raises(LegacyInputError, match="Grouped censor"):
        map_annual_series(
            (period,),
            (),
            group,
            discharge_unit="m^3/s",
            source_lines_by_year={},
            source_sha256=SOURCE_DIGEST,
        )


def test_decimal_censor_interval_must_retain_positive_width_after_float_conversion() -> None:
    collapsed = Observation(
        2001,
        ObservationKind.INTERVAL_CENSORED,
        None,
        "synthetic",
        lower=Decimal("1.00000000000000001"),
        upper=Decimal("1.00000000000000002"),
    )
    assert collapsed.lower is not None
    assert collapsed.upper is not None
    assert collapsed.lower < collapsed.upper
    with pytest.raises(LegacyInputError, match="collapsed censor interval"):
        _map(collapsed)

    representable = Observation(
        2001,
        ObservationKind.INTERVAL_CENSORED,
        None,
        "synthetic",
        lower=Decimal("1.0"),
        upper=Decimal("1.000000000000001"),
    )
    result = _map(representable)
    assert result.annual.lower_bound is not None
    assert result.annual.upper_bound is not None
    assert result.annual.lower_bound < result.annual.upper_bound


def test_historical_perception_thresholds_and_period_ids_are_not_discarded() -> None:
    observation = Observation(1900, ObservationKind.EXACT, Decimal(30), "historical_checked")
    provenance = {1900: "1900 30"}
    low = ExposurePeriod("threshold_10", ExposureKind.HISTORICAL, frozenset({1900}), Decimal(10))
    high = ExposurePeriod("threshold_20", ExposureKind.HISTORICAL, frozenset({1900}), Decimal(20))

    def convert(period: ExposurePeriod) -> MappedAnnualSeries:
        return map_annual_series(
            (period,),
            (observation,),
            (),
            discharge_unit="m^3/s",
            source_lines_by_year=provenance,
            source_sha256=SOURCE_DIGEST,
        )

    converted_low = convert(low)
    converted_high = convert(high)
    assert converted_low.source_periods == (low,)
    assert converted_high.source_periods == (high,)
    assert converted_low != converted_high
    assert converted_low.series.annual[0].exposure == ContractExposureKind.HISTORICAL
    assert converted_low.source_periods[0].identifier == "threshold_10"
    assert converted_low.source_periods[0].perception_threshold == Decimal(10)
    assert converted_high.source_periods[0].perception_threshold == Decimal(20)
    assert not converted_low.ready_for_likelihood
    assert not converted_high.ready_for_likelihood
