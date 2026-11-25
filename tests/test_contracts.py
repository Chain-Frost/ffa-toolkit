"""Validate the shared boundaries instead of inventing statistical outcomes."""

import pytest

from ffa_methods.compatibility import LEGACY_FLIKE_PROFILE
from ffa_methods.contracts import (
    AnnualObservation,
    CensorGroup,
    EvidenceStatus,
    ExposureKind,
    ObservationKind,
    ObservationSeries,
    ParameterCoordinate,
    ProbabilityKind,
    ResultMeaning,
    RunProvenance,
)


def test_real_zero_is_not_missing_or_excluded() -> None:
    zero = AnnualObservation(
        year=2001, kind=ObservationKind.EXACT, exposure=ExposureKind.SYSTEMATIC, value=0.0, source_flag="0"
    )
    excluded = AnnualObservation(
        year=2002, kind=ObservationKind.EXCLUDED, exposure=ExposureKind.SYSTEMATIC, value=7.0, source_flag="1"
    )
    missing = AnnualObservation(year=2003, kind=ObservationKind.MISSING, exposure=ExposureKind.SYSTEMATIC)
    assert zero.value == 0.0
    assert zero.kind != ObservationKind.AT_OR_BELOW_ZERO_THRESHOLD
    assert excluded.value == 7.0
    assert missing.value is None
    assert len({zero.kind, excluded.kind, missing.kind}) == 3


def test_interval_constraints_are_checked() -> None:
    valid = AnnualObservation(
        year=1990,
        kind=ObservationKind.INTERVAL_CENSORED,
        exposure=ExposureKind.HISTORICAL,
        lower_bound=5.0,
        upper_bound=8.0,
    )
    assert valid.upper_bound == 8.0
    with pytest.raises(ValueError, match="must not exceed"):
        AnnualObservation(
            year=1990,
            kind=ObservationKind.INTERVAL_CENSORED,
            exposure=ExposureKind.HISTORICAL,
            lower_bound=9.0,
            upper_bound=8.0,
        )
    with pytest.raises(ValueError, match="both bounds"):
        AnnualObservation(year=1990, kind=ObservationKind.INTERVAL_CENSORED, exposure=ExposureKind.HISTORICAL)


def test_censor_group_is_distinct_from_excluded_rows() -> None:
    group = CensorGroup(
        threshold=26.54,
        exposure=ExposureKind.HISTORICAL,
        count_at_or_below=12,
        count_above=0,
        start_year=1890,
        end_year=1901,
        source_record="original grouped threshold line",
    )
    series = ObservationSeries(discharge_unit="m3/s", annual=(), groups=(group,), zero_threshold=0.0)
    assert len(series.annual) == 0
    assert series.groups[0].count_at_or_below == 12
    assert series.zero_threshold == 0.0
    with pytest.raises(ValueError, match="cannot be negative"):
        CensorGroup(threshold=26.54, exposure=ExposureKind.HISTORICAL, count_above=-1)


def test_legacy_posterior_coordinates_are_not_alternate_parameter_scales() -> None:
    assert ParameterCoordinate.LP3_LOGE_SD_LOGE_FLOW.value == "lp3.loge_sd_loge_flow"
    assert "logit" in ParameterCoordinate.ZERO_THRESHOLD_LOGIT_PROBABILITY.value
    assert ProbabilityKind.NON_EXCEEDANCE != ProbabilityKind.ANNUAL_EXCEEDANCE
    assert ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS != ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY


def test_package_method_and_legacy_versions_remain_independent() -> None:
    provenance = RunProvenance(
        compatibility_profile=LEGACY_FLIKE_PROFILE.identifier,
        method_id="not_implemented",
        method_version="none",
        package_version="0.0.0.dev0",
        evidence_status=EvidenceStatus.CANDIDATE_PAIR,
    )
    assert provenance.package_version != LEGACY_FLIKE_PROFILE.legacy_program_version
    assert provenance.method_version != provenance.package_version
    assert provenance.evidence_status != EvidenceStatus.QUALIFIED_REFERENCE
