"""Independent WP02 -> WP03/WP05 synthetic integration and fail-closed tests."""

from dataclasses import replace
from decimal import Decimal
from math import exp, expm1, log, pi
from typing import TypeAlias

import pytest

from ffa_methods.contracts import ObservationKind as ContractObservationKind
from ffa_methods.contracts import ParameterCoordinate
from ffa_methods.distributions import GEV, GeneralizedPareto, Gumbel, LogCoordinates, Lognormal, LogPearsonIII
from ffa_methods.legacy.adapter import MappedAnnualSeries, map_annual_series
from ffa_methods.legacy.models import ExposureKind, ExposurePeriod, LegacyInputError, Observation, ObservationKind
from ffa_methods.likelihood import HistoricalPeriod, LikelihoodEvidenceError
from ffa_methods.likelihood_integration import (
    SyntheticEvidenceDeclaration,
    SyntheticIntegrationError,
    SyntheticRatingEpoch,
    evaluate_synthetic_likelihood,
    evaluate_synthetic_shared_rating,
    evaluate_synthetic_source_prior,
)
from ffa_methods.low_flow import MgbSweepSettings, audit_mgb_annual_candidates, select_mgb_candidates
from ffa_methods.priors import (
    CoordinateTransform,
    JointGaussianPrior,
    LegacyPriorMode,
    LegacyPriorSelection,
    PriorEvidenceError,
)
from ffa_methods.rating_errors import JointRatingScenario, RatingEvidenceError
from tests.numeric_assertions import near

Kernel: TypeAlias = Lognormal | LogPearsonIII | Gumbel | GEV | GeneralizedPareto

DIGEST = "3" * 64
SOURCE = "synthetic:wp05-contract-study"
SYNTHETIC_VERSION = "synthetic-component-v1"


def _mapped(
    *observations: Observation,
    unit: str = "m3/s",
    source_version: str = SYNTHETIC_VERSION,
    period_id: str = "systematic",
) -> MappedAnnualSeries:
    years = frozenset(observation.year for observation in observations)
    periods = (ExposurePeriod(period_id, ExposureKind.SYSTEMATIC, years),) if years else ()
    return map_annual_series(
        periods,
        observations,
        (),
        discharge_unit=unit,
        source_lines_by_year={
            observation.year: f"{observation.year}: original {observation.kind.value}" for observation in observations
        },
        source_sha256=DIGEST,
        source_file_version=source_version,
        year_definition="synthetic calendar years",
    )


def _exact(year: int, value: str, *, raw_flag: int | None = None, source: str = SOURCE) -> Observation:
    return Observation(year, ObservationKind.EXACT, Decimal(value), source, raw_flag=raw_flag)


def _declaration(
    *,
    unit: str = "m3/s",
    kernel_unit: str = "m3/s",
    source: str = SOURCE,
    historical_records: tuple[str, ...] = (),
    period_ids: tuple[str, ...] = ("systematic",),
    source_sha256: str = DIGEST,
) -> SyntheticEvidenceDeclaration:
    return SyntheticEvidenceDeclaration(
        purpose="independent_synthetic_mathematics",
        evidence_id=source,
        source_sha256=source_sha256,
        discharge_unit=unit,
        kernel_discharge_unit=kernel_unit,
        source_period_ids=period_ids,
        historical_record_ids=historical_records,
    )


def test_wp02_wp03_wp05_exact_censors_history_and_skipped_years() -> None:
    original = (
        _exact(2000, "2.000"),
        Observation(2001, ObservationKind.LEFT_CENSORED, None, SOURCE, upper=Decimal(1)),
        Observation(2002, ObservationKind.RIGHT_CENSORED, None, SOURCE, lower=Decimal(3)),
        Observation(2003, ObservationKind.INTERVAL_CENSORED, None, SOURCE, lower=Decimal(1), upper=Decimal(2)),
        Observation(2004, ObservationKind.MISSING, None, SOURCE),
        Observation(2005, ObservationKind.EXCLUDED, Decimal(9), SOURCE, reason="synthetic exclusion"),
    )
    mapped = _mapped(*original)
    history = HistoricalPeriod(
        start_year=1890,
        end_year=1893,
        threshold=1.5,
        known_exceedances=((1890, 4.0),),
        unmeasured_exceedances=1,
        source_record="synthetic:hist-1890-1893",
    )
    result = evaluate_synthetic_likelihood(
        Gumbel(0.0, 0.0),
        mapped,
        declaration=_declaration(historical_records=("synthetic:hist-1890-1893",)),
        historical_periods=(history,),
    )

    # Independent closed-form Gumbel density, CDF and SF. The exact known
    # historical year is not also an unlabelled exceedance: C(3, 1), not C(4, 2).
    def cdf(q: float) -> float:
        return exp(-exp(-q))

    def sf(q: float) -> float:
        return -expm1(-exp(-q))

    def logpdf(q: float) -> float:
        return -q - exp(-q)

    expected = (
        logpdf(2)
        + log(cdf(1))
        + log(sf(3))
        + log(cdf(2) - cdf(1))
        + logpdf(4)
        + log(3)
        + log(sf(1.5))
        + 2 * log(cdf(1.5))
    )
    assert near(result.likelihood.log_value, expected, abs_tol=2e-12, rel_tol=0.0)
    assert len(result.likelihood.terms) == 7
    assert [term.included for term in result.likelihood.terms] == [True, True, True, True, False, False, True]
    assert result.likelihood.terms[-1].source == "historical:0"
    assert result.annual_years == tuple(observation.year for observation in original)
    assert result.historical_records == ("synthetic:hist-1890-1893",)
    assert result.evidence_id == SOURCE
    assert result.source_sha256 == DIGEST
    assert not result.legacy_qualified
    assert mapped.source_records[0].legacy.value == Decimal("2.000")
    assert mapped.source_records[0].raw_source_line.startswith("2000:")
    assert not mapped.ready_for_likelihood
    assert mapped.series.groups == ()


@pytest.mark.parametrize(
    "kernel",
    [
        Lognormal(LogCoordinates(1.0, log(0.7))),
        LogPearsonIII(LogCoordinates(1.0, log(0.7)), 0.6),
        Gumbel(2.0, log(1.5)),
        GEV(2.0, log(1.5), -0.25),
        GeneralizedPareto(0.5, log(2.0), 0.1),
    ],
)
def test_each_actual_wp03_kernel_evaluates_mapped_source_records(kernel: Kernel) -> None:
    mapped = _mapped(
        _exact(2000, "3.25"),
        Observation(2001, ObservationKind.LEFT_CENSORED, None, SOURCE, upper=Decimal("2.5")),
        Observation(2002, ObservationKind.RIGHT_CENSORED, None, SOURCE, lower=Decimal("3.1")),
    )
    # The assertion is a cross-module composition check. Independent Gumbel
    # closed-form verification is in the preceding test.
    model = kernel
    likelihood = evaluate_synthetic_likelihood(model, mapped, declaration=_declaration()).likelihood
    expected = model.logpdf(3.25) + model.logcdf(2.5) + model.logsf(3.1)
    assert near(likelihood.log_value, expected, abs_tol=4e-12, rel_tol=0.0)


def test_wp03_narrow_interval_uses_wp05_density_fallback() -> None:
    mapped = _mapped(
        Observation(
            2000,
            ObservationKind.INTERVAL_CENSORED,
            None,
            SOURCE,
            lower=Decimal(0),
            upper=Decimal("1e-20"),
        )
    )
    model = Gumbel(0.0, 0.0)
    assert model.logcdf(0) == model.logcdf(1e-20)
    assert model.logsf(0) == model.logsf(1e-20)
    result = evaluate_synthetic_likelihood(model, mapped, declaration=_declaration())
    assert near(result.likelihood.log_value, log(1e-20) - 1.0, abs_tol=2e-13, rel_tol=0.0)


def test_impossible_support_stays_minus_infinity_not_a_fabricated_flood() -> None:
    model = GeneralizedPareto(2.0, 0.0, 0.0)
    result = evaluate_synthetic_likelihood(model, _mapped(_exact(2000, "1")), declaration=_declaration())
    assert result.likelihood.log_value == float("-inf")
    assert result.likelihood.terms[0].included


def test_invalid_tail_cannot_be_hidden_by_integrated_density() -> None:
    class InvalidTailGumbel(Gumbel):
        def logcdf(self, discharge: float) -> float:
            del discharge
            return float("nan")

        def logsf(self, discharge: float) -> float:
            del discharge
            return float("nan")

    mapped = _mapped(
        Observation(
            2000,
            ObservationKind.INTERVAL_CENSORED,
            None,
            SOURCE,
            lower=Decimal(0),
            upper=Decimal("1e-20"),
        )
    )
    with pytest.raises(ValueError, match="invalid log tail"):
        evaluate_synthetic_likelihood(InvalidTailGumbel(0.0, 0.0), mapped, declaration=_declaration())


def test_explicit_synthetic_identity_source_years_and_unit_must_match() -> None:
    mapped = _mapped(_exact(2000, "3"))
    good = _declaration()
    with pytest.raises(SyntheticIntegrationError, match="source observation discharge unit"):
        evaluate_synthetic_likelihood(
            Gumbel(0, 0), mapped, declaration=replace(good, discharge_unit="ft3/s", kernel_discharge_unit="ft3/s")
        )
    with pytest.raises(SyntheticIntegrationError, match="bound kernel"):
        evaluate_synthetic_likelihood(Gumbel(0, 0), mapped, declaration=replace(good, kernel_discharge_unit="ft3/s"))
    with pytest.raises(SyntheticIntegrationError, match="period IDs"):
        evaluate_synthetic_likelihood(Gumbel(0, 0), mapped, declaration=replace(good, source_period_ids=("wrong",)))
    with pytest.raises(SyntheticIntegrationError, match="SHA-256"):
        evaluate_synthetic_likelihood(Gumbel(0, 0), mapped, declaration=replace(good, source_sha256="4" * 64))
    with pytest.raises(SyntheticIntegrationError, match="synthetic likelihood"):
        evaluate_synthetic_likelihood(Gumbel(0, 0), _mapped(_exact(2000, "3"), source_version="3.10"), declaration=good)
    with pytest.raises(SyntheticIntegrationError, match="synthetic observations"):
        evaluate_synthetic_likelihood(
            Gumbel(0, 0), _mapped(_exact(2000, "3", source="legacy_file3.10")), declaration=good
        )


def test_unverified_flags_zero_controls_and_group_exposure_never_implicitly_map() -> None:
    flagged = _mapped(_exact(2000, "3", raw_flag=0))
    with pytest.raises(SyntheticIntegrationError, match="flagged project rows"):
        evaluate_synthetic_likelihood(Gumbel(0, 0), flagged, declaration=_declaration())

    with pytest.raises(LegacyInputError, match="unresolved observation"):
        _mapped(Observation(2000, ObservationKind.REPORTED_ZERO, Decimal(0), SOURCE))
    zero = _mapped(Observation(2000, ObservationKind.VERIFIED_TRUE_ZERO, Decimal(0), SOURCE))
    with pytest.raises(SyntheticIntegrationError, match="zero-flow"):
        evaluate_synthetic_likelihood(Gumbel(0, 0), zero, declaration=_declaration())

    # A manually modified view cannot override the original WP02 row mapping.
    clean = _mapped(_exact(2000, "3"))
    changed = replace(clean.series.annual[0], kind=ContractObservationKind.AT_OR_BELOW_ZERO_THRESHOLD)
    tampered = replace(clean, series=replace(clean.series, annual=(changed,)))
    with pytest.raises(SyntheticIntegrationError, match="original source-mapped"):
        evaluate_synthetic_likelihood(Gumbel(0, 0), tampered, declaration=_declaration())
    grouped = replace(clean, series=replace(clean.series, zero_threshold=0.0))
    with pytest.raises(SyntheticIntegrationError, match="zero-threshold"):
        evaluate_synthetic_likelihood(Gumbel(0, 0), grouped, declaration=_declaration())


def test_wp02_unaccounted_exposure_rejected_before_evaluation() -> None:
    observations = (_exact(2000, "3"),)
    with pytest.raises(LegacyInputError, match="Unresolved exposure"):
        map_annual_series(
            (ExposurePeriod("systematic", ExposureKind.SYSTEMATIC, frozenset({2000, 2001})),),
            observations,
            (),
            discharge_unit="m3/s",
            source_lines_by_year={2000: "2000,3"},
            source_sha256=DIGEST,
            source_file_version=SYNTHETIC_VERSION,
        )


def test_historical_source_identity_and_overlap_fail_closed() -> None:
    mapped = _mapped(_exact(2000, "3"))
    hist = HistoricalPeriod(
        start_year=1990, end_year=1992, threshold=1.0, unmeasured_exceedances=1, source_record="synthetic:hist"
    )
    with pytest.raises(SyntheticIntegrationError, match="historical record identity"):
        evaluate_synthetic_likelihood(Gumbel(0, 0), mapped, declaration=_declaration(), historical_periods=(hist,))
    overlapping = replace(hist, start_year=2000, end_year=2002)
    with pytest.raises(LikelihoodEvidenceError, match="overlapping"):
        evaluate_synthetic_likelihood(
            Gumbel(0, 0),
            mapped,
            declaration=_declaration(historical_records=("synthetic:hist",)),
            historical_periods=(overlapping,),
        )
    unverifiable = replace(hist, source_record=None)
    with pytest.raises(SyntheticIntegrationError, match="synthetic source-record"):
        evaluate_synthetic_likelihood(
            Gumbel(0, 0),
            mapped,
            declaration=_declaration(historical_records=("synthetic:hist",)),
            historical_periods=(unverifiable,),
        )


def test_joint_rating_across_mapped_epochs_preserves_jacobian_and_rejects_mixed_units() -> None:
    first = _mapped(_exact(2000, "1"))
    second = _mapped(_exact(2001, "2"))
    epochs = (
        SyntheticRatingEpoch(identifier="old", mapped=first, declaration=_declaration()),
        SyntheticRatingEpoch(identifier="new", mapped=second, declaration=_declaration()),
    )
    identity = (JointRatingScenario(weight=1.0, log_error_by_epoch=(("old", 0.0), ("new", 0.0))),)
    model = Gumbel(0, 0)
    result = evaluate_synthetic_shared_rating(model, epochs, identity)
    assert near(result.log_value, -1 - exp(-1) - 2 - exp(-2), abs_tol=2e-14, rel_tol=0.0)
    assert result.evidence_ids == (SOURCE, SOURCE)
    assert not result.legacy_qualified

    mixed = replace(
        epochs[1],
        mapped=_mapped(_exact(2001, "2"), unit="ft3/s"),
        declaration=_declaration(unit="ft3/s", kernel_unit="ft3/s"),
    )
    with pytest.raises(RatingEvidenceError, match="single recorded discharge unit"):
        evaluate_synthetic_shared_rating(model, (epochs[0], mixed), identity)
    with pytest.raises(RatingEvidenceError, match="multiple rating"):
        evaluate_synthetic_shared_rating(
            model,
            (epochs[0], replace(epochs[1], mapped=_mapped(_exact(2000, "2")))),
            identity,
        )


def test_rating_integration_keeps_shared_error_instead_of_independent_jitter() -> None:
    epoch = SyntheticRatingEpoch(
        identifier="shared",
        mapped=_mapped(_exact(2000, "1"), _exact(2001, "2")),
        declaration=_declaration(),
    )
    shifts = (log(0.8), log(1.2))
    scenarios = tuple(JointRatingScenario(weight=0.5, log_error_by_epoch=(("shared", shift),)) for shift in shifts)
    result = evaluate_synthetic_shared_rating(Gumbel(0.0, 0.0), (epoch,), scenarios)
    conditional = [
        exp((-exp(-shift) - exp(-exp(-shift)) - shift) + (-2 * exp(-shift) - exp(-2 * exp(-shift)) - shift))
        for shift in shifts
    ]
    assert near(exp(result.log_value), sum(conditional) / 2, abs_tol=1e-14, rel_tol=0.0)


def test_source_coordinate_prior_full_covariance_and_jacobian() -> None:
    coords = (ParameterCoordinate.LP3_MEAN_LOGE_FLOW, ParameterCoordinate.LP3_LOGE_SD_LOGE_FLOW)
    prior = JointGaussianPrior(
        coordinates=coords,
        means=(0.0, 0.0),
        covariance=((1.0, 0.3), (0.3, 1.0)),
        source="synthetic:source-coordinate-prior",
        mode=LegacyPriorMode.SUPPLIED,
    )
    selection = LegacyPriorSelection(mode=LegacyPriorMode.SUPPLIED, supplied=prior)
    x, sd = 0.4, 2.0
    y = log(sd)
    expected = -log(2 * pi) - 0.5 * log(1 - 0.3**2) - 0.5 * ((x * x - 0.6 * x * y + y * y) / (1 - 0.3**2)) - log(sd)
    actual = evaluate_synthetic_source_prior(
        selection,
        source="synthetic:source-coordinate-prior",
        coordinates=coords,
        values=(x, sd),
        transformations=(CoordinateTransform.IDENTITY, CoordinateTransform.RAW_POSITIVE_TO_LOG),
    )
    assert near(actual, expected, abs_tol=2e-14, rel_tol=0.0)
    with pytest.raises(PriorEvidenceError, match="order"):
        evaluate_synthetic_source_prior(
            selection,
            source="synthetic:source-coordinate-prior",
            coordinates=tuple(reversed(coords)),
            values=(x, sd),
            transformations=(CoordinateTransform.IDENTITY, CoordinateTransform.RAW_POSITIVE_TO_LOG),
        )
    with pytest.raises(PriorEvidenceError, match="identified"):
        evaluate_synthetic_source_prior(
            selection,
            source="legacy:opaque",
            coordinates=coords,
            values=(x, sd),
            transformations=(CoordinateTransform.IDENTITY, CoordinateTransform.RAW_POSITIVE_TO_LOG),
        )
    for mode in (LegacyPriorMode.NO_PRIOR, LegacyPriorMode.DEFAULT):
        with pytest.raises(PriorEvidenceError, match="remain unverified"):
            evaluate_synthetic_source_prior(
                LegacyPriorSelection(mode=mode),
                source="synthetic:source-coordinate-prior",
                coordinates=coords,
                values=(x, sd),
                transformations=(CoordinateTransform.IDENTITY, CoordinateTransform.RAW_POSITIVE_TO_LOG),
            )


def test_wp07_screening_is_auditable_not_automatic_likelihood_reclassification() -> None:
    mapped = _mapped(*(_exact(year, str(year - 1999)) for year in range(2000, 2005)))
    flows = tuple(float(year - 1999) for year in range(2000, 2005))
    decision = select_mgb_candidates(
        flows,
        (1.0, 1.0),
        settings=MgbSweepSettings.usgs_mgbt17c_1_0_7(),
        pvalue_provenance="synthetic:explicit-source-rank-pvalues",
    )
    audit = audit_mgb_annual_candidates(
        mapped.series,
        screened_years=tuple(range(2000, 2005)),
        decision=decision,
    )
    assert audit.original_series == mapped.series
    assert decision.low_outlier_count == 0
    before = mapped.series
    result = evaluate_synthetic_likelihood(Gumbel(0, 0), mapped, declaration=_declaration())
    assert len(result.likelihood.terms) == 5
    assert mapped.series is before  # No MGBT/censor change from the composition boundary.
