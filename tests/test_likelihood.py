"""Independent hand-calculated tests for the conditional continuous likelihood."""

from math import comb, erfc, exp, log, pi, sqrt
from statistics import NormalDist

import pytest

from ffa_methods.contracts import AnnualObservation, CensorGroup, ExposureKind, ObservationKind, ObservationSeries
from ffa_methods.likelihood import HistoricalPeriod, LikelihoodEvidenceError, log_likelihood
from tests.analytic_kernels import ExponentialKernel
from tests.numeric_assertions import near


class StandardNormalKernel:
    """Analytic unit Normal oracle; not a production FFA distribution family."""

    def logpdf(self, discharge: float) -> float:
        return -0.5 * discharge**2 - 0.5 * log(2 * pi)

    def logcdf(self, discharge: float) -> float:
        return log(self.cdf(discharge))

    def logsf(self, discharge: float) -> float:
        return log(self.sf(discharge))

    def cdf(self, discharge: float) -> float:
        return 0.5 * erfc(-discharge / sqrt(2))

    def sf(self, discharge: float) -> float:
        return 0.5 * erfc(discharge / sqrt(2))

    def ppf(self, non_exceedance_probability: float) -> float:
        return NormalDist().inv_cdf(non_exceedance_probability)

    def isf(self, survival_probability: float) -> float:
        return -NormalDist().inv_cdf(survival_probability)


@pytest.mark.parametrize(
    ("lower", "upper", "expected_log_probability"),
    [
        (9.0, 10.0, -43.628216632280818),
        (-10.0, -9.0, -43.628216632280818),
        (0.0, 1e-20, -46.970640393085588),
    ],
)
def test_independent_numerics_normal_intervals(lower: float, upper: float, expected_log_probability: float) -> None:
    """Pinned Numerics/R 4.4.3 assertions at PR22 fdf2893; 0BSD source, no code port."""
    kernel = StandardNormalKernel()
    row = AnnualObservation(
        year=2000,
        exposure=ExposureKind.HISTORICAL,
        kind=ObservationKind.INTERVAL_CENSORED,
        lower_bound=lower,
        upper_bound=upper,
    )
    result = log_likelihood(kernel, series(row))
    assert near(result.log_value, expected_log_probability, abs_tol=2e-13, rel_tol=0.0)


def test_narrow_normal_interval_really_coalesces_both_tail_logs() -> None:
    kernel = StandardNormalKernel()
    assert kernel.logcdf(0.0) == kernel.logcdf(1e-20)
    assert kernel.logsf(0.0) == kernel.logsf(1e-20)
    row = AnnualObservation(
        year=2000,
        exposure=ExposureKind.HISTORICAL,
        kind=ObservationKind.INTERVAL_CENSORED,
        lower_bound=0.0,
        upper_bound=1e-20,
    )
    result = log_likelihood(kernel, series(row))
    assert result.log_value > float("-inf")


def test_invalid_tail_logs_cannot_be_masked_by_density_integration() -> None:
    class InvalidTailNormal(StandardNormalKernel):
        def logcdf(self, discharge: float) -> float:
            del discharge
            return float("nan")

        def logsf(self, discharge: float) -> float:
            del discharge
            return float("nan")

    row = AnnualObservation(
        year=2000,
        exposure=ExposureKind.HISTORICAL,
        kind=ObservationKind.INTERVAL_CENSORED,
        lower_bound=0.0,
        upper_bound=1e-20,
    )
    with pytest.raises(ValueError, match="invalid log tail probability"):
        log_likelihood(InvalidTailNormal(), series(row))


def test_unresolved_narrow_interval_does_not_silently_become_zero() -> None:
    class InvalidDensityNormal(StandardNormalKernel):
        def logpdf(self, discharge: float) -> float:
            del discharge  # Required signature; intentionally inconsistent with the CDF/SF.
            return float("-inf")

    row = AnnualObservation(
        year=2000,
        exposure=ExposureKind.HISTORICAL,
        kind=ObservationKind.INTERVAL_CENSORED,
        lower_bound=0.0,
        upper_bound=1e-20,
    )
    with pytest.raises(LikelihoodEvidenceError, match="interval mass is unresolved"):
        log_likelihood(InvalidDensityNormal(), series(row))


def exact(year: int, value: float, exposure: ExposureKind = ExposureKind.SYSTEMATIC) -> AnnualObservation:
    return AnnualObservation(year=year, kind=ObservationKind.EXACT, exposure=exposure, value=value)


def series(*rows: AnnualObservation, groups: tuple[CensorGroup, ...] = ()) -> ObservationSeries:
    return ObservationSeries(discharge_unit="m3/s", annual=rows, groups=groups)


def test_exact_and_individual_censors_are_distinct() -> None:
    rows = (
        exact(2000, 2.0),
        AnnualObservation(
            year=2001, exposure=ExposureKind.SYSTEMATIC, kind=ObservationKind.LEFT_CENSORED, upper_bound=1
        ),
        AnnualObservation(
            year=2002, exposure=ExposureKind.SYSTEMATIC, kind=ObservationKind.RIGHT_CENSORED, lower_bound=3
        ),
        AnnualObservation(
            year=2003,
            exposure=ExposureKind.SYSTEMATIC,
            kind=ObservationKind.INTERVAL_CENSORED,
            lower_bound=1,
            upper_bound=2,
        ),
    )
    expected = -2 + log(1 - exp(-1)) - 3 + log(exp(-1) - exp(-2))
    result = log_likelihood(ExponentialKernel(), series(*rows))
    assert near(result.log_value, expected, abs_tol=1e-12)
    assert len(result.terms) == 4


def test_narrow_upper_tail_interval_uses_survival_subtraction() -> None:
    row = AnnualObservation(
        year=2000,
        kind=ObservationKind.INTERVAL_CENSORED,
        exposure=ExposureKind.HISTORICAL,
        lower_bound=35,
        upper_bound=36,
    )
    result = log_likelihood(ExponentialKernel(), series(row))
    assert near(result.log_value, -35 + log(1 - exp(-1)), abs_tol=1e-13)


def test_grouped_historical_counts_include_binomial_constant() -> None:
    group = CensorGroup(
        threshold=log(2),
        exposure=ExposureKind.HISTORICAL,
        count_above=2,
        count_at_or_below=8,
        start_year=1890,
        end_year=1899,
    )
    result = log_likelihood(ExponentialKernel(), series(groups=(group,)))
    assert near(result.log_value, log(comb(10, 2) / 2**10), abs_tol=1e-12)


def test_count_of_zero_above_does_not_multiply_zero_by_negative_infinity() -> None:
    group = CensorGroup(
        threshold=0.0,
        exposure=ExposureKind.HISTORICAL,
        count_above=1,
        count_at_or_below=0,
        start_year=1800,
        end_year=1800,
    )
    assert near(log_likelihood(ExponentialKernel(), series(groups=(group,))).log_value, 0)


def test_year_labelled_historical_exceedance_and_unmeasured_count() -> None:
    period = HistoricalPeriod(
        start_year=1900,
        end_year=1903,
        threshold=1.0,
        known_exceedances=((1900, 2.0),),
        unmeasured_exceedances=1,
    )
    # Year 1900 is explicitly known. Only the remaining 3 years contribute C(3,1).
    expected = -2 + log(3) - 1 + 2 * log(1 - exp(-1))
    result = log_likelihood(ExponentialKernel(), series(), historical_periods=(period,))
    assert near(result.log_value, expected, abs_tol=1e-12)


def test_year_labelled_historical_no_unmeasured_floods_has_no_multiplicity() -> None:
    period = HistoricalPeriod(start_year=1900, end_year=1901, threshold=1, known_exceedances=((1900, 2),))
    result = log_likelihood(ExponentialKernel(), series(), historical_periods=(period,))
    assert near(result.log_value, -2 + log(1 - exp(-1)))


@pytest.mark.parametrize(
    ("rows", "groups", "periods"),
    [
        ((exact(2000, 1), exact(2000, 2)), (), ()),
        (
            (exact(2000, 1),),
            (
                CensorGroup(
                    threshold=1,
                    exposure=ExposureKind.HISTORICAL,
                    count_above=1,
                    count_at_or_below=0,
                    start_year=2000,
                    end_year=2000,
                ),
            ),
            (),
        ),
        (
            (),
            (
                CensorGroup(
                    threshold=1,
                    exposure=ExposureKind.HISTORICAL,
                    count_above=0,
                    count_at_or_below=1,
                    start_year=2000,
                    end_year=2000,
                ),
            ),
            (HistoricalPeriod(start_year=2000, end_year=2000, threshold=1),),
        ),
        (
            (),
            (
                CensorGroup(
                    threshold=1,
                    exposure=ExposureKind.HISTORICAL,
                    count_above=0,
                    count_at_or_below=1,
                    start_year=2000,
                    end_year=2001,
                ),
            ),
            (),
        ),
        (
            (),
            (
                CensorGroup(
                    threshold=1,
                    exposure=ExposureKind.UNRESOLVED,
                    count_above=1,
                    count_at_or_below=0,
                    start_year=2000,
                    end_year=2000,
                ),
            ),
            (),
        ),
        (
            (),
            (
                CensorGroup(
                    threshold=1,
                    exposure=ExposureKind.HISTORICAL,
                    count_above=1,
                    count_at_or_below=None,
                    start_year=2000,
                    end_year=2000,
                ),
            ),
            (),
        ),
    ],
)
def test_exposure_ambiguity_is_not_silently_counted(
    rows: tuple[AnnualObservation, ...], groups: tuple[CensorGroup, ...], periods: tuple[HistoricalPeriod, ...]
) -> None:
    with pytest.raises(LikelihoodEvidenceError):
        log_likelihood(ExponentialKernel(), series(*rows, groups=groups), historical_periods=periods)


def test_excluded_missing_do_not_create_censor_information() -> None:
    rows = (
        AnnualObservation(year=2000, kind=ObservationKind.EXCLUDED, exposure=ExposureKind.SYSTEMATIC, value=1),
        AnnualObservation(year=2001, kind=ObservationKind.MISSING, exposure=ExposureKind.UNRESOLVED),
        exact(2002, 2),
    )
    result = log_likelihood(ExponentialKernel(), series(*rows))
    assert near(result.log_value, -2)
    assert [term.included for term in result.terms] == [False, False, True]


def test_unknown_zero_model_and_informative_unknown_exposure_reject() -> None:
    with pytest.raises(LikelihoodEvidenceError, match="zero-threshold"):
        log_likelihood(
            ExponentialKernel(),
            series(
                AnnualObservation(
                    year=2000,
                    kind=ObservationKind.AT_OR_BELOW_ZERO_THRESHOLD,
                    exposure=ExposureKind.SYSTEMATIC,
                    value=0,
                )
            ),
        )
    with pytest.raises(LikelihoodEvidenceError, match="unresolved exposure"):
        log_likelihood(ExponentialKernel(), series(exact(2000, 1, ExposureKind.UNRESOLVED)))


def test_impossible_exact_observation_gives_minus_infinity() -> None:
    assert log_likelihood(ExponentialKernel(), series(exact(2000, -1))).log_value == float("-inf")


def test_invalid_historical_magnitude_rejected() -> None:
    with pytest.raises(LikelihoodEvidenceError, match="above threshold"):
        HistoricalPeriod(start_year=1900, end_year=1901, threshold=2, known_exceedances=((1901, 2),))


def test_interval_with_equal_endpoints_is_zero_probability() -> None:
    row = AnnualObservation(
        year=2000,
        kind=ObservationKind.INTERVAL_CENSORED,
        exposure=ExposureKind.SYSTEMATIC,
        lower_bound=3,
        upper_bound=3,
    )
    assert log_likelihood(ExponentialKernel(), series(row)).log_value == float("-inf")
