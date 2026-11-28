"""Synthetic classified GEV/LP3 records shared by the posterior regression cases."""

from hashlib import sha256
from math import exp

from ffa_methods.contracts import AnnualObservation, ExposureKind, ObservationKind, ObservationSeries
from ffa_methods.family_posterior import GEV_LOCATION, LP3_MEAN, ClassifiedFixedFamilyPosterior, PosteriorFamily
from ffa_methods.lognormal_posterior import SOURCE_CLASSIFICATION
from ffa_methods.priors import JointGaussianPrior, LegacyPriorMode


def synthetic_flow(family: PosteriorFamily, value: float) -> float:
    return 3.0 + value if family == PosteriorFamily.GEV else 2.0 * exp(value)


def annual_row(
    year: int,
    kind: ObservationKind,
    *,
    value: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> AnnualObservation:
    return AnnualObservation(
        year=year,
        kind=kind,
        exposure=ExposureKind.SYSTEMATIC,
        value=value,
        lower_bound=lower,
        upper_bound=upper,
    )


def synthetic_model(family: PosteriorFamily, shape: float) -> ClassifiedFixedFamilyPosterior:
    q, upper, lower, low, high = (synthetic_flow(family, x) for x in (0.2, 0.8, -0.2, -0.3, 1.1))
    rows = (
        annual_row(2000, ObservationKind.EXACT, value=q),
        annual_row(2001, ObservationKind.LEFT_CENSORED, upper=upper),
        annual_row(2002, ObservationKind.RIGHT_CENSORED, lower=lower),
        annual_row(2003, ObservationKind.INTERVAL_CENSORED, lower=low, upper=high),
        annual_row(2004, ObservationKind.MISSING),
        annual_row(2005, ObservationKind.EXCLUDED, value=0.0),
    )
    is_gev = family == PosteriorFamily.GEV
    return ClassifiedFixedFamilyPosterior(
        family=family,
        series=ObservationSeries(discharge_unit="m3/s", annual=rows, year_definition="calendar year"),
        source="synthetic classified annual maxima",
        source_sha256=sha256(b"synthetic classified records").hexdigest(),
        source_classification=SOURCE_CLASSIFICATION,
        fixed_log_scale=0.0,
        fixed_shape=shape,
        reference_discharge=None if is_gev else 2.0,
        prior=JointGaussianPrior(
            coordinates=(GEV_LOCATION if is_gev else LP3_MEAN,),
            means=(3.0 if is_gev else 0.0,),
            covariance=((0.0625,),),
            source="synthetic supplied location prior",
            mode=LegacyPriorMode.SUPPLIED,
        ),
    )
