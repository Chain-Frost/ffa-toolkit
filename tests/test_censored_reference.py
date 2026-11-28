"""Independent identities and quadrature for deterministic censor enclosures."""

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from math import erfc, exp, log, nextafter, pi, sqrt
from statistics import NormalDist

import pytest

from ffa_methods.censored_reference import (
    PI_BOUNDS,
    CensorEvent,
    CensoredReferenceError,
    FixedScaleReference,
    FixedScaleReferenceInputs,
    ReferenceBounds,
    ReferenceControls,
    ReferenceKind,
    ReferenceRequest,
    fixed_scale_reference,
)


def _controls(level: int = 3) -> ReferenceControls:
    return ReferenceControls(
        extent=8, dyadic_level=level, maximum_cells=10000, maximum_cell_terms=100000, maximum_log_ratio=4.0
    )


def _mixed() -> FixedScaleReferenceInputs:
    return FixedScaleReferenceInputs(
        prior_mean=0.0,
        prior_variance=1.0,
        fixed_log_sd=log(0.8),
        reference_discharge=1.0,
        exact_flows=(exp(-0.4), exp(0.5)),
        censor_events=(
            CensorEvent(upper=exp(-0.2)),
            CensorEvent(lower=exp(0.1)),
            CensorEvent(lower=exp(-0.5), upper=exp(0.6)),
        ),
        proposal_mean=0.0,
        proposal_variance=1.0,
    )


def _value(result: FixedScaleReference, index: int) -> ReferenceBounds:
    bounds = result.values[index].bounds
    assert bounds is not None, result.values[index].reason
    return bounds


def _inside(bounds: ReferenceBounds, value: float) -> None:
    assert bounds.contains(Decimal.from_float(value)), (bounds, value)


def test_mixed_reference_refinement_and_output_containment() -> None:
    requests = (
        ReferenceRequest(kind=ReferenceKind.MEAN_MU),
        ReferenceRequest(kind=ReferenceKind.CENTERED_SECOND_MU, argument=0.0),
        ReferenceRequest(kind=ReferenceKind.EXP_MU),
        ReferenceRequest(kind=ReferenceKind.EXPECTED_AEP, argument=exp(0.35)),
        ReferenceRequest(kind=ReferenceKind.EXPECTED_F, argument=exp(0.35)),
        ReferenceRequest(kind=ReferenceKind.MU_CDF, argument=0.0),
        ReferenceRequest(kind=ReferenceKind.MU_SF, argument=0.0),
    )
    inputs = _mixed()
    coarse = fixed_scale_reference(inputs, _controls(3), requests)
    fine = fixed_scale_reference(inputs, _controls(5), requests)
    # Independent values from adaptive integration of the unnormalized real-line density.
    _inside(fine.normalizer, 0.0676240123107145)
    assert coarse.normalizer.lower < fine.normalizer.lower < fine.normalizer.upper < coarse.normalizer.upper
    assert fine.ratio_eligible and fine.log_ratio_upper is not None
    assert coarse.log_ratio_upper is not None and fine.log_ratio_upper < coarse.log_ratio_upper
    assert fine.log_ratio_upper >= Decimal("3.403272126898")
    exp_mu = 3.026592795635 / exp(0.8 * NormalDist().inv_cdf(0.9))
    expected = (0.0158223513737, 0.1329595261158, exp_mu, 0.3519021069439, 1 - 0.3519021069439)
    for index, value in enumerate(expected):
        bounds, old = _value(fine, index), _value(coarse, index)
        _inside(bounds, value)
        assert old.lower < bounds.lower <= bounds.upper < old.upper
    # For complementary indicators the returned (not normalized separately)
    # intervals must jointly enclose one, even for a threshold crossing cell.
    cdf, sf = _value(fine, 5), _value(fine, 6)
    _inside(cdf, 0.4827756703319943)
    _inside(sf, 0.5172243296680058)
    assert cdf.lower + sf.lower <= 1 <= cdf.upper + sf.upper
    assert fine.reserved_cell_terms <= fine.controls.maximum_cell_terms


def _all_censored() -> FixedScaleReferenceInputs:
    return FixedScaleReferenceInputs(
        prior_mean=0.0,
        prior_variance=1.0,
        fixed_log_sd=0.0,
        reference_discharge=1.0,
        exact_flows=(),
        censor_events=(CensorEvent(upper=1.0),),
        proposal_mean=0.0,
        proposal_variance=1.0,
    )


def test_single_event_exact_identities_and_identical_base_proposal() -> None:
    inputs = _all_censored()
    requests = (
        ReferenceRequest(kind=ReferenceKind.MEAN_MU),
        ReferenceRequest(kind=ReferenceKind.CENTERED_SECOND_MU, argument=0.0),
        ReferenceRequest(kind=ReferenceKind.EXPECTED_F, argument=1.0),
        ReferenceRequest(kind=ReferenceKind.EXPECTED_AEP, argument=1.0),
        ReferenceRequest(kind=ReferenceKind.EXP_MU),
        ReferenceRequest(kind=ReferenceKind.MU_CDF, argument=0.0),
    )
    result = fixed_scale_reference(inputs, _controls(), requests)
    _inside(result.normalizer, 0.5)
    assert result.ratio_eligible and result.log_ratio_upper is not None
    assert result.log_ratio_upper >= Decimal.from_float(log(2))
    # Truncated joint Gaussian and uniform-to-Beta transformations.
    expected = (-1 / sqrt(pi), 1.0, 2 / 3, 1 / 3, 2 * exp(0.5) * NormalDist().cdf(-1 / sqrt(2)), 0.75)
    for index, value in enumerate(expected):
        _inside(_value(result, index), value)
    displaced = fixed_scale_reference(replace(inputs, proposal_mean=0.1), _controls(1), ())
    assert not displaced.ratio_eligible and displaced.log_ratio_upper is None
    narrower = fixed_scale_reference(replace(inputs, proposal_variance=0.5), _controls(1), ())
    assert not narrower.ratio_eligible and narrower.log_ratio_upper is None
    ceiling = fixed_scale_reference(inputs, replace(_controls(1), maximum_log_ratio=0.1), ())
    assert not ceiling.ratio_eligible and ceiling.log_ratio_upper is not None
    # A changed ambient decimal precision must not change our enclosures.
    with localcontext() as ctx:
        ctx.prec = 7
        replay = fixed_scale_reference(inputs, _controls(), requests)
    assert result == replay


def test_all_censored_beta_normalizer_and_probability() -> None:
    inputs = replace(_all_censored(), censor_events=(CensorEvent(upper=1.0),) * 3 + (CensorEvent(lower=1.0),) * 2)
    result = fixed_scale_reference(inputs, _controls(), (ReferenceRequest(kind=ReferenceKind.EXPECTED_F, argument=1.0),))
    _inside(result.normalizer, 1 / 60)
    _inside(_value(result, 0), 4 / 7)


def test_narrow_interval_and_resource_limits_fail_closed() -> None:
    inputs = replace(_all_censored(), censor_events=(CensorEvent(lower=1.0, upper=nextafter(1.0, 2.0)),))
    requests = (ReferenceRequest(kind=ReferenceKind.MEAN_MU),)
    result = fixed_scale_reference(inputs, _controls(0), requests)
    assert result.normalizer.lower == 0 < result.normalizer.upper
    assert not result.ratio_eligible and result.values[0].bounds is None
    with pytest.raises(CensoredReferenceError, match="cell-term budget"):
        fixed_scale_reference(inputs, replace(_controls(0), maximum_cell_terms=1), requests)
    with pytest.raises(CensoredReferenceError, match="cell budget"):
        replace(_controls(), maximum_cells=1)
    with pytest.raises(CensoredReferenceError, match="duplicate"):
        fixed_scale_reference(inputs, _controls(0), requests * 2)


def test_exact_only_reference_and_direct_rare_survival() -> None:
    inputs = replace(_mixed(), censor_events=())
    requests = (
        ReferenceRequest(kind=ReferenceKind.MEAN_MU),
        ReferenceRequest(kind=ReferenceKind.EXPECTED_AEP, argument=exp(12)),
    )
    result = fixed_scale_reference(inputs, _controls(), requests)
    assert result.normalizer == ReferenceBounds(Decimal(1), Decimal(1))
    v = 1 / (1 + 2 / 0.8**2)
    m = v * 0.1 / 0.8**2
    _inside(_value(result, 0), m)
    # Independent direct erfc, not 1-CDF, even when the latter rounds to zero.
    expected = 0.5 * erfc((12 - m) / sqrt(2 * (v + 0.8**2)))
    bounds = _value(result, 1)
    _inside(bounds, expected)
    assert 0 < bounds.lower <= bounds.upper < Decimal("1e-12")


def _atan_bounds(denominator: int, pairs: int) -> tuple[Fraction, Fraction]:
    terms = 2 * pairs
    lower = sum(
        (Fraction((-1) ** k, (2 * k + 1) * denominator ** (2 * k + 1)) for k in range(terms)), Fraction(0)
    )
    return lower, lower + Fraction(1, (2 * terms + 1) * denominator ** (2 * terms + 1))


def test_pi_constant_encloses_machin_series() -> None:
    a, b = _atan_bounds(5, 25)
    c, d = _atan_bounds(239, 10)
    assert Fraction(PI_BOUNDS.lower) < 16 * a - 4 * d < 16 * b - 4 * c < Fraction(PI_BOUNDS.upper)


def test_unit_rescaling_and_small_normalizer() -> None:
    inputs = _all_censored()
    scaled = replace(inputs, reference_discharge=2.0, censor_events=(CensorEvent(upper=2.0),))
    result = fixed_scale_reference(scaled, _controls(2), (ReferenceRequest(kind=ReferenceKind.EXPECTED_F, argument=2.0),))
    _inside(result.normalizer, 0.5)
    _inside(_value(result, 0), 2 / 3)
    rare = replace(inputs, censor_events=(CensorEvent(upper=exp(-8)),))
    result = fixed_scale_reference(rare, _controls(2), ())
    expected = 0.5 * erfc(4.0)
    _inside(result.normalizer, expected)
    assert result.normalizer.lower > 0
    assert not result.ratio_eligible and result.log_ratio_upper is not None
