"""Independent finite-box and omitted-tail checks for estimated-scale references."""

from dataclasses import replace
from decimal import Decimal, FloatOperation, localcontext
from math import exp

import pytest

from ffa_methods.censored_reference import CensoredReferenceError, CensorEvent
from ffa_methods.estimated_scale_reference import (
    EstimatedScaleControls,
    EstimatedScaleInputs,
    EstimatedScaleKind,
    EstimatedScaleReference,
    EstimatedScaleRequest,
    estimated_scale_reference,
)


def _controls(level: int = 0, extent: int = 4) -> EstimatedScaleControls:
    return EstimatedScaleControls(
        extent=extent, dyadic_level=level, maximum_cells=2000, maximum_cell_terms=50000
    )


def _mixed() -> EstimatedScaleInputs:
    return EstimatedScaleInputs(
        prior_mean=(0.0, 0.0),
        prior_covariance=((0.16, 0.02), (0.02, 0.0144)),
        reference_discharge=1.0,
        exact_flows=(exp(-0.4), exp(0.5)),
        censor_events=(
            CensorEvent(upper=exp(-0.2)),
            CensorEvent(lower=exp(0.1)),
            CensorEvent(lower=exp(-0.5), upper=exp(0.5)),
        ),
    )


def _requests() -> tuple[EstimatedScaleRequest, ...]:
    return (
        EstimatedScaleRequest(kind=EstimatedScaleKind.MEAN_MU),
        EstimatedScaleRequest(kind=EstimatedScaleKind.MEAN_ETA),
        EstimatedScaleRequest(kind=EstimatedScaleKind.EXP_MU),
        EstimatedScaleRequest(kind=EstimatedScaleKind.EXP_ETA),
        EstimatedScaleRequest(kind=EstimatedScaleKind.CENTERED_SECOND_MU, argument=0.0),
        EstimatedScaleRequest(kind=EstimatedScaleKind.CENTERED_SECOND_ETA, argument=0.0),
        EstimatedScaleRequest(kind=EstimatedScaleKind.EXPECTED_F, argument=exp(0.35)),
        EstimatedScaleRequest(kind=EstimatedScaleKind.EXPECTED_AEP, argument=exp(0.35)),
        EstimatedScaleRequest(kind=EstimatedScaleKind.MU_CDF, argument=0.0),
        EstimatedScaleRequest(kind=EstimatedScaleKind.MU_SF, argument=0.0),
    )


def _check(result: EstimatedScaleReference, expected: tuple[float, ...]) -> None:
    for item, reference in zip(result.values, expected, strict=True):
        assert item.bounds is not None, item.reason
        assert item.bounds.contains(Decimal.from_float(reference)), (item, reference)


def test_correlated_mixed_reference_and_refinement() -> None:
    # Independent Gauss-Legendre calculation in conditional Gaussian coordinates,
    # integrating both sides of the mu=0 indicator separately. See the method note.
    expected = (
        -0.00827612383713244,
        -0.0222234418867801,
        1.03928232666411,
        0.984394213479628,
        0.0928259626422699,
        0.0134669252021155,
        0.640717343006427,
        0.359282656993573,
        0.516776416243289,
        0.483223583756711,
    )
    inputs = _mixed()
    requests = _requests()
    coarse = estimated_scale_reference(inputs, _controls(), requests)
    fine = estimated_scale_reference(inputs, _controls(2), requests)
    assert coarse.normalizer.contains(Decimal("0.0455818189049154"))
    assert fine.normalizer.contains(Decimal("0.0455818189049154"))
    _check(coarse, expected)
    _check(fine, expected)
    assert coarse.normalizer.lower < fine.normalizer.lower < fine.normalizer.upper < coarse.normalizer.upper
    for c, f in zip(coarse.values, fine.values, strict=True):
        assert c.bounds is not None and f.bounds is not None
        assert c.bounds.lower <= f.bounds.lower <= f.bounds.upper <= c.bounds.upper
    with localcontext() as ctx:
        ctx.prec = 100
        assert fine.tilted_mean[0].contains(-Decimal.from_float(0.02) * 2)
        assert fine.tilted_mean[1].contains(-Decimal.from_float(0.0144) * 2)
    # Independent binary64 cell-range calculation in the method note:
    # the old ratio interval was approximately [0.1944, 0.6646], while
    # using both integrals gives approximately [0.2266, 0.5180].
    aep = fine.values[7].bounds
    cdf = fine.values[6].bounds
    assert aep is not None and cdf is not None
    assert Decimal("0.22") < aep.lower <= aep.upper < Decimal("0.53")
    assert aep.lower + cdf.lower <= 1 <= aep.upper + cdf.upper
    assert fine.reserved_cell_terms == 20696 <= fine.controls.maximum_cell_terms
    # A budget that covers the former single-numerator reservation cannot
    # silently admit the added complementary integrands.
    with pytest.raises(CensoredReferenceError, match="cell-term budget"):
        estimated_scale_reference(inputs, replace(_controls(2), maximum_cell_terms=16600), requests)


def test_reflection_and_untruncated_tails() -> None:
    inputs = replace(
        _mixed(),
        prior_covariance=((0.16, 0.0), (0.0, 0.0144)),
        exact_flows=(),
        censor_events=(CensorEvent(upper=1.0), CensorEvent(lower=1.0)),
    )
    requests = (
        EstimatedScaleRequest(kind=EstimatedScaleKind.MEAN_MU),
        EstimatedScaleRequest(kind=EstimatedScaleKind.EXPECTED_F, argument=1.0),
        EstimatedScaleRequest(kind=EstimatedScaleKind.EXPECTED_AEP, argument=1.0),
        EstimatedScaleRequest(kind=EstimatedScaleKind.MU_CDF, argument=0.0),
        EstimatedScaleRequest(kind=EstimatedScaleKind.MU_SF, argument=0.0),
    )
    # The posterior is symmetric in mu at every eta; the exact values apply
    # to the full target, not only the finite box.
    result = estimated_scale_reference(inputs, _controls(1, extent=1), requests)
    assert result.omitted_mass.upper > Decimal("0.5")
    _check(result, (0.0, 0.5, 0.5, 0.5, 0.5))
    with localcontext() as ctx:
        ctx.prec = 7
        ctx.traps[FloatOperation] = True
        assert result == estimated_scale_reference(inputs, _controls(1, extent=1), requests)


def test_resource_and_invalid_covariance_refusal() -> None:
    inputs = _mixed()
    controls = _controls()
    with pytest.raises(CensoredReferenceError, match="cell budget"):
        replace(controls, maximum_cells=controls.cells - 1)
    with pytest.raises(CensoredReferenceError, match="cell-term budget"):
        estimated_scale_reference(inputs, replace(controls, maximum_cell_terms=1), _requests())
    with pytest.raises(CensoredReferenceError, match="duplicate"):
        estimated_scale_reference(inputs, controls, (_requests()[0],) * 2)
    with pytest.raises(CensoredReferenceError, match="positive-definite"):
        replace(inputs, prior_covariance=((1.0, 1.0), (1.0, 1.0)))
    with pytest.raises(CensoredReferenceError, match="positive-definite"):
        replace(inputs, prior_covariance=((1.0, 0.1), (0.2, 1.0)))
    with pytest.raises(CensoredReferenceError, match="invalid estimated-scale"):
        EstimatedScaleRequest(kind=EstimatedScaleKind.EXPECTED_AEP, argument=0.0)


def test_exact_only_unit_rescaling_and_rare_aep() -> None:
    inputs = replace(_mixed(), censor_events=())
    request = EstimatedScaleRequest(kind=EstimatedScaleKind.EXPECTED_AEP, argument=exp(4.0))
    result = estimated_scale_reference(inputs, _controls(1), (request,))
    scaled = replace(inputs, reference_discharge=2.0, exact_flows=tuple(2 * q for q in inputs.exact_flows))
    assert request.argument is not None
    scaled_result = estimated_scale_reference(scaled, _controls(1), (replace(request, argument=2 * request.argument),))
    assert result.normalizer.lower > 0
    assert result.values[0].bounds is not None and scaled_result.values[0].bounds is not None
    assert result.values[0].bounds.lower > 0
    assert result.values[0].bounds.lower <= scaled_result.values[0].bounds.upper
    assert scaled_result.values[0].bounds.lower <= result.values[0].bounds.upper
