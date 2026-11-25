"""Independent GEV endpoint examples, including exact binary64 boundary semantics."""

from math import inf, nextafter, ulp
from sys import float_info

from ffa_methods.gev_support import GEVSupportObservation, diagnose_gev_support


def test_upper_lower_and_gumbel_support() -> None:
    records = tuple(GEVSupportObservation(2000 + i, q) for i, q in enumerate((5.0, 6.0, 7.0, 14.0, 15.0)))
    upper = diagnose_gev_support(records, location=10.0, scale=2.0, shape=0.5, unit="m3/s")
    assert upper.kind == "finite_upper"
    assert upper.endpoint_exact == "14"
    assert upper.endpoint_approximate == 14.0
    assert upper.observation_count == 5
    assert upper.outside == (records[4],)
    assert upper.at_endpoint == (records[3],)
    lower = diagnose_gev_support(records, location=10.0, scale=2.0, shape=-0.5, unit="m3/s")
    assert lower.kind == "finite_lower"
    assert lower.endpoint_exact == "6"
    assert lower.outside == (records[0],)
    assert lower.at_endpoint == (records[1],)
    unbounded = diagnose_gev_support(records, location=10.0, scale=2.0, shape=-0.0, unit="m3/s")
    assert unbounded.kind == "unbounded"
    assert unbounded.endpoint_exact is None
    assert unbounded.endpoint_approximate is None
    assert unbounded.outside == unbounded.at_endpoint == ()
    assert unbounded.observation_count == 5


def test_rounded_endpoint_is_not_mathematical_equality() -> None:
    # 1/3 lies strictly between these adjacent binary64 numbers.
    below = GEVSupportObservation(2001, 1.0 / 3.0)
    above = GEVSupportObservation(2000, nextafter(below.discharge, inf))
    support = diagnose_gev_support((above, below), location=0.0, scale=1.0, shape=3.0, unit="m3/s")
    assert support.endpoint_exact == "1/3"
    assert support.endpoint_approximate == below.discharge
    assert support.outside == (above,)
    assert support.at_endpoint == ()


def test_adjacent_values_and_exact_binary_unit_scaling() -> None:
    records = tuple(
        GEVSupportObservation(2000 + i, q) for i, q in enumerate((nextafter(14.0, -inf), 14.0, nextafter(14.0, inf)))
    )
    for factor in (2.0**-100, 1.0, 2.0**100):
        scaled = tuple(GEVSupportObservation(row.year, row.discharge * factor) for row in records)
        support = diagnose_gev_support(scaled, location=10 * factor, scale=2 * factor, shape=0.5, unit="scaled")
        assert support.endpoint_approximate == 14 * factor
        assert support.at_endpoint == (scaled[1],)
        assert support.outside == (scaled[2],)


def test_finite_endpoint_beyond_binary64_is_not_unbounded() -> None:
    rows = (GEVSupportObservation(2000, float_info.max),)
    support = diagnose_gev_support(rows, location=0.0, scale=float_info.max, shape=ulp(0.0), unit="m3/s")
    assert support.kind == "finite_upper"
    assert support.endpoint_exact is not None
    assert support.endpoint_approximate is None
    assert support.outside == support.at_endpoint == ()
