"""Strictly typed numeric assertions for mathematical test fixtures."""

from math import isclose


def near(actual: float, expected: float, *, abs_tol: float = 1e-12, rel_tol: float = 1e-6) -> bool:
    """Compare finite analytic calculations without untyped pytest-approx calls."""
    return isclose(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol)
