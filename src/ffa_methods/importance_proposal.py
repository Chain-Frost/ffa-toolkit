"""Bounded smooth local-mode and inverse-curvature proposal construction (WP06).

The target and the resulting proposal use the same source coordinates. Coordinate
scales are numerical preconditioners only; no prior, density or Jacobian is changed.
"""

from dataclasses import dataclass
from math import fsum, isclose, isfinite, sqrt, ulp
from typing import TypeAlias

from .importance_sampling import ExplicitNormalProposal, ExplicitPosteriorTarget, SamplingError

Vector: TypeAlias = tuple[float, ...]
Matrix: TypeAlias = tuple[Vector, ...]


class ProposalConstructionError(SamplingError):
    """No numerically admitted local Gaussian proposal could be constructed."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CurvatureControls:
    """Explicit local Newton and adaptive central-difference settings."""

    scales: Vector
    initial_step: float
    max_refinements: int
    max_iterations: int
    gradient_tolerance: float
    mode_tolerance: float
    curvature_tolerance: float
    max_condition: float

    def __post_init__(self) -> None:
        if not self.scales or any(not 1e-12 <= value <= 1e12 for value in self.scales):
            msg = "coordinate scales must lie in [1e-12, 1e12]"
            raise ProposalConstructionError(msg)
        if not 1e-6 <= self.initial_step <= 0.1:
            msg = "initial scaled difference step must lie in [1e-6, 0.1]"
            raise ProposalConstructionError(msg)
        if type(self.max_refinements) is not int or not 1 <= self.max_refinements <= 8:
            msg = "difference refinements must be an integer in [1, 8]"
            raise ProposalConstructionError(msg)
        if type(self.max_iterations) is not int or not 1 <= self.max_iterations <= 100:
            msg = "Newton iterations must be an integer in [1, 100]"
            raise ProposalConstructionError(msg)
        if not (
            0 < self.gradient_tolerance <= 0.01
            and 0 < self.mode_tolerance <= 0.01
            and 0 < self.curvature_tolerance <= 0.01
            and 1 <= self.max_condition <= 1e8
        ):
            msg = "invalid gradient, mode, curvature or conditioning tolerance"
            raise ProposalConstructionError(msg)


@dataclass(frozen=True, slots=True)
class CurvatureProposalResult:
    """Local numerical result; neither global-mode nor posterior-coverage certification."""

    proposal: ExplicitNormalProposal
    negative_log_hessian: Matrix
    start: Vector
    controls: CurvatureControls
    log_posterior: float
    scaled_gradient_norm: float
    scaled_gradient_error: float
    scaled_curvature_error: float
    scaled_condition: float
    scaled_step: float
    iterations: int
    evaluations: int
    termination: str = "local stationarity and positive resolved curvature"


def _norm(matrix: Matrix) -> float:
    return max(fsum(abs(value) for value in row) for row in matrix)


def _max_abs(vector: Vector) -> float:
    return max(abs(value) for value in vector)


@dataclass(slots=True)
class _Objective:
    target: ExplicitPosteriorTarget
    evaluations: int = 0
    spacing: float = 0.0

    def evaluate(self, point: Vector) -> float:
        self.evaluations += 1
        if any(not isfinite(value) for value in point):
            msg = "nonfinite posterior evaluation coordinates"
            raise ProposalConstructionError(msg)
        try:
            likelihood = self.target.log_likelihood(point)
            prior = self.target.log_prior(point)
            value = -fsum((likelihood, prior))
        except (ArithmeticError, ValueError, TypeError) as exc:
            msg = "posterior density evaluation failed"
            raise ProposalConstructionError(msg) from exc
        if not all(isfinite(v) for v in (likelihood, prior, value)):
            msg = "nonfinite posterior density; boundary or invalid target is unsupported"
            raise ProposalConstructionError(msg)
        self.spacing = max(self.spacing, ulp(likelihood), ulp(prior), ulp(value))
        return value


def _shift(point: Vector, scales: Vector, delta: Vector) -> Vector:
    result = tuple(x + s * d for x, s, d in zip(point, scales, delta, strict=True))
    for x, y, s, d in zip(point, result, scales, delta, strict=True):
        if not isfinite(y) or (d != 0.0 and not isclose((y - x) / s, d, rel_tol=1e-6)):
            msg = "coordinate perturbation is not representable at the requested resolution"
            raise ProposalConstructionError(msg)
    return result


def _probe(
    objective: _Objective, point: Vector, scales: Vector, step: float, i: int, j: int, sign_i: int, sign_j: int
) -> float:
    delta = tuple(step * ((sign_i if k == i else 0) + (sign_j if k == j else 0)) for k in range(len(point)))
    return objective.evaluate(_shift(point, scales, delta))


@dataclass(frozen=True, slots=True)
class _Stencil:
    value: float
    gradient: Vector
    hessian: Matrix
    spacing: float


def _stencil(objective: _Objective, point: Vector, scales: Vector, step: float) -> _Stencil:
    objective.spacing = 0.0
    centre = objective.evaluate(point)
    n = len(point)
    gradient: list[float] = []
    hessian = [[0.0] * n for _ in point]
    for i in range(n):
        plus = _probe(objective, point, scales, step, i, i, 1, 0)
        minus = _probe(objective, point, scales, step, i, i, -1, 0)
        gradient.append(fsum((plus, -minus)) / (2 * step))
        hessian[i][i] = fsum((plus, minus, -centre, -centre)) / step**2
        for j in range(i):
            pp = _probe(objective, point, scales, step, i, j, 1, 1)
            pm = _probe(objective, point, scales, step, i, j, 1, -1)
            mp = _probe(objective, point, scales, step, i, j, -1, 1)
            mm = _probe(objective, point, scales, step, i, j, -1, -1)
            hessian[i][j] = hessian[j][i] = fsum((pp, mm, -pm, -mp)) / (4 * step**2)
    matrix = tuple(tuple(row) for row in hessian)
    if not all(isfinite(v) for row in matrix for v in row) or not all(isfinite(v) for v in gradient):
        msg = "nonfinite numerical derivatives"
        raise ProposalConstructionError(msg)
    return _Stencil(centre, tuple(gradient), matrix, objective.spacing)


@dataclass(frozen=True, slots=True)
class _Derivatives:
    stencil: _Stencil
    gradient_error: float
    curvature_error: float
    step: float


def _adaptive(objective: _Objective, point: Vector, controls: CurvatureControls) -> _Derivatives:
    step = controls.initial_step
    coarse = _stencil(objective, point, controls.scales, step)
    for _ in range(controls.max_refinements):
        step *= 0.5
        fine = _stencil(objective, point, controls.scales, step)
        spacing = max(coarse.spacing, fine.spacing)
        ge = _max_abs(tuple(a - b for a, b in zip(fine.gradient, coarse.gradient, strict=True)))
        ge += 16 * spacing / step
        he = _norm(
            tuple(
                tuple(a - b for a, b in zip(fine_row, coarse_row, strict=True))
                for fine_row, coarse_row in zip(fine.hessian, coarse.hessian, strict=True)
            )
        )
        he += len(point) * 32 * spacing / step**2
        if (
            fine.value == coarse.value
            and ge <= controls.gradient_tolerance / 4
            and he <= controls.curvature_tolerance * _norm(fine.hessian)
            and isfinite(he)
            and _norm(fine.hessian) > 0
        ):
            return _Derivatives(fine, ge, he, step)
        coarse = fine
    msg = "adaptive differences could not resolve gradient or curvature"
    raise ProposalConstructionError(msg)


def _cholesky(matrix: Matrix) -> Matrix:
    n = len(matrix)
    lower = [[0.0] * n for _ in matrix]
    for i in range(n):
        for j in range(i + 1):
            remainder = matrix[i][j] - fsum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if not isfinite(remainder) or remainder <= 0:
                    msg = "negative log-posterior curvature is not positive definite"
                    raise ProposalConstructionError(msg)
                lower[i][j] = sqrt(remainder)
            else:
                lower[i][j] = remainder / lower[j][j]
    return tuple(tuple(row) for row in lower)


def _solve(lower: Matrix, rhs: Vector) -> Vector:
    n = len(rhs)
    forward = [0.0] * n
    result = [0.0] * n
    for i in range(n):
        forward[i] = (rhs[i] - fsum(lower[i][k] * forward[k] for k in range(i))) / lower[i][i]
    for i in reversed(range(n)):
        result[i] = (forward[i] - fsum(lower[k][i] * result[k] for k in range(i + 1, n))) / lower[i][i]
    if any(not isfinite(v) for v in result):
        msg = "curvature solve is not representable"
        raise ProposalConstructionError(msg)
    return tuple(result)


def _inverse(lower: Matrix) -> Matrix:
    n = len(lower)
    columns = tuple(_solve(lower, tuple(float(i == j) for i in range(n))) for j in range(n))
    return tuple(tuple(columns[i][j] / 2 + columns[j][i] / 2 for j in range(n)) for i in range(n))


def _checked_inverse(derivatives: _Derivatives, controls: CurvatureControls) -> tuple[Matrix, float]:
    hessian = derivatives.stencil.hessian
    inverse = _inverse(_cholesky(hessian))
    inverse_norm = _norm(inverse)
    condition = _norm(hessian) * inverse_norm
    if not isfinite(condition) or condition > controls.max_condition:
        msg = "scaled curvature is ill-conditioned"
        raise ProposalConstructionError(msg)
    if inverse_norm * derivatives.curvature_error >= 0.05:
        msg = "curvature uncertainty is too large to establish positive resolved curvature"
        raise ProposalConstructionError(msg)
    n = len(hessian)
    residual = tuple(
        tuple(fsum(hessian[i][k] * inverse[k][j] for k in range(n)) - float(i == j) for j in range(n)) for i in range(n)
    )
    if _norm(residual) > 1e-7:
        msg = "curvature inverse residual exceeds numerical admission limit"
        raise ProposalConstructionError(msg)
    return inverse, condition


def _direction(inverse: Matrix, gradient: Vector) -> Vector:
    return tuple(-fsum(a * b for a, b in zip(row, gradient, strict=True)) for row in inverse)


def _trial_point(point: Vector, controls: CurvatureControls, delta: Vector) -> tuple[Vector, Vector]:
    candidate = tuple(x + s * d for x, s, d in zip(point, controls.scales, delta, strict=True))
    actual = tuple((y - x) / s for x, y, s in zip(point, candidate, controls.scales, strict=True))
    for y, achieved, requested in zip(candidate, actual, delta, strict=True):
        if not isfinite(y) or not isclose(achieved, requested, rel_tol=1e-6, abs_tol=controls.mode_tolerance / 16):
            msg = "Newton trial displacement is not representable at the requested resolution"
            raise ProposalConstructionError(msg)
    return candidate, actual


def _next_point(
    objective: _Objective, point: Vector, controls: CurvatureControls, derivatives: _Derivatives, direction: Vector
) -> Vector:
    slope = fsum(a * b for a, b in zip(derivatives.stencil.gradient, direction, strict=True))
    if not isfinite(slope) or slope >= 0:
        msg = "Newton direction is not a finite descent direction"
        raise ProposalConstructionError(msg)
    for attempt in range(24):
        fraction = 0.5**attempt
        candidate, actual = _trial_point(point, controls, tuple(fraction * v for v in direction))
        actual_slope = fsum(a * b for a, b in zip(derivatives.stencil.gradient, actual, strict=True))
        if not isfinite(actual_slope) or actual_slope >= 0:
            continue
        value = objective.evaluate(candidate)
        if value < derivatives.stencil.value and value <= derivatives.stencil.value + 1e-4 * actual_slope:
            return candidate
    msg = "local Newton line search failed"
    raise ProposalConstructionError(msg)


def _source_matrix(matrix: Matrix, scales: Vector, *, covariance: bool) -> Matrix:
    n = len(scales)
    rows: list[list[float]] = [[0.0] * n for _ in scales]
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j]
            factor = scales[i] * scales[j]
            converted = value * factor if covariance else value / factor
            if not isfinite(converted) or (value != 0 and converted == 0):
                msg = "source-coordinate curvature or covariance is not representable"
                raise ProposalConstructionError(msg)
            rows[i][j] = rows[j][i] = converted
    return tuple(tuple(row) for row in rows)


def _construct(target: ExplicitPosteriorTarget, start: Vector, controls: CurvatureControls) -> CurvatureProposalResult:
    objective = _Objective(target)
    point = start
    for iteration in range(controls.max_iterations + 1):
        derivatives = _adaptive(objective, point, controls)
        inverse, condition = _checked_inverse(derivatives, controls)
        direction = _direction(inverse, derivatives.stencil.gradient)
        if (
            _max_abs(derivatives.stencil.gradient) + derivatives.gradient_error <= controls.gradient_tolerance
            and _max_abs(direction) + _norm(inverse) * derivatives.gradient_error <= controls.mode_tolerance
        ):
            proposal = ExplicitNormalProposal(
                coordinates=target.coordinates,
                means=point,
                covariance=_source_matrix(inverse, controls.scales, covariance=True),
                source=f"WP06 local Newton/adaptive inverse Hessian; target={target.source}; {start=}; {controls!r}",
            )
            return CurvatureProposalResult(
                proposal=proposal,
                negative_log_hessian=_source_matrix(derivatives.stencil.hessian, controls.scales, covariance=False),
                start=start,
                controls=controls,
                log_posterior=-derivatives.stencil.value,
                scaled_gradient_norm=_max_abs(derivatives.stencil.gradient),
                scaled_gradient_error=derivatives.gradient_error,
                scaled_curvature_error=derivatives.curvature_error,
                scaled_condition=condition,
                scaled_step=derivatives.step,
                iterations=iteration,
                evaluations=objective.evaluations,
            )
        if iteration < controls.max_iterations:
            point = _next_point(objective, point, controls, derivatives, direction)
    msg = "local Newton iteration limit reached without admitted stationarity"
    raise ProposalConstructionError(msg)


def build_curvature_proposal(
    *, target: ExplicitPosteriorTarget, starting_point: tuple[tuple[str, float], ...], controls: CurvatureControls
) -> CurvatureProposalResult:
    """Construct an explicitly configured local Gaussian; fail closed outside the smooth SPD basin."""
    coordinates = tuple(name for name, _ in starting_point)
    start = tuple(value for _, value in starting_point)
    if (
        coordinates != target.coordinates
        or not 1 <= len(coordinates) <= 8
        or len(controls.scales) != len(coordinates)
        or any(not name.strip() for name in coordinates)
        or any(not isfinite(value) for value in start)
    ):
        msg = "one to eight finite named start coordinates and matching scales are required in target order"
        raise ProposalConstructionError(msg)
    try:
        return _construct(target, start, controls)
    except ProposalConstructionError:
        raise
    except (ArithmeticError, SamplingError, ValueError) as exc:
        msg = "proposal construction arithmetic or sampler admission failed"
        raise ProposalConstructionError(msg) from exc
