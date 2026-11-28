"""Joint Gaussian priors in declared legacy/source parameter coordinates.

Supplied/regional priors can be evaluated only once coordinates and covariance
are established by evidence. The observed Flike no-prior/default density is not
known; those selections intentionally fail closed rather than inventing one.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Context, Decimal, localcontext
from enum import StrEnum
from fractions import Fraction
from math import fsum, isclose, isfinite, log, log1p, pi, sqrt

from ffa_methods.contracts import ParameterCoordinate


class PriorEvidenceError(ValueError):
    """A purported legacy prior has inadequate evidence or invalid parameters."""


class LegacyPriorMode(StrEnum):
    """The selection is distinct from an independently chosen statistical prior."""

    SUPPLIED = "supplied"
    REGIONAL = "regional"
    NO_PRIOR = "no_prior"
    DEFAULT = "default"


class CoordinateTransform(StrEnum):
    """Map evaluator coordinates to the original prior coordinates."""

    IDENTITY = "identity"
    RAW_POSITIVE_TO_LOG = "raw_positive_to_log"
    PROBABILITY_TO_LOGIT = "probability_to_logit"


@dataclass(frozen=True, slots=True, kw_only=True)
class JointGaussianPrior:
    """Full joint Normal density with covariance in the **source** coordinates.

    `source` and `coordinates` are required: never parse unknown .fld numeric
    blocks as a prior merely because they have the right number of values.
    """

    coordinates: tuple[ParameterCoordinate | str, ...]
    means: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    source: str
    mode: LegacyPriorMode
    site_in_regional_estimate: bool | None = None
    _conditional_slope: Fraction | None = field(init=False, default=None, repr=False, compare=False)
    _factor: tuple[tuple[float, ...], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Enforce SPD covariance and provenance at construction, not in a solver."""
        n = len(self.coordinates)
        if self.mode not in (LegacyPriorMode.SUPPLIED, LegacyPriorMode.REGIONAL):
            msg = "Gaussian prior requires supplied or regional mode"
            raise PriorEvidenceError(msg)
        if not self.source.strip() or not n or len(self.means) != n or len(self.covariance) != n:
            msg = "prior requires provenance, coordinates, mean vector and square covariance"
            raise PriorEvidenceError(msg)
        if len(set(self.coordinates)) != n:
            msg = "prior coordinates must be unique"
            raise PriorEvidenceError(msg)
        if any(not str(coordinate).strip() for coordinate in self.coordinates):
            msg = "empty coordinate name"
            raise PriorEvidenceError(msg)
        if not all(isfinite(value) for value in self.means):
            msg = "prior means must be finite"
            raise PriorEvidenceError(msg)
        factor = _cholesky(self.covariance)
        object.__setattr__(self, "_factor", factor)
        if n == 2 and self.covariance[1][0] != 0:
            object.__setattr__(
                self, "_conditional_slope", Fraction(self.covariance[1][0]) / Fraction(self.covariance[0][0])
            )

    def logpdf(self, values: Sequence[float]) -> float:
        """Log density in *exactly* the source's vector order and coordinates."""
        n = len(self.means)
        if len(values) != n or not all(isfinite(value) for value in values):
            msg = "prior input must be a finite vector of the declared dimension"
            raise PriorEvidenceError(msg)
        factor = self._factor
        residuals: list[float] = []
        if self._conditional_slope is not None:
            # A rounded regression coefficient or a rounded first residual can
            # consume the conditional variation in an almost dependent pair.
            first = Fraction(values[0]) - Fraction(self.means[0])
            second = Fraction(values[1]) - Fraction(self.means[1]) - self._conditional_slope * first
            try:
                residuals.extend((float(first / Fraction(factor[0][0])), float(second / Fraction(factor[1][1]))))
            except OverflowError:
                # Its squared standardized residual is also outside binary64.
                return -float("inf")
        else:
            for i in range(n):
                solved = values[i] - self.means[i] - fsum(factor[i][j] * residuals[j] for j in range(i))
                residuals.append(solved / factor[i][i])
        quadratic = fsum(value * value for value in residuals)
        return -0.5 * (n * log(2.0 * pi) + 2.0 * fsum(log(factor[i][i]) for i in range(n)) + quadratic)

    def logpdf_in_evaluator_coordinates(
        self, values: Sequence[float], *, transformations: tuple[CoordinateTransform, ...]
    ) -> float:
        """Evaluate the *transformed density*, including full Jacobian.

        Example: for a prior on eta=log(sd), evaluating at raw sd requires
        log(pi(log(sd))) - log(sd), not simply pi(log(sd)).
        Likewise z=logit(p) contributes -log(p)-log(1-p).
        """
        if len(transformations) != len(self.means) or len(values) != len(self.means):
            msg = "one explicit transform is required for each source coordinate"
            raise PriorEvidenceError(msg)
        source_values: list[float] = []
        log_jacobians: list[float] = []
        for value, transform in zip(values, transformations, strict=True):
            if not isfinite(value):
                msg = "transformed coordinate must be finite"
                raise PriorEvidenceError(msg)
            if transform == CoordinateTransform.IDENTITY:
                source_values.append(value)
                log_jacobians.append(0.0)
            elif transform == CoordinateTransform.RAW_POSITIVE_TO_LOG:
                if value <= 0:
                    msg = "raw standard deviation/scale must be positive"
                    raise PriorEvidenceError(msg)
                source_values.append(log(value))
                log_jacobians.append(-log(value))
            elif transform == CoordinateTransform.PROBABILITY_TO_LOGIT:
                if value <= 0 or value >= 1:
                    msg = "logit coordinate requires a probability strictly inside (0, 1)"
                    raise PriorEvidenceError(msg)
                source_values.append(log(value) - log1p(-value))
                log_jacobians.append(-log(value) - log1p(-value))
            else:
                msg = "unsupported prior coordinate transformation"
                raise PriorEvidenceError(msg)
        return self.logpdf(source_values) + fsum(log_jacobians)


def _bivariate_cholesky(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    a, b, d = Fraction(matrix[0][0]), Fraction(matrix[1][0]), Fraction(matrix[1][1])
    determinant = a * d - b * b
    if a <= 0 or determinant <= 0:
        msg = "covariance must be positive definite"
        raise PriorEvidenceError(msg)
    conditional_variance = determinant / a
    # The variance can underflow binary64 even when its SD is representable.
    with localcontext(Context(prec=80)):
        sd = float((Decimal(conditional_variance.numerator) / Decimal(conditional_variance.denominator)).sqrt())
    if not sd:
        msg = "conditional prior standard deviation is not representable"
        raise PriorEvidenceError(msg)
    first_sd = sqrt(matrix[0][0])
    return ((first_sd, 0.0), (matrix[1][0] / first_sd, sd))


def _cholesky(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """Dependency-free Cholesky: retain *all* covariance off-diagonal entries."""
    n = len(matrix)
    if not n or any(len(row) != n or any(not isfinite(value) for value in row) for row in matrix):
        msg = "covariance must be a finite square matrix"
        raise PriorEvidenceError(msg)
    if any(not isclose(matrix[i][j], matrix[j][i], rel_tol=1e-12, abs_tol=1e-14) for i in range(n) for j in range(i)):
        msg = "covariance must be symmetric"
        raise PriorEvidenceError(msg)
    if n == 2 and matrix[1][0] != 0:
        return _bivariate_cholesky(matrix)
    rows: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            remainder = matrix[i][j] - fsum(rows[i][k] * rows[j][k] for k in range(j))
            if i == j:
                if remainder <= 0:
                    msg = "covariance must be positive definite"
                    raise PriorEvidenceError(msg)
                rows[i][j] = sqrt(remainder)
            else:
                rows[i][j] = remainder / rows[j][j]
    return tuple(tuple(row) for row in rows)


@dataclass(frozen=True, slots=True, kw_only=True)
class LegacyPriorSelection:
    """Require established source evidence before assigning a legacy prior."""

    mode: LegacyPriorMode
    supplied: JointGaussianPrior | None = None

    def __post_init__(self) -> None:
        """A regional/supplied prior cannot be silently swapped or omitted."""
        if self.mode in (LegacyPriorMode.SUPPLIED, LegacyPriorMode.REGIONAL):
            if self.supplied is None or self.supplied.mode != self.mode:
                msg = "prior mode and supplied covariance evidence must match"
                raise PriorEvidenceError(msg)
        elif self.supplied is not None:
            msg = "no-prior/default settings cannot silently use a supplied prior"
            raise PriorEvidenceError(msg)

    def logpdf(self, values: Sequence[float]) -> float:
        """Do not invent a density for the legacy unverified defaults."""
        if self.supplied is None:
            msg = "legacy no-prior/default density and bounds remain unverified for Flike 5.0.306.0"
            raise PriorEvidenceError(msg)
        return self.supplied.logpdf(values)
