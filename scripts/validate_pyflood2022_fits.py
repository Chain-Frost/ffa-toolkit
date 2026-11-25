"""Read-only all-eligible-station deterministic stress matrix; not MCMC/native replay."""

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from math import comb, erf, exp, expm1, fsum, inf, isfinite, isnan, lgamma, log, log1p, pi, sqrt
from pathlib import Path
from statistics import NormalDist
from time import perf_counter
from typing import TypeAlias

from scipy.special import beta, betainc
from scipy.stats import genextreme, pearson3

from ffa_methods.distributions import GEV, GeneralizedPareto, Gumbel, Lognormal, LogPearsonIII
from ffa_methods.estimators.gev_lh import fit_gev_lh
from ffa_methods.estimators.ordinary_lmoments import (
    fit_gp_lmoments,
    fit_gumbel_lmoments,
    fit_lognormal_lmoments,
    fit_lp3_lmoments,
)
from scripts.audit_pyflood2022 import (
    ASSETS,
    REVISION,
    FloodTable,
    archive_csv,
    flood_table,
    group_series,
    pair_audit,
    parse_table,
    verify_asset,
)

METHODS = (
    "gev-h0",
    "gev-h1",
    "gev-h2",
    "gev-h3",
    "gev-h4",
    "gumbel",
    "lognormal",
    "lp3",
    "gp-free",
    "gp-fixed-zero",
)
LEVELS = (0.5, 0.9, 0.99, 0.999)
UNIT_FACTOR = 1024.0  # Exactly representable change of physical units.
EULER = 0.5772156649015329
Kernel: TypeAlias = GEV | Gumbel | Lognormal | LogPearsonIII | GeneralizedPareto
Case: TypeAlias = tuple[str, tuple[float, ...], int]


def _lh_from_maxima(b: tuple[float, ...], h: int) -> tuple[float, float, float, float]:
    """Wang (1997) Appendix A2-A5, with B_j the expected maximum of j+1 values."""
    a, c, d, e = b
    return (
        a,
        (h + 2) / 2 * fsum((c, -a)),
        (h + 3) / 6 * fsum(((h + 4) * d, -2 * (h + 3) * c, (h + 2) * a)),
        (h + 4)
        / 24
        * fsum(
            (
                (h + 6) * (h + 5) * e,
                -3 * (h + 5) * (h + 4) * d,
                3 * (h + 4) * (h + 3) * c,
                -(h + 3) * (h + 2) * a,
            )
        ),
    )


def independent_lh(values: tuple[float, ...], h: int) -> tuple[float, float, float, float]:
    """Independent normalized ascending-rank PWM evaluation, not adjacent-gap sums."""
    ordered = sorted(values)
    n = len(ordered)
    if not 0 <= h <= 4 or n < h + 4:
        msg = "need H=0..4 and at least H+4 observations"
        raise ValueError(msg)
    maxima = tuple(
        fsum(comb(i, j) / comb(n, j + 1) * q for i, q in enumerate(ordered) if i >= j) for j in range(h, h + 4)
    )
    return _lh_from_maxima(maxima, h)


def _log_gamma_one_plus(k: float) -> float:
    # Taylor series of log Gamma(1+k), independently avoiding lgamma cancellation.
    if abs(k) < 1e-5:
        return -EULER * k + pi**2 / 12 * k**2 - 1.2020569031595942 / 3 * k**3
    return lgamma(1 + k)


def gev_population(kernel: GEV, h: int) -> tuple[float, ...]:
    """Expected maxima from GEV(max of j) and the independent PWM transform."""
    k = -kernel.xi
    scale = exp(kernel.log_scale)
    maxima = tuple(
        EULER + log(j) if k == 0 else -expm1(_log_gamma_one_plus(k) - k * log(j)) / k for j in range(h + 1, h + 5)
    )
    first, second, third, fourth = _lh_from_maxima(maxima, h)
    return kernel.location + scale * first, scale * second, scale * third, scale * fourth


def ordinary_population(kernel: Gumbel | Lognormal | LogPearsonIII | GeneralizedPareto) -> tuple[float, ...]:
    """Independent closed-form population L-moments in each fitting coordinate."""
    if isinstance(kernel, Gumbel):
        scale = exp(kernel.log_scale)
        return kernel.location + EULER * scale, scale * log(2)
    if isinstance(kernel, Lognormal):
        sd = exp(kernel.log.natural_log_sd)
        mean = kernel.log.reference_discharge * exp(kernel.log.natural_mean + sd**2 / 2)
        return mean, mean * erf(sd / 2)
    if isinstance(kernel, LogPearsonIII):
        mean = kernel.log.natural_mean
        sd = exp(kernel.log.natural_log_sd)
        if kernel.skew == 0:
            return mean, sd / sqrt(pi), 0.0
        shape = (2 / kernel.skew) ** 2
        second = sd / (sqrt(shape) * float(beta(shape, 0.5)))
        # Complementary beta form of Pearson III L-skew, signed by reflection.
        tau = 3 - 6 * float(betainc(2 * shape, shape, 2 / 3))
        return mean, second, second * tau * (1 if kernel.skew > 0 else -1)
    scale = exp(kernel.log_scale)
    first = kernel.location + scale / (1 - kernel.xi)
    second = scale / ((1 - kernel.xi) * (2 - kernel.xi))
    return first, second, second * (1 + kernel.xi) / (3 - kernel.xi)


def independent_quantile(kernel: Kernel, p: float) -> float:
    """Direct quantile equations or SciPy reference distribution in source units."""
    if isinstance(kernel, GEV):
        return float(genextreme.ppf(p, -kernel.xi, loc=kernel.location, scale=exp(kernel.log_scale)))
    if isinstance(kernel, Gumbel):
        return kernel.location - exp(kernel.log_scale) * log(-log(p))
    if isinstance(kernel, GeneralizedPareto):
        tail = -log1p(-p)
        reduced = tail if kernel.xi == 0 else expm1(kernel.xi * tail) / kernel.xi
        return kernel.location + exp(kernel.log_scale) * reduced
    mean, sd = kernel.log.natural_mean, exp(kernel.log.natural_log_sd)
    if isinstance(kernel, Lognormal) or kernel.skew == 0:
        log_flow = NormalDist(mu=mean, sigma=sd).inv_cdf(p)
    else:
        log_flow = float(pearson3.ppf(p, kernel.skew, loc=mean, scale=sd))
    return kernel.log.reference_discharge * exp(log_flow)


@dataclass(frozen=True, slots=True)
class Fitted:
    kernel: Kernel
    quantile: Callable[[float], float]
    sample: tuple[float, ...]
    population: tuple[float, ...] | None = None
    residual: float | None = None


def fit(method: str, values: tuple[float, ...], *, reference: float) -> Fitted:
    """Invoke existing estimators; no extra fits, priors, or parameter repair."""
    if method.startswith("gev-h"):
        gev = fit_gev_lh(values, shift=int(method[-1]))
        kernel = GEV(gev.location, log(gev.scale), -gev.shape)
        return Fitted(kernel, gev.quantile, gev.sample_moments, gev.fitted_moments, gev.lh_kurtosis_residual)
    if method == "gumbel":
        gumbel = fit_gumbel_lmoments(values)
        return Fitted(gumbel.distribution, gumbel.quantile, gumbel.sample_moments)
    if method == "lognormal":
        ln = fit_lognormal_lmoments(values, reference_discharge=reference)
        return Fitted(ln.distribution, ln.quantile, ln.sample_moments)
    if method == "lp3":
        lp3 = fit_lp3_lmoments(values, reference_discharge=reference)
        return Fitted(lp3.distribution, lp3.quantile, lp3.sample_moments)
    if method in ("gp-free", "gp-fixed-zero"):
        gp = fit_gp_lmoments(values, location=0.0 if method == "gp-fixed-zero" else None)
        return Fitted(gp.distribution, gp.quantile, gp.sample_moments)
    msg = "unknown stress-matrix method"
    raise ValueError(msg)


def _ratio(actual: float, expected: float, *, absolute: float, relative: float) -> float:
    if not isfinite(actual) or not isfinite(expected):
        return inf
    return abs(actual - expected) / max(absolute, relative * abs(expected))


def _moment_ratio(actual: tuple[float, ...], expected: tuple[float, ...], scale: float) -> float:
    return max(_ratio(a, b, absolute=scale * 1e-10, relative=1e-8) for a, b in zip(actual, expected, strict=True))


def moment_checks(method: str, values: tuple[float, ...], fitted: Fitted) -> dict[str, float]:
    h = int(method[-1]) if method.startswith("gev-h") else 0
    coordinates = tuple(log(q) for q in values) if method == "lp3" else values
    scale = max(abs(q) for q in coordinates)
    expected = independent_lh(coordinates, h)
    results = {"sample_moments": _moment_ratio(fitted.sample, expected[: len(fitted.sample)], scale)}
    if isinstance(fitted.kernel, GEV):
        population = gev_population(fitted.kernel, h)
        results["matched_population"] = _moment_ratio(population[:3], expected[:3], scale)
        if fitted.population is None or fitted.residual is None:
            msg = "GEV fitted moment and raw residual diagnostics missing"
            raise ValueError(msg)
        results["population_diagnostics"] = _moment_ratio(fitted.population, population, scale)
        residual = expected[3] / expected[1] - population[3] / population[1]
        results["fourth_ratio_residual"] = _ratio(fitted.residual, residual, absolute=1e-8, relative=1e-8)
    else:
        population = ordinary_population(fitted.kernel)
        count = len(fitted.sample)
        results["matched_population"] = _moment_ratio(population[:count], expected[:count], scale)
    return results


@dataclass(slots=True)
class CheckEvidence:
    """Retain completed ratios and errors independently, including partial failures."""

    ratios: dict[str, float] = field(default_factory=dict[str, float])
    exceptions: dict[str, str] = field(default_factory=dict[str, str])

    def record(self, name: str, ratio: float) -> None:
        self.ratios[name] = max(self.ratios.get(name, 0.0), ratio)

    def failed(self, name: str, exc: Exception) -> None:
        self.exceptions[name] = str(exc)


def _reference_check(evidence: CheckEvidence, fitted: Fitted, q: float, p: float, scale: float) -> None:
    try:
        reference = independent_quantile(fitted.kernel, p)
        evidence.record("reference_quantile", _ratio(q, reference, absolute=scale * 1e-12, relative=1e-8))
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        evidence.failed(f"reference_quantile:{p}", exc)


def _probability_checks(evidence: CheckEvidence, fitted: Fitted, q: float, p: float) -> None:
    try:
        evidence.record("cdf_and_survival", _ratio(fitted.kernel.cdf(q), p, absolute=1e-11, relative=1e-9))
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        evidence.failed(f"cdf:{p}", exc)
    try:
        evidence.record("cdf_and_survival", _ratio(fitted.kernel.sf(q), 1 - p, absolute=1e-11, relative=1e-9))
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        evidence.failed(f"survival:{p}", exc)


def _unit_check(evidence: CheckEvidence, scaled: Fitted, q: float, p: float, scale: float) -> None:
    try:
        evidence.record(
            "unit_scaling", _ratio(scaled.quantile(p) / UNIT_FACTOR, q, absolute=scale * 1e-10, relative=1e-8)
        )
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        evidence.failed(f"unit_quantile:{p}", exc)


def quantile_checks(values: tuple[float, ...], fitted: Fitted, scaled: Fitted | None, evidence: CheckEvidence) -> None:
    """Keep earlier levels and other checks even when a particular evaluation fails."""
    scale = max(values)
    for p in LEVELS:
        try:
            q = fitted.quantile(p)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            evidence.failed(f"base_quantile:{p}", exc)
            continue
        _reference_check(evidence, fitted, q, p, scale)
        _probability_checks(evidence, fitted, q, p)
        if scaled is not None:
            _unit_check(evidence, scaled, q, p, scale)


def cohort(table: FloodTable, sites: set[str]) -> tuple[list[Case], dict[str, list[str]]]:
    """All eligible sites, sorted by source ID; count each exclusion once."""
    accepted: list[Case] = []
    excluded: dict[str, list[str]] = {}
    for site, rows in sorted(group_series(table).items()):
        reason = _exclusion(site, rows, table, sites)
        if reason is not None:
            excluded.setdefault(reason, []).append(site)
            continue
        values = tuple(float(value) for _, value in sorted(rows) if value is not None)
        gaps = max(year for year, _ in rows) - min(year for year, _ in rows) + 1 - len(rows)
        if min(values) == max(values):
            excluded.setdefault("constant", []).append(site)
        else:
            accepted.append((site, values, gaps))
    return accepted, excluded


def _exclusion(site: str, rows: list[tuple[int, Decimal | None]], table: FloodTable, sites: set[str]) -> str | None:
    if site not in sites:
        return "missing_metadata"
    if site in table.duplicate_sites:
        return "duplicate_water_year"
    if len(rows) < 30:
        return "fewer_than_30_records"
    for _, value in rows:
        if value is None or not value.is_finite() or value <= 0 or not 0 < float(value) < inf:
            return "invalid_peak"
    return None


@dataclass(frozen=True, slots=True)
class Outcome:
    site: str
    status: str
    reason: str
    errors: dict[str, float]
    outside_support: bool | None = None
    check_exceptions: dict[str, str] = field(default_factory=dict[str, str])


def _support_check(fitted: Fitted, values: tuple[float, ...], evidence: CheckEvidence) -> bool | None:
    try:
        lower, upper = fitted.kernel.support()
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        evidence.failed("support", exc)
        return None
    if isnan(lower) or isnan(upper) or lower > upper:
        evidence.failed("support", ValueError("invalid fitted support endpoints"))
        return None
    return min(values) < lower or max(values) > upper


def _scaled_fit(method: str, values: tuple[float, ...], evidence: CheckEvidence) -> Fitted | None:
    try:
        return fit(method, tuple(q * UNIT_FACTOR for q in values), reference=UNIT_FACTOR)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        evidence.failed("scaled_refit", exc)
        return None


def check(method: str, case: Case) -> Outcome:
    site, values, _ = case
    try:
        fitted = fit(method, values, reference=1.0)
    except ValueError as exc:
        return Outcome(site, "fit_rejected", str(exc), {})
    except (ArithmeticError, RuntimeError) as exc:
        return Outcome(site, "fit_error", str(exc), {})
    evidence = CheckEvidence()
    outside = _support_check(fitted, values, evidence)
    try:
        evidence.ratios.update(moment_checks(method, values, fitted))
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        evidence.failed("moments", exc)
    scaled = _scaled_fit(method, values, evidence)
    quantile_checks(values, fitted, scaled, evidence)
    failures = [name for name, ratio in evidence.ratios.items() if not isfinite(ratio) or ratio > 1]
    status = "check_error" if evidence.exceptions else "mismatch" if failures else "pass"
    reason = ",".join([*failures, *evidence.exceptions])
    return Outcome(site, status, reason, evidence.ratios, outside, evidence.exceptions)


def report_method(method: str, cases: list[Case]) -> dict[str, object]:
    start = perf_counter()
    results = [check(method, case) for case in cases]
    counts: dict[str, int] = {}
    maxima: dict[str, float] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
        for name, ratio in result.errors.items():
            maxima[name] = max(maxima.get(name, 0.0), ratio)
    return {
        "method": method,
        "attempted": len(results),
        "outcomes": counts,
        "elapsed_seconds": perf_counter() - start,
        "max_tolerance_ratio": {name: ratio if isfinite(ratio) else None for name, ratio in maxima.items()},
        "outside_fitted_support_sites": [result.site for result in results if result.outside_support],
        "support_unresolved_sites": [
            result.site
            for result in results
            if result.outside_support is None and result.status not in ("fit_rejected", "fit_error")
        ],
        "nonpasses": [
            {
                "site": result.site,
                "status": result.status,
                "reason": result.reason,
                "completed_tolerance_ratios": {
                    name: ratio if isfinite(ratio) else None for name, ratio in result.errors.items()
                },
                "check_exceptions": result.check_exceptions,
            }
            for result in results
            if result.status != "pass"
        ],
    }


def run(directory: Path) -> None:
    assets = {name: verify_asset(directory / name, size=size, blob=blob) for name, (size, blob) in ASSETS.items()}
    header, records, _ = parse_table(assets["sites_info.csv"])
    index = header.index("STATIONID")
    sites = {row[index] for row in records}
    if not sites or "" in sites or len(sites) != len(records):
        msg = "invalid metadata station IDs"
        raise ValueError(msg)
    first = flood_table(archive_csv(assets["flood_data_v5.zip"])[1])
    second = flood_table(archive_csv(assets["flood_data_censored_v5.zip"])[1])
    if not pair_audit(first, second)["exact_unique_raw_pair"]:
        msg = "pinned archives do not have identical unique raw observations"
        raise ValueError(msg)
    cases, excluded = cohort(first, sites)
    if not cases:
        msg = "no eligible station records"
        raise ValueError(msg)
    print(
        json.dumps(
            {
                "scope": "independent deterministic numerical stress; no MCMC or native qualification",
                "revision": REVISION,
                "eligible": len(cases),
                "eligible_sites": [site for site, _, _ in cases],
                "eligible_peaks": sum(len(values) for _, values, _ in cases),
                "eligible_gap_sites": sum(gaps > 0 for _, _, gaps in cases),
                "excluded": excluded,
                "levels": LEVELS,
                "unit_factor": UNIT_FACTOR,
            },
            allow_nan=False,
        ),
        flush=True,
    )
    for method in METHODS:
        print(json.dumps(report_method(method, cases), allow_nan=False), flush=True)
    # Completion is not universal scientific acceptance: outcomes are explicitly retained.
    print("Stress matrix completed; inspect all outcome counts and nonpasses.", flush=True)


def main() -> int:
    if len(sys.argv) != 2:
        msg = "usage: python -m scripts.validate_pyflood2022_fits PINNED_ASSET_DIRECTORY"
        raise SystemExit(msg)
    run(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
