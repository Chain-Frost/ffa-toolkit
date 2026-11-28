"""Synthetic order-statistic, report and bootstrap checks independent of native data."""

import csv
import json
from collections.abc import Callable
from math import comb, expm1, fsum, gamma, isclose, log
from pathlib import Path
from random import Random
from statistics import covariance, quantiles, variance
from typing import Never, cast

import numpy as np
import pytest
from scipy.optimize import brentq

from ffa_methods.cli import main
from ffa_methods.deterministic_workflow import EXACT_ASSERTION
from ffa_methods.estimators.gev_lh_calibration import calibrate_fitted_lh
from ffa_methods.estimators.parametric_bootstrap import (
    BootstrapAttempt,
    BootstrapRefit,
    BootstrapRefusalError,
    BootstrapSettings,
    parametric_bootstrap,
)


def test_direct_gev_calibration_matches_rank_weight_covariance() -> None:
    count, shift, shape, seed = 50, 4, 0.0, 7
    actual = calibrate_fitted_lh(count, shape, shift, seed=seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    peaks = -np.log(-np.log(rng.random((200_000, count))))
    peaks.sort(axis=1)

    def choose(n: int, k: int) -> int:
        return comb(n, k) if 0 <= k <= n else 0

    weights = np.array(
        [
            [
                sum(
                    (-1) ** j * comb(r - 1, j) * choose(i, shift + r - j - 1) * choose(count - 1 - i, j)
                    for j in range(r)
                )
                / (r * comb(count, shift + r))
                for r in (2, 3, 4)
            ]
            for i in range(count)
        ]
    )
    moments = peaks @ weights
    skew, kurtosis = ((moments[:, order] / moments[:, 0]).tolist() for order in (1, 2))
    conditional = variance(kurtosis) - covariance(skew, kurtosis) ** 2 / variance(skew)
    assert isclose(actual.covariance.conditional_variance, conditional, rel_tol=1e-12, abs_tol=0)
    assert actual.variance_lower < conditional < actual.variance_upper


def test_bootstrap_percentiles_and_rank_precision() -> None:
    settings = BootstrapSettings(200, 42, 0.8, 0.0, 1.0)
    result = parametric_bootstrap(
        lambda u: u,
        lambda record: lambda _: (record[0] + record[1]) / 2,
        count=2,
        probabilities=(0.9,),
        settings=settings,
    )
    rng = Random(42)  # noqa: S311  # Replay the declared Monte Carlo stream.
    samples = sorted((rng.random() + rng.random()) / 2 for _ in range(200))
    lo, hi = result.endpoints[0]
    assert isclose(lo.estimate, 0.1 * samples[19] + 0.9 * samples[20], rel_tol=1e-14)
    assert isclose(hi.estimate, 0.9 * samples[179] + 0.1 * samples[180], rel_tol=1e-14)
    for endpoint, p in ((lo, 0.1), (hi, 0.9)):
        cdf = 0.0
        lower, upper = None, None
        for k in range(200):
            cdf += comb(200, k) * p**k * (1 - p) ** (200 - k)
            if lower is None and cdf >= 0.0125:
                lower = samples[k - 1]
            if cdf >= 0.9875:
                upper = samples[k]
                break
        assert (endpoint.mc_lower, endpoint.mc_upper) == (lower, upper)
    assert result.failures == ()


def test_bootstrap_gates_refuse_failed_and_imprecise_results() -> None:
    def no_selection(_: tuple[float, ...]) -> Never:
        msg = "no candidate passed"
        raise ValueError(msg)

    with pytest.raises(BootstrapRefusalError, match=r"4/4.*no candidate passed") as refusal:
        parametric_bootstrap(
            lambda u: u, no_selection, count=2, probabilities=(0.9,), settings=BootstrapSettings(4, 42, 0.8, 0.0, 1.0)
        )
    assert refusal.value.code == "failure_fraction"
    assert refusal.value.attempts == (BootstrapAttempt(None, "refit", "no candidate passed"),) * 4
    with pytest.raises(BootstrapRefusalError, match="endpoint Monte Carlo precision failed") as refusal:
        parametric_bootstrap(
            lambda u: u,
            lambda record: lambda _: record[0],
            count=2,
            probabilities=(0.9,),
            settings=BootstrapSettings(200, 42, 0.8, 0.0, 1e-9),
        )
    assert refusal.value.code == "endpoint_precision"
    assert refusal.value.attempts == (BootstrapAttempt(None),) * 200


def test_bootstrap_attempts_retain_reselection_and_use_successful_refits_only() -> None:
    def refit(record: tuple[float, ...]) -> BootstrapRefit:
        first = record[0]
        if first < 0.1:
            msg = "no candidate passed"
            raise ValueError(msg)
        return BootstrapRefit(lambda _: first if first > 0.2 else float("nan"), 4 if first < 0.5 else 1)

    result = parametric_bootstrap(
        lambda u: u, refit, count=2, probabilities=(0.9,), settings=BootstrapSettings(200, 42, 0.8, 0.5, 1.0)
    )
    rng = Random(42)  # noqa: S311  # Replay the declared Monte Carlo stream.
    first_draws: list[float] = []
    for _ in range(200):
        first_draws.append(rng.random())
        rng.random()
    ordered = sorted(first for first in first_draws if first > 0.2)
    deciles = quantiles(ordered, n=10, method="inclusive")
    assert isclose(result.endpoints[0][0].estimate, deciles[0], rel_tol=1e-14)
    assert isclose(result.endpoints[0][1].estimate, deciles[-1], rel_tol=1e-14)
    assert result.attempts == tuple(
        BootstrapAttempt(None, "refit", "no candidate passed")
        if first < 0.1
        else BootstrapAttempt(4, "quantiles", "nonfinite replicate quantiles")
        if first <= 0.2
        else BootstrapAttempt(4 if first < 0.5 else 1)
        for first in first_draws
    )


def _gev_reference_fit(values: list[float], shift: int) -> Callable[[float], float]:
    """Fit from direct ranks and the binomial expansion of order-statistic CDFs."""
    ordered = sorted(values)
    count = len(ordered)

    def choose(n: int, k: int) -> int:
        return comb(n, k) if 0 <= k <= n else 0

    sample = [
        fsum(
            value
            * sum(
                (-1) ** j * comb(r - 1, j) * choose(i, shift + r - j - 1) * choose(count - 1 - i, j) for j in range(r)
            )
            for i, value in enumerate(ordered)
        )
        / (r * comb(count, shift + r))
        for r in (1, 2, 3)
    ]

    def moments(shape: float) -> tuple[float, ...]:
        maxima = {
            size: (1 - gamma(1 + shape) * size**-shape) / shape if shape else 0.5772156649015329 + log(size)
            for size in range(1, shift + 4)
        }

        def expected_order(rank: int, size: int) -> float:
            return fsum(
                comb(size, j) * (-1) ** k * comb(size - j, k) * maxima[j + k]
                for j in range(rank, size + 1)
                for k in range(size - j + 1)
            )

        return tuple(
            fsum((-1) ** j * comb(r - 1, j) * expected_order(shift + r - j, shift + r) for j in range(r)) / r
            for r in (1, 2, 3)
        )

    def skew_error(shape: float) -> float:
        _, second, third = moments(shape)
        return third / second - sample[2] / sample[1]

    shape = float(brentq(skew_error, -0.9, 2))
    first, second, _ = moments(shape)
    scale = sample[1] / second
    location = sample[0] - scale * first
    return lambda f: location - scale * expm1(shape * log(-log(f))) / shape


def _gumbel_reference_fit(values: list[float]) -> Callable[[float], float]:
    count = len(values)
    scale = fsum((2 * i - count + 1) * a for i, a in enumerate(sorted(values))) / (count * (count - 1) * log(2))
    location = fsum(values) / count - scale * 0.5772156649015329
    return lambda f: location - scale * log(-log(f))


@pytest.mark.parametrize(
    ("schema", "cutoff"),
    [
        ("ffa-gev-selected-h/1", 1.96),
        ("ffa-gev-selected-h/1", 1e-6),
        ("ffa-ordinary-lmoments/1", None),
        ("ffa-gev-fixed-h/1", None),
    ],
)
def test_cli_candidate_and_bootstrap_reports(tmp_path: Path, schema: str, cutoff: float | None) -> None:
    raw_ams = tmp_path / "annual.csv"
    records = [20 - log(-log((i - 0.35) / 50.3)) for i in range(1, 51)]
    raw_ams.write_text(
        "Year,AMS\n" + "\n".join(f"{1950 + i},{record}" for i, record in enumerate(records, start=1)),
        encoding="utf-8",
    )
    path = tmp_path / "settings.json"
    settings: dict[str, object] = {
        "schema_version": schema,
        "discharge_unit": "m3/s",
        "year_definition": "calendar",
        "source_classification": EXACT_ASSERTION,
        "non_exceedance_probabilities": [0.9],
    }
    if schema == "ffa-gev-selected-h/1":
        settings.update(estimator="gev_lh_selected_h", critical_z=cutoff, calibration_seed=7)
    else:
        if schema == "ffa-gev-fixed-h/1":
            settings.update(estimator="gev_lh_fixed_h", lh_shift=1)
        else:
            settings["estimator"] = "gumbel_lmoments"
        settings["bootstrap"] = {
            "replications": 200,
            "seed": 42,
            "confidence": 0.8,
            "maximum_failure_fraction": 0.0,
            "endpoint_tolerance": 1.0,
        }
    path.write_text(json.dumps(settings), encoding="utf-8")
    target = tmp_path / "report"
    args = ["fit", "--ams", str(raw_ams), "--settings", str(path), "--output", str(target)]
    # Preserve the no-selected-fit numerical report; it cannot produce a frequency curve.
    if cutoff != 1e-6:
        args.append("--frequency-report")
    assert main(args) == 0
    report = cast(dict[str, object], json.loads((target / "results.json").read_text(encoding="utf-8")))
    values = cast(list[dict[str, object]], report["results"])
    if schema == "ffa-gev-selected-h/1":
        candidate_support = cast(dict[str, object], report["candidate_gev_support"])
        assert set(candidate_support) == set("01234")
        with (target / "results.csv").open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        assert json.loads(row["candidate_gev_support_json"]) == candidate_support
        expected_h = 0 if cutoff == 1.96 else None
        assert cast(dict[str, object], report["settings"])["lh_shift"] == expected_h
        assert len(values) == int(expected_h is not None)
        assert bool(row["meaning"]) == (expected_h is not None)
    else:
        assert [value["probability_limit_side"] for value in values] == [None, "lower", "upper"]
        assert values[1]["meaning"] == "quantile_parametric_bootstrap_percentile_limit"
        rng = Random(42)  # noqa: S311  # Replay the declared Monte Carlo stream.

        def reference(sample: list[float]) -> Callable[[float], float]:
            return _gev_reference_fit(sample, 1) if schema == "ffa-gev-fixed-h/1" else _gumbel_reference_fit(sample)

        generate = reference(records)
        draws = [reference([generate(rng.random()) for _ in records])(0.9) for _ in range(200)]
        deciles = quantiles(draws, n=10, method="inclusive")
        for result, expected in zip(values, (generate(0.9), deciles[0], deciles[-1]), strict=True):
            measured = result["value"]
            assert isinstance(measured, float)
            assert isclose(measured, expected, abs_tol=1e-9, rel_tol=0)

    if cutoff != 1e-6:
        with (target / "frequency-coordinates.csv").open(encoding="utf-8", newline="") as stream:
            plotted = [row for row in csv.DictReader(stream) if row["series"] != "observed"]
        assert len(plotted) == len(values)
        for marker in plotted:
            row = values[int(marker["result_row"]) - 1]
            assert float(marker["discharge"]) == row["value"]
            assert float(marker["F"]) == row["input_non_exceedance_probability"]
            assert float(marker["AEP"]) == row["input_annual_exceedance_probability"]
            assert marker["meaning"] == row["meaning"]
            if row["probability_limit_side"] is not None:
                assert marker["probability_limit_side"] == row["probability_limit_side"]
                assert float(marker["reported_probability_level"]) == row["reported_probability_level"]
                assert marker["probability_limit_construction"] == row["probability_limit_construction"]
        svg = (target / "frequency.svg").read_text(encoding="utf-8")
        assert ('data-interval="parametric_bootstrap"' in svg) == (schema != "ffa-gev-selected-h/1")
