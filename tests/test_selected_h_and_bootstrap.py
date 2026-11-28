"""Synthetic order-statistic, report and bootstrap checks independent of native data."""

import csv
import json
from math import comb, isclose, log
from pathlib import Path
from random import Random
from statistics import covariance, variance
from typing import Never, cast

import numpy as np
import pytest

from ffa_methods.cli import main
from ffa_methods.deterministic_workflow import EXACT_ASSERTION
from ffa_methods.estimators.gev_lh_calibration import calibrate_fitted_lh
from ffa_methods.estimators.parametric_bootstrap import BootstrapSettings, parametric_bootstrap


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
    assert lo.mc_lower < lo.estimate < lo.mc_upper
    assert hi.mc_lower < hi.estimate < hi.mc_upper
    assert result.failures == ()


def test_bootstrap_gates_refuse_failed_and_imprecise_results() -> None:
    def no_selection(_: tuple[float, ...]) -> Never:
        msg = "no candidate passed"
        raise ValueError(msg)

    with pytest.raises(ValueError, match=r"4/4.*no candidate passed"):
        parametric_bootstrap(
            lambda u: u, no_selection, count=2, probabilities=(0.9,), settings=BootstrapSettings(4, 42, 0.8, 0.0, 1.0)
        )
    with pytest.raises(ValueError, match="endpoint Monte Carlo precision failed"):
        parametric_bootstrap(
            lambda u: u,
            lambda record: lambda _: record[0],
            count=2,
            probabilities=(0.9,),
            settings=BootstrapSettings(200, 42, 0.8, 0.0, 1e-9),
        )


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
    raw_ams.write_text(
        "Year,AMS\n" + "\n".join(f"{1950 + i},{20 - log(-log((i - 0.35) / 50.3))}" for i in range(1, 51)),
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
    assert main(["fit", "--ams", str(raw_ams), "--settings", str(path), "--output", str(target)]) == 0
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
